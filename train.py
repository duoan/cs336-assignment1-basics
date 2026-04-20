import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.profiler import ProfilerActivity, profile
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from cs336_basics import checkpointing, layers
from cs336_basics.data import NumpyBinaryTokenDataset
from cs336_basics.functions import clip_gradient, cross_entropy
from cs336_basics.optimizers import AdamW, get_lr_cosine_schedule


def get_device():
    if torch.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_loss(model, inputs, targets):
    logits = model(inputs)
    return cross_entropy(logits, targets)


def train_step(model, optimizer, compute_loss_fn, all_params, max_l2_norm, inputs, targets):
    optimizer.zero_grad(set_to_none=True)
    loss = compute_loss_fn(model, inputs, targets)
    loss.backward()
    clip_gradient(all_params, max_l2_norm=max_l2_norm)
    optimizer.step()
    return loss.item()


@torch.inference_mode()
def eval_step(model, compute_loss_fn, inputs, targets):
    return compute_loss_fn(model, inputs, targets)


def evaluate(model, compute_loss_fn, dataloader, device):
    total_loss = 0.0
    count = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        total_loss += eval_step(model, compute_loss_fn, inputs, targets).item()
        count += 1
    return total_loss / count


def measure_flops(model, vocab_size, batch_size, context_length, device):
    from cs336_basics.functions import cross_entropy as ce_fn

    dummy_input = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    dummy_target = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, with_flops=True) as prof:
        logits = model(dummy_input)
        loss = ce_fn(logits, dummy_target)
        loss.backward()

    model.zero_grad(set_to_none=True)
    return sum(e.flops for e in prof.key_averages() if e.flops > 0)


@hydra.main(config_path="configs", config_name="default", version_base=None)
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = get_device()
    m = cfg.model
    t = cfg.training

    tokens_per_step = t.batch_size * m.context_length
    max_steps = t.total_tokens // tokens_per_step
    warmup_iters = int(max_steps * t.warmup_ratio)
    cosine_cycle_iters = max_steps

    print(
        f"tokens_per_step={tokens_per_step:,}, max_steps={max_steps:,}, "
        f"warmup_iters={warmup_iters:,}, total_tokens={t.total_tokens:,}"
    )

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model = layers.model.TransformerLanguageModel(
        vocab_size=m.vocab_size,
        context_length=m.context_length,
        d_model=m.d_model,
        num_layers=m.num_layers,
        num_heads=m.num_heads,
        d_ff=m.d_ff,
        rope_theta=m.rope_theta,
        device=device,
        dtype=getattr(torch, t.dtype),
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total_params={total_params:,}")

    gpu_peak_flops = t.get("gpu_peak_tflops", None)
    if gpu_peak_flops is not None:
        gpu_peak_flops = gpu_peak_flops * 1e12

    flops_per_step = measure_flops(model, m.vocab_size, t.batch_size, m.context_length, device)
    flops_per_step_approx = 6 * total_params * tokens_per_step
    print(f"flops_per_step (profiler): {flops_per_step:,.0f}")
    print(f"flops_per_step (6NBS approx): {flops_per_step_approx:,.0f}")
    print(f"ratio (profiler / 6NBS): {flops_per_step / flops_per_step_approx:.2f}")

    all_params = list(model.parameters())

    if cfg.training.get("compile", False):
        compile_mode = cfg.training.get("compile_mode", "default")
        if cfg.training.get("clear_compile_cache", False):
            import shutil
            cache_dir = os.path.join("/tmp", "torchinductor_" + os.environ.get("USER", ""))
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                print(f"Cleared compile cache: {cache_dir}")
        if compile_mode == "max-autotune":
            torch._inductor.config.coordinate_descent_tuning = True
        if torch.cuda.is_available():
            compute_loss_fn = torch.compile(compute_loss, mode=compile_mode)
        else:
            compute_loss_fn = torch.compile(compute_loss, backend="aot_eager")
    else:
        compute_loss_fn = compute_loss

    num_workers = t.get("num_workers", 4 if torch.cuda.is_available() else 0)
    pin_memory = torch.cuda.is_available()

    train_dataset = NumpyBinaryTokenDataset(cfg.data.train_path, context_length=m.context_length)
    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=max_steps * t.batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=t.batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_dataset = NumpyBinaryTokenDataset(cfg.data.valid_path, context_length=m.context_length)
    valid_sampler = RandomSampler(valid_dataset, replacement=True, num_samples=t.eval_steps * t.batch_size)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=t.batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    optimizer = AdamW(
        params=all_params, lr=t.lr,
        betas=tuple(t.betas), eps=t.eps, weight_decay=t.weight_decay,
    )

    max_l2_norm = t.max_l2_norm
    start_step = 0
    if cfg.resume:
        start_step = checkpointing.load_checkpoint(cfg.resume, model, optimizer)
        print(f"Resumed from {cfg.resume}, starting at step {start_step}")

    # --- training loop ---
    interval_start = time.perf_counter()
    interval_steps = 0
    data_time_accum = 0.0
    compute_time_accum = 0.0
    train_iter = iter(train_dataloader)

    for step in tqdm(range(start_step, max_steps)):
        data_tick = time.perf_counter()
        inputs, targets = next(train_iter)
        inputs, targets = inputs.to(device), targets.to(device)
        data_time_accum += time.perf_counter() - data_tick

        lr = get_lr_cosine_schedule(step, t.min_lr, t.max_lr, warmup_iters, cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        compute_tick = time.perf_counter()
        train_loss = train_step(
            model, optimizer, compute_loss_fn, all_params,
            max_l2_norm, inputs, targets,
        )
        compute_time_accum += time.perf_counter() - compute_tick
        interval_steps += 1

        if (step + 1) % t.log_interval == 0:
            now = time.perf_counter()
            elapsed = now - interval_start
            avg_dt = elapsed / interval_steps
            tokens_per_sec = tokens_per_step / avg_dt
            mfu_str = ""
            if gpu_peak_flops is not None:
                mfu = flops_per_step / (avg_dt * gpu_peak_flops) * 100
                mfu_str = f", mfu={mfu:.1f}%"
            avg_data_ms = data_time_accum / interval_steps * 1000
            avg_compute_ms = compute_time_accum / interval_steps * 1000
            tqdm.write(
                f"step {step + 1}, lr={lr:.6f}, train_loss={train_loss:.4f}, "
                f"tok/s={tokens_per_sec:,.0f}, avg_dt={avg_dt * 1000:.0f}ms{mfu_str}, "
                f"data={avg_data_ms:.0f}ms, compute={avg_compute_ms:.0f}ms"
            )
            interval_start = now
            data_time_accum = 0.0
            compute_time_accum = 0.0
            interval_steps = 0

        if (step + 1) % t.eval_interval == 0:
            val_loss = evaluate(model, compute_loss_fn, valid_dataloader, device)
            tqdm.write(f"step {step + 1}, val_loss={val_loss:.4f}")
            interval_start = time.perf_counter()
            interval_steps = 0
            data_time_accum = 0.0
            compute_time_accum = 0.0

        if (step + 1) % t.save_interval == 0:
            checkpointing.save_checkpoint(model, optimizer, step, f"./out/checkpoint_{step + 1}.pt")
            tqdm.write(f"step {step + 1}, checkpoint saved")
            interval_start = time.perf_counter()
            interval_steps = 0
            data_time_accum = 0.0
            compute_time_accum = 0.0


if __name__ == "__main__":
    train()
