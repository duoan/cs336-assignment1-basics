import hydra
import torch
from omegaconf import DictConfig, OmegaConf
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


@hydra.main(config_path="configs", config_name="default", version_base=None)
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    device = get_device()
    m = cfg.model
    t = cfg.training

    model_kwargs = dict(
        vocab_size=m.vocab_size,
        context_length=m.context_length,
        d_model=m.d_model,
        num_layers=m.num_layers,
        num_heads=m.num_heads,
        d_ff=m.d_ff,
        rope_theta=m.rope_theta,
        device=device,
        dtype=torch.float32,
    )

    train_dataset = NumpyBinaryTokenDataset(cfg.data.train_path, context_length=m.context_length)
    valid_dataset = NumpyBinaryTokenDataset(cfg.data.valid_path, context_length=m.context_length)

    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=t.max_steps * t.batch_size)
    train_dataloader = DataLoader(train_dataset, batch_size=t.batch_size, sampler=train_sampler, num_workers=0)

    valid_sampler = RandomSampler(valid_dataset, replacement=True, num_samples=t.eval_steps * t.batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=t.batch_size, sampler=valid_sampler, num_workers=0)

    model = layers.model.TransformerLanguageModel(**model_kwargs)

    if torch.mps.is_available():
        model = torch.compile(model, backend="aot_eager")
    elif torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)
    else:
        model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"total_params={total_params:,}")

    optimizer = AdamW(
        params=model.parameters(),
        lr=t.lr,
        betas=tuple(t.betas),
        eps=t.eps,
        weight_decay=t.weight_decay,
    )

    start_step = 0
    if cfg.resume:
        start_step = checkpointing.load_checkpoint(cfg.resume, model, optimizer)
        print(f"Resumed from {cfg.resume}, starting at step {start_step}")

    @torch.inference_mode()
    def evaluate():
        total_loss = 0.0
        count = 0
        for inputs, targets in valid_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            total_loss += loss.item()
            count += 1
        return total_loss / count

    for step, (inputs, targets) in enumerate(tqdm(train_dataloader, initial=start_step, total=t.max_steps)):
        if step < start_step:
            continue

        lr = get_lr_cosine_schedule(step, t.min_lr, t.max_lr, t.warmup_iters, t.cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        clip_gradient(model.parameters(), max_l2_norm=t.max_l2_norm)
        optimizer.step()

        train_loss = loss.item()
        if (step + 1) % t.log_interval == 0:
            tqdm.write(f"step {step + 1}, lr={lr:.6f}, train_loss={train_loss:.4f}")
        if (step + 1) % t.eval_interval == 0:
            val_loss = evaluate()
            tqdm.write(f"step {step + 1}, val_loss={val_loss:.4f}")
        if (step + 1) % t.save_interval == 0:
            checkpointing.save_checkpoint(model, optimizer, step, f"./out/checkpoint_{step + 1}.pt")
            tqdm.write(f"step {step + 1}, checkpoint saved")


if __name__ == "__main__":
    train()
