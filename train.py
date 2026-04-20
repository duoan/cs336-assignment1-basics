import argparse
from dataclasses import asdict, dataclass

import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from cs336_basics import checkpointing, layers
from cs336_basics.data import NumpyBinaryTokenDataset
from cs336_basics.functions import clip_gradient, cross_entropy
from cs336_basics.optimizers import AdamW, get_lr_cosine_schedule


@dataclass
class Config:
    # model parameters
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    device: str = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32

    # training parameters
    batch_size: int = 16
    max_steps: int = 80_000  # total_token / batch_size / context_length

    lr: float = 6e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 2_000
    cosine_cycle_iters: int = 80_000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume from")
    args = parser.parse_args()

    config = Config(
        vocab_size=10_000,
        context_length=256,
        d_model=512,
        d_ff=1344,
        num_layers=4,
        num_heads=16,
        rope_theta=10_000,
    )

    config_dict = asdict(config)

    train_dataset = NumpyBinaryTokenDataset(
        "./out/TinyStoriesV2-GPT4-train.ids.bin", context_length=config.context_length
    )
    valid_dataset = NumpyBinaryTokenDataset(
        "./out/TinyStoriesV2-GPT4-valid.ids.bin", context_length=config.context_length
    )

    train_sampler = RandomSampler(train_dataset, replacement=True, num_samples=config.max_steps * config.batch_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=0,
    )
    eval_steps = 50
    valid_sampler = RandomSampler(valid_dataset, replacement=True, num_samples=eval_steps * config.batch_size)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        sampler=valid_sampler,
        num_workers=0,
    )

    model = layers.model.TransformerLanguageModel(**config_dict)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params=:,}, \n{total_trainable_params=:,}")

    optimizer = AdamW(params=model.parameters(), **config_dict)

    start_step = 0
    if args.resume:
        start_step = checkpointing.load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume}, starting at step {start_step}")

    @torch.inference_mode()
    def eval():
        total_loss = 0.0
        count = 0
        for inputs, targets in valid_dataloader:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            total_loss += loss.item()
            count += 1
        return total_loss / count

    for step, (inputs, targets) in enumerate(tqdm(train_dataloader, initial=start_step, total=config.max_steps)):
        if step < start_step:
            continue
        lr = get_lr_cosine_schedule(step, config.min_lr, config.max_lr, config.warmup_iters, config.cosine_cycle_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        inputs = inputs.to(config.device)
        targets = targets.to(config.device)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        clip_gradient(model.parameters(), max_l2_norm=1.0)
        optimizer.step()

        train_loss = loss.item()
        if (step + 1) % 100 == 0:
            tqdm.write(f"step {step + 1}, lr={lr:.6f}, train_loss={train_loss:.4f}")
        if (step + 1) % 1_000 == 0:
            val_loss = eval()
            tqdm.write(f"step {step + 1}, val_loss={val_loss:.4f}")
        if (step + 1) % 10_000 == 0:
            checkpointing.save_checkpoint(model, optimizer, step, f"./out/checkpoint_{step + 1}.out")
            tqdm.write(f"step {step + 1}, checkpoint saved")
