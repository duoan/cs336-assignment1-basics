import math
from collections.abc import Callable
from typing import Any

import torch


def get_lr_cosine_schedule(
    t: int,
    lr_min: float,
    lr_max: float,
    t_w: int,
    t_c: int,
) -> float:
    if t < t_w:
        return t / t_w * lr_max
    elif t >= t_w and t <= t_c:
        return lr_min + 0.5 * (1 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        *,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        **kwargs: Any,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure
        for group in self.param_groups:
            lr: float = group["lr"]
            beta_1, beta_2 = group["betas"]
            weight_decay: float = group["weight_decay"]
            eps: float = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or 1.

                if "m" not in state:
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)

                m = state["m"]
                v = state["v"]
                grad = p.grad.data  # Get the gradient of loss with respect to p.

                lr_t = lr * math.sqrt(1.0 - beta_2**t) / (1.0 - beta_1**t)  # Compute adjusted learning rate for iteration t
                p.data -= lr * weight_decay * p.data  # Apply weight decay
                m.mul_(beta_1).add_(grad, alpha=1.0 - beta_1)  # Update the first moment estimate (in-place)
                v.mul_(beta_2).addcmul_(grad, grad, value=1.0 - beta_2)  # Update the second moment estimate (in-place)
                p.data -= lr_t * m / (v.sqrt() + eps)  # Apply moment-adjusted weight updates
                state["t"] = t + 1

        return loss


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1
        return loss


if __name__ == "__main__":
    torch.manual_seed(42)
    for lr in [1, 1e-1, 1e-2, 1e-3]:
        weights = torch.nn.Parameter(5 * torch.randn(10, 10))
        opt = AdamW([weights], lr=lr)

        for t in range(10):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean()  # Compute a scalar loss value.

            print(loss.cpu().item())
            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()

        print("=" * 30)
