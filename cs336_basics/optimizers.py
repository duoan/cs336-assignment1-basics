import math
from collections.abc import Callable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        *,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
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

                m = state.get("m", torch.nn.Parameter(torch.zeros_like(p, device=p.device, dtype=torch.float32)))
                v = state.get("v", torch.nn.Parameter(torch.zeros_like(p, device=p.device, dtype=torch.float32)))

                lr_t = (
                    lr * math.sqrt(1.0 - beta_2**t) / (1.0 - beta_1**t)
                )  # Compute adjusted learning rate for iteration t
                p.data -= lr * weight_decay * p.data  # Apply weight decay
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                m = beta_1 * m + (1.0 - beta_1) * grad  # Update the first moment estimate
                v = beta_2 * v + (1.0 - beta_2) * grad**2  # Update the second moment estimate
                p.data -= lr_t / torch.sqrt(v + eps) * m  # Apply mement-adjusted weight updates
                state["m"] = m  # store the first moment
                state["v"] = v  # store the second moment
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
