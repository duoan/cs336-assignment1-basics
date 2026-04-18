import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        original_dtype = x.dtype

        # upcast to float32 to prevent overflow
        if original_dtype != torch.float32:
            x = x.to(dtype=torch.float32, non_blocking=True)

        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = x * rms * self.weight

        # cast to original dtype
        if original_dtype != torch.float32:
            out = out.to(dtype=original_dtype)

        return out
