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
            # activation: B x L x D
            x = x.to(dtype=torch.float32, non_blocking=True)

        # activation: ( 3 x B x L x D + B x L ) x 4 bytes ~= 12*B*L*D bytes
        # x: (B x L x D)
        # inv_rms: (B x L)
        # x_norm: (B x L x D)
        # out: (B x L x D)
        inv_rms = x.pow(2).mean(-1, keepdim=True).add_(self.eps).rsqrt_()
        x_norm = x * inv_rms
        out = x_norm * self.weight

        # cast to original dtype
        if original_dtype != torch.float32:
            out = out.to(dtype=original_dtype)

        return out
