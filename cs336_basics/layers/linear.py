import math

import einx
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class Linear(nn.Module):
    def __init__(self, d_out: int, d_in: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((d_out, d_in), device=device, dtype=dtype))
        self.reset_parameters(d_out, d_in)

    def reset_parameters(self, d_out: int, d_in: int) -> None:
        std = math.sqrt(2.0 / (d_out + d_in))
        nn.init.trunc_normal_(
            self.weight,
            mean=0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )

    def forward(self, in_features: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        return einx.dot("d_out d_in, ... d_in -> ... d_out", self.weight, in_features)
