import einx
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from cs336_basics.functions import silu

from .linear import Linear


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w2 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.active = silu

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        """
        assume d_ff = 8/3 * d_model
        activation: 4xBxSxD + 4xBxSxDx8/3 + 4xBxSxDx8/3 + 4xBxSxDx8/3 = 4xBxSxDx(1+8)= 36BSD bytes
        """
        x_w1 = self.active(self.w1(x))
        x_w3 = self.w3(x)
        x_w1_w3 = einx.dot("... d_ff, ... d_ff -> ... d_ff", x_w1, x_w3)
        out = self.w2(x_w1_w3)
        return out
