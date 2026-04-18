import einx
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from .attention import MultiHeadSelfAttention
from .feed_forward import PositionWiseFeedForwardNetwork
from .norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # layer1, x + attn(ln1(x))
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )

        # layer2, x + ffn(ln2(x))
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = PositionWiseFeedForwardNetwork(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        seq_len = x.size(1)
        token_positions = einx.id("seq_len -> 1 seq_len", torch.arange(seq_len))
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
