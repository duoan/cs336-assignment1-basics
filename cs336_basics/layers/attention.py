import einx
import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.functions import scaled_dot_product_attention

from .embedding import RotaryPositionalEmbedding
from .linear import Linear


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divided by num_head"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device)

        # all parameter shape is (d_out, d_in)
        # parameters: d_model x d_model * 4
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "batch_size seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "batch_size seq_len d_model"]:
        """
        activation: 4BSD + 4BS + 12BSD + 4BHSS + 4BHSS + 4BSD => 4BS + 20BSD + 8BHSS
        """

        seq_len = x.size(1)

        # activation: x: 4BSD bytes

        # activation: 4BS bytes
        mask: Bool[Tensor, "batch_size seq_len"] = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        # activation: 12BSD bytes
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = einx.id("b s (h d) -> b h s d", q, h=self.num_heads)
        k = einx.id("b s (h d) -> b h s d", k, h=self.num_heads)
        v = einx.id("b s (h d) -> b h s d", v, h=self.num_heads)

        if hasattr(self, "rope") and token_positions is not None:
            assert token_positions.size(1) == seq_len
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # activation: 4BHSS + 4BHSS + 4BSD

        attention = scaled_dot_product_attention(q, k, v, mask)  # (b h s d)
        attention = einx.id("b h s d -> b s (h d)", attention)
        out = self.output_proj(attention)
        return out
