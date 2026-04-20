from typing import Any

import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor

from .embedding import TokenEmbedding
from .linear import Linear
from .norm import RMSNorm
from .transformer import TransformerBlock


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.token_embeddings = TokenEmbedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(vocab_size, d_model, device=device, dtype=dtype)

    def forward(
        self,
        in_indices: Int[Tensor, "batch_size sequence_length"],
    ) -> Float[Tensor, "batch_size vocab_size"]:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
