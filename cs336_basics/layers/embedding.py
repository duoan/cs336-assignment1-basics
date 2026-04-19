import einx
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Float[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        return einx.get_at("[vocab_size] d_model, ... -> ... d_model", self.weight, token_ids)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        """
        buffer memory: max_seq_len * d_k
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even for RoPE"

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 1. Compute frequencies: θ_i = 1 / theta^(2i/d_k), for i = 0..d_k/2-1
        # Shape: (d_k/2,)
        i = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)  # [0, 2, 4, ..., d_k-2]
        inv_freq = 1.0 / (theta ** (i / d_k))

        # 2. Compute angles for every (position, frequency) pair: m * θ_i
        # Shape: (max_seq_len, d_k/2)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("m,i->mi", positions, inv_freq)

        # 3. Precompute cos / sin tables and cache them as buffers.
        # Buffers move with model.to(device) but are not trainable parameters.
        # persistent=False: don't write to state_dict (can be recomputed from theta/d_k).
        cos = freqs.cos()  # (max_seq_len, d_k/2)
        sin = freqs.sin()  # (max_seq_len, d_k/2)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        # 4. Look up the precomputed cos / sin for each token position.
        # token_positions: (..., seq_len) -> indexed: (..., seq_len, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # pre-allocate memory
        # activation: (batch_size, seq_len, d_k)
        out = torch.empty_like(x)

        # x'_{2i}   = x_{2i} * cos - x_{2i+1} * sin
        # x'_{2i+1} = x_{2i} * sin + x_{2i+1} * cos
        out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos

        # Preserve input dtype (cos/sin are float32 for numerical stability).
        return out.to(x.dtype)
