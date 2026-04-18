import einx
import torch
from jaxtyping import Bool, Float
from torch import Tensor


def silu(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return x / (1 + torch.exp(-x))


def softmax(x: Float[Tensor, " ..."], dim: int = -1) -> Float[Tensor, " ..."]:
    x_max = x.max(dim=dim, keepdim=True).values
    x = x - x_max
    x = x.exp()
    return x / x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.size(-1)
    scores: Float[Tensor, " ... queries keys"] = (
        einx.dot(" ... queries d_k,  ... keys d_k -> ... queries keys", Q, K) * d_k**-0.5
    )

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    scores = softmax(scores)
    attention = einx.dot(" ... queries keys,  ... keys d_v -> ... queries d_v", scores, V)
    return attention
