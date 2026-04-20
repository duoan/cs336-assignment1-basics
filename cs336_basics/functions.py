import einx
import torch
from jaxtyping import Bool, Float, Int
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
    """
    activation: 4BHSS + 4BHSS + 4BSD
    """
    d_k = Q.size(-1)

    # activation: (B, H, S, S) -> 4BHSS bytes
    scores: Float[Tensor, " ... queries keys"] = (
        einx.dot(" ... queries d_k,  ... keys d_k -> ... queries keys", Q, K) * d_k**-0.5
    )

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # activation: (B, H, S, S) -> 4BHSS bytes
    probs = softmax(scores)
    # activation" (B, H, S, D) -> 4BSD bytes
    attention = einx.dot(" ... queries keys,  ... keys d_v -> ... queries d_v", probs, V)
    return attention


def cross_entropy(
    logits: Float[Tensor, " ... vocab_size"],  # current sequence logits, next token prediction o_i
    targets: Int[Tensor, " ..."],  # next token actual tokens x_(i+1)
) -> Float[Tensor, " ..."]:
    """Calculate the cross entropy loss for a given batch sequence next token prediction"""
    vocab_size = logits.size(-1)
    logits = logits.view(-1, vocab_size)

    # log(softmax(x_i)) = log(exp(x_i - max) / sum(exp(x_j - max)))
    #                = (x_i - max) - log(sum(exp(x_j - max)))
    logits = logits - logits.max(dim=-1)[0].unsqueeze(-1)
    log_probs = logits - torch.exp(logits).sum(dim=-1, keepdim=True).log()

    targets = targets.view(-1)

    return -log_probs[range(len(targets)), targets].mean()


@torch.no_grad()
def clip_gradient(parameters: list[Float[Tensor, " ..."]], max_l2_norm=1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach().float(), 2) for g in grads]), 2
    )
    if total_norm <= max_l2_norm:
        return total_norm.item()

    clip_coef = max_l2_norm / (total_norm + 1e-6)
    for g in grads:
        g.detach().mul_(clip_coef)
