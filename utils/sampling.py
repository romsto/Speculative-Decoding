import torch
from torch import Tensor
from torch.nn import functional as F


def topp(logits: Tensor, top_p: float) -> Tensor:
    """
    Filter a distribution of logits using top-p filtering.

    Args:
        logits: logits distribution.
        top_p: keep top tokens with cumulative probability <= top_p.

    Returns:
        logits distribution.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    to_keep = cumulative_probs <= top_p
    indices_to_keep = sorted_indices[to_keep]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[indices_to_keep] = False
    logits[mask] = -1e20
    return logits


def top_k_top_p_filter(logits: Tensor, top_k: int = 0, top_p: float = 0.0) -> Tensor:
    """
    Filter a distribution of logits using top-k and top-p filtering.

    Args:
        logits: logits distribution.
        top_k: keep top k tokens with highest probability.
        top_p: keep top tokens with cumulative probability <= top_p.

    Returns:
        logits distribution.
    """
    if top_k > 0:
        top_k_tokens = torch.topk(logits, top_k, dim=-1).values
        min_logits = torch.min(top_k_tokens)
        logits = torch.where(
            logits < min_logits, torch.ones_like(logits) * -1e20, logits
        )
    if top_p > 0.0:
        logits = topp(logits, top_p)
    return logits


def norm_logits(
    logits: Tensor, temperature: float, top_k: float, top_p: float
) -> Tensor:
    """
    Normalize logits.

    Args:
        logits: logits distribution.
        temperature: temperature.
        top_k: keep top k tokens with highest probability.
        top_p: keep top tokens with cumulative probability <= top_p.

    Returns:
        normalized logits distribution.
    """
    if temperature != 1.0:
        logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    return probs


def sample(logits: Tensor, num_samples: int = 1) -> Tensor:
    """
    Sample from a distribution.

    Args:
        logits: logits distribution.
        num_samples: number of samples.

    Returns:
        sampled tokens.
    """
    idx_next = torch.multinomial(logits, num_samples=num_samples)
    return idx_next


def greedy(logits: Tensor) -> Tensor:
    """
    Greedy sample from a distribution.

    Args:
        logits: logits distribution.

    Returns:
        sampled tokens.
    """
    idx_next = torch.argmax(logits)
    return idx_next.unsqueeze(0).unsqueeze(0).to(logits.device)
