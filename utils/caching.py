from typing import Tuple
from torch import Tensor


def prune_cache(cache: Tuple[Tuple[Tensor, Tensor]], num_tokens_to_discard: int):
    """
    Prune the cache by removing the specified number of tokens from the end. This pruning works for most models.
    It works for models having past_key_values such as Tuple of tuple(Tensor) of length n_layers, with each tuple having 2 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)

    Args:
        cache Tuple(Tuple[Tensor, Tensor]): The KV cache to be pruned.
        num_tokens_to_discard (int): The number of tokens to discard from the end of the cache.

    Returns:
        Tuple[Tensor, Tensor]: The pruned KV cache.
    """
    if cache is None:
        return None

    new_cache = []
    for layer_cache in cache:
        if layer_cache is None:
            new_cache.append(None)
            continue

        key, value = layer_cache
        new_key = key[:, :, :-num_tokens_to_discard, :]
        new_value = value[:, :, :-num_tokens_to_discard, :]
        new_cache.append((new_key, new_value))

    return tuple(new_cache)
