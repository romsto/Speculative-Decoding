from typing import Tuple, Union
from torch import Tensor
from transformers.cache_utils import DynamicCache


def prune_cache(cache: Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache], num_tokens_to_discard: int):
    """
    Prune the cache by removing the specified number of tokens from the end.

    Args:
        cache (Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache]): The KV cache to be pruned.
        num_tokens_to_discard (int): The number of tokens to discard from the end of the cache.

    Returns:
        Union[Tuple[Tuple[Tensor, Tensor]], DynamicCache]: The pruned KV cache.
    """
    if cache is None:
        return None
    if isinstance(cache, tuple):
        return prune_tuple_cache(cache, num_tokens_to_discard)
    elif isinstance(cache, DynamicCache):
        return prune_dynamic_cache(cache, num_tokens_to_discard)
    else:
        raise ValueError("Unsupported cache type.")


def prune_tuple_cache(cache: Tuple[Tuple[Tensor, Tensor]], num_tokens_to_discard: int):
    """
    Prune the cache by removing the specified number of tokens from the end. This pruning works for most models.
    It works for models having past_key_values such as Tuple of tuple(Tensor) of length n_layers, containing 2 or 4 tensors of shape (batch_size, num_heads, sequence_length, embed_size_per_head)

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

        layer = []
        for i in range(len(layer_cache)):
            tensor = layer_cache[i]
            new_tensor = tensor[:, :, :-num_tokens_to_discard, :]
            layer.append(new_tensor)
        new_cache.append(tuple(layer))

    return tuple(new_cache)


def prune_dynamic_cache(cache: DynamicCache, num_tokens_to_discard: int):
    """
    Prune the cache by removing the specified number of tokens from the end. This pruning works for models using DynamicCache.

    Args:
        cache (DynamicCache): The KV cache to be pruned.
        num_tokens_to_discard (int): The number of tokens to discard from the end of the cache.

    Returns:
        DynamicCache: The pruned KV cache. (same instance as the input cache, but modified in place)
    """
    if cache is None:
        return None

    for layer in range(len(cache)):
        cache.key_cache[layer] = cache.key_cache[layer][:, :, :-num_tokens_to_discard, :]
        cache.value_cache[layer] = cache.value_cache[layer][:, :, :-num_tokens_to_discard, :]
    cache._seen_tokens -= num_tokens_to_discard

    return cache
