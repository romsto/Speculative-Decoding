from torch import Tensor, tensor, cat
import torch
from torch.nn import Module
from typing import Optional


def forward(
    model: Module,
    input_ids: Tensor,
    decoded_ids: Optional[Tensor] = None,
    cache=None,
    use_cache: bool = False,
    decoder_only: bool = None,
):
    """General forward function for models with encoder-decoder or decoder only architecture.

    Args:
        model (Module): Model to forward pass.
        input_ids (Tensor): Input tensor.
        decoded_ids (Tensor, optional): Decoded tensor. Defaults to None.
        cache ([type], optional): Cache for faster decoding. Defaults to None.
        use_cache (bool, optional): Whether to use cache. Defaults to False.
        decoder_only (bool, optional): Whether the model is decoder only. If not provided, it will automatically be deduced.

    Returns:
        Tensor: Model output.
    """
    if decoder_only is None:
        decoder_only = is_decoder_only(model)

    if decoder_only:
        if decoded_ids is not None:
            input_ids = cat((input_ids, decoded_ids), dim=1)
        return model(input_ids, past_key_values=cache, use_cache=use_cache)
    else:
        if decoded_ids is None:
            decoded_ids = tensor([[model.config.decoder_start_token_id]]).to(
                input_ids.device
            )
        return model(
            input_ids,
            decoder_input_ids=decoded_ids,
            past_key_values=cache,
            use_cache=use_cache,
        )


def is_decoder_only(model: Module):
    """Check if the model is decoder only.

    Args:
        model (Module): Model.

    Returns:
        bool: True if decoder only, False otherwise.
    """
    return not hasattr(model, "encoder") or (
        not hasattr(model, "encoder") and not hasattr(model, "decoder")
    )


def decoder_start_token(model: Module, device: str = None):
    """Get the decoder start token for a model.

    Args:
        model (Module): Model.
        device (str, optional): Device. Defaults to None.

    Returns:
        Tensor: Encoded start token.
    """
    if device is None:
        device = model.device

    start_token_id = None
    if hasattr(model, "config") and hasattr(model.config, "decoder_start_token_id"):
        start_token_id = model.config.decoder_start_token_id

    return (
        tensor([[start_token_id]], dtype=torch.int64).to(device)
        if start_token_id is not None
        else tensor([[]], dtype=torch.int64).to(device)
    )
