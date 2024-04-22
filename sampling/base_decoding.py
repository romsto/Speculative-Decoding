from json import decoder
import torch
from torch import Tensor
from torch.nn import Module
from utils.sampling import norm_logits, sample
from utils.generation import forward, decoder_start_token, is_decoder_only


@torch.no_grad()
def autoregressive_decoding(
    input_ids: Tensor,
    model: Module,
    max_len: int = 40,
    temperature: float = 1,
    top_k: int = 0,
    top_p: float = 0,
    end_token_id: int = 1,
    use_cache: bool = True,
) -> Tensor:
    """
    Autoregressive decoding. Sample method depends on the top_k and top_p parameters. If top_k = 1, then greedy decoding is used. Otherwise, multinomial sampling is used.

    Args:
        input_ids: input/prefix sequence.
        model: model.
        max_len: maximum length of the output sequence.
        temperature: temperature for sampling.
        top_k: top_k for sampling.
        top_p: top_p for sampling.
        end_token_id: end token id to stop generating.
        use_cache: whether to use KV cache.
        
    Returns:
        generated sequence.
    """
    seq_len = 0
    cache = None

    decoder_only = is_decoder_only(model)
    decoded_input_ids = decoder_start_token(model)

    while seq_len < max_len:
        output = forward(
            model,
            input_ids=input_ids,
            decoded_ids=decoded_input_ids,
            cache=cache,
            use_cache=use_cache,
            decoder_only=decoder_only,
        )
        logits = output.logits[..., -1, :]
        cache = output.past_key_values
        logits = norm_logits(logits, temperature, top_k, top_p)
        xi = sample(logits)
        decoded_input_ids = torch.cat((decoded_input_ids, xi), dim=-1)
        seq_len += 1
        if xi == end_token_id:
            break

    return decoded_input_ids
