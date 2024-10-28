import torch
from torch.nn import Module
from utils.logits_processor import LogitsProcessor, GreedyProcessor
import utils.printing as printing
from typing import List


@torch.no_grad()
def autoregressive_generate_encoder_decoder(
    inputs: List[int],
    model: Module,
    max_gen_len: int = 40,
    logits_processor: LogitsProcessor = GreedyProcessor(),
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    use_cache: bool = False,
    debug: bool = False,
) -> List[int]:
    """
    Generate text sequence autoregressively based on the input sequence.

    Args:
        inputs (List[int]): input sequence of batch size 1.
        model (Module): model to use for inference.
        max_gen_len (int): maximum length of the generated sequence.
        logits_processor (LogitsProcessor): logits processor for sampling.
        eos_token_id (int): end token id.
        pad_token_id (int): pad token id.
        use_cache (bool): whether to use cache.

    Returns:
        List[int]: generated sequence.

    Note:
        This generation methods only works for encoder-decoder models.
    """
    cache = None
    prompt = torch.tensor(inputs, dtype=torch.long, device=model.device).unsqueeze(0)
    prompt_len = len(inputs)

    decoder_start_token = model.config.decoder_start_token_id

    # prepare input tensor
    max_sequece_length = 1024
    total_len = min(max_sequece_length - prompt_len - 1, max_gen_len + 1)
    decoded_ids = torch.full(
        (1, total_len), pad_token_id, dtype=torch.long, device=model.device
    )
    decoded_ids[0, 0] = decoder_start_token

    list_tokens_id = (eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id])
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=model.device)

    for curr in range(1, total_len):
        o = model(
            input_ids=prompt,
            decoder_input_ids=decoded_ids[..., :curr],
            past_key_values=cache,
            use_cache=use_cache,
        )
        logits = o.logits[..., -1, :]  # [1, vocab_size]
        probs = logits_processor(logits)  # [1, vocab_size]
        x = logits_processor.sample(probs)  # [1, 1]
        decoded_ids[0, curr] = x
        cache = o.past_key_values

        # check for end token
        if torch.isin(x, stop_tokens):
            if debug:
                printing.end_token_found(curr)
            break

    return decoded_ids[0, : curr + 1].tolist()
