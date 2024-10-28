from math import inf
import torch
from torch.nn import Module
from utils.logits_processor import LogitsProcessor, GreedyProcessor
import utils.printing as printing
from typing import List


@torch.no_grad()
def autoregressive_generate(
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
        This generation methods only works for decoder-only models.
    """
    cache = None
    prompt_len = len(inputs)
    # prepare input tensor
    max_seq_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else (model.config.max_context_length if hasattr(model.config, 'max_context_length') else 1024)
    total_len = min(max_seq_length, prompt_len + max_gen_len)
    input_ids = torch.full((1, total_len), pad_token_id, dtype=torch.long, device=model.device)
    input_ids[0, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=model.device)

    list_tokens_id = (
        eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]
    )
    stop_tokens = torch.tensor(list_tokens_id, dtype=torch.long, device=model.device)

    for curr in range(prompt_len, total_len):
        o = model(input_ids[..., :curr], past_key_values=cache, use_cache=use_cache)
        logits = o.logits[..., -1, :]  # [1, vocab_size]
        probs = logits_processor(logits)  # [1, vocab_size]
        x = logits_processor.sample(probs)  # [1, 1]
        input_ids[0, curr] = x
        cache = o.past_key_values

        # check for end token
        if torch.isin(x, stop_tokens):
            if debug:
                printing.end_token_found(curr)
            break

    return input_ids[0, prompt_len : curr + 1].tolist()


@torch.no_grad()
def beam_search_generate(
    inputs: List[int],
    model: Module,
    max_gen_len: int = 40,
    num_beams: int = 4,
    top_k: int = 3,
    min_length: float = 5.0,
    alpha: float = 1.2,
    eos_tokens_id: int | List[int] = 1,
    pad_token_id: int = 0,
    debug: bool = False,
    tokenizer=None,
) -> List[int]:
    """
    Generate text sequence using beam search based on the input sequence.

    Args:
        inputs (List[int]): input sequence of batch size 1.
        model (Module): model to use for inference.
        max_gen_len (int): maximum length of the generated sequence.
        num_beams (int): number of beams.
        top_k (int): number of top k to consider at each beam.
        min_length (float): length penalty.
        alpha (float): alpha parameter of beam search decoding.
        eos_token_id (int): end token id.
        pad_token_id (int): pad token id.
        debug (bool): whether to print debug information.
        tokenizer: tokenizer for debug.

    Returns:
        List[int]: generated sequence.

    Note:
        This generation methods only works for decoder-only models.
        Cache is not available yet.
    """

    def _length_penalty_fn(length, alpha, min_length):
        return ((min_length + length) / (min_length + 1)) ** alpha

    prompt_len = len(inputs)
    max_seq_length = model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else (model.config.max_context_length if hasattr(model.config, 'max_context_length') else 1024)

    assert prompt_len < max_seq_length, "Prompt length exceeds maximum sequence length."

    total_len = min(max_seq_length, prompt_len + max_gen_len)
    input_ids = torch.full((num_beams, total_len), pad_token_id, dtype=torch.long, device=model.device)
    input_ids[:, :prompt_len] = torch.tensor(inputs, dtype=torch.long, device=model.device)
    probs = torch.full((num_beams, total_len), torch.finfo(torch.float).min, dtype=torch.float, device=model.device)
    beams_probs = torch.full((num_beams,), torch.finfo(torch.float).min, dtype=torch.float, device=model.device)
    last_indexes = torch.full((num_beams,), -1, dtype=torch.long, device=model.device)

    stop_tokens = torch.tensor((eos_tokens_id if isinstance(eos_tokens_id, list) else [eos_tokens_id]), dtype=torch.long, device=model.device)

    # prefill
    probs[:, :prompt_len] = 1.0
    beams_probs[:] = 1.0
    o = model(input_ids[:, :prompt_len])
    curr_prob = torch.nn.functional.log_softmax(o.logits[0, -1, :], dim=-1)
    top_probs, top_tokens = torch.topk(curr_prob, num_beams, dim=-1)
    input_ids[:, prompt_len] = top_tokens
    probs[:, prompt_len] = probs[:, prompt_len - 1] + top_probs
    beams_probs[:] = probs[:, prompt_len] / _length_penalty_fn(1, alpha, min_length)
    
    for curr in range(prompt_len + 1, total_len):
        o = model(input_ids[:, :curr])
        logits = o.logits[:, -1, :]
        probs_curr = torch.nn.functional.log_softmax(logits, dim=-1)
        top_probs, top_tokens = torch.topk(probs_curr, top_k, dim=-1)
        possibilities = []
        for beam in range(num_beams):
            if last_indexes[beam] != -1:
                prob_vec = probs[beam].detach().clone()
                input_vec = input_ids[beam].detach().clone()
                possibilities.append(
                    (beams_probs[beam], input_vec, prob_vec, last_indexes[beam])
                )
                continue
            
            for possibility in range(top_k):
                new_prob = probs[beam, curr - 1] + top_probs[beam, possibility]
                lp = _length_penalty_fn(curr - prompt_len, alpha, min_length)
                prob_vec = probs[beam].detach().clone()
                prob_vec[curr] = new_prob
                input_vec = input_ids[beam].detach().clone()
                input_vec[curr] = top_tokens[beam, possibility]
                last_token_idx = -1
                if torch.isin(input_vec[curr], stop_tokens) or input_vec[curr] == pad_token_id:
                    last_token_idx = curr
                    
                already_in = False
                for p in possibilities:
                    if torch.equal(p[1], input_vec):
                        already_in = True
                        break
                if not already_in:
                    possibilities.append((new_prob / (lp if lp != 0 else 1), input_vec, prob_vec, last_token_idx))

        possibilities.sort(key=lambda x: x[0], reverse=True)

        if debug:
            printing.beam_search_step(possibilities, curr, tokenizer)

        possibilities = possibilities[:num_beams]

        for beam in range(num_beams):
            beams_probs[beam] = possibilities[beam][0]
            input_ids[beam] = possibilities[beam][1]
            probs[beam] = possibilities[beam][2]
            last_indexes[beam] = possibilities[beam][3]

        if torch.all(last_indexes != -1):
            if debug:
                printing.end_token_found(curr)
            break

    last_indexes[last_indexes == -1] = total_len - 1

    return input_ids[0, prompt_len : last_indexes[0] + 1].tolist()
