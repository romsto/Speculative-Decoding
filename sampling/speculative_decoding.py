import torch
from torch import Tensor
from torch.nn import Module
from transformers import T5ForConditionalGeneration
from typing import Tuple
from utils.sampling import norm_logits, sample
from utils.generation import forward, decoder_start_token, is_decoder_only
from utils.caching import prune_cache_general, prune_cache_t5
from utils.printing import token_ids_to_string
from termcolor import colored


def max_fn(x: Tensor) -> Tensor:
    """
    Max function.
        x: input tensor.
    Returns:
        tensor norm(max(0, x)).
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / x_max_sum


@torch.no_grad()
def speculative_decoding(
    input_ids: Tensor,
    drafter: Module,
    target: Module,
    min_len: int = 20,
    gamma: int = 5,
    temperature: float = 1,
    top_k: int = 0,
    top_p: float = 0,
    skip_sample_adjustment: bool = False,
    end_token_id: int = 1,
    use_cache: bool = True,
    target_first:bool = True,
    debug: bool = False,
    debug_tokenizer=None,
) -> Tuple[Tensor, float]:
    """
    Implementation of Speculative Decoding. (https://arxiv.org/pdf/2211.17192.pdf)
    Sampling method depends on the top_k and top_p parameters. If top_k = 0, then greedy decoding is used. Otherwise, multinomial sampling is used.

    Args:
        input_ids: input/prefix sequence.
        drafter: drafter model.
        target: target model.
        min_len: minimum length of the output sequence. The maximum length will be of length min_len + gamma + 1.
        gamma: gamma, number of drafts generated by the drafter.
        temperature: temperature for sampling.
        top_k: top_k for sampling.
        top_p: top_p for sampling.
        skip_sample_adjustment: whether to skip the sample adjustment step when some drafts are discarded.
        end_token_id: end token id to stop generating.
        use_cache: whether to use KV cache.
        target_first: whether to do one target pass before running the speculative algorithm.
        debug: debug mode.
        debug_tokenizer: tokenizer for debug mode. tokens will be output in string format.
    
    Returns:
        generated sequence, acceptance rate (number of accepted drafts divided by the number of total drafts).
    """

    drafter_cache = None
    target_cache = None
    prune_cache_drafter = prune_cache_general if not isinstance(drafter, T5ForConditionalGeneration) else prune_cache_t5
    prune_cache_target = prune_cache_general if not isinstance(target, T5ForConditionalGeneration) else prune_cache_t5

    seq_len = 0
    total_accept = 0
    number_of_trials = 0
    device = input_ids.device

    target_decoder_only = is_decoder_only(target)
    drafter_decoder_only = is_decoder_only(drafter)

    target_decoder_start_id = decoder_start_token(target, device)
    drafter_decoder_start_id = decoder_start_token(drafter, device)

    decoded_input_ids = torch.tensor([[]], dtype=torch.int64).to(device)

    if target_first:
        # run the target model before the speculative algorithm. Allows to prefill the kvcache and get a first token.
        Mp = forward(
            target,
            input_ids=input_ids,
            decoded_ids=torch.cat((target_decoder_start_id, decoded_input_ids), dim=-1),
            cache=target_cache,
            use_cache=use_cache,
            decoder_only=target_decoder_only,
        )
        target_cache = Mp.past_key_values
        p_p = norm_logits(Mp.logits[..., -1, :], temperature, top_k, top_p)
        t = sample(p_p)
        decoded_input_ids = torch.cat((decoded_input_ids, t), dim=-1)
        seq_len += 1
        if debug:
            print(f"{colored('Initiale Step', on_color='on_dark_grey', color='white')} 1 token:")
            print(colored(token_ids_to_string(t[0], debug_tokenizer), 'blue'))

    while seq_len < min_len:
        drafts_probs = []
        drafter_decoded_input_ids = torch.tensor([[]], dtype=torch.int64).to(device)

        # generate gamma drafts
        for _ in range(gamma):
            Mq = forward(
                drafter,
                input_ids=input_ids,
                decoded_ids=torch.cat(
                    (
                        drafter_decoder_start_id,
                        decoded_input_ids,
                        drafter_decoded_input_ids,
                    ),
                    dim=-1,
                ),
                cache=drafter_cache,
                use_cache=use_cache,
                decoder_only=drafter_decoder_only,
            )
            draft_logits = Mq.logits[..., -1, :]
            drafter_cache = Mq.past_key_values
            draft_logits = norm_logits(draft_logits, temperature, top_k, top_p)
            xi = sample(draft_logits)  # x_i ~ q_i(x)
            drafts_probs.append(draft_logits)  # q_i(x)
            drafter_decoded_input_ids = torch.cat(
                (drafter_decoded_input_ids, xi), dim=-1
            )
            number_of_trials += 1

        # run target model on drafts and get logits of the previous tokens plus one more token
        Mp = forward(
            target,
            input_ids=input_ids,
            decoded_ids=torch.cat(
                (target_decoder_start_id, decoded_input_ids, drafter_decoded_input_ids),
                dim=-1,
            ),
            cache=target_cache,
            use_cache=use_cache,
            decoder_only=target_decoder_only,
        )
        target_cache = Mp.past_key_values
        # apply normalization
        vocabulary_limit = drafts_probs[0].shape[1]

        p = [
            norm_logits(
                Mp.logits[..., -gamma - 1 + i, :vocabulary_limit],
                temperature,
                top_k,
                top_p,
            )
            for i in range(gamma)
        ]

        # compute the last accepted draft position (rejection sampling)
        r = torch.rand(gamma, device=device)
        # n = \min(\left\{i - 1 \, \middle| \, 1 \leq i \leq \gamma, \, r_i > \frac{p_i(x)}{q_i(x)} \right\} \cup \{\gamma\})
        n = min(
            [
                i
                for i in range(gamma)
                if r[i] > p[i][..., drafter_decoded_input_ids[0, i]] / drafts_probs[i][..., drafter_decoded_input_ids[0, i]]
            ]
            + [gamma]
        )

        # if the end token is in drafts(0, n), stop and return the sequence with the drafts until end token
        if n == gamma:
            drafts = drafter_decoded_input_ids[0]
        else:
            drafts = drafter_decoded_input_ids[0, : -gamma + n]
        if torch.any(drafts == end_token_id):
            positions = torch.where(drafts == end_token_id)[0]
            if positions.shape[0] > 1:
                position = positions[0].item()
            else:
                position = positions.item()
            total_accept += position + 1
            decoded_input_ids = torch.cat(
                (
                    drafter_decoder_start_id,
                    decoded_input_ids,
                    drafts[:position].unsqueeze(0),
                ),
                dim=-1,
            )
            if debug:
                print(colored(f"End token found at position {position}", "red"))
            return decoded_input_ids, total_accept / number_of_trials

        # prune the cache
        if use_cache:
            drafter_cache = prune_cache_drafter(drafter_cache, gamma - n)
            target_cache = prune_cache_target(target_cache, gamma - n + 1)

        total_accept += n

        # adjut the distribution from Mp
        p_p = norm_logits(
            Mp.logits[..., -(gamma - n + 1), :vocabulary_limit], temperature, top_k, top_p
        )
        if n < gamma and not skip_sample_adjustment:
            p_p = max_fn(p[n] - drafts_probs[n])

        # sample from the adjusted distribution
        t = sample(p_p)

        # cat seq with n drafts and the sampled token
        generated = torch.cat((drafts.unsqueeze(0), t), dim=-1)
        if debug:
            print(f"{colored('Speculative Step', on_color='on_dark_grey', color='white')} {n} draft{'s' if n > 1 else ''} + 1 token:")
            print(token_ids_to_string(decoded_input_ids[0], debug_tokenizer), end=' ')
            print(colored(token_ids_to_string(drafts, debug_tokenizer), 'green'), end=(' ' if n > 0 else ''))
            print(colored(token_ids_to_string(drafter_decoded_input_ids[0, n:], debug_tokenizer), 'red'), end=(' ' if n < gamma else ''))
            print(colored(token_ids_to_string(t[0], debug_tokenizer), 'blue'))

        decoded_input_ids = torch.cat((decoded_input_ids, generated), dim=-1)
        seq_len = decoded_input_ids.shape[1]

        # stop if the sampled token is the end token
        if t.item() == end_token_id:
            break

    return decoded_input_ids, total_accept / number_of_trials
