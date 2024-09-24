from typing import List, Tuple
from termcolor import colored
from torch import Tensor


def token_ids_to_string(token_ids, tokenizer):
    """Convert token ids to string.

    Args:
        token_ids (List[int]): List of token ids.
        tokenizer (Tokenizer): Tokenizer.

    Returns:
        str: String representation of token ids.
    """
    strings = tokenizer.convert_ids_to_tokens(token_ids)
    return " ".join(strings)


def end_token_found(location: int):
    print(colored(f"End token found at position {location}", "red"))


def initial_step(token: Tensor, tokenizer):
    print(f"{colored('Initiale Step', on_color='on_dark_grey', color='white')} 1 token:")
    print(colored(token_ids_to_string(token, tokenizer), "blue"))


def speculative_step(
    tokenizer,
    current_inputs: Tensor,
    inputs: Tensor,
    n: int,
    prompt_end: int,
    current_position: int,
    corrected_gamma: int,
):
    print(f"{colored('Speculative Step', on_color='on_dark_grey', color='white')} {n} draft{'s' if n > 1 else ''} + 1 token:")
    print(token_ids_to_string(inputs[0, prompt_end:current_position], tokenizer), end=" ")
    print(colored(token_ids_to_string(inputs[0, current_position : current_position + n], tokenizer), "green"), end=(" " if n > 0 else ""))
    print(colored(token_ids_to_string(current_inputs[0, current_position + n : current_position + corrected_gamma], tokenizer), "red"), end=(" " if n < corrected_gamma else ""))
    print(colored(token_ids_to_string(inputs[..., current_position + n], tokenizer), "blue"))


def beam_search_step(possibilities: List[Tuple[float, Tensor, Tensor]], current_position: int, tokenizer):
    print(f"{colored('Beam Search Step', on_color='on_dark_grey', color='white')} Token {current_position}:")
    
    for i, (prob, tokens, _) in enumerate(possibilities):
        print(f"{i+1}. {prob:.3f}\t{token_ids_to_string(tokens[:current_position - 1], tokenizer)} {colored(token_ids_to_string(tokens[current_position - 1:current_position], tokenizer), 'blue')}")
