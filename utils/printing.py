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
