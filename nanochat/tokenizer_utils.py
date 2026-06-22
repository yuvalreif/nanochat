"""Tokenizer utility helpers kept out of the core tokenizer wrappers."""

import re

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE


NORMALIZED_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|\p{L}+|\p{N}{1,2}|[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
PUNCT_TO_WORD_BOUNDARY_PATTERN = re.compile(r"([^\w\s])(?=\w)")
WORD_TO_PUNCT_BOUNDARY_PATTERN = re.compile(r"(?<=\w)([^\w\s])")


def lowercase_standard_capitalization(text: str) -> str:
    def _replace(match: re.Match) -> str:
        word = match.group(0)
        if len(word) >= 2 and word[0].isupper() and word[1:].islower():
            return word.lower()
        return word

    return WORD_PATTERN.sub(_replace, text)


def separate_affix_punctuation(text: str) -> str:
    text = PUNCT_TO_WORD_BOUNDARY_PATTERN.sub(r"\1 ", text)
    text = WORD_TO_PUNCT_BOUNDARY_PATTERN.sub(r" \1", text)
    return text


def apply_space_cap_normalization(text: str) -> str:
    return lowercase_standard_capitalization(separate_affix_punctuation(text))


def _token_bytes_to_string(token_bytes: bytes) -> str:
    return token_bytes.decode("latin-1")


def _extract_vocab_and_merges(mergeable_ranks: dict[bytes, int]) -> tuple[dict[str, int], list[tuple[str, str]]]:
    vocab: dict[str, int] = {}
    merges_local: list[tuple[bytes, bytes, int]] = []
    for token, rank in mergeable_ranks.items():
        vocab[_token_bytes_to_string(token)] = rank
        if len(token) == 1:
            continue
        local: list[tuple[bytes, bytes, int]] = []
        for idx in range(1, len(token)):
            left = token[:idx]
            right = token[idx:]
            if left in mergeable_ranks and right in mergeable_ranks and (left + right) in mergeable_ranks:
                local.append((left, right, rank))
        local = sorted(local, key=lambda x: (mergeable_ranks[x[0]], mergeable_ranks[x[1]]))
        merges_local.extend(local)
    merges_local = sorted(merges_local, key=lambda x: x[2])
    merges = [
        (_token_bytes_to_string(left), _token_bytes_to_string(right))
        for (left, right, _) in merges_local
    ]
    return vocab, merges


def build_backend_tokenizer(
    *,
    mergeable_ranks: dict[bytes, int],
    pattern: str,
    special_tokens_in_id_order: list[str],
) -> HFTokenizer:
    vocab, merges = _extract_vocab_and_merges(mergeable_ranks)
    tokenizer = HFTokenizer(BPE(vocab, merges, fuse_unk=False))
    if hasattr(tokenizer.model, "ignore_merges"):
        tokenizer.model.ignore_merges = True
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(pattern), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(special_tokens_in_id_order)
    return tokenizer
