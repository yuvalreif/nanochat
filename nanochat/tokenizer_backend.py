"""Build a Hugging Face tokenizer backend from rustbpe merge ranks."""

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE


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
