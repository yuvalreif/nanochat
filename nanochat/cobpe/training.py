"""CoBPE tokenizer training helpers."""

import json
import os
import re
import rustbpe
import tiktoken

from nanochat.cobpe.tokenizer import build_cobpe_metadata
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS
COBPE_VOCAB_BUFFER_SIZE = 512
COBPE_METADATA_FILENAMES = ("compositional.json", "cobpe_config.json")
NORMALIZED_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|\p{L}+|\p{N}{1,2}|[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
PUNCT_TO_WORD_BOUNDARY_PATTERN = re.compile(r"([^\w\s])(?=\w)")
WORD_TO_PUNCT_BOUNDARY_PATTERN = re.compile(r"(?<=\w)([^\w\s])")


def _lowercase_standard_capitalization(text: str) -> str:
    def replace(match: re.Match) -> str:
        word = match.group(0)
        return word.lower() if len(word) >= 2 and word[0].isupper() and word[1:].islower() else word
    return WORD_PATTERN.sub(replace, text)


def _separate_affix_punctuation(text: str) -> str:
    text = PUNCT_TO_WORD_BOUNDARY_PATTERN.sub(r"\1 ", text)
    return WORD_TO_PUNCT_BOUNDARY_PATTERN.sub(r" \1", text)


def normalize_cobpe_training_text(text: str) -> str:
    """Expose spaces, standard capitalization, and punctuation as modifiers."""
    return _lowercase_standard_capitalization(_separate_affix_punctuation(text))


def train_normalized_space_cap_tokenizer(text_iterator, vocab_size):
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
    assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
    tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=NORMALIZED_SPLIT_PATTERN)
    pattern = tokenizer.get_pattern()
    mergeable_ranks_list = tokenizer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(name="rustbpe_normalized_space_cap", pat_str=pattern, mergeable_ranks=mergeable_ranks, special_tokens=special_tokens)
    return RustBPETokenizer(enc, "<|bos|>")


def train_cobpe_tokenizer(text_iterator, target_vocab_size: int, *, vocab_buffer_size: int = COBPE_VOCAB_BUFFER_SIZE):
    train_vocab_size = int(target_vocab_size) + int(vocab_buffer_size)
    tokenizer = train_normalized_space_cap_tokenizer(text_iterator, train_vocab_size)
    redundant = detect_redundant_token_ids(tokenizer)
    redundant_ids = sorted({
        *redundant["space_prefixed_with_unspaced_counterpart"],
        *redundant["capitalized_with_lowercase_counterpart"],
    })
    tokenizer, finalize_stats = finalize_tokenizer_vocab(tokenizer, target_vocab_size=int(target_vocab_size), redundant_token_ids=redundant_ids)
    build_report = {
        "vocab_size_target": int(target_vocab_size),
        "vocab_buffer_size": int(vocab_buffer_size),
        "trained_vocab_size": int(train_vocab_size),
        "redundant_counts": {
            "space_prefixed_with_unspaced_counterpart": len(redundant["space_prefixed_with_unspaced_counterpart"]),
            "capitalized_with_lowercase_counterpart": len(redundant["capitalized_with_lowercase_counterpart"]),
        },
        **finalize_stats,
    }
    return tokenizer, build_report


def save_cobpe_metadata(tokenizer_dir: str, build_report: dict):
    metadata_path = os.path.join(tokenizer_dir, "compositional.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(build_cobpe_metadata(), f, indent=2, sort_keys=True)

    config_path = os.path.join(tokenizer_dir, "cobpe_config.json")
    config = {
        "version": 1,
        "normalized_space_cap": True,
        "normalization": {
            "separate_affix_punctuation": True,
            "case_normalization": "standard_titlecase_to_lower",
            "split_pattern": "normalized",
        },
        **build_report,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return metadata_path, config_path


def remove_cobpe_metadata(tokenizer_dir: str) -> list[str]:
    """Remove CoBPE sidecar files so a plain BPE tokenizer cannot load as CoBPE."""
    removed = []
    for filename in COBPE_METADATA_FILENAMES:
        path = os.path.join(tokenizer_dir, filename)
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)
    return removed


def detect_redundant_token_ids(tokenizer) -> dict[str, list[int]]:
    vocab_size = tokenizer.get_vocab_size()
    special_ids = {int(tokenizer.encode_special(tok)) for tok in tokenizer.get_special_tokens()}
    id_to_surface = {}
    surface_to_ids = {}
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        surface = tokenizer.decode([token_id])
        if not surface:
            continue
        id_to_surface[token_id] = surface
        surface_to_ids.setdefault(surface, []).append(token_id)

    def _first_alpha_index(text: str) -> int:
        for idx, ch in enumerate(text):
            if ch.isalpha():
                return idx
        return -1

    def _is_standard_titlecase(text: str, alpha_idx: int) -> bool:
        alpha_chars = [ch for ch in text[alpha_idx:] if ch.isalpha()]
        if len(alpha_chars) < 2:
            return False
        return alpha_chars[0].isupper() and all(ch.islower() for ch in alpha_chars[1:])

    space_prefixed = []
    capitalized = []
    for token_id, surface in id_to_surface.items():
        if len(surface) >= 2 and surface.startswith(" ") and not surface.startswith("  "):
            counterpart = surface[1:]
            if counterpart and counterpart in surface_to_ids:
                space_prefixed.append(int(token_id))
        alpha_idx = _first_alpha_index(surface)
        if alpha_idx >= 0 and _is_standard_titlecase(surface, alpha_idx):
            counterpart = surface[:alpha_idx] + surface[alpha_idx].lower() + surface[alpha_idx + 1 :]
            if counterpart in surface_to_ids:
                capitalized.append(int(token_id))
    return {
        "space_prefixed_with_unspaced_counterpart": sorted(set(space_prefixed)),
        "capitalized_with_lowercase_counterpart": sorted(set(capitalized)),
    }


def finalize_tokenizer_vocab(
    rust_tokenizer: RustBPETokenizer,
    *,
    target_vocab_size: int,
    redundant_token_ids: list[int],
) -> tuple[RustBPETokenizer, dict]:
    enc = rust_tokenizer.enc
    mergeable_ranks = getattr(enc, "_mergeable_ranks", None)
    pattern = getattr(enc, "_pat_str", None)
    special_tokens_map = getattr(enc, "_special_tokens", None)
    if not isinstance(mergeable_ranks, dict) or not pattern or not isinstance(special_tokens_map, dict):
        raise ValueError("Unexpected tokenizer internals while finalizing tokenizer vocab.")

    num_special = int(len(special_tokens_map))
    target_mergeable_size = int(target_vocab_size) - num_special
    if target_mergeable_size < 256:
        raise ValueError(f"target vocab size {target_vocab_size} leaves fewer than 256 mergeables")

    existing_mergeable_size = int(len(mergeable_ranks))
    if target_mergeable_size > existing_mergeable_size:
        raise ValueError(
            f"target mergeable size {target_mergeable_size} exceeds existing {existing_mergeable_size}"
        )

    redundant_mergeable = sorted(int(tid) for tid in redundant_token_ids if 0 <= int(tid) < existing_mergeable_size)
    redundant_set = set(redundant_mergeable)
    keep = set(range(min(256, existing_mergeable_size)))
    for tid in range(existing_mergeable_size):
        if tid in keep or tid in redundant_set:
            continue
        keep.add(int(tid))
        if len(keep) >= target_mergeable_size:
            break
    fallback_added = 0
    if len(keep) < target_mergeable_size:
        for tid in redundant_mergeable:
            if tid in keep:
                continue
            keep.add(int(tid))
            fallback_added += 1
            if len(keep) >= target_mergeable_size:
                break
    if len(keep) != target_mergeable_size:
        raise RuntimeError(
            f"failed selecting target mergeables: got {len(keep)} expected {target_mergeable_size}"
        )

    rank_to_token = {int(rank): token_bytes for token_bytes, rank in mergeable_ranks.items()}
    selected_old_ranks = sorted(keep)
    new_mergeable_ranks = {
        rank_to_token[old_rank]: new_rank
        for new_rank, old_rank in enumerate(selected_old_ranks)
    }
    specials_in_order = [t for t, _ in sorted(special_tokens_map.items(), key=lambda kv: kv[1])]
    new_specials = {tok: len(new_mergeable_ranks) + i for i, tok in enumerate(specials_in_order)}
    finalized_enc = tiktoken.Encoding(name=getattr(enc, "name", "rustbpe"), pat_str=pattern, mergeable_ranks=new_mergeable_ranks, special_tokens=new_specials)
    stats = {
        "target_vocab_size": int(target_vocab_size),
        "existing_vocab_size": int(existing_mergeable_size + num_special),
        "target_mergeable_size": int(target_mergeable_size),
        "existing_mergeable_size": int(existing_mergeable_size),
        "redundant_candidates": int(len(redundant_mergeable)),
        "retained_redundant_mergeables": int(sum(1 for x in selected_old_ranks if x in redundant_set)),
        "dropped_redundant_mergeables": int(sum(1 for x in redundant_mergeable if x not in keep)),
        "fallback_added_mergeables": int(fallback_added),
    }
    return RustBPETokenizer(finalized_enc, "<|bos|>"), stats
