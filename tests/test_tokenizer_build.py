import os

import tiktoken

from nanochat.tokenizer import (
    RustBPETokenizer,
    SPLIT_PATTERN,
    apply_space_cap_normalization,
    detect_redundant_token_ids,
    finalize_tokenizer_vocab,
)


def _build_test_tokenizer():
    mergeable_ranks = {bytes([i]): i for i in range(256)}
    mergeable_ranks[b"table"] = 256
    mergeable_ranks[b" table"] = 257
    mergeable_ranks[b"Table"] = 258
    mergeable_ranks[b"chair"] = 259
    enc = tiktoken.Encoding(
        name="test_rustbpe",
        pat_str=SPLIT_PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens={"<|bos|>": len(mergeable_ranks)},
    )
    return RustBPETokenizer(enc, "<|bos|>")


def test_apply_space_cap_normalization_detaches_affix_punctuation_and_lowercases_titlecase():
    text = "Hello,World! NASA stays."
    normalized = apply_space_cap_normalization(text)
    assert normalized == "hello , world ! NASA stays ."


def test_detect_redundant_token_ids_finds_space_and_cap_duplicates():
    tokenizer = _build_test_tokenizer()
    redundant = detect_redundant_token_ids(tokenizer)

    assert redundant["space_prefixed_with_unspaced_counterpart"] == [257]
    assert redundant["capitalized_with_lowercase_counterpart"] == [258]


def test_finalize_tokenizer_vocab_prefers_nonredundant_tokens():
    tokenizer = _build_test_tokenizer()
    finalized, stats = finalize_tokenizer_vocab(
        tokenizer,
        target_vocab_size=260,
        redundant_token_ids=[257, 258],
    )

    mergeable_ranks = getattr(finalized.enc, "_mergeable_ranks")
    assert b"table" in mergeable_ranks
    assert b" table" in mergeable_ranks
    assert b"Table" not in mergeable_ranks
    assert stats["dropped_redundant_mergeables"] == 1


def test_rust_bpe_save_writes_portable_tokenizer_json(tmp_path):
    tokenizer = _build_test_tokenizer()
    tokenizer.save(str(tmp_path))

    assert os.path.exists(tmp_path / "tokenizer.pkl")
    assert os.path.exists(tmp_path / "tokenizer.json")
