"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
import os
import json
import time
import argparse
import torch
from nanochat.tokenizer import (
    RustBPETokenizer,
    apply_space_cap_normalization,
    detect_redundant_token_ids,
    finalize_tokenizer_vocab,
)
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 2B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
parser.add_argument('--normalized-space-cap', action='store_true', help='Train on normalized text for space-prefix and capitalization modifiers')
parser.add_argument('--vocab-buffer-size', type=int, default=0, help='Extra vocab slots to train before pruning redundant space/cap tokens')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")
print(f"normalized_space_cap: {args.normalized_space_cap}")
print(f"vocab_buffer_size: {args.vocab_buffer_size:,}")

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            if args.normalized_space_cap:
                doc_text = apply_space_cap_normalization(doc_text)
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
train_vocab_size = int(args.vocab_size) + int(args.vocab_buffer_size)
if args.normalized_space_cap:
    tokenizer = RustBPETokenizer.train_from_iterator_normalized_space_cap(text_iter, train_vocab_size)
else:
    tokenizer = RustBPETokenizer.train_from_iterator(text_iter, train_vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

buffer_report = {}
if args.vocab_buffer_size > 0:
    redundant = detect_redundant_token_ids(tokenizer)
    redundant_ids = sorted({
        *redundant["space_prefixed_with_unspaced_counterpart"],
        *redundant["capitalized_with_lowercase_counterpart"],
    })
    tokenizer, finalize_stats = finalize_tokenizer_vocab(
        tokenizer,
        target_vocab_size=int(args.vocab_size),
        redundant_token_ids=redundant_ids,
    )
    buffer_report = {
        "vocab_size_target": int(args.vocab_size),
        "vocab_buffer_size": int(args.vocab_buffer_size),
        "trained_vocab_size": int(train_vocab_size),
        "normalized_space_cap": bool(args.normalized_space_cap),
        "redundant_counts": {
            "space_prefixed_with_unspaced_counterpart": len(redundant["space_prefixed_with_unspaced_counterpart"]),
            "capitalized_with_lowercase_counterpart": len(redundant["capitalized_with_lowercase_counterpart"]),
        },
        **finalize_stats,
    }
    print(f"Finalized buffered tokenizer to target vocab size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)
if args.normalized_space_cap or args.vocab_buffer_size > 0:
    config_path = os.path.join(tokenizer_dir, "vd_buffer_config.json")
    payload = {
        "version": 1,
        "normalized_space_cap": bool(args.normalized_space_cap),
        "normalization": {
            "separate_affix_punctuation": bool(args.normalized_space_cap),
            "case_normalization": "standard_titlecase_to_lower" if args.normalized_space_cap else "none",
            "split_pattern": "normalized" if args.normalized_space_cap else "baseline",
        },
        "vocab_size_target": int(args.vocab_size),
        "vocab_buffer_size": int(args.vocab_buffer_size),
        "trained_vocab_size": int(train_vocab_size),
    }
    payload.update(buffer_report)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Saved compositional tokenizer config to {config_path}")

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])
