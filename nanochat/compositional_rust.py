"""
Rustbpe-backed compositional tokenizer bridge.

This module keeps the Python side very small:
- try to load the packaged Rust tokenizer
- pass a compact JSON config derived from compositional metadata
- normalize the returned ids / modifier rows
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Optional

try:
    import rustbpe as _rustbpe
except Exception:
    _rustbpe = None


def _normalize_result(result: Any) -> tuple[list[int], list[list[int]]]:
    if isinstance(result, dict):
        token_ids = [int(v) for v in result["output_ids"]]
        modifier_rows = [[int(x) for x in row] for row in result["modifier_rows"]]
        return token_ids, modifier_rows
    token_ids, modifier_rows = result
    return [int(v) for v in token_ids], [[int(x) for x in row] for row in modifier_rows]


class RustCompositionalBackend:
    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def process_text(self, text: str) -> tuple[list[int], list[list[int]]]:
        return _normalize_result(self._tokenizer.process_text(text))

    def process_text_batch(self, texts: list[str]) -> list[tuple[list[int], list[list[int]]]]:
        return [
            _normalize_result(item)
            for item in self._tokenizer.process_text_batch(texts)
        ]

    def decode_with_modifiers(self, token_ids: list[int], modifier_rows: list[list[int]]) -> str:
        return str(self._tokenizer.decode_with_modifiers(token_ids, modifier_rows))

    def decode_token_with_modifiers(self, token_id: int, modifier_row: list[int]) -> str:
        return str(
            self._tokenizer.decode_with_modifiers(
                [int(token_id)], [[int(v) for v in modifier_row]]
            )
        )

    def utf8_len_with_modifiers_batch(
        self,
        token_ids: list[int],
        modifier_rows: list[list[int]],
    ) -> list[int]:
        return [
            int(v)
            for v in self._tokenizer.utf8_len_with_modifiers_batch(
                token_ids, modifier_rows
            )
        ]

    def debug_tokenize_text(self, text: str) -> dict[str, Any]:
        return json.loads(self._tokenizer.debug_tokenize_text_json(text))

    def debug_process_text(self, text: str) -> dict[str, Any]:
        return json.loads(self._tokenizer.debug_process_text_json(text))


def _extract_base_bpe_config(tokenizer_dir: Optional[str]) -> Optional[dict[str, Any]]:
    if tokenizer_dir is None:
        return None
    pickle_path = Path(tokenizer_dir) / "tokenizer.pkl"
    if not pickle_path.exists():
        return None
    try:
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
    except Exception:
        return None

    mergeable_ranks = getattr(enc, "_mergeable_ranks", None)
    pattern = getattr(enc, "_pat_str", None)
    special_tokens = getattr(enc, "_special_tokens", None)
    if not isinstance(mergeable_ranks, dict) or not isinstance(pattern, str):
        return None

    rank_entries = []
    for token_bytes, rank in sorted(mergeable_ranks.items(), key=lambda kv: int(kv[1])):
        if not isinstance(token_bytes, (bytes, bytearray)):
            return None
        rank_entries.append(
            {
                "token": bytes(token_bytes).decode("latin-1"),
                "rank": int(rank),
            }
        )

    special_token_map: dict[str, int] = {}
    if isinstance(special_tokens, dict):
        for token, token_id in special_tokens.items():
            special_token_map[str(token)] = int(token_id)

    return {
        "pattern": pattern,
        "mergeable_ranks": rank_entries,
        "special_tokens": special_token_map,
    }


def build_rust_backend(
    spec, *, tokenizer_dir: Optional[str] = None
) -> Optional[RustCompositionalBackend]:
    if _rustbpe is None or not hasattr(_rustbpe, "CompositionalTokenizer"):
        return None

    payload = spec.to_rust_config()
    base_bpe = _extract_base_bpe_config(tokenizer_dir)
    if base_bpe is None:
        return None
    payload["base_bpe"] = base_bpe

    config_json = json.dumps(payload, separators=(",", ":"))
    tokenizer = _rustbpe.CompositionalTokenizer(config_json)
    return RustCompositionalBackend(tokenizer)
