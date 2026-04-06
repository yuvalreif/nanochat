"""
Optional Rust-backed compositional runtime bridge.

This module keeps the Python side very small:
- try to load a compiled Rust processor
- pass a compact JSON config derived from compositional metadata
- normalize the returned ids / modifier rows
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

try:
    import nanochat_compositional_rust as _nanochat_compositional_rust  # type: ignore[import-not-found]
except Exception:
    _nanochat_compositional_rust = None


def _normalize_result(result: Any) -> tuple[list[int], list[list[int]]]:
    if isinstance(result, dict):
        token_ids = [int(v) for v in result["output_ids"]]
        modifier_rows = [[int(x) for x in row] for row in result["modifier_rows"]]
        return token_ids, modifier_rows
    token_ids, modifier_rows = result
    return [int(v) for v in token_ids], [[int(x) for x in row] for row in modifier_rows]


class RustCompositionalBackend:
    def __init__(self, processor: Any):
        self._processor = processor

    def process_text(self, text: str) -> tuple[list[int], list[list[int]]]:
        return _normalize_result(self._processor.process_text(text))

    def process_text_batch(self, texts: list[str]) -> list[tuple[list[int], list[list[int]]]]:
        return [_normalize_result(item) for item in self._processor.process_text_batch(texts)]


def build_rust_backend(spec, *, tokenizer_dir: Optional[str] = None) -> Optional[RustCompositionalBackend]:
    if _nanochat_compositional_rust is None or not hasattr(_nanochat_compositional_rust, "CompositionalProcessor"):
        return None

    tokenizer_json = None
    if tokenizer_dir is not None:
        tokenizer_json_path = Path(tokenizer_dir) / "tokenizer.json"
        if tokenizer_json_path.exists():
            tokenizer_json = tokenizer_json_path.read_text(encoding="utf-8")

    config_json = json.dumps(spec.to_rust_config(tokenizer_json=tokenizer_json), separators=(",", ":"))
    processor = _nanochat_compositional_rust.CompositionalProcessor(config_json)
    return RustCompositionalBackend(processor)
