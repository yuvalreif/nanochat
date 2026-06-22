"""CoBPE metadata helpers and Rust-backed tokenizer wrapper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from nanochat.compositional_rust import build_rust_backend
from nanochat.token_codec import TokenSequenceMixin


MULTI_TOKEN_FIRST_GROUPS = {
    "space_prefix",
    "base_capitalization",
    "determiners",
    "articles",
    "article_det",
    "article_space_prefix",
    "article_capitalization",
    "prepositions",
    "prep_space_prefix",
    "prep_capitalization",
    "prefix_punctuation",
}

COBPE_DETERMINERS = (
    "the", "a", "an",
    "my", "your", "his", "her", "our", "their", "its",
)
COBPE_PREPOSITIONS = ("by", "at", "of", "to", "in", "on", "with", "for", "from")
COBPE_PREFIX_PUNCTUATION = (
    "'", "\u2018", "\u2019", '"', "\u201c", "\u201d", "`", "(", "[", "{", "-",
)
COBPE_SUFFIX_PUNCTUATION = (
    "'", "\u2018", "\u2019", '"', "\u201c", "\u201d", "`",
    ")", "]", "}", ".", "!", "?", ",", ";", ":", "-", "'s", "s'",
)


def build_cobpe_metadata() -> dict[str, Any]:
    """Return the canonical metadata used by nanochat's CoBPE tokenizer mode."""
    group_value_names = {
        "space_prefix": ["no_space_prefix", "with_space_prefix"],
        "base_capitalization": ["no_capitalization", "add_capitalization"],
        "determiners": ["no_det", *(f"det_{value}" for value in COBPE_DETERMINERS)],
        "article_capitalization": ["no_article_cap", "add_article_cap"],
        "prepositions": ["no_prep", *(f"prep_{value}" for value in COBPE_PREPOSITIONS)],
        "prep_capitalization": ["no_prep_cap", "add_prep_cap"],
        "prefix_punctuation": [
            "no_prefix_punct",
            *(f"punct_prefix_{value}" for value in COBPE_PREFIX_PUNCTUATION),
        ],
        "suffix_punctuation": [
            "no_suffix_punct",
            *(f"punct_suffix_{value}" for value in COBPE_SUFFIX_PUNCTUATION),
        ],
    }
    group_names = list(group_value_names)
    return {
        "version": 1,
        "group_names": group_names,
        "group_value_names": group_value_names,
        "num_modifier_groups": len(group_names),
        "modifier_group_sizes": [len(group_value_names[name]) for name in group_names],
        "default_modifier": [0] * len(group_names),
        "entries": [],
        "inverse_entries": [],
    }


def _as_int_list(values: Iterable[Any]) -> list[int]:
    return [int(v) for v in values]


def _normalize_modifier_row(row: Iterable[Any], *, num_groups: int) -> tuple[int, ...]:
    values = tuple(int(v) for v in row)
    if len(values) != num_groups:
        raise ValueError(
            f"Modifier row length mismatch: expected {num_groups}, got {len(values)}"
        )
    return values


def _normalize_modifier_rows(
    *,
    num_groups: int,
    base_ids: list[int],
    modifier_rows: Optional[Iterable[Iterable[Any]]],
    modifier: Optional[Iterable[Any]],
) -> tuple[tuple[int, ...], ...]:
    if modifier_rows is not None:
        rows = tuple(
            _normalize_modifier_row(row, num_groups=num_groups) for row in modifier_rows
        )
        if len(rows) != len(base_ids):
            raise ValueError(
                "modifier_rows length must match base_ids length: "
                f"{len(rows)} != {len(base_ids)}"
            )
        return rows
    if modifier is not None:
        row = _normalize_modifier_row(modifier, num_groups=num_groups)
        return tuple(row for _ in base_ids)
    raise ValueError("Entry must provide modifier_rows or modifier.")


@dataclass(frozen=True)
class _SequenceEntry:
    token_ids: tuple[int, ...]
    base_ids: tuple[int, ...]
    modifier_rows: tuple[tuple[int, ...], ...]
    surface: Optional[str] = None


class CompositionalSpec:
    def __init__(
        self,
        *,
        modifier_group_sizes: Iterable[Any],
        num_modifier_groups: int,
        default_modifier: Iterable[Any],
        group_names: Iterable[str],
        group_value_names: dict[str, list[str]],
        sequence_entries: list[_SequenceEntry],
    ) -> None:
        self.modifier_group_sizes = tuple(int(v) for v in modifier_group_sizes)
        self.num_modifier_groups = int(num_modifier_groups)
        if self.num_modifier_groups <= 0:
            raise ValueError("num_modifier_groups must be > 0")
        if len(self.modifier_group_sizes) != self.num_modifier_groups:
            raise ValueError(
                "modifier_group_sizes length mismatch: "
                f"{len(self.modifier_group_sizes)} != {self.num_modifier_groups}"
            )
        self.default_modifier = _normalize_modifier_row(
            default_modifier, num_groups=self.num_modifier_groups
        )
        self.group_names = tuple(str(name) for name in group_names)
        if len(self.group_names) != self.num_modifier_groups:
            raise ValueError(
                "group_names length mismatch: "
                f"{len(self.group_names)} != {self.num_modifier_groups}"
            )
        self.group_to_idx = {name: idx for idx, name in enumerate(self.group_names)}
        self.group_value_names = {
            str(group): [str(v) for v in values]
            for group, values in (group_value_names or {}).items()
        }
        self.sequence_entries = list(sequence_entries)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CompositionalSpec":
        version = int(payload.get("version", 1))
        if version != 1:
            raise ValueError(f"Unsupported compositional metadata version: {version}")

        raw_group_sizes = payload.get("modifier_group_sizes")
        if raw_group_sizes is None:
            num_groups = int(payload["num_modifier_groups"])
            raw_group_sizes = [2] * num_groups
        else:
            raw_group_sizes = _as_int_list(raw_group_sizes)
            num_groups = len(raw_group_sizes)
        default_modifier = payload.get("default_modifier", [0] * num_groups)
        group_names = payload.get("group_names") or [f"group_{idx}" for idx in range(num_groups)]
        group_value_names = {
            str(group): [str(v) for v in values]
            for group, values in (payload.get("group_value_names") or {}).items()
        }

        sequence_entries: list[_SequenceEntry] = []

        raw_entries = list(payload.get("entries", [])) + list(payload.get("sequence_entries", []))
        for raw_entry in raw_entries:
            token_ids = tuple(_as_int_list(raw_entry["token_ids"]))
            base_ids = tuple(_as_int_list(raw_entry["base_ids"]))
            modifier_rows = _normalize_modifier_rows(
                num_groups=num_groups,
                base_ids=list(base_ids),
                modifier_rows=raw_entry.get("modifier_rows"),
                modifier=raw_entry.get("modifier"),
            )
            surface = raw_entry.get("surface")
            entry = _SequenceEntry(
                token_ids=token_ids,
                base_ids=base_ids,
                modifier_rows=modifier_rows,
                surface=str(surface) if surface is not None else None,
            )
            sequence_entries.append(entry)

        return cls(
            modifier_group_sizes=raw_group_sizes,
            num_modifier_groups=num_groups,
            default_modifier=default_modifier,
            group_names=group_names,
            group_value_names=group_value_names,
            sequence_entries=sequence_entries,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "CompositionalSpec":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    def _rust_literal_map(
        self,
        group_names: tuple[str, ...],
        prefixes: tuple[str, ...],
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for group_name in group_names:
            names = self.group_value_names.get(group_name, [])
            for rel_idx, value_name in enumerate(names):
                for prefix in prefixes:
                    if value_name.startswith(prefix):
                        out[value_name[len(prefix) :]] = {
                            "group_name": group_name,
                            "rel_idx": int(rel_idx),
                        }
                        break
        return out

    def to_rust_config(self) -> dict[str, Any]:
        entries = []
        for entry in self.sequence_entries:
            entries.append(
                {
                    "token_ids": list(entry.token_ids),
                    "base_ids": list(entry.base_ids),
                    "modifier_rows": [list(row) for row in entry.modifier_rows],
                }
            )
        reverse_entries = []
        for entry in self.sequence_entries:
            reverse_entries.append(
                {
                    "token_ids": list(entry.token_ids),
                    "base_ids": list(entry.base_ids),
                    "modifier_rows": [list(row) for row in entry.modifier_rows],
                    "surface": entry.surface,
                }
            )
        group_indices = {
            group_name: int(group_idx)
            for group_name, group_idx in self.group_to_idx.items()
        }
        payload = {
            "version": 1,
            "num_modifier_groups": int(self.num_modifier_groups),
            "modifier_group_sizes": list(self.modifier_group_sizes),
            "default_modifier": list(self.default_modifier),
            "group_names": list(self.group_names),
            "group_value_names": self.group_value_names,
            "entries": entries,
            "reverse_entries": reverse_entries,
            "token_meta": [],
            "runtime": {
                "group_indices": group_indices,
                "literal_maps": {
                    "determiners": self._rust_literal_map(
                        ("determiners", "article_det", "articles"),
                        ("det_", "article_"),
                    ),
                    "prepositions": self._rust_literal_map(("prepositions",), ("prep_",)),
                    "prefix_punctuation": self._rust_literal_map(
                        ("prefix_punctuation",),
                        ("punct_prefix_",),
                    ),
                    "suffix_punctuation": self._rust_literal_map(
                        ("suffix_punctuation",),
                        ("punct_suffix_",),
                    ),
                },
                "multi_token_first_group_indices": sorted(
                    int(group_idx)
                    for group_name, group_idx in self.group_to_idx.items()
                    if group_name in MULTI_TOKEN_FIRST_GROUPS
                ),
                "attachment_limits": {
                    "max_prefix_punctuation": 1,
                    "max_suffix_punctuation": 1,
                },
            },
        }
        return payload


class RustCoBPETokenizer(TokenSequenceMixin):
    """Rust-backed CoBPE tokenizer wrapper."""

    def __init__(self, base_tokenizer, spec: CompositionalSpec, *, tokenizer_dir: Optional[str] = None):
        self.base_tokenizer = base_tokenizer
        self.spec = spec
        self.rust_backend = build_rust_backend(spec, tokenizer_dir=tokenizer_dir)
        if self.rust_backend is None:
            raise RuntimeError(
                "Compositional tokenizer requires the Rust backend at runtime. "
                "Install rustbpe with CoBPE tokenizer support and load from a "
                "tokenizer directory containing tokenizer.pkl."
            )

    def __getattr__(self, name: str):
        return getattr(self.base_tokenizer, name)

    def has_compositional_mode(self) -> bool:
        return True

    def get_num_modifier_groups(self) -> int:
        return self.spec.num_modifier_groups

    def get_modifier_group_sizes(self) -> list[int]:
        return list(self.spec.modifier_group_sizes)

    def get_default_modifier(self) -> list[int]:
        return list(self.spec.default_modifier)

    def _prepend_append_rows(
        self,
        token_ids: list[int],
        modifier_rows: list[list[int]],
        *,
        prepend=None,
        append=None,
    ) -> tuple[list[int], list[list[int]]]:
        out_ids = list(token_ids)
        out_mods = [list(row) for row in modifier_rows]
        zero_row = list(self.spec.default_modifier)
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            out_ids.insert(0, int(prepend_id))
            out_mods.insert(0, list(zero_row))
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            out_ids.append(int(append_id))
            out_mods.append(list(zero_row))
        return out_ids, out_mods

    def _encode_one_with_modifiers(self, text: str, prepend=None, append=None):
        token_ids, modifier_rows = self.rust_backend.process_text(text)
        return self._prepend_append_rows(
            token_ids,
            modifier_rows,
            prepend=prepend,
            append=append,
        )

    def encode_with_modifiers(self, text, prepend=None, append=None, num_threads=8):
        if isinstance(text, str):
            return self._encode_one_with_modifiers(text, prepend=prepend, append=append)
        if isinstance(text, list):
            encoded = self.rust_backend.process_text_batch(text)
            return [
                self._prepend_append_rows(token_ids, modifier_rows, prepend=prepend, append=append)
                for token_ids, modifier_rows in encoded
            ]
        raise ValueError(f"Invalid input type: {type(text)}")

    def decode_token_with_modifiers(self, token_id: int, modifier_row: Iterable[Any]) -> str:
        row = [int(v) for v in modifier_row]
        return self.rust_backend.decode_token_with_modifiers(int(token_id), row)

    def utf8_len_with_modifiers_batch(
        self,
        token_ids: list[int],
        modifier_rows: list[list[int]],
    ) -> list[int]:
        return self.rust_backend.utf8_len_with_modifiers_batch(token_ids, modifier_rows)

    def decode_with_modifiers(self, token_ids: list[int], modifier_ids: list[list[int]]) -> str:
        return self.rust_backend.decode_with_modifiers(token_ids, modifier_ids)
