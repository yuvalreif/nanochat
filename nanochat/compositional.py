"""
Minimal compositional-token runtime helpers.

This module intentionally stays small:
- load a compact metadata artifact
- apply longest-match sequence replacements over raw tokenizer ids
- carry per-output-token modifier rows
- optionally reconstruct surfaces for sampled outputs
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from nanochat.compositional_rust import build_rust_backend


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


def _is_invalid_utf8_fragment(token_bytes: bytes) -> bool:
    try:
        token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


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


class _TrieNode:
    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.entry: Optional[_SequenceEntry] = None


class _ReverseTrieNode:
    def __init__(self) -> None:
        self.children: dict[tuple[int, tuple[int, ...]], _ReverseTrieNode] = {}
        self.entry: Optional[_SequenceEntry] = None


class CompositionalSpec:
    def __init__(
        self,
        *,
        modifier_group_sizes: Iterable[Any],
        num_modifier_groups: int,
        default_modifier: Iterable[Any],
        group_names: Iterable[str],
        group_value_names: dict[str, list[str]],
        direct_entries: dict[int, _SequenceEntry],
        sequence_entries: list[_SequenceEntry],
        inverse_surfaces: dict[tuple[int, tuple[int, ...]], str],
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
        self.direct_entries = direct_entries
        self.sequence_entries = list(sequence_entries)
        self.inverse_surfaces = inverse_surfaces
        self._root = _TrieNode()
        self._max_sequence_len = 1
        self._reverse_root = _ReverseTrieNode()
        self._max_reverse_sequence_len = 1
        for entry in sequence_entries:
            self._insert(entry)
            self._insert_reverse(entry)

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

        direct_entries: dict[int, _SequenceEntry] = {}
        sequence_entries: list[_SequenceEntry] = []
        inverse_surfaces: dict[tuple[int, tuple[int, ...]], str] = {}

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
            if len(token_ids) == 1:
                direct_entries[token_ids[0]] = entry
            sequence_entries.append(entry)

        for raw_entry in payload.get("inverse_entries", []):
            base_id = int(raw_entry["base_id"])
            modifier_row = _normalize_modifier_row(
                raw_entry["modifier"], num_groups=num_groups
            )
            inverse_surfaces[(base_id, modifier_row)] = str(raw_entry["surface"])

        return cls(
            modifier_group_sizes=raw_group_sizes,
            num_modifier_groups=num_groups,
            default_modifier=default_modifier,
            group_names=group_names,
            group_value_names=group_value_names,
            direct_entries=direct_entries,
            sequence_entries=sequence_entries,
            inverse_surfaces=inverse_surfaces,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "CompositionalSpec":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    def _insert(self, entry: _SequenceEntry) -> None:
        node = self._root
        for token_id in entry.token_ids:
            if token_id not in node.children:
                node.children[token_id] = _TrieNode()
            node = node.children[token_id]
        node.entry = entry
        self._max_sequence_len = max(self._max_sequence_len, len(entry.token_ids))

    def _insert_reverse(self, entry: _SequenceEntry) -> None:
        node = self._reverse_root
        for base_id, modifier_row in zip(entry.base_ids, entry.modifier_rows):
            key = (int(base_id), tuple(int(v) for v in modifier_row))
            if key not in node.children:
                node.children[key] = _ReverseTrieNode()
            node = node.children[key]
        node.entry = entry
        self._max_reverse_sequence_len = max(self._max_reverse_sequence_len, len(entry.base_ids))

    def _longest_match(self, token_ids: list[int], start_idx: int) -> Optional[_SequenceEntry]:
        node = self._root
        best_entry = None
        stop = min(start_idx + self._max_sequence_len, len(token_ids))
        for pos in range(start_idx, stop):
            token_id = int(token_ids[pos])
            child = node.children.get(token_id)
            if child is None:
                break
            node = child
            if node.entry is not None:
                best_entry = node.entry
        return best_entry

    def _longest_reverse_match(
        self,
        token_ids: list[int],
        modifier_ids: list[list[int]],
        start_idx: int,
    ) -> Optional[_SequenceEntry]:
        node = self._reverse_root
        best_entry = None
        stop = min(start_idx + self._max_reverse_sequence_len, len(token_ids))
        for pos in range(start_idx, stop):
            key = (
                int(token_ids[pos]),
                _normalize_modifier_row(modifier_ids[pos], num_groups=self.num_modifier_groups),
            )
            child = node.children.get(key)
            if child is None:
                break
            node = child
            if node.entry is not None:
                best_entry = node.entry
        return best_entry

    def apply(self, token_ids: list[int]) -> tuple[list[int], list[list[int]]]:
        out_ids: list[int] = []
        out_mods: list[list[int]] = []
        idx = 0
        while idx < len(token_ids):
            entry = self._longest_match(token_ids, idx)
            if entry is None:
                token_id = int(token_ids[idx])
                direct_entry = self.direct_entries.get(token_id)
                if direct_entry is None:
                    out_ids.append(token_id)
                    out_mods.append(list(self.default_modifier))
                else:
                    out_ids.extend(direct_entry.base_ids)
                    out_mods.extend([list(row) for row in direct_entry.modifier_rows])
                idx += 1
                continue
            out_ids.extend(entry.base_ids)
            out_mods.extend([list(row) for row in entry.modifier_rows])
            idx += len(entry.token_ids)
        return out_ids, out_mods

    def _combine_modifier_rows(
        self,
        modifier_rows: tuple[tuple[int, ...], ...] | list[list[int]],
    ) -> tuple[int, ...]:
        combined = list(self.default_modifier)
        for row in modifier_rows:
            normalized = _normalize_modifier_row(row, num_groups=self.num_modifier_groups)
            for group_idx, value in enumerate(normalized):
                if int(value) != int(self.default_modifier[group_idx]):
                    combined[group_idx] = int(value)
        return tuple(combined)

    def _value_name(self, group_name: str, value: int) -> Optional[str]:
        names = self.group_value_names.get(group_name, [])
        if 0 <= int(value) < len(names):
            return names[int(value)]
        return None

    def _capitalize_first_alpha(self, text: str) -> str:
        chars = list(text)
        for idx, ch in enumerate(chars):
            if ch.isalpha():
                chars[idx] = ch.upper()
                break
        return "".join(chars)

    def _lowercase_first_alpha(self, text: str) -> str:
        chars = list(text)
        for idx, ch in enumerate(chars):
            if ch.isalpha():
                chars[idx] = ch.lower()
                break
        return "".join(chars)

    def _first_alpha_is_upper(self, text: str) -> Optional[bool]:
        for ch in text:
            if ch.isalpha():
                return bool(ch.isupper())
        return None

    def _space_setting(self, modifiers: tuple[int, ...], group_name: str) -> bool:
        group_idx = self.group_to_idx.get(group_name)
        if group_idx is None:
            return False
        value = int(modifiers[group_idx])
        if value == int(self.default_modifier[group_idx]):
            return False
        value_name = (self._value_name(group_name, value) or "").lower()
        if value_name.startswith(("with_", "add_")):
            return True
        if value_name.startswith(("remove_", "no_", "na_", "none")):
            return False
        return value == 1

    def _literal_from_group(self, modifiers: tuple[int, ...], group_name: str, prefixes: tuple[str, ...]) -> Optional[str]:
        group_idx = self.group_to_idx.get(group_name)
        if group_idx is None:
            return None
        value = int(modifiers[group_idx])
        if value == int(self.default_modifier[group_idx]):
            return None
        value_name = self._value_name(group_name, value)
        if not value_name:
            return None
        for prefix in prefixes:
            if value_name.startswith(prefix):
                return value_name[len(prefix):]
        return None

    def _apply_base_capitalization(self, surface: str, modifiers: tuple[int, ...]) -> str:
        group_idx = self.group_to_idx.get("base_capitalization")
        if group_idx is None:
            return surface
        value = int(modifiers[group_idx])
        if value == int(self.default_modifier[group_idx]):
            return surface
        value_name = (self._value_name("base_capitalization", value) or "").lower()
        if value_name.startswith(("add_", "with_")) or value == 1:
            return self._capitalize_first_alpha(surface)
        if value_name.startswith(("remove_", "lower_")):
            return self._lowercase_first_alpha(surface)
        return surface

    def _synthesize_surface(self, lexical_surface: str, modifiers: tuple[int, ...]) -> str:
        if not lexical_surface:
            surface = ""
        elif lexical_surface.strip() == "":
            surface = lexical_surface
        else:
            # Only drop explicit leading ASCII spaces that are modeled separately
            # by the space_prefix modifier. Preserve newlines and other whitespace.
            surface = lexical_surface.lstrip(" ")
        surface = self._apply_base_capitalization(surface, modifiers)

        prefix_punct = self._literal_from_group(modifiers, "prefix_punctuation", ("punct_prefix_",))
        preposition = self._literal_from_group(modifiers, "prepositions", ("prep_",))
        determiner = self._literal_from_group(modifiers, "determiners", ("det_", "article_"))
        if determiner is None:
            determiner = self._literal_from_group(modifiers, "article_det", ("det_", "article_"))
        if determiner is None:
            determiner = self._literal_from_group(modifiers, "articles", ("article_", "det_"))
        suffix_punct = self._literal_from_group(modifiers, "suffix_punctuation", ("punct_suffix_",))

        if preposition and self._space_setting(modifiers, "prep_capitalization"):
            preposition = self._capitalize_first_alpha(preposition)
        if determiner and self._space_setting(modifiers, "article_capitalization"):
            determiner = self._capitalize_first_alpha(determiner)

        pieces = [piece for piece in (preposition, determiner, surface) if piece]
        expr = " ".join(pieces)
        if prefix_punct:
            expr = f"{prefix_punct}{expr}"
        if suffix_punct:
            expr = f"{expr}{suffix_punct}"
        if self._space_setting(modifiers, "space_prefix") and expr and not expr[:1].isspace():
            expr = " " + expr
        return expr

    def _lexical_surface_for_entry(self, entry: _SequenceEntry, decode_token) -> str:
        # Single-token decorated variants should reconstruct from the normalized base
        # token so we do not duplicate detachable modifiers like articles.
        if len(entry.base_ids) == 1:
            return decode_token(list(entry.base_ids))
        # Multi-token spans may only be faithfully recoverable through the stored
        # fused surface; decoding base_ids directly can reintroduce false boundaries.
        if entry.surface is not None:
            return entry.surface
        return decode_token(list(entry.base_ids))

    def reconstruct_surface(self, token_ids: list[int], modifier_ids: list[list[int]], decode_token) -> str:
        if len(token_ids) != len(modifier_ids):
            raise ValueError(
                f"token_ids and modifier_ids length mismatch: {len(token_ids)} != {len(modifier_ids)}"
            )
        chunks: list[str] = []
        idx = 0
        while idx < len(token_ids):
            entry = self._longest_reverse_match(token_ids, modifier_ids, idx)
            if entry is not None:
                lexical_surface = self._lexical_surface_for_entry(entry, decode_token)
                chunks.append(self._synthesize_surface(lexical_surface, self._combine_modifier_rows(entry.modifier_rows)))
                idx += len(entry.base_ids)
                continue
            normalized_row = tuple(int(v) for v in modifier_ids[idx])
            if normalized_row == self.default_modifier:
                literal_start = idx
                idx += 1
                while idx < len(token_ids):
                    if tuple(int(v) for v in modifier_ids[idx]) != self.default_modifier:
                        break
                    if self._longest_reverse_match(token_ids, modifier_ids, idx) is not None:
                        break
                    idx += 1
                chunks.append(decode_token([int(token_id) for token_id in token_ids[literal_start:idx]]))
                continue
            chunks.append(self.surface_for_token(token_ids[idx], modifier_ids[idx], decode_token))
            idx += 1
        return "".join(chunks)

    def surface_for_token(self, token_id: int, modifier_row: Iterable[Any], decode_token) -> str:
        normalized_row = tuple(int(v) for v in modifier_row)
        lexical_surface = decode_token([int(token_id)])
        if normalized_row == self.default_modifier:
            return lexical_surface
        return self._synthesize_surface(lexical_surface, normalized_row)

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

    def to_rust_config(self, tokenizer_json: Optional[str] = None) -> dict[str, Any]:
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
        if tokenizer_json is not None:
            payload["tokenizer_json"] = tokenizer_json
        return payload


class CompositionalTokenizer:
    """
    Thin wrapper around a base tokenizer.

    `encode` / `decode` remain baseline-compatible on purpose.
    The compositional path is available through `encode_with_modifiers` and
    `decode_with_modifiers`.
    """

    def __init__(self, base_tokenizer, spec: CompositionalSpec, *, tokenizer_dir: Optional[str] = None):
        self.base_tokenizer = base_tokenizer
        self.spec = spec
        self.rust_backend = build_rust_backend(spec, tokenizer_dir=tokenizer_dir)
        if tokenizer_dir is not None and self.rust_backend is None:
            raise RuntimeError(
                "Compositional tokenizer requires the Rust backend at runtime. "
                "Build/install the nanochat compositional Rust extension first."
            )
        self._use_rust_backend = (
            self.rust_backend is not None
            and os.environ.get("COBPE_MATCH_BACKEND", "").lower() != "python"
        )
        self._token_text_cache: dict[int, str] = {}
        self._token_has_space_prefix_cache: dict[int, bool] = {}
        self._token_has_word_char_cache: dict[int, bool] = {}
        self._token_is_whitespace_only_cache: dict[int, bool] = {}
        self._token_is_byte_fallback_cache: dict[int, bool] = {}
        self._rank_to_token_bytes: Optional[dict[int, bytes]] = None
        self._space_group_idx = self.spec.group_to_idx.get("space_prefix", -1)
        self._base_cap_group_idx = self.spec.group_to_idx.get("base_capitalization", -1)
        self._article_cap_group_idx = self.spec.group_to_idx.get("article_capitalization", -1)
        self._prep_cap_group_idx = self.spec.group_to_idx.get("prep_capitalization", -1)
        self._article_space_group_idx = self.spec.group_to_idx.get("article_space_prefix", -1)
        self._prep_space_group_idx = self.spec.group_to_idx.get("prep_space_prefix", -1)
        self._determiner_group_name = self._first_existing_group("determiners", "article_det", "articles")
        self._preposition_group_name = "prepositions" if "prepositions" in self.spec.group_to_idx else None
        self._prefix_punct_group_name = "prefix_punctuation" if "prefix_punctuation" in self.spec.group_to_idx else None
        self._suffix_punct_group_name = "suffix_punctuation" if "suffix_punctuation" in self.spec.group_to_idx else None
        self._determiner_literals = self._build_literal_map(
            ("determiners", "article_det", "articles"),
            ("det_", "article_"),
        )
        self._preposition_literals = self._build_literal_map(("prepositions",), ("prep_",))
        self._prefix_punct_literals = self._build_literal_map(("prefix_punctuation",), ("punct_prefix_",))
        self._suffix_punct_literals = self._build_literal_map(("suffix_punctuation",), ("punct_suffix_",))

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

    def _first_existing_group(self, *group_names: str) -> Optional[str]:
        for group_name in group_names:
            if group_name in self.spec.group_to_idx:
                return group_name
        return None

    def _build_literal_map(
        self,
        group_names: tuple[str, ...],
        prefixes: tuple[str, ...],
    ) -> dict[str, tuple[str, int]]:
        out: dict[str, tuple[str, int]] = {}
        for group_name in group_names:
            names = self.spec.group_value_names.get(group_name, [])
            for rel_idx, value_name in enumerate(names):
                for prefix in prefixes:
                    if value_name.startswith(prefix):
                        out[value_name[len(prefix):]] = (group_name, rel_idx)
                        break
        return out

    def _token_text(self, token_id: int) -> str:
        token_id = int(token_id)
        cached = self._token_text_cache.get(token_id)
        if cached is None:
            cached = self.base_tokenizer.decode([token_id])
            self._token_text_cache[token_id] = cached
        return cached

    def _token_has_space_prefix_id(self, token_id: int) -> bool:
        token_id = int(token_id)
        cached = self._token_has_space_prefix_cache.get(token_id)
        if cached is None:
            token = self._token_text(token_id)
            cached = bool(token.startswith(" ") or token.startswith("Ġ") or token.startswith("▁"))
            self._token_has_space_prefix_cache[token_id] = cached
        return cached

    def _token_has_word_char_id(self, token_id: int) -> bool:
        token_id = int(token_id)
        cached = self._token_has_word_char_cache.get(token_id)
        if cached is None:
            stripped = self._token_text(token_id).lstrip(" Ġ▁")
            cached = any(ch.isalnum() for ch in stripped)
            self._token_has_word_char_cache[token_id] = cached
        return cached

    def _token_is_whitespace_only_id(self, token_id: int) -> bool:
        token_id = int(token_id)
        cached = self._token_is_whitespace_only_cache.get(token_id)
        if cached is None:
            cached = self._token_text(token_id).strip() == ""
            self._token_is_whitespace_only_cache[token_id] = cached
        return cached

    def _token_bytes_by_id(self) -> dict[int, bytes]:
        if self._rank_to_token_bytes is None:
            mergeable_ranks = getattr(getattr(self.base_tokenizer, "enc", None), "_mergeable_ranks", None)
            if isinstance(mergeable_ranks, dict):
                self._rank_to_token_bytes = {
                    int(rank): bytes(token_bytes)
                    for token_bytes, rank in mergeable_ranks.items()
                    if isinstance(token_bytes, (bytes, bytearray))
                }
            else:
                self._rank_to_token_bytes = {}
        return self._rank_to_token_bytes

    def _token_is_byte_fallback_id(self, token_id: int) -> bool:
        token_id = int(token_id)
        cached = self._token_is_byte_fallback_cache.get(token_id)
        if cached is None:
            token_bytes = self._token_bytes_by_id().get(token_id)
            cached = bool(token_bytes and _is_invalid_utf8_fragment(token_bytes))
            self._token_is_byte_fallback_cache[token_id] = cached
        return cached

    def _token_bytes(self, token_id: int) -> Optional[bytes]:
        return self._token_bytes_by_id().get(int(token_id))

    def _decode_token_bytes(self, token_ids: list[int]) -> Optional[str]:
        chunks = []
        for token_id in token_ids:
            token_bytes = self._token_bytes(token_id)
            if token_bytes is None:
                return None
            chunks.append(token_bytes)
        try:
            return b"".join(chunks).decode("utf-8")
        except UnicodeDecodeError:
            return None

    def _byte_component_end(self, token_ids: list[int], start_idx: int) -> int:
        if start_idx >= len(token_ids):
            return start_idx
        first_bytes = self._token_bytes(token_ids[start_idx])
        if not first_bytes or not _is_invalid_utf8_fragment(first_bytes):
            return start_idx + 1
        pending = bytearray()
        for end_idx in range(start_idx, len(token_ids)):
            token_bytes = self._token_bytes(token_ids[end_idx])
            if token_bytes is None:
                break
            pending.extend(token_bytes)
            try:
                pending.decode("utf-8")
            except UnicodeDecodeError:
                continue
            return end_idx + 1
        return start_idx + 1

    def _byte_component_has_word_char(self, token_ids: list[int], start_idx: int) -> bool:
        end_idx = self._byte_component_end(token_ids, start_idx)
        decoded = self._decode_token_bytes(token_ids[start_idx:end_idx])
        return bool(decoded and any(ch.isalnum() for ch in decoded))

    def _decode_with_modifiers_python(self, token_ids: list[int], modifier_ids: list[list[int]]) -> str:
        if len(token_ids) != len(modifier_ids):
            raise ValueError(
                f"token_ids and modifier_ids length mismatch: {len(token_ids)} != {len(modifier_ids)}"
            )
        chunks: list[str] = []
        idx = 0
        while idx < len(token_ids):
            entry = self.spec._longest_reverse_match(token_ids, modifier_ids, idx)
            if entry is not None:
                lexical_surface = self.spec._lexical_surface_for_entry(entry, self.base_tokenizer.decode)
                chunks.append(self.spec._synthesize_surface(lexical_surface, self.spec._combine_modifier_rows(entry.modifier_rows)))
                idx += len(entry.base_ids)
                continue

            if self._token_is_byte_fallback_id(token_ids[idx]):
                component_end = self._byte_component_end(token_ids, idx)
                lexical_surface = self._decode_token_bytes(token_ids[idx:component_end])
                if lexical_surface is not None:
                    combined = self.spec._combine_modifier_rows(
                        tuple(tuple(int(v) for v in row) for row in modifier_ids[idx:component_end])
                    )
                    if combined == self.spec.default_modifier:
                        chunks.append(lexical_surface)
                    else:
                        chunks.append(self.spec._synthesize_surface(lexical_surface, combined))
                    idx = component_end
                    continue

            normalized_row = tuple(int(v) for v in modifier_ids[idx])
            if normalized_row == self.spec.default_modifier:
                literal_start = idx
                idx += 1
                while idx < len(token_ids):
                    if tuple(int(v) for v in modifier_ids[idx]) != self.spec.default_modifier:
                        break
                    if self._token_is_byte_fallback_id(token_ids[idx]):
                        break
                    if self.spec._longest_reverse_match(token_ids, modifier_ids, idx) is not None:
                        break
                    idx += 1
                chunks.append(self.base_tokenizer.decode([int(token_id) for token_id in token_ids[literal_start:idx]]))
                continue

            chunks.append(self.spec.surface_for_token(token_ids[idx], modifier_ids[idx], self.base_tokenizer.decode))
            idx += 1
        return "".join(chunks)

    def _modifier_utf8_delta(self, modifier_row: Iterable[Any]) -> Optional[int]:
        row = tuple(int(v) for v in modifier_row)
        if len(row) != self.spec.num_modifier_groups:
            raise ValueError(
                f"Modifier row length mismatch: expected {self.spec.num_modifier_groups}, got {len(row)}"
            )

        supported_zero_delta = {
            "base_capitalization",
            "article_capitalization",
            "prep_capitalization",
            "article_space_prefix",
            "prep_space_prefix",
        }
        supported_groups = supported_zero_delta | {
            "space_prefix",
            "determiners",
            "article_det",
            "articles",
            "prepositions",
            "prefix_punctuation",
            "suffix_punctuation",
        }
        for group_idx, value in enumerate(row):
            if int(value) == int(self.spec.default_modifier[group_idx]):
                continue
            if self.spec.group_names[group_idx] not in supported_groups:
                return None

        delta = 0
        if self.spec._space_setting(row, "space_prefix"):
            delta += 1
        for group_name, prefixes, separator_bytes in (
            ("prepositions", ("prep_",), 1),
            ("determiners", ("det_", "article_"), 1),
            ("article_det", ("det_", "article_"), 1),
            ("articles", ("article_", "det_"), 1),
            ("prefix_punctuation", ("punct_prefix_",), 0),
            ("suffix_punctuation", ("punct_suffix_",), 0),
        ):
            literal = self.spec._literal_from_group(row, group_name, prefixes)
            if literal:
                delta += len(literal.encode("utf-8")) + separator_bytes
        return delta

    def _canonical_token_surface(self, token_id: int) -> str:
        if self._token_is_byte_fallback_id(token_id):
            return ""
        return self._token_text(token_id).lstrip(" ").lower()

    def _is_capitalized_surface(self, text: str) -> bool:
        stripped = text.lstrip(" ")
        for ch in stripped:
            if ch.isalpha():
                return ch.isupper()
        return False

    def _is_base_cap_representable_surface(self, text: str) -> bool:
        alpha_chars = [ch for ch in text if ch.isalpha()]
        if not alpha_chars:
            return True
        if all(ch.islower() for ch in alpha_chars):
            return True
        return bool(alpha_chars[0].isupper() and all(ch.islower() for ch in alpha_chars[1:]))

    def _empty_modifier(self) -> list[int]:
        return list(self.spec.default_modifier)

    def _set_group_value(self, modifier: list[int], group_name: Optional[str], rel_idx: int) -> None:
        if group_name is None:
            return
        group_idx = self.spec.group_to_idx.get(group_name, -1)
        if 0 <= group_idx < len(modifier):
            modifier[group_idx] = int(rel_idx)

    def _modifier_has_active_group(self, modifier: list[int], group_name: Optional[str]) -> bool:
        if group_name is None:
            return False
        group_idx = self.spec.group_to_idx.get(group_name, -1)
        if not (0 <= group_idx < len(modifier)):
            return False
        return int(modifier[group_idx]) != int(self.spec.default_modifier[group_idx])

    def _combine_pending(self, base_modifier: list[int], pending_groups: list[tuple[str, int]]) -> list[int]:
        if not pending_groups:
            return list(base_modifier)
        combined = list(base_modifier)
        applied_groups: set[str] = set()
        for group_name, rel_idx in reversed(pending_groups):
            if group_name in applied_groups:
                continue
            self._set_group_value(combined, group_name, rel_idx)
            applied_groups.add(group_name)
        return combined

    def _spread_multi_token_modifiers(self, combined_modifier: list[int], base_len: int) -> list[list[int]]:
        if base_len <= 1:
            return [list(combined_modifier)]
        empty = self._empty_modifier()
        first_modifier = self._empty_modifier()
        last_modifier = self._empty_modifier()
        for idx, group_name in enumerate(self.spec.group_names):
            value = int(combined_modifier[idx])
            if value <= 0:
                continue
            if group_name in MULTI_TOKEN_FIRST_GROUPS:
                first_modifier[idx] = value
            else:
                last_modifier[idx] = value
        if base_len == 2:
            return [first_modifier, last_modifier]
        return [first_modifier] + [list(empty) for _ in range(base_len - 2)] + [last_modifier]

    def _modifier_has_only_surface_groups(self, modifier: list[int]) -> bool:
        for group_idx, value in enumerate(modifier):
            if int(value) == int(self.spec.default_modifier[group_idx]):
                continue
            group_name = self.spec.group_names[group_idx]
            if group_name in {"space_prefix", "base_capitalization"}:
                continue
            return False
        return True

    def _is_intra_word_cap_alias_match(self, match_length: int, modifier: list[int]) -> bool:
        if match_length != 1:
            return False
        if not (0 <= self._base_cap_group_idx < len(modifier)):
            return False
        if int(modifier[self._base_cap_group_idx]) != 1:
            return False
        return self._modifier_has_only_surface_groups(modifier)

    def _find_longest_boundary_safe_match(
        self,
        raw_token_ids: list[int],
        start_idx: int,
        token_has_space_prefix: list[bool],
        token_has_word_char: list[bool],
        space_prefix_prefix_sum: list[int],
    ) -> Optional[_SequenceEntry]:
        n_tokens = len(raw_token_ids)
        if start_idx >= n_tokens:
            return None
        start_inside_word = bool(
            start_idx > 0
            and token_has_word_char[start_idx]
            and (not token_has_space_prefix[start_idx])
            and token_has_word_char[start_idx - 1]
        )
        node = self.spec._root
        max_end = min(start_idx + self.spec._max_sequence_len, n_tokens)
        best_entry: Optional[_SequenceEntry] = None
        best_length = 0
        for end_idx in range(start_idx, max_end):
            token_id = int(raw_token_ids[end_idx])
            child = node.children.get(token_id)
            if child is None:
                break
            node = child
            entry = node.entry
            if entry is None:
                continue
            span_end = end_idx + 1
            match_length = span_end - start_idx
            combined_modifier = list(self.spec._combine_modifier_rows(entry.modifier_rows))
            allow_intra_word_cap_alias = self._is_intra_word_cap_alias_match(match_length, combined_modifier)
            if start_inside_word and (not allow_intra_word_cap_alias):
                continue
            if (
                span_end < n_tokens
                and token_has_word_char[end_idx]
                and (not token_has_space_prefix[span_end])
                and token_has_word_char[span_end]
                and (not allow_intra_word_cap_alias)
            ):
                continue
            if match_length > 1 and all(token_has_word_char[j] for j in range(start_idx, span_end)):
                if (space_prefix_prefix_sum[span_end] - space_prefix_prefix_sum[start_idx + 1]) > 0:
                    continue
            best_entry = entry
            best_length = match_length
        return best_entry if best_length > 0 else None

    def _should_prefer_cap_fallback_over_match(
        self,
        raw_token_ids: list[int],
        start_idx: int,
        entry: _SequenceEntry,
        token_has_space_prefix: list[bool],
        token_has_word_char: list[bool],
    ) -> bool:
        match_length = len(entry.token_ids)
        modifier = list(self.spec._combine_modifier_rows(entry.modifier_rows))
        if not self._is_intra_word_cap_alias_match(match_length, modifier):
            return False
        if not (0 <= start_idx < len(raw_token_ids)) or not token_has_word_char[start_idx]:
            return False
        prev_continues_word = bool(
            start_idx > 0 and token_has_word_char[start_idx - 1] and not token_has_space_prefix[start_idx]
        )
        next_idx = start_idx + match_length
        next_continues_word = bool(
            next_idx < len(raw_token_ids) and token_has_word_char[next_idx] and not token_has_space_prefix[next_idx]
        )
        return prev_continues_word or next_continues_word

    def _split_camel_case_segments(self, surface: str) -> Optional[list[str]]:
        if not surface:
            return None
        boundaries = [0]
        has_internal_upper = False
        for idx in range(1, len(surface)):
            current = surface[idx]
            if not current.isupper():
                continue
            has_internal_upper = True
            prev = surface[idx - 1]
            next_is_lower = (idx + 1) < len(surface) and surface[idx + 1].islower()
            if prev.islower() or (prev.isupper() and next_is_lower):
                boundaries.append(idx)
        if not has_internal_upper:
            return None
        boundaries.append(len(surface))
        if len(boundaries) <= 2:
            return None
        segments = [surface[left:right] for left, right in zip(boundaries[:-1], boundaries[1:]) if left < right]
        return segments if len(segments) > 1 else None

    def _expand_caps_segments(self, segments: list[str]) -> list[str]:
        expanded: list[str] = []
        for segment in segments:
            if len(segment) > 1 and all(ch.isupper() for ch in segment):
                expanded.extend(list(segment))
            else:
                expanded.append(segment)
        return expanded

    def _try_lowercase_cap_fallback(
        self,
        raw_token_ids: list[int],
        start_idx: int,
        token_has_space_prefix: list[bool],
        token_has_word_char: list[bool],
        pending_groups: list[tuple[str, int]],
        pending_leading_space: bool,
    ) -> Optional[tuple[int, list[int], list[list[int]]]]:
        if self._base_cap_group_idx < 0 or start_idx >= len(raw_token_ids):
            return None
        if not token_has_word_char[start_idx]:
            return None
        end_idx = start_idx + 1
        while end_idx < len(raw_token_ids):
            token_id = int(raw_token_ids[end_idx])
            if token_has_space_prefix[end_idx] or (not token_has_word_char[end_idx]) or self._token_is_whitespace_only_id(token_id):
                break
            end_idx += 1
        surface = self.base_tokenizer.decode(raw_token_ids[start_idx:end_idx])
        if not surface:
            return None
        surface = surface.strip()
        if (not surface) or (not surface.isalpha()) or (not any(ch.isupper() for ch in surface)):
            return None
        if not self._is_base_cap_representable_surface(surface):
            return None
        split_segments = self._split_camel_case_segments(surface)
        is_title_surface = bool(surface[0].isupper() and (len(surface) == 1 or surface[1:].islower()))
        if split_segments is None and (not is_title_surface):
            return None
        segments = self._expand_caps_segments(split_segments or [surface])
        output_ids: list[int] = []
        output_mods: list[list[int]] = []
        first_output = True
        for segment in segments:
            lower_ids = [int(v) for v in self.base_tokenizer.encode(segment.lower())]
            if not lower_ids:
                return None
            base_modifier = self._empty_modifier()
            base_modifier[self._base_cap_group_idx] = 1
            if first_output and pending_leading_space and self._space_group_idx >= 0:
                base_modifier[self._space_group_idx] = 1
            if first_output:
                base_modifier = self._combine_pending(base_modifier, pending_groups)
            per_token_mods = self._spread_multi_token_modifiers(base_modifier, len(lower_ids))
            output_ids.extend(lower_ids)
            output_mods.extend(per_token_mods)
            first_output = False
        return end_idx - start_idx, output_ids, output_mods

    def _can_attach_detached_modifier(
        self,
        raw_token_ids: list[int],
        start_idx: int,
        consumed_len: int,
        token_has_space_prefix: list[bool],
        token_has_word_char: list[bool],
        pending_has_prefix_punct: bool,
    ) -> bool:
        if consumed_len <= 0 or start_idx < 0 or start_idx >= len(raw_token_ids):
            return False
        span_end = min(start_idx + consumed_len, len(raw_token_ids))
        left_ok = False
        if start_idx == 0:
            left_ok = True
        else:
            left_ok = bool(
                token_has_space_prefix[start_idx]
                or self._token_is_whitespace_only_id(raw_token_ids[start_idx - 1])
            )
        if (not left_ok) and (not pending_has_prefix_punct):
            return False
        j = span_end
        saw_whitespace_between = False
        while j < len(raw_token_ids) and self._token_is_whitespace_only_id(raw_token_ids[j]):
            if self._token_text(raw_token_ids[j]) != " ":
                return False
            saw_whitespace_between = True
            j += 1
        if j >= len(raw_token_ids):
            return False
        if not token_has_word_char[j]:
            return False
        next_surface = self._canonical_token_surface(raw_token_ids[j])
        current_surface = self._canonical_token_surface(raw_token_ids[start_idx])
        if next_surface in self._preposition_literals:
            return False
        if current_surface in self._preposition_literals and next_surface in self._determiner_literals:
            return True
        if next_surface in self._determiner_literals:
            return False
        return bool(saw_whitespace_between or token_has_space_prefix[j])

    def _strip_invalid_detached_modifier_groups(
        self,
        modifier_values: list[int],
        raw_token_ids: list[int],
        start_idx: int,
        consumed_len: int,
        token_has_space_prefix: list[bool],
        token_has_word_char: list[bool],
        pending_has_prefix_punct: bool,
    ) -> list[int]:
        if self._can_attach_detached_modifier(
            raw_token_ids,
            start_idx,
            consumed_len,
            token_has_space_prefix,
            token_has_word_char,
            pending_has_prefix_punct,
        ):
            return modifier_values
        cleaned = list(modifier_values)
        for group_name in (
            self._determiner_group_name,
            self._preposition_group_name,
            "article_capitalization",
            "prep_capitalization",
            "article_space_prefix",
            "prep_space_prefix",
        ):
            group_idx = self.spec.group_to_idx.get(group_name or "", -1)
            if 0 <= group_idx < len(cleaned):
                cleaned[group_idx] = int(self.spec.default_modifier[group_idx])
        return cleaned

    def _strip_nonlexical_surface_groups(
        self,
        modifier_values: list[int],
        base_ids: Iterable[int],
    ) -> list[int]:
        if any(self._token_has_word_char_id(int(base_id)) for base_id in base_ids):
            return modifier_values
        cleaned = list(modifier_values)
        for group_name in (
            self._determiner_group_name,
            self._preposition_group_name,
            "article_capitalization",
            "prep_capitalization",
            self._prefix_punct_group_name,
            self._suffix_punct_group_name,
        ):
            group_idx = self.spec.group_to_idx.get(group_name or "", -1)
            if 0 <= group_idx < len(cleaned):
                cleaned[group_idx] = int(self.spec.default_modifier[group_idx])
        return cleaned

    def _raw_expr_has_leading_space(
        self,
        raw_token_ids: list[int],
        start_idx: int,
        token_has_space_prefix: list[bool],
    ) -> bool:
        if start_idx <= 0:
            return False
        prev_token_id = raw_token_ids[start_idx - 1]
        if self._token_is_whitespace_only_id(prev_token_id):
            return False
        return bool(token_has_space_prefix[start_idx])

    def _next_non_whitespace_idx(self, raw_token_ids: list[int], start_idx: int) -> Optional[int]:
        idx = int(start_idx)
        while idx < len(raw_token_ids):
            if not self._token_is_whitespace_only_id(raw_token_ids[idx]):
                return idx
            idx += 1
        return None

    def _token_can_host_expr_space(
        self,
        raw_token_ids: list[int],
        token_has_space_prefix: list[bool],
        token_has_word_char: list[bool],
        start_idx: int,
    ) -> bool:
        idx = int(start_idx)
        saw_ascii_space = False
        while idx < len(raw_token_ids) and self._token_is_whitespace_only_id(raw_token_ids[idx]):
            if self._token_text(raw_token_ids[idx]) != " ":
                return False
            if saw_ascii_space:
                return False
            saw_ascii_space = True
            idx += 1
        next_idx = self._next_non_whitespace_idx(raw_token_ids, start_idx)
        if next_idx is None:
            return False
        if token_has_space_prefix[next_idx]:
            return False
        token_id = raw_token_ids[next_idx]
        if token_has_word_char[next_idx]:
            return True
        token_surface = self._canonical_token_surface(token_id)
        if token_surface in self._determiner_literals or token_surface in self._preposition_literals:
            return True
        if self._prefix_punct_group_name and token_surface in self._prefix_punct_literals:
            lookahead_idx = self._next_non_whitespace_idx(raw_token_ids, next_idx + 1)
            return lookahead_idx is not None and bool(token_has_word_char[lookahead_idx])
        return False

    def _apply_contextual_space_prefix(
        self,
        modifier_values: list[int],
        *,
        raw_token_ids: list[int],
        start_idx: int,
        token_has_space_prefix: list[bool],
        use_pending_space: bool,
        pending_leading_space: bool,
    ) -> list[int]:
        if self._space_group_idx < 0:
            return modifier_values
        normalized = list(modifier_values)
        expr_has_space = (
            bool(pending_leading_space)
            if use_pending_space
            else self._raw_expr_has_leading_space(raw_token_ids, start_idx, token_has_space_prefix)
        )
        normalized[self._space_group_idx] = 1 if expr_has_space else int(self.spec.default_modifier[self._space_group_idx])
        return normalized

    def _apply_contextual_base_cap(
        self,
        modifier_values: list[int],
        *,
        raw_token_ids: list[int],
        start_idx: int,
        consumed_len: int,
    ) -> list[int]:
        if self._base_cap_group_idx < 0:
            return modifier_values
        normalized = list(modifier_values)
        cap_value = int(normalized[self._base_cap_group_idx])
        if cap_value == int(self.spec.default_modifier[self._base_cap_group_idx]):
            return normalized
        raw_surface = self.base_tokenizer.decode(raw_token_ids[start_idx : start_idx + consumed_len])
        if (not self._is_capitalized_surface(raw_surface)) or (not self._is_base_cap_representable_surface(raw_surface)):
            normalized[self._base_cap_group_idx] = int(self.spec.default_modifier[self._base_cap_group_idx])
        return normalized

    def _entry_has_case_mismatch(
        self,
        entry: _SequenceEntry,
        modifier_values: list[int],
        *,
        raw_token_ids: list[int],
        start_idx: int,
        consumed_len: int,
    ) -> bool:
        if self._base_cap_group_idx < 0:
            return False
        if int(modifier_values[self._base_cap_group_idx]) != int(self.spec.default_modifier[self._base_cap_group_idx]):
            return False
        raw_surface = self.base_tokenizer.decode(raw_token_ids[start_idx : start_idx + consumed_len])
        base_surface = self.base_tokenizer.decode(list(entry.base_ids))
        if raw_surface == base_surface:
            return False
        raw_letters = "".join(ch.lower() for ch in raw_surface if ch.isalpha())
        base_letters = "".join(ch.lower() for ch in base_surface if ch.isalpha())
        if not raw_letters or raw_letters != base_letters:
            return False
        if not self._is_base_cap_representable_surface(raw_surface):
            return True
        raw_is_upper = self.spec._first_alpha_is_upper(raw_surface)
        base_is_upper = self.spec._first_alpha_is_upper(base_surface)
        if raw_is_upper is None or base_is_upper is None:
            return False
        if raw_is_upper == base_is_upper:
            return False
        return True

    def _python_process_text(self, text: str) -> tuple[list[int], list[list[int]]]:
        raw_ids = [int(v) for v in self.base_tokenizer.encode(text)]
        out_ids: list[int] = []
        out_mods: list[list[int]] = []
        raw_texts = [self._token_text(tok_id) for tok_id in raw_ids]
        token_has_space_prefix = [self._token_has_space_prefix_id(tok_id) for tok_id in raw_ids]
        token_has_word_char = [self._token_has_word_char_id(tok_id) for tok_id in raw_ids]
        token_is_byte_fallback = [self._token_is_byte_fallback_id(tok_id) for tok_id in raw_ids]
        byte_component_end = [idx + 1 for idx in range(len(raw_ids))]
        idx = 0
        while idx < len(raw_ids):
            end_idx = self._byte_component_end(raw_ids, idx)
            byte_component_end[idx] = end_idx
            if end_idx > idx + 1 or token_is_byte_fallback[idx]:
                token_has_word_char[idx] = self._byte_component_has_word_char(raw_ids, idx)
                for inner_idx in range(idx + 1, end_idx):
                    token_has_word_char[inner_idx] = False
                    byte_component_end[inner_idx] = end_idx
            idx = max(end_idx, idx + 1)
        space_prefix_prefix_sum = [0] * (len(raw_ids) + 1)
        running_space = 0
        for idx, has_space in enumerate(token_has_space_prefix, start=1):
            if has_space:
                running_space += 1
            space_prefix_prefix_sum[idx] = running_space

        pending_groups: list[tuple[str, int]] = []
        pending_token_records: list[tuple[int, int]] = []
        pending_leading_space = False

        def _literal_modifier(start_idx: int, token_id: int, *, force_leading_space: bool = False) -> list[int]:
            modifier = self._empty_modifier()
            if (
                self._space_group_idx >= 0
                and (not self._token_is_whitespace_only_id(token_id))
                and (
                    force_leading_space
                    or self._raw_expr_has_leading_space(raw_ids, start_idx, token_has_space_prefix)
                )
            ):
                modifier[self._space_group_idx] = 1
            return modifier

        def _emit_literal(start_idx: int, token_id: int, *, force_leading_space: bool = False) -> None:
            out_ids.append(int(token_id))
            out_mods.append(_literal_modifier(start_idx, token_id, force_leading_space=force_leading_space))

        def _flush_pending_literal() -> None:
            nonlocal pending_leading_space
            emit_leading_space = bool(pending_leading_space)
            for raw_idx, raw_token_id in pending_token_records:
                is_whitespace_only = self._token_is_whitespace_only_id(raw_token_id)
                _emit_literal(
                    raw_idx,
                    raw_token_id,
                    force_leading_space=bool(emit_leading_space and not is_whitespace_only),
                )
                if emit_leading_space:
                    emit_leading_space = False
            pending_groups.clear()
            pending_token_records.clear()
            pending_leading_space = False

        def _mark_pending_detached_prefix(start_idx: int) -> None:
            nonlocal pending_leading_space
            if pending_leading_space:
                return
            if self._raw_expr_has_leading_space(raw_ids, start_idx, token_has_space_prefix):
                pending_leading_space = True

        i = 0
        while i < len(raw_ids):
            token_id = int(raw_ids[i])
            token_text = raw_texts[i]
            token_surface = token_text.lstrip(" ")

            if self._token_is_whitespace_only_id(token_id):
                if token_text == " ":
                    if pending_groups:
                        pending_token_records.append((i, token_id))
                    elif self._token_can_host_expr_space(raw_ids, token_has_space_prefix, token_has_word_char, i + 1):
                        pending_leading_space = True
                    else:
                        _emit_literal(i, token_id)
                    i += 1
                    continue
                if pending_groups or pending_token_records:
                    _flush_pending_literal()
                _emit_literal(i, token_id)
                i += 1
                continue

            if token_is_byte_fallback[i]:
                if pending_groups or pending_token_records:
                    first_modifier = self._combine_pending(self._empty_modifier(), pending_groups)
                else:
                    first_modifier = self._empty_modifier()
                if pending_leading_space and self._space_group_idx >= 0:
                    first_modifier[self._space_group_idx] = 1
                component_end = byte_component_end[i]
                out_ids.extend(int(v) for v in raw_ids[i:component_end])
                out_mods.append(first_modifier)
                out_mods.extend(self._empty_modifier() for _ in range(i + 1, component_end))
                pending_groups.clear()
                pending_token_records.clear()
                pending_leading_space = False
                i = component_end
                continue

            if self._suffix_punct_group_name and token_surface in self._suffix_punct_literals and out_mods:
                if i == 0 or self._token_is_whitespace_only_id(raw_ids[i - 1]) or token_has_space_prefix[i]:
                    pass
                elif self._modifier_has_active_group(out_mods[-1], self._suffix_punct_group_name):
                    pass
                else:
                    group_name, rel_idx = self._suffix_punct_literals[token_surface]
                    self._set_group_value(out_mods[-1], group_name, rel_idx)
                    i += 1
                    continue

            if self._prefix_punct_group_name and token_surface in self._prefix_punct_literals:
                next_idx = i + 1
                if (
                    next_idx < len(raw_ids)
                    and (not self._token_is_whitespace_only_id(raw_ids[next_idx]))
                    and (not token_has_space_prefix[next_idx])
                    and token_has_word_char[next_idx]
                ):
                    group_name, rel_idx = self._prefix_punct_literals[token_surface]
                    pending_groups.append((group_name, rel_idx))
                    pending_token_records.append((i, token_id))
                    i += 1
                    continue
                if pending_groups or pending_token_records:
                    _flush_pending_literal()

            canonical_surface = self._canonical_token_surface(token_id)
            if (
                canonical_surface in self._determiner_literals
                and self._is_base_cap_representable_surface(token_text)
                and self._can_attach_detached_modifier(
                    raw_ids,
                    i,
                    1,
                    token_has_space_prefix,
                    token_has_word_char,
                    bool(pending_groups),
                )
            ):
                group_name, rel_idx = self._determiner_literals[canonical_surface]
                _mark_pending_detached_prefix(i)
                pending_groups.append((group_name, rel_idx))
                pending_token_records.append((i, token_id))
                if self._is_capitalized_surface(token_text) and self._is_base_cap_representable_surface(token_text):
                    pending_groups.append(("article_capitalization", 1))
                i += 1
                continue
            if (
                canonical_surface in self._preposition_literals
                and self._is_base_cap_representable_surface(token_text)
                and self._can_attach_detached_modifier(
                    raw_ids,
                    i,
                    1,
                    token_has_space_prefix,
                    token_has_word_char,
                    bool(pending_groups),
                )
            ):
                group_name, rel_idx = self._preposition_literals[canonical_surface]
                _mark_pending_detached_prefix(i)
                pending_groups.append((group_name, rel_idx))
                pending_token_records.append((i, token_id))
                if self._is_capitalized_surface(token_text) and self._is_base_cap_representable_surface(token_text):
                    pending_groups.append(("prep_capitalization", 1))
                i += 1
                continue
            if pending_groups and (
                canonical_surface in self._determiner_literals or canonical_surface in self._preposition_literals
            ):
                _flush_pending_literal()
                continue

            entry = self._find_longest_boundary_safe_match(
                raw_ids,
                i,
                token_has_space_prefix,
                token_has_word_char,
                space_prefix_prefix_sum,
            )

            if entry is not None:
                combined_modifier = list(self.spec._combine_modifier_rows(entry.modifier_rows))
                if self._should_prefer_cap_fallback_over_match(
                    raw_ids,
                    i,
                    entry,
                    token_has_space_prefix,
                    token_has_word_char,
                ):
                    entry = None
                else:
                    base_surface = self.base_tokenizer.decode(list(entry.base_ids)).lstrip(" ").lower()
                    if (
                        len(entry.base_ids) == 1
                        and self._modifier_has_only_surface_groups(combined_modifier)
                        and base_surface in self._determiner_literals
                        and self._is_base_cap_representable_surface(self.base_tokenizer.decode(raw_ids[i:i + len(entry.token_ids)]))
                        and self._can_attach_detached_modifier(
                            raw_ids, i, len(entry.token_ids), token_has_space_prefix, token_has_word_char, bool(pending_groups)
                        )
                    ):
                        group_name, rel_idx = self._determiner_literals[base_surface]
                        _mark_pending_detached_prefix(i)
                        pending_groups.append((group_name, rel_idx))
                        pending_token_records.extend((i + offset, raw_ids[i + offset]) for offset in range(len(entry.token_ids)))
                        raw_surface = self.base_tokenizer.decode(raw_ids[i:i + len(entry.token_ids)])
                        if self._is_capitalized_surface(raw_surface) and self._is_base_cap_representable_surface(raw_surface):
                            pending_groups.append(("article_capitalization", 1))
                        i += len(entry.token_ids)
                        continue
                    if (
                        len(entry.base_ids) == 1
                        and self._modifier_has_only_surface_groups(combined_modifier)
                        and base_surface in self._preposition_literals
                        and self._is_base_cap_representable_surface(self.base_tokenizer.decode(raw_ids[i:i + len(entry.token_ids)]))
                        and self._can_attach_detached_modifier(
                            raw_ids, i, len(entry.token_ids), token_has_space_prefix, token_has_word_char, bool(pending_groups)
                        )
                    ):
                        group_name, rel_idx = self._preposition_literals[base_surface]
                        _mark_pending_detached_prefix(i)
                        pending_groups.append((group_name, rel_idx))
                        pending_token_records.extend((i + offset, raw_ids[i + offset]) for offset in range(len(entry.token_ids)))
                        raw_surface = self.base_tokenizer.decode(raw_ids[i:i + len(entry.token_ids)])
                        if self._is_capitalized_surface(raw_surface) and self._is_base_cap_representable_surface(raw_surface):
                            pending_groups.append(("prep_capitalization", 1))
                        i += len(entry.token_ids)
                        continue

                    combined_modifier = self._strip_invalid_detached_modifier_groups(
                        combined_modifier,
                        raw_ids,
                        i,
                        len(entry.token_ids),
                        token_has_space_prefix,
                        token_has_word_char,
                        bool(pending_groups),
                    )
                    combined_modifier = self._apply_contextual_base_cap(
                        combined_modifier,
                        raw_token_ids=raw_ids,
                        start_idx=i,
                        consumed_len=len(entry.token_ids),
                    )
                    combined_modifier = self._apply_contextual_space_prefix(
                        combined_modifier,
                        raw_token_ids=raw_ids,
                        start_idx=i,
                        token_has_space_prefix=token_has_space_prefix,
                        use_pending_space=bool(pending_groups or pending_leading_space),
                        pending_leading_space=pending_leading_space,
                    )
                    if self._entry_has_case_mismatch(
                        entry,
                        combined_modifier,
                        raw_token_ids=raw_ids,
                        start_idx=i,
                        consumed_len=len(entry.token_ids),
                    ):
                        entry = None
                    if entry is None:
                        pass
                    else:
                        combined_modifier = self._strip_nonlexical_surface_groups(combined_modifier, entry.base_ids)
                        if pending_groups or pending_leading_space:
                            if len(entry.base_ids) == 1:
                                merged = list(entry.modifier_rows[0])
                                merged = self._strip_invalid_detached_modifier_groups(
                                    merged,
                                    raw_ids,
                                    i,
                                    len(entry.token_ids),
                                    token_has_space_prefix,
                                    token_has_word_char,
                                    bool(pending_groups),
                                )
                                merged = self._apply_contextual_base_cap(
                                    merged,
                                    raw_token_ids=raw_ids,
                                    start_idx=i,
                                    consumed_len=len(entry.token_ids),
                                )
                                merged = self._apply_contextual_space_prefix(
                                    merged,
                                    raw_token_ids=raw_ids,
                                    start_idx=i,
                                    token_has_space_prefix=token_has_space_prefix,
                                    use_pending_space=bool(pending_groups or pending_leading_space),
                                    pending_leading_space=pending_leading_space,
                                )
                                merged = self._strip_nonlexical_surface_groups(merged, entry.base_ids)
                                merged = self._combine_pending(merged, pending_groups)
                                out_ids.extend(int(v) for v in entry.base_ids)
                                out_mods.append(merged)
                            else:
                                combined = combined_modifier
                                if pending_leading_space and self._space_group_idx >= 0:
                                    combined[self._space_group_idx] = 1
                                combined = self._combine_pending(combined, pending_groups)
                                out_ids.extend(int(v) for v in entry.base_ids)
                                out_mods.extend(self._spread_multi_token_modifiers(combined, len(entry.base_ids)))
                        else:
                            out_ids.extend(int(v) for v in entry.base_ids)
                            if len(entry.base_ids) == 1:
                                out_mods.append(list(combined_modifier))
                            else:
                                normalized_rows = [list(row) for row in entry.modifier_rows]
                                if normalized_rows:
                                    normalized_rows[0] = self._apply_contextual_base_cap(
                                        normalized_rows[0],
                                        raw_token_ids=raw_ids,
                                        start_idx=i,
                                        consumed_len=len(entry.token_ids),
                                    )
                                    normalized_rows[0] = self._apply_contextual_space_prefix(
                                        normalized_rows[0],
                                        raw_token_ids=raw_ids,
                                        start_idx=i,
                                        token_has_space_prefix=token_has_space_prefix,
                                        use_pending_space=bool(pending_leading_space),
                                        pending_leading_space=pending_leading_space,
                                    )
                                    if self._space_group_idx >= 0:
                                        for row in normalized_rows[1:]:
                                            row[self._space_group_idx] = int(self.spec.default_modifier[self._space_group_idx])
                                out_mods.extend(normalized_rows)
                        pending_groups.clear()
                        pending_token_records.clear()
                        pending_leading_space = False
                        i += len(entry.token_ids)
                        continue

            if token_has_word_char[i]:
                fallback = self._try_lowercase_cap_fallback(
                    raw_ids,
                    i,
                    token_has_space_prefix,
                    token_has_word_char,
                    pending_groups,
                    pending_leading_space,
                )
                if fallback is not None:
                    consumed_len, fallback_ids, fallback_mods = fallback
                    if pending_groups:
                        fallback_mods[0] = self._apply_contextual_space_prefix(
                            fallback_mods[0],
                            raw_token_ids=raw_ids,
                            start_idx=i,
                            token_has_space_prefix=token_has_space_prefix,
                            use_pending_space=True,
                            pending_leading_space=pending_leading_space,
                        )
                    else:
                        fallback_mods[0] = self._apply_contextual_space_prefix(
                            fallback_mods[0],
                            raw_token_ids=raw_ids,
                            start_idx=i,
                            token_has_space_prefix=token_has_space_prefix,
                            use_pending_space=bool(pending_leading_space),
                            pending_leading_space=pending_leading_space,
                        )
                    out_ids.extend(fallback_ids)
                    out_mods.extend(fallback_mods)
                    pending_groups.clear()
                    pending_token_records.clear()
                    pending_leading_space = False
                    i += consumed_len
                    continue

            base_modifier = self._empty_modifier()
            if pending_leading_space and self._space_group_idx >= 0 and token_has_word_char[i]:
                base_modifier[self._space_group_idx] = 1
            base_modifier = self._apply_contextual_base_cap(
                base_modifier,
                raw_token_ids=raw_ids,
                start_idx=i,
                consumed_len=1,
            )
            base_modifier = self._apply_contextual_space_prefix(
                base_modifier,
                raw_token_ids=raw_ids,
                start_idx=i,
                token_has_space_prefix=token_has_space_prefix,
                use_pending_space=bool(pending_leading_space),
                pending_leading_space=pending_leading_space,
            )
            base_modifier = self._combine_pending(base_modifier, pending_groups)
            out_ids.append(token_id)
            out_mods.append(base_modifier)
            pending_groups.clear()
            pending_token_records.clear()
            pending_leading_space = False
            i += 1
        if pending_groups or pending_token_records:
            _flush_pending_literal()
        return out_ids, out_mods

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
        if self._use_rust_backend:
            token_ids, modifier_rows = self.rust_backend.process_text(text)
        else:
            token_ids, modifier_rows = self._python_process_text(text)
        return self._prepend_append_rows(
            token_ids,
            modifier_rows,
            prepend=prepend,
            append=append,
        )

    def encode_with_modifiers(self, text, prepend=None, append=None, num_threads=8):
        if self._use_rust_backend and isinstance(text, list):
            encoded = self.rust_backend.process_text_batch(text)
            return [
                self._prepend_append_rows(token_ids, modifier_rows, prepend=prepend, append=append)
                for token_ids, modifier_rows in encoded
            ]
        if isinstance(text, str):
            return self._encode_one_with_modifiers(text, prepend=prepend, append=append)
        if isinstance(text, list):
            return [
                self._encode_one_with_modifiers(t, prepend=prepend, append=append)
                for t in text
            ]
        raise ValueError(f"Invalid input type: {type(text)}")

    def decode_token_with_modifiers(self, token_id: int, modifier_row: Iterable[Any]) -> str:
        row = [int(v) for v in modifier_row]
        if self._use_rust_backend:
            return self.rust_backend.decode_token_with_modifiers(int(token_id), row)
        return self.spec.surface_for_token(token_id, row, self.base_tokenizer.decode)

    def utf8_len_with_modifiers_batch(
        self,
        token_ids: list[int],
        modifier_rows: list[list[int]],
    ) -> list[int]:
        if self._use_rust_backend:
            return self.rust_backend.utf8_len_with_modifiers_batch(token_ids, modifier_rows)
        out = []
        for token_id, row in zip(token_ids, modifier_rows):
            token_bytes = self._token_bytes(int(token_id))
            if token_bytes is not None and self._token_is_byte_fallback_id(int(token_id)):
                delta = self._modifier_utf8_delta(row)
                if delta is not None:
                    out.append(len(token_bytes) + delta)
                    continue
            chunk = self.spec.surface_for_token(int(token_id), row, self.base_tokenizer.decode)
            out.append(len(chunk.encode("utf-8")))
        return out

    def decode_with_modifiers(self, token_ids: list[int], modifier_ids: list[list[int]]) -> str:
        if self._use_rust_backend:
            return self.rust_backend.decode_with_modifiers(token_ids, modifier_ids)
        return self._decode_with_modifiers_python(token_ids, modifier_ids)
