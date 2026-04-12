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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from nanochat.compositional_rust import build_rust_backend


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
        surface = lexical_surface.strip()
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
                lexical_surface = entry.surface if entry.surface is not None else decode_token(list(entry.token_ids))
                chunks.append(self._synthesize_surface(lexical_surface, self._combine_modifier_rows(entry.modifier_rows)))
                idx += len(entry.base_ids)
                continue
            chunks.append(self.surface_for_token(token_ids[idx], modifier_ids[idx], decode_token))
            idx += 1
        return "".join(chunks)

    def surface_for_token(self, token_id: int, modifier_row: Iterable[Any], decode_token) -> str:
        normalized_row = tuple(int(v) for v in modifier_row)
        key = (int(token_id), normalized_row)
        lexical_surface = self.inverse_surfaces.get(key)
        if lexical_surface is None:
            lexical_surface = decode_token([int(token_id)])
        return self._synthesize_surface(lexical_surface, normalized_row)

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
        payload = {
            "version": 1,
            "num_modifier_groups": int(self.num_modifier_groups),
            "modifier_group_sizes": list(self.modifier_group_sizes),
            "default_modifier": list(self.default_modifier),
            "entries": entries,
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
        if self.rust_backend is not None:
            token_ids, modifier_rows = self.rust_backend.process_text(text)
        else:
            raw_ids = self.base_tokenizer.encode(text)
            token_ids, modifier_rows = self.spec.apply(list(raw_ids))
        return self._prepend_append_rows(
            token_ids,
            modifier_rows,
            prepend=prepend,
            append=append,
        )

    def encode_with_modifiers(self, text, prepend=None, append=None, num_threads=8):
        if self.rust_backend is not None and isinstance(text, list):
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
        return self.spec.surface_for_token(token_id, modifier_row, self.base_tokenizer.decode)

    def decode_with_modifiers(self, token_ids: list[int], modifier_ids: list[list[int]]) -> str:
        return self.spec.reconstruct_surface(token_ids, modifier_ids, self.base_tokenizer.decode)
