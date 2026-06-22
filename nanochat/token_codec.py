"""
Common encoded-token interface for regular BPE and CoBPE.

Regular BPE identifies a token with one integer. CoBPE uses the same base-token
integer plus one modifier value from each configured group. EncodedSequence and
EncodedBatch keep those parallel values together without copying the underlying
lists or tensors on construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, NamedTuple, Sequence

import torch


class EncodedBatch(NamedTuple):
    """Tensor batch with optional CoBPE modifier ids."""
    ids: torch.Tensor
    modifiers: torch.Tensor | None = None


class EncodedSequence:
    """List-backed token ids with optional per-token CoBPE modifier rows."""
    __slots__ = ("ids", "modifiers")

    def __init__(self, ids: list[int], modifiers: list[list[int]] | None = None):
        self.ids = ids
        self.modifiers = modifiers

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx]

    def __eq__(self, other) -> bool:
        if not isinstance(other, EncodedSequence):
            return False
        return self.ids == other.ids and self.modifiers == other.modifiers

    def __repr__(self) -> str:
        return f"EncodedSequence(ids={self.ids!r}, modifiers={self.modifiers!r})"

    @property
    def has_modifiers(self) -> bool:
        return self.modifiers is not None

    def copy(self) -> "EncodedSequence":
        return EncodedSequence(self.ids.copy(), None if self.modifiers is None else [row.copy() for row in self.modifiers])

    def units(self):
        if self.modifiers is None:
            return self.ids
        return [
            (token_id, tuple(modifier))
            for token_id, modifier in zip(self.ids, self.modifiers)
        ]

    def slice(self, start=None, stop=None) -> "EncodedSequence":
        return EncodedSequence(self.ids[slice(start, stop)], None if self.modifiers is None else self.modifiers[slice(start, stop)])

    def append_item(self, item: "TokenItem") -> None:
        self.ids.append(item.id)
        if self.modifiers is not None:
            if item.modifier is None:
                raise ValueError("modifier is required for this encoded sequence")
            self.modifiers.append(item.modifier)

    def append(self, token_id: int, modifier: Sequence[int] | None = None) -> None:
        self.ids.append(int(token_id))
        if self.modifiers is not None:
            if modifier is None:
                raise ValueError("modifier is required for this encoded sequence")
            self.modifiers.append([int(v) for v in modifier])

    def extend(self, other: "EncodedSequence") -> None:
        if self.modifiers is None and other.modifiers is not None:
            raise ValueError("cannot append modifier-bearing tokens to plain encoded sequence")
        if self.modifiers is not None and other.modifiers is None:
            raise ValueError("modifier-bearing encoded sequence requires modifiers")
        self.ids.extend(other.ids)
        if self.modifiers is not None:
            self.modifiers.extend(other.modifiers)

    def token_items(self) -> list["TokenItem"]:
        if self.modifiers is None:
            return [TokenItem(token_id) for token_id in self.ids]
        return [
            TokenItem(token_id, modifier)
            for token_id, modifier in zip(self.ids, self.modifiers)
        ]

    def item_at(self, idx: int) -> "TokenItem":
        modifier = None if self.modifiers is None else self.modifiers[idx]
        return TokenItem(self.ids[idx], modifier)


def tokenizer_has_modifiers(tokenizer) -> bool:
    return bool(hasattr(tokenizer, "has_compositional_mode") and tokenizer.has_compositional_mode())


def normalize_encoded_sequence(tokenizer, tokens) -> EncodedSequence:
    has_modifiers = tokenizer_has_modifiers(tokenizer)
    if isinstance(tokens, EncodedSequence):
        seq = tokens
    elif isinstance(tokens, tuple):
        token_ids, modifiers = tokens
        seq = EncodedSequence(token_ids, modifiers)
    else:
        seq = EncodedSequence(tokens if isinstance(tokens, list) else list(tokens))
    if has_modifiers and seq.modifiers is None:
        default = tokenizer.get_default_modifier()
        seq = EncodedSequence(seq.ids, [list(default) for _ in seq.ids])
    if not has_modifiers and seq.modifiers is not None:
        raise ValueError("modifier-bearing EncodedSequence requires a compositional tokenizer")
    return seq


def token_item_for_tokenizer(tokenizer, token_id: int, modifier: Sequence[int] | None = None) -> "TokenItem":
    if tokenizer_has_modifiers(tokenizer):
        if modifier is None:
            modifier = tokenizer.get_default_modifier()
        return TokenItem(int(token_id), [int(v) for v in modifier])
    if modifier is not None:
        raise ValueError("modifier-bearing TokenItem requires a compositional tokenizer")
    return TokenItem(int(token_id))


def empty_encoded_sequence_for_tokenizer(tokenizer) -> EncodedSequence:
    return EncodedSequence([], [] if tokenizer_has_modifiers(tokenizer) else None)


def encode_encoded_sequence(tokenizer, text: str, prepend=None, append=None, **kwargs) -> EncodedSequence:
    if tokenizer_has_modifiers(tokenizer):
        token_ids, modifiers = tokenizer.encode_with_modifiers(text, prepend=prepend, append=append, **kwargs)
        return EncodedSequence(token_ids, modifiers)
    return EncodedSequence(tokenizer.encode(text, prepend=prepend, append=append, **kwargs))


def encode_encoded_sequences(tokenizer, texts: list[str], prepend=None, append=None, **kwargs) -> list[EncodedSequence]:
    if tokenizer_has_modifiers(tokenizer):
        return [
            EncodedSequence(token_ids, modifiers)
            for token_ids, modifiers in tokenizer.encode_with_modifiers(texts, prepend=prepend, append=append, **kwargs)
        ]
    return [
        EncodedSequence(token_ids)
        for token_ids in tokenizer(texts, prepend=prepend, append=append, **kwargs)
    ]


def decode_encoded_sequence(tokenizer, sequence) -> str:
    seq = normalize_encoded_sequence(tokenizer, sequence)
    if seq.modifiers is not None:
        return tokenizer.decode_with_modifiers(seq.ids, seq.modifiers)
    return tokenizer.decode(seq.ids)


class EncodedSequenceMixin:
    """Add structured sequence helpers without changing a tokenizer's BPE API."""
    def has_compositional_mode(self) -> bool:
        return False

    def normalize_sequence(self, tokens) -> EncodedSequence:
        return normalize_encoded_sequence(self, tokens)

    def token_item(self, token_id: int, modifier: Sequence[int] | None = None) -> "TokenItem":
        return token_item_for_tokenizer(self, token_id, modifier)

    def empty_sequence(self) -> EncodedSequence:
        return empty_encoded_sequence_for_tokenizer(self)

    def encode_sequence(self, text: str, prepend=None, append=None, **kwargs) -> EncodedSequence:
        return encode_encoded_sequence(self, text, prepend=prepend, append=append, **kwargs)

    def encode_sequences(self, texts: list[str], prepend=None, append=None, **kwargs) -> list[EncodedSequence]:
        return encode_encoded_sequences(self, texts, prepend=prepend, append=append, **kwargs)

    def decode_sequence(self, sequence) -> str:
        return decode_encoded_sequence(self, sequence)


@dataclass(frozen=True)
class TokenItem:
    """One generated base token and its optional CoBPE modifier values."""
    id: int
    modifier: list[int] | None = None


class TokenCodec:
    """Convert tokenizer outputs and generation state to the common encoded form."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.has_modifiers = tokenizer_has_modifiers(tokenizer)

    def default_modifier(self) -> list[int] | None:
        if not self.has_modifiers:
            return None
        return list(self.tokenizer.get_default_modifier())

    def normalize(self, tokens) -> EncodedSequence:
        if hasattr(self.tokenizer, "normalize_sequence"):
            return self.tokenizer.normalize_sequence(tokens)
        return normalize_encoded_sequence(self.tokenizer, tokens)

    def encode_text(self, text: str, prepend=None, append=None, **kwargs) -> EncodedSequence:
        if hasattr(self.tokenizer, "encode_sequence"):
            return self.tokenizer.encode_sequence(text, prepend=prepend, append=append, **kwargs)
        return encode_encoded_sequence(self.tokenizer, text, prepend=prepend, append=append, **kwargs)

    def encode_texts(self, texts: list[str], prepend=None, append=None, **kwargs) -> list[EncodedSequence]:
        if hasattr(self.tokenizer, "encode_sequences"):
            return self.tokenizer.encode_sequences(texts, prepend=prepend, append=append, **kwargs)
        return encode_encoded_sequences(self.tokenizer, texts, prepend=prepend, append=append, **kwargs)

    def decode(self, sequence) -> str:
        if hasattr(self.tokenizer, "decode_sequence"):
            return self.tokenizer.decode_sequence(sequence)
        return decode_encoded_sequence(self.tokenizer, sequence)

    def item(self, token_id: int, modifier: Sequence[int] | None = None) -> TokenItem:
        if hasattr(self.tokenizer, "token_item"):
            return self.tokenizer.token_item(token_id, modifier)
        return token_item_for_tokenizer(self.tokenizer, token_id, modifier)

    def empty_sequence(self) -> EncodedSequence:
        if hasattr(self.tokenizer, "empty_sequence"):
            return self.tokenizer.empty_sequence()
        return empty_encoded_sequence_for_tokenizer(self.tokenizer)

    def empty_step(self) -> EncodedSequence:
        return EncodedSequence([], [] if self.has_modifiers else None)

    def sequence_tensor(self, sequence: EncodedSequence, device) -> EncodedBatch:
        seq = self.normalize(sequence)
        ids = torch.tensor([seq.ids], dtype=torch.long, device=device)
        modifiers = None
        if seq.modifiers is not None:
            modifiers = torch.tensor([seq.modifiers], dtype=torch.long, device=device)
        return EncodedBatch(ids, modifiers)

    def step_tensor(self, step: EncodedSequence, device) -> EncodedBatch:
        ids = torch.tensor(step.ids, dtype=torch.long, device=device).unsqueeze(1)
        modifiers = None
        if step.modifiers is not None:
            modifiers = torch.tensor(step.modifiers, dtype=torch.long, device=device).unsqueeze(1)
        return EncodedBatch(ids, modifiers)


def stack_sequences(sequences: Iterable[EncodedSequence | Sequence[int]], pad_token_id: int, default_modifier: Sequence[int] | None = None) -> EncodedBatch:
    """Pad plain or modifier-bearing token sequences to a common length."""
    seqs = list(sequences)
    bsz, seq_len = len(seqs), max(len(seq) for seq in seqs)
    input_ids = torch.full((bsz, seq_len), int(pad_token_id), dtype=torch.long)
    has_modifiers = any(isinstance(seq, EncodedSequence) and seq.modifiers is not None for seq in seqs)
    modifier_ids = None
    if has_modifiers:
        if default_modifier is None:
            raise ValueError("default_modifier is required when stacking modifiers")
        modifier_ids = torch.full((bsz, seq_len, len(default_modifier)), 0, dtype=torch.long)
        modifier_ids[:] = torch.tensor(default_modifier, dtype=torch.long)
    for i, seq in enumerate(seqs):
        ids = seq.ids if isinstance(seq, EncodedSequence) else seq
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        if modifier_ids is not None:
            if not isinstance(seq, EncodedSequence) or seq.modifiers is None:
                raise ValueError("all stacked sequences must provide modifiers")
            modifier_ids[i, :len(seq)] = torch.tensor(seq.modifiers, dtype=torch.long)
    return EncodedBatch(input_ids, modifier_ids)
