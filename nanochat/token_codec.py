"""Common token sequence interface for regular BPE and CoBPE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import torch


def tokenizer_has_modifiers(tokenizer) -> bool:
    return bool(
        hasattr(tokenizer, "has_compositional_mode")
        and tokenizer.has_compositional_mode()
    )


def normalize_token_sequence(tokenizer, tokens) -> "TokenSequence":
    has_modifiers = tokenizer_has_modifiers(tokenizer)
    if isinstance(tokens, TokenSequence):
        seq = tokens.copy()
    elif isinstance(tokens, tuple):
        token_ids, modifiers = tokens
        seq = TokenSequence(list(token_ids), [list(row) for row in modifiers])
    else:
        seq = TokenSequence(list(tokens))
    if has_modifiers and seq.modifiers is None:
        default = tokenizer.get_default_modifier()
        seq.modifiers = [list(default) for _ in seq.ids]
    if not has_modifiers and seq.modifiers is not None:
        raise ValueError("modifier-bearing TokenSequence requires a compositional tokenizer")
    return seq


def token_item_for_tokenizer(tokenizer, token_id: int, modifier: Sequence[int] | None = None) -> "TokenItem":
    if tokenizer_has_modifiers(tokenizer):
        if modifier is None:
            modifier = tokenizer.get_default_modifier()
        return TokenItem(int(token_id), [int(v) for v in modifier])
    if modifier is not None:
        raise ValueError("modifier-bearing TokenItem requires a compositional tokenizer")
    return TokenItem(int(token_id))


def empty_token_sequence_for_tokenizer(tokenizer) -> "TokenSequence":
    return TokenSequence([], [] if tokenizer_has_modifiers(tokenizer) else None)


def encode_token_sequence(tokenizer, text: str, prepend=None, append=None, **kwargs) -> "TokenSequence":
    if tokenizer_has_modifiers(tokenizer):
        token_ids, modifiers = tokenizer.encode_with_modifiers(
            text,
            prepend=prepend,
            append=append,
            **kwargs,
        )
        return TokenSequence(token_ids, modifiers)
    return TokenSequence(tokenizer.encode(text, prepend=prepend, append=append, **kwargs))


def encode_token_sequences(tokenizer, texts: list[str], prepend=None, append=None, **kwargs) -> list["TokenSequence"]:
    if tokenizer_has_modifiers(tokenizer):
        return [
            TokenSequence(token_ids, modifiers)
            for token_ids, modifiers in tokenizer.encode_with_modifiers(
                texts,
                prepend=prepend,
                append=append,
                **kwargs,
            )
        ]
    return [
        TokenSequence(token_ids)
        for token_ids in tokenizer(texts, prepend=prepend, append=append, **kwargs)
    ]


def decode_token_sequence(tokenizer, sequence) -> str:
    seq = normalize_token_sequence(tokenizer, sequence)
    if seq.modifiers is not None:
        return tokenizer.decode_with_modifiers(seq.ids, seq.modifiers)
    return tokenizer.decode(seq.ids)


class TokenSequenceMixin:
    def has_compositional_mode(self) -> bool:
        return False

    def normalize_sequence(self, tokens) -> "TokenSequence":
        return normalize_token_sequence(self, tokens)

    def token_item(self, token_id: int, modifier: Sequence[int] | None = None) -> "TokenItem":
        return token_item_for_tokenizer(self, token_id, modifier)

    def empty_sequence(self) -> "TokenSequence":
        return empty_token_sequence_for_tokenizer(self)

    def encode_sequence(self, text: str, prepend=None, append=None, **kwargs) -> "TokenSequence":
        return encode_token_sequence(self, text, prepend=prepend, append=append, **kwargs)

    def encode_sequences(self, texts: list[str], prepend=None, append=None, **kwargs) -> list["TokenSequence"]:
        return encode_token_sequences(self, texts, prepend=prepend, append=append, **kwargs)

    def decode_sequence(self, sequence) -> str:
        return decode_token_sequence(self, sequence)


@dataclass(frozen=True)
class TokenItem:
    id: int
    modifier: list[int] | None = None


@dataclass
class TokenSequence:
    ids: list[int]
    modifiers: list[list[int]] | None = None

    def __post_init__(self):
        self.ids = [int(v) for v in self.ids]
        if self.modifiers is not None:
            self.modifiers = [[int(x) for x in row] for row in self.modifiers]
            if len(self.modifiers) != len(self.ids):
                raise ValueError(
                    "modifier length must match token length: "
                    f"{len(self.modifiers)} != {len(self.ids)}"
                )

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx]

    @property
    def has_modifiers(self) -> bool:
        return self.modifiers is not None

    def copy(self) -> "TokenSequence":
        return TokenSequence(
            self.ids.copy(),
            None if self.modifiers is None else [row.copy() for row in self.modifiers],
        )

    def units(self):
        if self.modifiers is None:
            return self.ids
        return [
            (int(token_id), tuple(int(v) for v in modifier))
            for token_id, modifier in zip(self.ids, self.modifiers)
        ]

    def slice(self, start=None, stop=None) -> "TokenSequence":
        return TokenSequence(
            self.ids[slice(start, stop)],
            None if self.modifiers is None else self.modifiers[slice(start, stop)],
        )

    def append_item(self, item: TokenItem) -> None:
        self.ids.append(int(item.id))
        if self.modifiers is not None:
            if item.modifier is None:
                raise ValueError("modifier is required for this token sequence")
            self.modifiers.append([int(v) for v in item.modifier])

    def extend(self, other: "TokenSequence") -> None:
        if self.modifiers is None and other.modifiers is not None:
            raise ValueError("cannot append modifier-bearing tokens to plain token sequence")
        if self.modifiers is not None and other.modifiers is None:
            raise ValueError("modifier-bearing token sequence requires modifiers")
        self.ids.extend(int(v) for v in other.ids)
        if self.modifiers is not None:
            self.modifiers.extend([list(row) for row in other.modifiers])

    def token_items(self) -> list[TokenItem]:
        if self.modifiers is None:
            return [TokenItem(token_id) for token_id in self.ids]
        return [
            TokenItem(token_id, list(modifier))
            for token_id, modifier in zip(self.ids, self.modifiers)
        ]


@dataclass
class TokenStep:
    ids: list[int]
    modifiers: list[list[int]] | None = None

    def __post_init__(self):
        self.ids = [int(v) for v in self.ids]
        if self.modifiers is not None:
            self.modifiers = [[int(x) for x in row] for row in self.modifiers]
            if len(self.modifiers) != len(self.ids):
                raise ValueError(
                    "modifier length must match token step length: "
                    f"{len(self.modifiers)} != {len(self.ids)}"
                )

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self) -> Iterator[int]:
        return iter(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx]

    def append(self, token_id: int, modifier: Sequence[int] | None = None) -> None:
        self.ids.append(int(token_id))
        if self.modifiers is not None:
            if modifier is None:
                raise ValueError("modifier is required for this token step")
            self.modifiers.append([int(v) for v in modifier])

    def item_at(self, idx: int) -> TokenItem:
        modifier = None if self.modifiers is None else list(self.modifiers[idx])
        return TokenItem(self.ids[idx], modifier)


class TokenCodec:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.has_modifiers = tokenizer_has_modifiers(tokenizer)

    def default_modifier(self) -> list[int] | None:
        if not self.has_modifiers:
            return None
        return list(self.tokenizer.get_default_modifier())

    def normalize(self, tokens) -> TokenSequence:
        if hasattr(self.tokenizer, "normalize_sequence"):
            return self.tokenizer.normalize_sequence(tokens)
        return normalize_token_sequence(self.tokenizer, tokens)

    def encode_text(self, text: str, prepend=None, append=None, **kwargs) -> TokenSequence:
        if hasattr(self.tokenizer, "encode_sequence"):
            return self.tokenizer.encode_sequence(text, prepend=prepend, append=append, **kwargs)
        return encode_token_sequence(self.tokenizer, text, prepend=prepend, append=append, **kwargs)

    def encode_texts(self, texts: list[str], prepend=None, append=None, **kwargs) -> list[TokenSequence]:
        if hasattr(self.tokenizer, "encode_sequences"):
            return self.tokenizer.encode_sequences(texts, prepend=prepend, append=append, **kwargs)
        return encode_token_sequences(self.tokenizer, texts, prepend=prepend, append=append, **kwargs)

    def decode(self, sequence) -> str:
        if hasattr(self.tokenizer, "decode_sequence"):
            return self.tokenizer.decode_sequence(sequence)
        return decode_token_sequence(self.tokenizer, sequence)

    def item(self, token_id: int, modifier: Sequence[int] | None = None) -> TokenItem:
        if hasattr(self.tokenizer, "token_item"):
            return self.tokenizer.token_item(token_id, modifier)
        return token_item_for_tokenizer(self.tokenizer, token_id, modifier)

    def empty_sequence(self) -> TokenSequence:
        if hasattr(self.tokenizer, "empty_sequence"):
            return self.tokenizer.empty_sequence()
        return empty_token_sequence_for_tokenizer(self.tokenizer)

    def empty_step(self) -> TokenStep:
        return TokenStep([], [] if self.has_modifiers else None)

    def sequence_tensor(self, sequence: TokenSequence, device) -> tuple[torch.Tensor, torch.Tensor | None]:
        seq = self.normalize(sequence)
        ids = torch.tensor([seq.ids], dtype=torch.long, device=device)
        modifiers = None
        if seq.modifiers is not None:
            modifiers = torch.tensor([seq.modifiers], dtype=torch.long, device=device)
        return ids, modifiers

    def step_tensor(self, step: TokenStep, device) -> tuple[torch.Tensor, torch.Tensor | None]:
        ids = torch.tensor(step.ids, dtype=torch.long, device=device).unsqueeze(1)
        modifiers = None
        if step.modifiers is not None:
            modifiers = torch.tensor(step.modifiers, dtype=torch.long, device=device).unsqueeze(1)
        return ids, modifiers


def stack_token_sequences(
    sequences: Iterable[TokenSequence],
    pad_token_id: int,
    default_modifier: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    seqs = list(sequences)
    bsz, seq_len = len(seqs), max(len(seq) for seq in seqs)
    input_ids = torch.full((bsz, seq_len), int(pad_token_id), dtype=torch.long)
    has_modifiers = any(seq.modifiers is not None for seq in seqs)
    modifier_ids = None
    if has_modifiers:
        if default_modifier is None:
            raise ValueError("default_modifier is required when stacking modifiers")
        modifier_ids = torch.full(
            (bsz, seq_len, len(default_modifier)),
            0,
            dtype=torch.long,
        )
        modifier_ids[:] = torch.tensor(default_modifier, dtype=torch.long)
    for i, seq in enumerate(seqs):
        input_ids[i, :len(seq)] = torch.tensor(seq.ids, dtype=torch.long)
        if modifier_ids is not None:
            if seq.modifiers is None:
                raise ValueError("all stacked sequences must provide modifiers")
            modifier_ids[i, :len(seq)] = torch.tensor(seq.modifiers, dtype=torch.long)
    return input_ids, modifier_ids
