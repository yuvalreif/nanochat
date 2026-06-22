"""Modifier-stream operations used by nanochat dataloaders."""

import torch

from nanochat.token_codec import normalize_token_sequence


def resolve_num_modifier_groups(tokenizer, *, with_modifiers: bool) -> int:
    """Validate the CoBPE dataloader mode and return its modifier width."""
    if not with_modifiers:
        return 0
    if not (hasattr(tokenizer, "has_compositional_mode") and tokenizer.has_compositional_mode()):
        raise ValueError("with_modifiers=True requires a compositional tokenizer.")
    get_num_modifier_groups = getattr(tokenizer, "get_num_modifier_groups", None)
    if get_num_modifier_groups is None:
        raise ValueError("with_modifiers=True requires tokenizer.get_num_modifier_groups() support.")
    num_modifier_groups = int(get_num_modifier_groups())
    if num_modifier_groups <= 0:
        raise ValueError(f"with_modifiers=True requires num_modifier_groups > 0, got {num_modifier_groups}")
    return num_modifier_groups


def encode_doc_batch(tokenizer, doc_batch, *, bos_token, tokenizer_threads, with_modifiers):
    """Encode documents into structured sequences for the packing buffer."""
    encoded = tokenizer(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
    encoded = [normalize_token_sequence(tokenizer, tokens) for tokens in encoded]
    if with_modifiers:
        for seq in encoded:
            if seq.modifiers is None:
                raise ValueError("Compositional dataloader expected modifier rows.")
    return encoded


def copy_doc_span(row_buffer, row_mod_buffer, *, row_idx, pos, doc, take):
    """Copy matching base-token and modifier spans into a packed row."""
    row_buffer[row_idx, pos:pos + take] = torch.tensor(doc.ids[:take], dtype=torch.long)
    if row_mod_buffer is not None:
        if doc.modifiers is None:
            raise ValueError("modifier rows are required when row_mod_buffer is set")
        row_mod_buffer[row_idx, pos:pos + take] = torch.tensor(doc.modifiers[:take], dtype=torch.long)
