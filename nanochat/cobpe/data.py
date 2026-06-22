"""Modifier-stream operations used by nanochat dataloaders."""

import torch


def resolve_num_modifier_groups(tokenizer, *, with_modifiers: bool) -> int:
    """Validate the CoBPE dataloader mode and return its modifier width."""
    if not with_modifiers:
        return 0
    if not hasattr(tokenizer, "encode_with_modifiers"):
        raise ValueError("with_modifiers=True requires tokenizer.encode_with_modifiers(...) support.")
    get_num_modifier_groups = getattr(tokenizer, "get_num_modifier_groups", None)
    if get_num_modifier_groups is None:
        raise ValueError("with_modifiers=True requires tokenizer.get_num_modifier_groups() support.")
    num_modifier_groups = int(get_num_modifier_groups())
    if num_modifier_groups <= 0:
        raise ValueError(f"with_modifiers=True requires num_modifier_groups > 0, got {num_modifier_groups}")
    return num_modifier_groups


def encode_doc_batch(tokenizer, doc_batch, *, bos_token, tokenizer_threads, with_modifiers):
    """Encode documents into structured sequences for the packing buffer."""
    if with_modifiers:
        encoded = tokenizer.encode_with_modifiers(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        out = []
        for token_ids, modifier_rows in encoded:
            if len(token_ids) != len(modifier_rows):
                raise ValueError(
                    "Compositional tokenizer returned mismatched token/modifier lengths: "
                    f"{len(token_ids)} != {len(modifier_rows)}"
                )
            out.append((token_ids, modifier_rows))
        return out

    return tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)


def copy_doc_span(row_buffer, row_mod_buffer, *, row_idx, pos, token_ids, modifier_rows, take):
    """Copy matching base-token and modifier spans into a packed row."""
    row_buffer[row_idx, pos:pos + take] = torch.tensor(token_ids[:take], dtype=torch.long)
    if row_mod_buffer is not None:
        if modifier_rows is None:
            raise ValueError("modifier rows are required when row_mod_buffer is set")
        row_mod_buffer[row_idx, pos:pos + take] = torch.tensor(modifier_rows[:take], dtype=torch.long)
