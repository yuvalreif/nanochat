"""
A number of functions that help with evaluating a base model.
"""
import math
import torch.nn.functional as F
import torch
import torch.distributed as dist


def _target_counted_mask(y, token_bytes):
    valid = y >= 0
    y_safe = torch.where(valid, y, torch.zeros_like(y))
    return valid & (token_bytes[y_safe] > 0)


def _space_value_adds_prefix_space(value_name: str, rel_idx: int, default_idx: int) -> bool:
    if rel_idx == default_idx:
        return False
    name = (value_name or "").lower()
    if name.startswith("with_") or name.startswith("add_"):
        return True
    if name.startswith(("remove_", "lower_", "no_", "na_", "none")):
        return False
    return rel_idx == 1


def _build_compositional_bpb_tables(tokenizer, token_bytes, *, device):
    spec = getattr(tokenizer, "spec", None)
    if spec is None:
        return None, None, None, None
    cache = getattr(tokenizer, "_bpb_modifier_tables_cache", {})
    cache_key = (str(device), str(token_bytes.dtype), int(token_bytes.numel()))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    base_bytes = token_bytes.detach().to("cpu").clone()
    stripped_base_bytes = base_bytes.clone()
    token_bytes_by_id = getattr(tokenizer, "_token_bytes_by_id", None)
    if callable(token_bytes_by_id):
        for token_id, raw_bytes in token_bytes_by_id().items():
            if 0 <= int(token_id) < int(base_bytes.numel()) and int(base_bytes[int(token_id)]) > 0:
                raw_bytes = bytes(raw_bytes)
                base_bytes[int(token_id)] = len(raw_bytes)
                try:
                    raw_text = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    stripped_base_bytes[int(token_id)] = len(raw_bytes)
                else:
                    if raw_text and raw_text.strip() != "":
                        stripped_base_bytes[int(token_id)] = len(raw_bytes.lstrip(b" "))
                    else:
                        stripped_base_bytes[int(token_id)] = len(raw_bytes)
    base_bytes = base_bytes.to(device=device)
    stripped_base_bytes = stripped_base_bytes.to(device=device)

    group_sizes = [int(v) for v in spec.modifier_group_sizes]
    num_groups = int(spec.num_modifier_groups)
    max_group_size = max(group_sizes) if group_sizes else 0
    delta_table = torch.zeros((num_groups, max_group_size), dtype=torch.int64, device=device)
    supported_table = torch.zeros((num_groups, max_group_size), dtype=torch.bool, device=device)

    determiner_groups = {"determiners", "article_det", "articles"}
    capitalization_groups = {"base_capitalization", "article_capitalization", "prep_capitalization"}
    zero_delta_groups = {"article_space_prefix", "prep_space_prefix"}
    for group_idx, group_name in enumerate(spec.group_names):
        group_size = int(group_sizes[group_idx])
        default_rel = int(spec.default_modifier[group_idx])
        value_names = list(spec.group_value_names.get(group_name, []))
        for rel_idx in range(group_size):
            if rel_idx == default_rel:
                supported_table[group_idx, rel_idx] = True
                continue
            value_name = value_names[rel_idx] if rel_idx < len(value_names) else ""
            delta = 0
            supported = False

            if group_name in capitalization_groups:
                # Case transforms can change UTF-8 byte length for non-ASCII text.
                supported = False
            elif group_name in zero_delta_groups:
                supported = True
            elif group_name in determiner_groups:
                for prefix in ("det_", "article_"):
                    if value_name.startswith(prefix):
                        delta = len(value_name[len(prefix):].encode("utf-8")) + 1
                        supported = True
                        break
            elif group_name == "prepositions" and value_name.startswith("prep_"):
                delta = len(value_name[len("prep_"):].encode("utf-8")) + 1
                supported = True
            elif group_name == "prefix_punctuation" and value_name.startswith("punct_prefix_"):
                delta = len(value_name[len("punct_prefix_"):].encode("utf-8"))
                supported = True
            elif group_name == "suffix_punctuation" and value_name.startswith("punct_suffix_"):
                delta = len(value_name[len("punct_suffix_"):].encode("utf-8"))
                supported = True
            elif group_name == "space_prefix":
                delta = 1 if _space_value_adds_prefix_space(value_name, rel_idx, default_rel) else 0
                supported = True

            if supported:
                delta_table[group_idx, rel_idx] = int(delta)
                supported_table[group_idx, rel_idx] = True

    cached = (base_bytes, stripped_base_bytes, delta_table, supported_table)
    cache[cache_key] = cached
    setattr(tokenizer, "_bpb_modifier_tables_cache", cache)
    return cached


def _unpack_eval_batch(batch):
    x, y = batch
    if isinstance(x, tuple) and isinstance(y, tuple):
        x_ids, x_mods = x
        y_ids, y_mods = y
        return x_ids, y_ids, x_mods, y_mods
    return x, y, None, None


def _modified_utf8_lengths(tokenizer, token_ids, modifier_rows):
    if hasattr(tokenizer, "utf8_len_with_modifiers_batch"):
        return tokenizer.utf8_len_with_modifiers_batch(token_ids, modifier_rows)
    if hasattr(tokenizer, "decode_token_with_modifiers"):
        return [
            len(tokenizer.decode_token_with_modifiers(token_id, row).encode("utf-8"))
            for token_id, row in zip(token_ids, modifier_rows)
        ]
    raise ValueError(
        "Compositional BPB evaluation requires tokenizer.utf8_len_with_modifiers_batch() "
        "or tokenizer.decode_token_with_modifiers()."
    )


def _compositional_target_bytes(y, y_mods, token_bytes, tokenizer):
    if tokenizer is None:
        raise ValueError(
            "Compositional BPB evaluation requires a tokenizer."
        )
    y_flat = y.view(-1)
    mods_flat = y_mods.view(-1, y_mods.size(-1))
    valid = y_flat >= 0
    y_safe = torch.where(valid, y_flat, torch.zeros_like(y_flat))
    num_bytes = torch.zeros_like(y_flat, dtype=token_bytes.dtype)
    base_bytes, stripped_base_bytes, delta_table, supported_table = _build_compositional_bpb_tables(
        tokenizer,
        token_bytes,
        device=token_bytes.device,
    )
    if base_bytes is not None and stripped_base_bytes is not None and delta_table is not None and supported_table is not None:
        group_ids = torch.arange(mods_flat.size(1), device=mods_flat.device).view(1, -1)
        max_group_size = delta_table.size(1)
        mods_safe = torch.clamp(mods_flat.to(torch.long), min=0, max=max_group_size - 1)
        deltas = delta_table[group_ids, mods_safe]
        supported = supported_table[group_ids, mods_safe].all(dim=-1)
        default_modifier = torch.tensor(
            [int(v) for v in tokenizer.spec.default_modifier],
            dtype=torch.long,
            device=mods_flat.device,
        ).view(1, -1)
        is_default = (mods_flat.to(torch.long) == default_modifier).all(dim=-1)
        base_for_row = torch.where(is_default, base_bytes[y_safe], stripped_base_bytes[y_safe])
        num_bytes = torch.where(valid, base_for_row, num_bytes)
        counted = valid & (base_bytes[y_safe] > 0) & supported
        if counted.any():
            delta_sum = deltas.sum(dim=-1).to(dtype=token_bytes.dtype)
            num_bytes[counted] = num_bytes[counted] + delta_sum[counted]
        fallback = valid & (base_bytes[y_safe] > 0) & (~supported)
    elif valid.any():
        fallback = valid & (token_bytes[y_safe] > 0)
    else:
        fallback = valid
    if fallback.any():
        idxs = fallback.nonzero(as_tuple=False).view(-1)
        token_ids = [int(v) for v in y_flat[idxs].tolist()]
        modifier_rows = [[int(x) for x in row] for row in mods_flat[idxs].tolist()]
        byte_lengths = _modified_utf8_lengths(tokenizer, token_ids, modifier_rows)
        num_bytes[idxs] = torch.tensor(byte_lengths, dtype=token_bytes.dtype, device=token_bytes.device)
    return num_bytes


def _compositional_joint_nll_sum_groups(model, x, y, x_mods, y_mods):
    """
    Joint compositional NLL per token for BPB:
      base CE + sum over modifier-group CEs.
    """
    # Pass 1: base CE from the regular model path (no hidden retention).
    base_loss = model(x, y, modifier_ids=x_mods, loss_reduction='none').view_as(y)

    # Pass 2: hidden only for modifier CE.
    hidden = model(x, modifier_ids=x_mods, return_hidden_only=True)
    batch_size, seq_len = y.shape
    valid_targets = y >= 0
    safe_targets = torch.where(valid_targets, y, torch.zeros_like(y))

    modifier_logits = model.get_modifier_logits(hidden, safe_targets)
    modifier_loss_sum = torch.zeros_like(base_loss)
    for group_idx, group_logits in enumerate(modifier_logits):
        group_targets = y_mods[..., group_idx].long()
        group_targets = torch.where(
            valid_targets,
            group_targets,
            torch.full_like(group_targets, -1),
        )
        group_loss = F.cross_entropy(
            group_logits.view(batch_size * seq_len, -1),
            group_targets.reshape(batch_size * seq_len),
            ignore_index=-1,
            reduction="none",
        ).view(batch_size, seq_len)
        modifier_loss_sum = modifier_loss_sum + group_loss
    return base_loss + modifier_loss_sum


@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes, tokenizer=None):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    apples:apples if you change the vocab size. The way this works is that instead of just
    calculating the average loss as usual, you calculate the sum loss, and independently
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    the number of bytes that the target tokens represent.

    The added complexity is so that:
    1) All "normal" tokens are normalized by the length of the token in bytes
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.

    In addition to evaluate_loss, we need the token_bytes tensor:
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    """
    # record the losses
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y, x_mods, y_mods = _unpack_eval_batch(next(batch_iter))
        if x_mods is None:
            loss2d = model(x, y, loss_reduction='none') # (B, T) or flattened
        else:
            # For compositional BPB, use joint token likelihood:
            # base loss + sum over modifier-group losses.
            loss2d = _compositional_joint_nll_sum_groups(model, x, y, x_mods, y_mods)
        loss2d = loss2d.view(-1) # flatten
        if y_mods is not None:
            counted = _target_counted_mask(y.view(-1), token_bytes)
            total_nats += loss2d[counted].sum()
            num_bytes2d = _compositional_target_bytes(y.view_as(loss2d), y_mods, token_bytes, tokenizer)
            total_bytes += num_bytes2d.sum()
        elif (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            y = y.view(-1) # flatten
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            counted = _target_counted_mask(y, token_bytes)
            y_safe = torch.where(y >= 0, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                counted,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += loss2d[counted].sum()
            total_bytes += num_bytes2d.sum()
        else:
            y = y.view(-1) # flatten
            # fast path: no ignored targets, safe to index directly
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # sum reduce across all ranks
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
