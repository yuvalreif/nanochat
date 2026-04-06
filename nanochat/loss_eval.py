"""
A number of functions that help with evaluating a base model.
"""
import math
import torch
import torch.distributed as dist


def _unpack_eval_batch(batch):
    x, y = batch
    if isinstance(x, tuple) and isinstance(y, tuple):
        x_ids, x_mods = x
        y_ids, y_mods = y
        return x_ids, y_ids, x_mods, y_mods
    return x, y, None, None


def _compositional_target_bytes(y, y_mods, token_bytes, tokenizer):
    if tokenizer is None or not hasattr(tokenizer, "decode_token_with_modifiers"):
        raise ValueError(
            "Compositional BPB evaluation requires a tokenizer with "
            "decode_token_with_modifiers()."
        )
    default_modifier = tuple(int(v) for v in tokenizer.get_default_modifier())
    y_flat = y.view(-1)
    mods_flat = y_mods.view(-1, y_mods.size(-1))
    num_bytes = torch.zeros_like(y_flat, dtype=token_bytes.dtype)
    valid = y_flat >= 0
    for idx in valid.nonzero(as_tuple=False).view(-1).tolist():
        token_id = int(y_flat[idx].item())
        modifier_row = tuple(int(v) for v in mods_flat[idx].tolist())
        if modifier_row == default_modifier:
            num_bytes[idx] = token_bytes[token_id]
            continue
        chunk = tokenizer.decode_token_with_modifiers(token_id, modifier_row)
        num_bytes[idx] = len(chunk.encode("utf-8"))
    return num_bytes


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
            loss2d = model(
                x,
                y,
                loss_reduction='none',
                modifier_ids=x_mods,
                target_modifier_ids=y_mods,
            )
        loss2d = loss2d.view(-1) # flatten
        y = y.view(-1) # flatten
        if y_mods is not None:
            num_bytes2d = _compositional_target_bytes(y.view_as(loss2d), y_mods, token_bytes, tokenizer)
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        elif (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
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
