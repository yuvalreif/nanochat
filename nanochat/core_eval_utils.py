"""Utilities shared by CORE evaluation paths."""

from dataclasses import dataclass

import torch


def _format_prompt_debug(prompt):
    flat = prompt.replace("\n", "\\n")
    head = flat[:180]
    tail = flat[-180:] if len(flat) > 180 else flat
    return f"chars={len(prompt)} head='{head}' tail='{tail}'"


def find_changed_token_span(tokens_without, tokens_with, prompts):
    """
    Return the changed token span when comparing a prompt without/with continuation.
    This intentionally compares base token ids only, because adding continuation text
    can change boundary tokenization near the join point.
    """
    min_len = min(len(tokens_without), len(tokens_with))
    prefix_len = 0
    while prefix_len < min_len and int(tokens_without[prefix_len]) == int(tokens_with[prefix_len]):
        prefix_len += 1
    max_suffix = min_len - prefix_len
    suffix_len = 0
    while suffix_len < max_suffix:
        i_wo = len(tokens_without) - 1 - suffix_len
        i_w = len(tokens_with) - 1 - suffix_len
        if int(tokens_without[i_wo]) != int(tokens_with[i_w]):
            break
        suffix_len += 1
    start_idx = prefix_len
    end_idx = len(tokens_with) - suffix_len
    if start_idx >= end_idx:
        raise ValueError(
            "LM prompt tokenization produced no changed span between without/with prompts. "
            f"len(tokens_without)={len(tokens_without)} len(tokens_with)={len(tokens_with)} "
            f"prefix_len={prefix_len} suffix_len={suffix_len}. "
            f"prompt_without[{_format_prompt_debug(prompts[0])}] "
            f"prompt_with[{_format_prompt_debug(prompts[1])}]"
        )
    assert tokens_without[:start_idx] == tokens_with[:start_idx]
    if suffix_len > 0:
        assert tokens_without[-suffix_len:] == tokens_with[-suffix_len:]
    return start_idx, end_idx


def find_strict_prefix_span(tokens_without, tokens_with):
    start_idx = len(tokens_without)
    end_idx = len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    return start_idx, end_idx


@dataclass
class EvalForwardOutput:
    losses: torch.Tensor
    predictions: torch.Tensor
    modifier_predictions: torch.Tensor | None
    modifier_group_losses: list[torch.Tensor] | None


def _modifier_predictions(model, hidden, prediction_base_ids):
    pred_modifier_logits = model.get_modifier_logits(hidden, prediction_base_ids)
    return torch.stack([group_logits.argmax(dim=-1) for group_logits in pred_modifier_logits], dim=-1)


def _modifier_losses(model, hidden, target_ids, target_modifier_ids, batch_size, seq_len):
    modifier_logits = model.get_modifier_logits(hidden, target_ids)
    modifier_loss = None
    modifier_group_losses = []
    for group_idx, group_logits in enumerate(modifier_logits):
        group_loss = torch.nn.functional.cross_entropy(
            group_logits.view(batch_size * seq_len, -1),
            target_modifier_ids[..., group_idx].reshape(batch_size * seq_len),
            reduction='none',
        ).view(batch_size, seq_len)
        modifier_group_losses.append(group_loss)
        modifier_loss = group_loss if modifier_loss is None else modifier_loss + group_loss
    return modifier_loss, modifier_group_losses


@torch.no_grad()
def forward_model(model, input_ids, modifier_ids=None):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.
    """
    batch_size, seq_len = input_ids.size()
    if modifier_ids is None:
        outputs = model(input_ids)
        modifier_predictions = None
        modifier_group_losses = None
    else:
        outputs, hidden = model(input_ids, modifier_ids=modifier_ids, return_hidden=True)
        prediction_base_ids = outputs.argmax(dim=-1)
        modifier_predictions = _modifier_predictions(model, hidden, prediction_base_ids)
    # Roll the tensor to the left by one position to get the (autoregressive) target ids
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # Calculate cross entropy at all positions
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    if modifier_ids is not None:
        target_modifier_ids = torch.roll(modifier_ids, shifts=-1, dims=1)
        safe_target_ids = target_ids.clone()
        modifier_loss, modifier_group_losses = _modifier_losses(model, hidden, safe_target_ids, target_modifier_ids, batch_size, seq_len)
        losses = losses + modifier_loss
    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float('nan')
    # Get the argmax predictions at each position
    predictions = outputs.argmax(dim=-1)
    return EvalForwardOutput(
        losses=losses,
        predictions=predictions,
        modifier_predictions=modifier_predictions,
        modifier_group_losses=modifier_group_losses,
    )


def option_mean_loss_with_suffix_boundary_rule(
    losses: torch.Tensor,
    modifier_group_losses: list[torch.Tensor] | None,
    input_modifier_ids: torch.Tensor | None,
    tokenizer,
    option_idx: int,
    start_idx: int,
    end_idx: int,
) -> float:
    """
    Score an option with one boundary rule:
    for the final scored token, do not count suffix_punctuation loss when
    the target suffix punctuation is default (i.e. there is no explicit
    suffix punctuation in the option text).
    """
    s0, e0 = start_idx - 1, end_idx - 1
    span_losses = losses[option_idx, s0:e0].clone()
    if (
        modifier_group_losses is not None
        and input_modifier_ids is not None
        and hasattr(tokenizer, "spec")
        and ("suffix_punctuation" in tokenizer.spec.group_to_idx)
    ):
        suffix_group_idx = int(tokenizer.spec.group_to_idx["suffix_punctuation"])
        default_suffix = int(tokenizer.get_default_modifier()[suffix_group_idx])
        target_last_idx = end_idx - 1
        pred_last_idx = end_idx - 2
        if pred_last_idx >= s0 and target_last_idx >= 0:
            target_suffix_value = int(input_modifier_ids[option_idx, target_last_idx, suffix_group_idx].item())
            if target_suffix_value == default_suffix:
                span_losses[-1] = span_losses[-1] - modifier_group_losses[suffix_group_idx][option_idx, pred_last_idx]
    return span_losses.mean().item()
