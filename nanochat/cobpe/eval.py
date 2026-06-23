"""CoBPE-specific span and scoring rules used by CORE evaluation."""

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


def modifier_predictions_match_with_suffix_boundary_rule(
    predicted_modifiers: torch.Tensor,
    actual_modifiers: torch.Tensor,
    tokenizer,
) -> bool:
    """
    Exact modifier match for LM tasks, except final default suffix punctuation.

    If the gold answer does not include suffix punctuation, BPE-style generation
    would not have generated that suffix yet, so the final default suffix value is
    outside the answer boundary. Non-default gold suffix punctuation still counts.
    """
    matches = predicted_modifiers == actual_modifiers
    if (
        predicted_modifiers.numel() > 0
        and hasattr(tokenizer, "spec")
        and ("suffix_punctuation" in tokenizer.spec.group_to_idx)
    ):
        suffix_group_idx = int(tokenizer.spec.group_to_idx["suffix_punctuation"])
        default_suffix = int(tokenizer.get_default_modifier()[suffix_group_idx])
        target_suffix_value = int(actual_modifiers[-1, suffix_group_idx].item())
        if target_suffix_value == default_suffix:
            matches[-1, suffix_group_idx] = True
    return bool(torch.all(matches).item())
