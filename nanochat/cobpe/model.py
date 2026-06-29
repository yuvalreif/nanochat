"""Input and output layers for CoBPE modifier groups."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(x):
    return F.rms_norm(x, (x.size(-1),))


def _round_up(value, multiple):
    return ((value + multiple - 1) // multiple) * multiple


class CoBPEModule(nn.Module):
    """
    Add modifier embeddings to base-token inputs and predict modifier values.

    CoBPE first predicts a regular BPE token. Its hidden state and the selected
    base-token unembedding then condition a modifier head that predicts one
    value independently for each modifier group.
    """

    def __init__(
        self,
        group_sizes,
        n_embd,
        linear_cls,
        pad_total_size_to=64,
        pad_refine_dim_to=64,
        conditioning_mode="mlp",
    ):
        super().__init__()
        self.conditioning_mode = str(conditioning_mode).lower()
        if self.conditioning_mode not in {"mlp", "concat_gated", "concat_gated_refine"}:
            raise ValueError(
                f"Unsupported CoBPE modifier conditioning_mode={conditioning_mode!r}. "
                "Expected one of: mlp, concat_gated, concat_gated_refine."
            )
        self.group_sizes = tuple(int(size) for size in group_sizes)
        self.num_groups = len(self.group_sizes)
        self.total_size = sum(self.group_sizes)
        self.padded_total_size = _round_up(max(1, self.total_size), pad_total_size_to)
        offsets = []
        offset = 0
        for group_size in self.group_sizes:
            offsets.append(offset)
            offset += group_size
        self.group_offsets = tuple(offsets)
        # AdamW reduce_scatter shards on dim 0, so keep modifier matrices shard-friendly.
        refine_dim = min(n_embd, _round_up(max(32, 4 * self.total_size), pad_refine_dim_to))
        self.embed = nn.Embedding(self.padded_total_size, n_embd)
        self.refine_fc = None
        self.refine_out = None
        self.hidden_head = None
        self.base_proj = None
        self.gate = None
        self.logit_refine = None
        self.gate_size = 0
        if self.conditioning_mode == "mlp":
            self.refine_fc = linear_cls(2 * n_embd, refine_dim, bias=False)
            self.refine_out = linear_cls(refine_dim, self.padded_total_size, bias=False)
        elif self.conditioning_mode in {"concat_gated", "concat_gated_refine"}:
            self.hidden_head = linear_cls(n_embd, self.padded_total_size, bias=False)
            self.base_proj = linear_cls(n_embd, self.padded_total_size, bias=False)
            self.gate_size = 8
            self.gate = linear_cls(n_embd, self.gate_size, bias=False)
            if self.conditioning_mode == "concat_gated_refine":
                self.logit_refine = linear_cls(self.padded_total_size, self.padded_total_size, bias=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.embed.weight, mean=0.0, std=0.8)
        if self.refine_fc is not None:
            torch.nn.init.normal_(self.refine_fc.weight, mean=0.0, std=0.001)
        if self.refine_out is not None:
            torch.nn.init.normal_(self.refine_out.weight, mean=0.0, std=0.001)
        if self.hidden_head is not None:
            torch.nn.init.normal_(self.hidden_head.weight, mean=0.0, std=0.001)
        if self.base_proj is not None:
            torch.nn.init.normal_(self.base_proj.weight, mean=0.0, std=0.001)
        if self.gate is not None:
            torch.nn.init.zeros_(self.gate.weight)
        if self.logit_refine is not None:
            torch.nn.init.zeros_(self.logit_refine.weight)
        if self.padded_total_size > self.total_size:
            self.embed.weight[self.total_size:].zero_()
            if self.refine_out is not None:
                self.refine_out.weight[self.total_size:].zero_()
            if self.hidden_head is not None:
                self.hidden_head.weight[self.total_size:].zero_()
            if self.base_proj is not None:
                self.base_proj.weight[self.total_size:].zero_()
            if self.logit_refine is not None:
                self.logit_refine.weight[self.total_size:].zero_()

    def _offset_ids(self, modifier_ids):
        if modifier_ids.dim() != 3:
            raise ValueError(f"modifier_ids must have shape [B, T, G], got {tuple(modifier_ids.shape)}")
        if modifier_ids.size(-1) != self.num_groups:
            raise ValueError(f"modifier_ids group mismatch: expected {self.num_groups}, got {modifier_ids.size(-1)}")
        modifier_ids = modifier_ids.long()
        group_sizes = torch.tensor(self.group_sizes, dtype=torch.long, device=modifier_ids.device).view(1, 1, -1)
        bad_mask = (modifier_ids < 0) | (modifier_ids >= group_sizes)
        if bad_mask.any():
            bad_indices = bad_mask.nonzero(as_tuple=False)[0].detach().cpu().tolist()
            bad_value = int(modifier_ids[tuple(bad_indices)].detach().cpu().item())
            group_idx = int(bad_indices[-1])
            raise ValueError(f"modifier_ids out of range for group {group_idx}: group_size={self.group_sizes[group_idx]}, sample_bad_id={bad_value}")
        offsets = torch.tensor(self.group_offsets, dtype=torch.long, device=modifier_ids.device).view(1, 1, -1)
        return modifier_ids + offsets

    def embed_sum(self, modifier_ids):
        return self.embed(self._offset_ids(modifier_ids)).sum(dim=-2)

    def logits(self, hidden, token_ids, base_unembedding):
        if hidden.dim() != 3:
            raise ValueError(f"hidden must have shape [B, T, C], got {tuple(hidden.shape)}")
        if token_ids.shape != hidden.shape[:2]:
            raise ValueError(f"token_ids must match hidden shape: {tuple(token_ids.shape)} != {tuple(hidden.shape[:2])}")
        hidden_flat = hidden.view(-1, hidden.size(-1))
        base_rep = F.embedding(token_ids.view(-1).long(), base_unembedding).to(dtype=hidden_flat.dtype)
        if self.conditioning_mode == "mlp":
            refine_in = torch.cat([_norm(hidden_flat), _norm(base_rep)], dim=-1)
            all_logits = self.refine_out(F.silu(self.refine_fc(refine_in)))
        elif self.conditioning_mode in {"concat_gated", "concat_gated_refine"}:
            hidden_logits = self.hidden_head(_norm(hidden_flat))
            base_logits = self.base_proj(_norm(base_rep))
            gate = torch.sigmoid(self.gate(hidden_flat)[:, :1])
            if gate.dtype != base_logits.dtype:
                gate = gate.to(dtype=base_logits.dtype)
            all_logits = hidden_logits + gate * base_logits
            if self.conditioning_mode == "concat_gated_refine":
                all_logits = all_logits + self.logit_refine(F.silu(all_logits))
        else:
            raise ValueError(f"Unknown CoBPE modifier conditioning_mode: {self.conditioning_mode}")
        outputs = []
        for group_idx, group_size in enumerate(self.group_sizes):
            start = self.group_offsets[group_idx]
            outputs.append(all_logits[:, start:start + group_size].view(*token_ids.shape, group_size))
        return outputs

    def loss(self, hidden, targets, target_modifier_ids, base_unembedding, loss_reduction):
        if target_modifier_ids is None:
            return None
        if target_modifier_ids.dim() != 3:
            raise ValueError(f"target_modifier_ids must have shape [B, T, G], got {tuple(target_modifier_ids.shape)}")
        if target_modifier_ids.shape[:2] != targets.shape:
            raise ValueError(f"target_modifier_ids must match targets shape: {tuple(target_modifier_ids.shape[:2])} != {tuple(targets.shape)}")
        if target_modifier_ids.size(-1) != self.num_groups:
            raise ValueError(f"target_modifier_ids group mismatch: expected {self.num_groups}, got {target_modifier_ids.size(-1)}")
        valid_targets = targets >= 0
        safe_targets = torch.where(valid_targets, targets, torch.zeros_like(targets))
        modifier_loss = None
        for group_idx, group_logits in enumerate(self.logits(hidden, safe_targets, base_unembedding)):
            group_targets = target_modifier_ids[..., group_idx].long()
            group_targets = torch.where(valid_targets, group_targets, torch.full_like(group_targets, -1))
            group_loss = F.cross_entropy(group_logits.view(-1, group_logits.size(-1)), group_targets.reshape(-1), ignore_index=-1, reduction=loss_reduction)
            modifier_loss = group_loss if modifier_loss is None else modifier_loss + group_loss
        return modifier_loss
