import torch

from nanochat.gpt import GPT, GPTConfig


def _build_model(*, modifier_group_sizes=()):
    cfg = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=2,
        n_head=4,
        n_kv_head=4,
        n_embd=32,
        window_pattern="L",
        modifier_group_sizes=tuple(modifier_group_sizes),
        modifier_loss_weight=1.0,
    )
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def test_gpt_baseline_forward_still_works_without_modifiers():
    model = _build_model()
    ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
    targets = torch.randint(0, 32, (2, 8), dtype=torch.long)
    loss = model(ids, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gpt_compositional_forward_accepts_modifier_ids_and_targets():
    model = _build_model(modifier_group_sizes=(3, 4))
    ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
    targets = torch.randint(0, 32, (2, 8), dtype=torch.long)
    modifier_ids = torch.stack(
        [
            torch.randint(0, 3, (2, 8), dtype=torch.long),
            torch.randint(0, 4, (2, 8), dtype=torch.long),
        ],
        dim=-1,
    )
    loss = model(ids, targets, modifier_ids=modifier_ids, target_modifier_ids=modifier_ids)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gpt_rejects_modifier_ids_when_model_has_no_modifier_groups():
    model = _build_model()
    ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
    modifier_ids = torch.zeros((2, 8, 1), dtype=torch.long)
    try:
        model(ids, modifier_ids=modifier_ids)
    except ValueError as exc:
        assert "no modifier groups" in str(exc)
    else:
        raise AssertionError("Expected ValueError when passing modifier_ids to baseline model")


def test_gpt_rejects_wrong_modifier_group_count():
    model = _build_model(modifier_group_sizes=(3, 4))
    ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
    modifier_ids = torch.zeros((2, 8, 1), dtype=torch.long)
    try:
        model(ids, modifier_ids=modifier_ids)
    except ValueError as exc:
        assert "group mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for wrong modifier group count")
