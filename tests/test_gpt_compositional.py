import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.cobpe.model import CoBPEModule
from nanochat.gpt import GPT, GPTConfig, Linear


def _build_model(
    *,
    modifier_group_sizes=(),
    n_layer=2,
    cobpe_smear=False,
    cobpe_backout=False,
    cobpe_smear_backout_scope="full",
    cobpe_modifier_conditioning="mlp",
):
    cfg = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=n_layer,
        n_head=4,
        n_kv_head=4,
        n_embd=32,
        window_pattern="L",
        modifier_group_sizes=tuple(modifier_group_sizes),
        modifier_loss_weight=1.0,
        cobpe_smear=cobpe_smear,
        cobpe_backout=cobpe_backout,
        cobpe_smear_backout_scope=cobpe_smear_backout_scope,
        cobpe_modifier_conditioning=cobpe_modifier_conditioning,
    )
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


def _norm(x):
    return F.rms_norm(x, (x.size(-1),))


class _CaptureBlock(nn.Module):
    def __init__(self, delta=None):
        super().__init__()
        self.delta = delta
        self.inputs = []

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        self.inputs.append(x.detach().clone())
        if self.delta is None:
            return x
        return x + self.delta.to(device=x.device, dtype=x.dtype)


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


def test_gpt_baseline_smear_still_runs_for_regular_bpe():
    model = _build_model()
    capture = _CaptureBlock()
    model.transformer.h = nn.ModuleList([capture, _CaptureBlock()])
    with torch.no_grad():
        model.resid_lambdas.fill_(1.0)
        model.x0_lambdas.zero_()
        model.smear_lambda.fill_(1.0)
        model.smear_gate.weight.zero_()

    ids = torch.tensor([[2, 5, 7]], dtype=torch.long)
    model(ids, return_hidden_only=True)

    base = _norm(model.transformer.wte(ids))
    expected = torch.cat([base[:, :1], base[:, 1:] + 0.5 * base[:, :-1]], dim=1)
    assert torch.allclose(capture.inputs[0], expected, atol=1e-5, rtol=1e-5)


def test_gpt_compositional_skips_smear():
    model = _build_model(modifier_group_sizes=(3, 4))
    capture = _CaptureBlock()
    model.transformer.h = nn.ModuleList([capture, _CaptureBlock()])
    with torch.no_grad():
        model.resid_lambdas.fill_(1.0)
        model.x0_lambdas.zero_()
    assert not hasattr(model, "smear_lambda")
    assert not hasattr(model, "smear_gate")

    ids = torch.tensor([[2, 5, 7]], dtype=torch.long)
    modifier_ids = torch.tensor([[[1, 2], [2, 3], [0, 1]]], dtype=torch.long)
    model(ids, modifier_ids=modifier_ids, return_hidden_only=True)

    composed = _norm(model.transformer.wte(ids) + model.cobpe.embed_sum(modifier_ids))
    assert torch.allclose(capture.inputs[0], composed, atol=1e-5, rtol=1e-5)


def test_gpt_compositional_can_enable_smear():
    model = _build_model(modifier_group_sizes=(3, 4), cobpe_smear=True)
    capture = _CaptureBlock()
    model.transformer.h = nn.ModuleList([capture, _CaptureBlock()])
    with torch.no_grad():
        model.resid_lambdas.fill_(1.0)
        model.x0_lambdas.zero_()
        model.smear_lambda.fill_(1.0)
        model.smear_gate.weight.zero_()
    assert hasattr(model, "smear_lambda")
    assert hasattr(model, "smear_gate")
    assert not hasattr(model, "backout_lambda")

    ids = torch.tensor([[2, 5, 7]], dtype=torch.long)
    modifier_ids = torch.tensor([[[1, 2], [2, 3], [0, 1]]], dtype=torch.long)
    model(ids, modifier_ids=modifier_ids, return_hidden_only=True)

    base = _norm(model.transformer.wte(ids) + model.cobpe.embed_sum(modifier_ids))
    expected = torch.cat([base[:, :1], base[:, 1:] + 0.5 * base[:, :-1]], dim=1)
    assert torch.allclose(capture.inputs[0], expected, atol=1e-5, rtol=1e-5)


def test_gpt_compositional_smear_can_apply_to_base_only():
    model = _build_model(
        modifier_group_sizes=(3, 4),
        cobpe_smear=True,
        cobpe_smear_backout_scope="base",
    )
    capture = _CaptureBlock()
    model.transformer.h = nn.ModuleList([capture, _CaptureBlock()])
    with torch.no_grad():
        model.resid_lambdas.fill_(1.0)
        model.x0_lambdas.zero_()
        model.smear_lambda.fill_(1.0)
        model.smear_gate.weight.zero_()

    ids = torch.tensor([[2, 5, 7]], dtype=torch.long)
    modifier_ids = torch.tensor([[[1, 2], [2, 3], [0, 1]]], dtype=torch.long)
    model(ids, modifier_ids=modifier_ids, return_hidden_only=True)

    base = _norm(model.transformer.wte(ids))
    smeared_base = torch.cat([base[:, :1], base[:, 1:] + 0.5 * base[:, :-1]], dim=1)
    expected = _norm(smeared_base + model.cobpe.embed_sum(modifier_ids))
    assert torch.allclose(capture.inputs[0], expected, atol=1e-5, rtol=1e-5)


def test_gpt_baseline_backout_still_runs_for_regular_bpe():
    model = _build_model(n_layer=3)
    delta = torch.linspace(-0.5, 0.5, model.config.n_embd).view(1, 1, -1)
    model.transformer.h = nn.ModuleList([_CaptureBlock(), _CaptureBlock(), _CaptureBlock(delta)])
    with torch.no_grad():
        model.resid_lambdas.fill_(1.0)
        model.x0_lambdas.zero_()
        model.smear_lambda.zero_()
        model.backout_lambda.fill_(0.5)

    ids = torch.tensor([[2, 5, 7]], dtype=torch.long)
    logits, hidden = model(ids, return_hidden=True)

    trunk_input = _norm(model.transformer.wte(ids))
    expected_hidden = _norm(trunk_input + delta - 0.5 * trunk_input)
    assert torch.allclose(hidden, expected_hidden, atol=1e-5, rtol=1e-5)
    expected_logits = model.lm_head(expected_hidden)[..., :model.config.vocab_size].float()
    expected_logits = 15 * torch.tanh(expected_logits / 15)
    assert torch.allclose(logits, expected_logits, atol=1e-5, rtol=1e-5)


def test_gpt_compositional_skips_backout():
    model = _build_model(modifier_group_sizes=(3, 4), n_layer=3)
    delta = torch.linspace(-0.5, 0.5, model.config.n_embd).view(1, 1, -1)
    model.transformer.h = nn.ModuleList([_CaptureBlock(), _CaptureBlock(), _CaptureBlock(delta)])
    with torch.no_grad():
        model.resid_lambdas.fill_(1.0)
        model.x0_lambdas.zero_()
    assert not hasattr(model, "smear_lambda")
    assert not hasattr(model, "backout_lambda")

    ids = torch.tensor([[2, 5, 7]], dtype=torch.long)
    modifier_ids = torch.tensor([[[1, 2], [2, 3], [0, 1]]], dtype=torch.long)
    logits, hidden = model(ids, modifier_ids=modifier_ids, return_hidden=True)

    trunk_input = _norm(model.transformer.wte(ids) + model.cobpe.embed_sum(modifier_ids))
    expected_hidden = _norm(trunk_input + delta)

    assert torch.allclose(hidden, expected_hidden, atol=1e-5, rtol=1e-5)
    expected_logits = model.lm_head(expected_hidden)[..., :model.config.vocab_size].float()
    expected_logits = 15 * torch.tanh(expected_logits / 15)
    assert torch.allclose(logits, expected_logits, atol=1e-5, rtol=1e-5)


def test_gpt_compositional_can_enable_backout():
    model = _build_model(modifier_group_sizes=(3, 4), n_layer=3, cobpe_backout=True)
    delta = torch.linspace(-0.5, 0.5, model.config.n_embd).view(1, 1, -1)
    model.transformer.h = nn.ModuleList([_CaptureBlock(), _CaptureBlock(), _CaptureBlock(delta)])
    with torch.no_grad():
        model.resid_lambdas.fill_(1.0)
        model.x0_lambdas.zero_()
        model.backout_lambda.fill_(0.5)
    assert not hasattr(model, "smear_lambda")
    assert hasattr(model, "backout_lambda")

    ids = torch.tensor([[2, 5, 7]], dtype=torch.long)
    modifier_ids = torch.tensor([[[1, 2], [2, 3], [0, 1]]], dtype=torch.long)
    logits, hidden = model(ids, modifier_ids=modifier_ids, return_hidden=True)

    trunk_input = _norm(model.transformer.wte(ids) + model.cobpe.embed_sum(modifier_ids))
    expected_hidden = _norm(trunk_input + delta - 0.5 * trunk_input)

    assert torch.allclose(hidden, expected_hidden, atol=1e-5, rtol=1e-5)
    expected_logits = model.lm_head(expected_hidden)[..., :model.config.vocab_size].float()
    expected_logits = 15 * torch.tanh(expected_logits / 15)
    assert torch.allclose(logits, expected_logits, atol=1e-5, rtol=1e-5)


def test_gpt_compositional_backout_can_apply_to_base_logits_only():
    model = _build_model(
        modifier_group_sizes=(3, 4),
        n_layer=3,
        cobpe_backout=True,
        cobpe_smear_backout_scope="base",
    )
    delta = torch.linspace(-0.5, 0.5, model.config.n_embd).view(1, 1, -1)
    model.transformer.h = nn.ModuleList([_CaptureBlock(), _CaptureBlock(), _CaptureBlock(delta)])
    with torch.no_grad():
        model.resid_lambdas.fill_(1.0)
        model.x0_lambdas.zero_()
        model.backout_lambda.fill_(0.5)

    ids = torch.tensor([[2, 5, 7]], dtype=torch.long)
    modifier_ids = torch.tensor([[[1, 2], [2, 3], [0, 1]]], dtype=torch.long)
    logits, hidden = model(ids, modifier_ids=modifier_ids, return_hidden=True)

    trunk_input = _norm(model.transformer.wte(ids) + model.cobpe.embed_sum(modifier_ids))
    expected_modifier_hidden = _norm(trunk_input + delta)
    expected_base_hidden = _norm(trunk_input + delta - 0.5 * trunk_input)

    assert torch.allclose(hidden, expected_modifier_hidden, atol=1e-5, rtol=1e-5)
    expected_logits = model.lm_head(expected_base_hidden)[..., :model.config.vocab_size].float()
    expected_logits = 15 * torch.tanh(expected_logits / 15)
    assert torch.allclose(logits, expected_logits, atol=1e-5, rtol=1e-5)


def test_gpt_compositional_has_no_smear_backout_params():
    model = _build_model(modifier_group_sizes=(3, 4))
    assert not hasattr(model, "smear_gate")
    assert not hasattr(model, "smear_lambda")
    assert not hasattr(model, "backout_lambda")
    assert model.estimate_flops() > 0
    params = model.num_scaling_params()
    assert params["total"] == sum(p.numel() for p in model.parameters())
    model.setup_optimizer()


def test_gpt_compositional_smear_backout_params_are_optional():
    model = _build_model(modifier_group_sizes=(3, 4), cobpe_smear=True, cobpe_backout=True)
    assert hasattr(model, "smear_gate")
    assert hasattr(model, "smear_lambda")
    assert hasattr(model, "backout_lambda")
    params = model.num_scaling_params()
    assert params["total"] == sum(p.numel() for p in model.parameters())
    model.setup_optimizer()


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


def test_gpt_compositional_pads_modifier_tables_for_ddp_sharding():
    model = _build_model(modifier_group_sizes=(31, 30))
    assert model.cobpe.total_size == 61
    assert model.cobpe.padded_total_size == 64
    assert model.cobpe.embed.weight.shape == (64, model.config.n_embd)
    assert model.cobpe.refine_out.weight.shape[0] == 64
    assert torch.any(model.cobpe.embed.weight[0] != 0)
    assert torch.any(model.cobpe.embed.weight[31] != 0)
    assert torch.equal(model.cobpe.embed.weight[61:], torch.zeros_like(model.cobpe.embed.weight[61:]))
    assert torch.equal(model.cobpe.refine_out.weight[61:], torch.zeros_like(model.cobpe.refine_out.weight[61:]))

    ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
    modifier_ids = torch.stack(
        [
            torch.randint(0, 31, (2, 8), dtype=torch.long),
            torch.randint(0, 30, (2, 8), dtype=torch.long),
        ],
        dim=-1,
    )
    logits = model.get_modifier_logits(torch.randn(2, 8, model.config.n_embd), ids)
    assert [group_logits.shape for group_logits in logits] == [(2, 8, 31), (2, 8, 30)]
    assert model.cobpe.embed_sum(modifier_ids).shape == (2, 8, model.config.n_embd)


def test_gpt_compositional_concat_gated_modifier_conditioning():
    model = _build_model(modifier_group_sizes=(3, 4), cobpe_modifier_conditioning="concat_gated")
    assert model.cobpe.conditioning_mode == "concat_gated"
    assert model.cobpe.refine_fc is None
    assert model.cobpe.refine_out is None
    assert model.cobpe.hidden_head.weight.shape == (64, model.config.n_embd)
    assert model.cobpe.base_proj.weight.shape == (64, model.config.n_embd)
    assert model.cobpe.gate.weight.shape == (1, model.config.n_embd)

    ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
    modifier_ids = torch.stack(
        [
            torch.randint(0, 3, (2, 8), dtype=torch.long),
            torch.randint(0, 4, (2, 8), dtype=torch.long),
        ],
        dim=-1,
    )
    loss = model(ids, ids, modifier_ids=modifier_ids, target_modifier_ids=modifier_ids)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    logits = model.get_modifier_logits(torch.randn(2, 8, model.config.n_embd), ids)
    assert [group_logits.shape for group_logits in logits] == [(2, 8, 3), (2, 8, 4)]


def test_cobpe_canonical_modifier_head_pads_refine_dim_for_adamw_sharding():
    module = CoBPEModule((31, 30), 768, Linear)
    assert module.refine_fc.out_features == 256
    assert module.refine_out.out_features == 64
    assert module.refine_fc.out_features % 8 == 0
    assert min(module.refine_out.in_features, module.refine_out.out_features) < 128


def test_cobpe_concat_gated_modifier_head_pads_for_adamw_sharding():
    module = CoBPEModule((31, 30), 768, Linear, conditioning_mode="concat_gated")
    assert module.hidden_head.weight.shape == (64, 768)
    assert module.base_proj.weight.shape == (64, 768)
    assert module.gate.weight.shape == (1, 768)
    assert module.refine_fc is None
    assert module.refine_out is None


def test_cobpe_modifier_logits_are_not_softcapped():
    module = CoBPEModule((3,), 32, Linear)
    with torch.no_grad():
        module.refine_fc.weight.fill_(1.0)
        module.refine_out.weight.fill_(1.0)
    hidden = torch.ones(1, 2, 32)
    token_ids = torch.zeros(1, 2, dtype=torch.long)
    base_unembedding = torch.ones(32, 32)

    (logits,) = module.logits(hidden, token_ids, base_unembedding)
    assert torch.all(logits > 15)


def test_cobpe_rejects_unknown_modifier_conditioning_mode():
    try:
        CoBPEModule((3,), 32, Linear, conditioning_mode="unknown")
    except ValueError as exc:
        assert "conditioning_mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported CoBPE modifier conditioning mode")


def test_gpt_compositional_adamw_params_are_8gpu_shardable():
    model = _build_model(modifier_group_sizes=(31, 30))
    optimizer = model.setup_optimizer()

    for group in optimizer.param_groups:
        if group["kind"] != "adamw":
            continue
        for param in group["params"]:
            if param.numel() >= 1024 and param.ndim > 0:
                assert param.shape[0] % 8 == 0


def test_gpt_modifier_parameters_use_unembedding_optimizer_bucket():
    model = _build_model(modifier_group_sizes=(3, 4))
    unembedding_lr = 0.004
    embedding_lr = 0.2
    optimizer = model.setup_optimizer(
        unembedding_lr=unembedding_lr,
        embedding_lr=embedding_lr,
    )
    dmodel_lr_scale = (model.config.n_embd / 768) ** -0.5
    expected_unembedding_lr = unembedding_lr * dmodel_lr_scale
    expected_embedding_lr = embedding_lr * dmodel_lr_scale

    def group_for(param):
        for group in optimizer.param_groups:
            if any(p is param for p in group["params"]):
                return group
        raise AssertionError("parameter not found in optimizer groups")

    modifier_params = [
        *model.cobpe.parameters(),
    ]
    for param in modifier_params:
        group = group_for(param)
        assert group["kind"] == "adamw"
        assert group["lr"] == expected_unembedding_lr
        assert group["initial_lr"] == expected_unembedding_lr
        assert group["betas"] == (0.8, 0.96)
        assert group["weight_decay"] == 0.01

    wte_group = group_for(model.transformer.wte.weight)
    assert wte_group["kind"] == "adamw"
    assert wte_group["lr"] == expected_embedding_lr
    assert wte_group["betas"] == (0.8, 0.995)
    assert wte_group["weight_decay"] == 0.001


def test_gpt_concat_gated_modifier_parameters_use_unembedding_optimizer_bucket():
    model = _build_model(modifier_group_sizes=(3, 4), cobpe_modifier_conditioning="concat_gated")
    optimizer = model.setup_optimizer()

    def group_for(param):
        for group in optimizer.param_groups:
            if any(p is param for p in group["params"]):
                return group
        raise AssertionError("parameter not found in optimizer groups")

    for param in model.cobpe.parameters():
        group = group_for(param)
        assert group["kind"] == "adamw"
        assert group["betas"] == (0.8, 0.96)
        assert group["weight_decay"] == 0.01
