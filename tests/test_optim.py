import torch

from nanochat import optim as optim_mod
from nanochat.optim import DistMuonAdamW


class _FakeWork:
    def get_future(self):
        return object()


def test_dist_adamw_all_reduces_large_nondivisible_params(monkeypatch):
    calls = []

    def fake_all_reduce(grad, op, async_op):
        calls.append(("all_reduce", tuple(grad.shape)))
        return _FakeWork()

    def fake_reduce_scatter_tensor(output, input, op, async_op):
        calls.append(("reduce_scatter", tuple(output.shape), tuple(input.shape)))
        return _FakeWork()

    monkeypatch.setattr(optim_mod.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(optim_mod.dist, "reduce_scatter_tensor", fake_reduce_scatter_tensor)

    nondivisible = torch.nn.Parameter(torch.zeros(61, 32))
    shardable = torch.nn.Parameter(torch.zeros(64, 32))
    small = torch.nn.Parameter(torch.zeros(3, 32))
    unused = torch.nn.Parameter(torch.zeros(64, 32))
    for param in (nondivisible, shardable, small):
        param.grad = torch.ones_like(param)

    optimizer = DistMuonAdamW([
        dict(
            kind="adamw",
            params=[nondivisible, shardable, small, unused],
            lr=0.1,
            betas=(0.8, 0.96),
            eps=1e-10,
            weight_decay=0.01,
        )
    ])
    info = optimizer._reduce_adamw(optimizer.param_groups[0], world_size=4)

    assert info["param_infos"][nondivisible]["is_small"]
    assert not info["param_infos"][shardable]["is_small"]
    assert info["param_infos"][small]["is_small"]
    assert unused not in info["param_infos"]
    assert ("all_reduce", (61, 32)) in calls
    assert ("all_reduce", (3, 32)) in calls
    assert ("reduce_scatter", (16, 32), (64, 32)) in calls
