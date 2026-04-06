import math

import torch

from nanochat.loss_eval import evaluate_bpb


class MockEvalModel:
    def __init__(self, loss_value=1.0):
        self.loss_value = float(loss_value)
        self.calls = []
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def __call__(
        self,
        input_ids,
        targets=None,
        loss_reduction="mean",
        modifier_ids=None,
        target_modifier_ids=None,
    ):
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "targets": None if targets is None else targets.clone(),
                "modifier_ids": None if modifier_ids is None else modifier_ids.clone(),
                "target_modifier_ids": None if target_modifier_ids is None else target_modifier_ids.clone(),
                "loss_reduction": loss_reduction,
            }
        )
        assert targets is not None
        assert loss_reduction == "none"
        return torch.full(targets.shape, self.loss_value, dtype=torch.float32)


class MockCompositionalTokenizer:
    def get_default_modifier(self):
        return [0]

    def decode_token_with_modifiers(self, token_id, modifier_row):
        lookup = {
            (10, (1,)): "AB",
            (11, (2,)): "xyz",
        }
        return lookup[(int(token_id), tuple(int(v) for v in modifier_row))]


def test_evaluate_bpb_passes_modifier_batches_and_counts_modified_bytes():
    model = MockEvalModel(loss_value=1.0)
    token_bytes = torch.zeros(32, dtype=torch.int64)
    token_bytes[11] = 1
    batch = (
        (torch.tensor([[99, 10]]), torch.tensor([[[0], [1]]])),
        (torch.tensor([[10, 11]]), torch.tensor([[[1], [2]]])),
    )

    bpb = evaluate_bpb(
        model,
        [batch],
        steps=1,
        token_bytes=token_bytes,
        tokenizer=MockCompositionalTokenizer(),
    )

    assert len(model.calls) == 1
    assert model.calls[0]["modifier_ids"].tolist() == [[[0], [1]]]
    assert model.calls[0]["target_modifier_ids"].tolist() == [[[1], [2]]]
    expected_bytes = len("AB".encode("utf-8")) + len("xyz".encode("utf-8"))
    expected_bpb = 2.0 / (math.log(2) * expected_bytes)
    assert math.isclose(bpb, expected_bpb)
