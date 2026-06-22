import math

import torch

from nanochat.cobpe.tokenizer import CompositionalSpec
from nanochat.cobpe.bpb import compositional_target_bytes
from nanochat.loss_eval import evaluate_bpb


class MockEvalModel:
    def __init__(self, loss_value=1.0, modifier_group_sizes=(3,)):
        self.loss_value = float(loss_value)
        self.modifier_group_sizes = tuple(int(v) for v in modifier_group_sizes)
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
        return_hidden_only=False,
    ):
        if return_hidden_only:
            return torch.zeros((*input_ids.shape, 1), dtype=torch.float32)
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

    def get_modifier_logits(self, hidden, safe_targets):
        return [
            torch.zeros((*safe_targets.shape, group_size), dtype=torch.float32)
            for group_size in self.modifier_group_sizes
        ]


class MockCoBPETokenizer:
    def get_default_modifier(self):
        return [0]

    def decode_token_with_modifiers(self, token_id, modifier_row):
        lookup = {
            (1, (0,)): "<|bos|>",
            (10, (1,)): "AB",
            (10, (1, 3)): "A",
            (11, (2,)): "xyz",
        }
        return lookup[(int(token_id), tuple(int(v) for v in modifier_row))]


class MockSpaceStrippingTokenizer:
    def __init__(self):
        self.base_id = 10
        self.spec = CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 5,
                "modifier_group_sizes": [2, 2, 2, 2, 2],
                "group_names": [
                    "space_prefix",
                    "determiners",
                    "prepositions",
                    "prefix_punctuation",
                    "suffix_punctuation",
                ],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "determiners": ["no_determiner", "article_the"],
                    "prepositions": ["no_preposition", "prep_on"],
                    "prefix_punctuation": ["no_prefix", 'punct_prefix_"'],
                    "suffix_punctuation": ["no_suffix", "punct_suffix_."],
                },
                "default_modifier": [0, 0, 0, 0, 0],
                "entries": [],
            }
        )

    def _token_bytes_by_id(self):
        return {self.base_id: b" dog"}

    def _decode_base(self, token_ids):
        assert token_ids == [self.base_id]
        return " dog"

    def decode_token_with_modifiers(self, token_id, modifier_row):
        assert int(token_id) == self.base_id
        surfaces = {
            (0, 0, 0, 0, 0): " dog",
            (0, 0, 0, 0, 1): "dog.",
            (1, 0, 0, 0, 0): " dog",
            (0, 1, 0, 0, 0): "the dog",
            (0, 0, 1, 0, 0): "on dog",
            (0, 0, 0, 1, 0): '"dog',
            (1, 1, 1, 1, 1): ' "on the dog.',
        }
        return surfaces[tuple(int(v) for v in modifier_row)]


def test_evaluate_bpb_passes_modifier_batches_and_counts_modified_bytes():
    model = MockEvalModel(loss_value=1.0)
    token_bytes = torch.zeros(32, dtype=torch.int64)
    token_bytes[10] = 99
    token_bytes[11] = 1
    batch = (
        (torch.tensor([[99, 10, 11]]), torch.tensor([[[0], [1], [2]]])),
        (torch.tensor([[10, 11, 1]]), torch.tensor([[[1], [2], [0]]])),
    )

    bpb = evaluate_bpb(
        model,
        [batch],
        steps=1,
        token_bytes=token_bytes,
        tokenizer=MockCoBPETokenizer(),
    )

    assert len(model.calls) == 1
    assert model.calls[0]["modifier_ids"].tolist() == [[[0], [1], [2]]]
    assert model.calls[0]["target_modifier_ids"] is None
    expected_bytes = len("AB".encode("utf-8")) + len("xyz".encode("utf-8"))
    expected_nats = 2.0 * (1.0 + math.log(3.0))
    expected_bpb = expected_nats / (math.log(2) * expected_bytes)
    assert math.isclose(bpb, expected_bpb, rel_tol=1e-6)


def test_evaluate_bpb_sums_modifier_group_losses():
    model = MockEvalModel(loss_value=0.5, modifier_group_sizes=(2, 4))
    token_bytes = torch.zeros(32, dtype=torch.int64)
    token_bytes[10] = 1
    batch = (
        (torch.tensor([[99]]), torch.tensor([[[0, 0]]])),
        (torch.tensor([[10]]), torch.tensor([[[1, 3]]])),
    )

    bpb = evaluate_bpb(
        model,
        [batch],
        steps=1,
        token_bytes=token_bytes,
        tokenizer=MockCoBPETokenizer(),
    )

    expected_nats = 0.5 + math.log(2.0) + math.log(4.0)
    expected_bpb = expected_nats / math.log(2)
    assert math.isclose(bpb, expected_bpb, rel_tol=1e-6)


def test_compositional_target_bytes_strip_base_space_for_non_default_modifiers():
    tokenizer = MockSpaceStrippingTokenizer()
    rows = [
        [0, 0, 0, 0, 0],  # default row preserves the literal base token surface: " dog"
        [0, 0, 0, 0, 1],  # suffix punctuation: "dog."
        [1, 0, 0, 0, 0],  # explicit space prefix: " dog"
        [0, 1, 0, 0, 0],  # determiner/article: "the dog"
        [0, 0, 1, 0, 0],  # preposition: "on dog"
        [0, 0, 0, 1, 0],  # prefix punctuation: '"dog'
        [1, 1, 1, 1, 1],  # combined additive modifiers: ' "on the dog.'
    ]
    token_bytes = torch.zeros(32, dtype=torch.int64)
    token_bytes[tokenizer.base_id] = len(b" dog")
    y = torch.full((1, len(rows)), tokenizer.base_id, dtype=torch.long)
    y_mods = torch.tensor([rows], dtype=torch.long)

    fast_lengths = compositional_target_bytes(y, y_mods, token_bytes, tokenizer)
    decoded_lengths = torch.tensor(
        [
            len(tokenizer.decode_token_with_modifiers(tokenizer.base_id, row).encode("utf-8"))
            for row in rows
        ],
        dtype=torch.int64,
    )

    assert fast_lengths.tolist() == decoded_lengths.tolist()
    assert decoded_lengths.tolist() == [4, 4, 4, 7, 6, 4, 13]
