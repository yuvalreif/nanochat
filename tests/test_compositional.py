import json

import pytest

from nanochat.compositional import CompositionalSpec, CompositionalTokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit


class ToyTokenizer:
    def __init__(self):
        self._bos = 99
        self._enc = {
            "a": [1],
            "b": [2],
            "ab": [1, 2],
            "cab": [3, 1, 2],
        }
        self._dec = {
            1: "a",
            2: "b",
            3: "c",
            10: "A",
            11: "B",
            12: "AB",
            99: "<bos>",
        }

    def encode(self, text, prepend=None, append=None, num_threads=8):
        if isinstance(text, list):
            return [self.encode(t, prepend=prepend, append=append, num_threads=num_threads) for t in text]
        ids = list(self._enc[text])
        if prepend is not None:
            ids = [prepend] + ids
        if append is not None:
            ids = ids + [append]
        return ids

    def decode(self, ids):
        return "".join(self._dec[int(token_id)] for token_id in ids)

    def encode_special(self, text):
        if text == "<|bos|>":
            return self._bos
        raise KeyError(text)

    def get_bos_token_id(self):
        return self._bos


def test_compositional_spec_apply_longest_match_and_fallback():
    spec = CompositionalSpec.from_dict(
        {
            "version": 1,
            "num_modifier_groups": 2,
            "default_modifier": [0, 0],
            "entries": [
                {
                    "token_ids": [1],
                    "base_ids": [10],
                    "modifier": [1, 0],
                    "surface": "A",
                },
                {
                    "token_ids": [1, 2],
                    "base_ids": [12],
                    "modifier": [2, 0],
                    "surface": "AB",
                },
            ],
        }
    )

    token_ids, modifier_rows = spec.apply([3, 1, 2])
    assert token_ids == [3, 12]
    assert modifier_rows == [[0, 0], [2, 0]]


def test_compositional_tokenizer_batch_encode_and_decode():
    tokenizer = CompositionalTokenizer(
        ToyTokenizer(),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 2,
                "default_modifier": [0, 0],
                "entries": [
                    {
                        "token_ids": [1, 2],
                        "base_ids": [12],
                        "modifier": [2, 0],
                        "surface": "AB",
                    }
                ],
                "inverse_entries": [
                    {"base_id": 12, "modifier": [2, 0], "surface": "AB"},
                ],
            }
        ),
    )

    encoded = tokenizer.encode_with_modifiers(["ab"], prepend=tokenizer.get_bos_token_id())
    assert encoded == [([99, 12], [[0, 0], [2, 0]])]

    decoded = tokenizer.decode_with_modifiers([12], [[2, 0]])
    assert decoded == "AB"


def test_compositional_decode_uses_reverse_span_match_for_multi_token_entries():
    tokenizer = CompositionalTokenizer(
        ToyTokenizer(),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 2,
                "group_names": ["space_prefix", "base_capitalization"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "base_capitalization": ["no_capitalization", "add_capitalization"],
                },
                "default_modifier": [0, 0],
                "entries": [
                    {
                        "token_ids": [1, 2],
                        "base_ids": [10, 11],
                        "modifier_rows": [[1, 0], [0, 2]],
                        "surface": "ab",
                    }
                ],
            }
        ),
    )

    decoded = tokenizer.decode_with_modifiers([10, 11], [[1, 0], [0, 2]])
    assert decoded == " ab"


def test_compositional_decode_synthesizes_space_and_case_from_modifier_names():
    tokenizer = CompositionalTokenizer(
        ToyTokenizer(),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 2,
                "group_names": ["space_prefix", "base_capitalization"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "base_capitalization": ["no_capitalization", "add_capitalization"],
                },
                "default_modifier": [0, 0],
                "entries": [
                    {
                        "token_ids": [1, 2],
                        "base_ids": [12],
                        "modifier_rows": [[1, 1]],
                        "surface": "ab",
                    }
                ],
            }
        ),
    )

    decoded = tokenizer.decode_with_modifiers([12], [[1, 1]])
    assert decoded == " Ab"


def test_compositional_tokenizer_uses_rust_backend_when_available(monkeypatch, tmp_path):
    from nanochat import compositional as compositional_mod

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    class MockBackend:
        def process_text(self, text):
            assert text == "ab"
            return [12], [[2, 0]]

        def process_text_batch(self, texts):
            assert texts == ["ab", "ab"]
            return [([12], [[2, 0]]), ([12], [[2, 0]])]

    monkeypatch.setattr(compositional_mod, "build_rust_backend", lambda spec, tokenizer_dir=None: MockBackend())

    tokenizer = compositional_mod.CompositionalTokenizer(
        ToyTokenizer(),
        compositional_mod.CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 2,
                "default_modifier": [0, 0],
                "entries": [
                    {
                        "token_ids": [1, 2],
                        "base_ids": [12],
                        "modifier": [2, 0],
                        "surface": "AB",
                    }
                ],
            }
        ),
        tokenizer_dir=str(tokenizer_dir),
    )

    single = tokenizer.encode_with_modifiers("ab", prepend=tokenizer.get_bos_token_id())
    batch = tokenizer.encode_with_modifiers(["ab", "ab"], prepend=tokenizer.get_bos_token_id())

    assert single == ([99, 12], [[0, 0], [2, 0]])
    assert batch == [([99, 12], [[0, 0], [2, 0]]), ([99, 12], [[0, 0], [2, 0]])]


def test_get_tokenizer_loads_compositional_metadata(tmp_path, monkeypatch):
    from nanochat import tokenizer as tokenizer_mod
    from nanochat import compositional as compositional_mod

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tokenizer_dir / "compositional.json").write_text(
        json.dumps(
            {
                "version": 1,
                "num_modifier_groups": 1,
                "default_modifier": [0],
                "entries": [],
            }
        ),
        encoding="utf-8",
    )

    class DummyBaseTokenizer:
        pass

    monkeypatch.setattr(tokenizer_mod.RustBPETokenizer, "from_directory", classmethod(lambda cls, path: DummyBaseTokenizer()))

    class DummyCommon:
        @staticmethod
        def get_base_dir():
            return str(tmp_path)

    monkeypatch.setattr("nanochat.common.get_base_dir", DummyCommon.get_base_dir)
    monkeypatch.setattr(compositional_mod, "build_rust_backend", lambda spec, tokenizer_dir=None: object())
    tok = tokenizer_mod.get_tokenizer()
    assert isinstance(tok, CompositionalTokenizer)
    assert tok.get_num_modifier_groups() == 1


def test_compositional_tokenizer_requires_rust_backend_when_loading_from_dir(tmp_path):
    with pytest.raises(RuntimeError, match="requires the Rust backend"):
        CompositionalTokenizer(
            ToyTokenizer(),
            CompositionalSpec.from_dict(
                {
                    "version": 1,
                    "num_modifier_groups": 1,
                    "default_modifier": [0],
                    "entries": [],
                }
            ),
            tokenizer_dir=str(tmp_path),
        )


def test_dataloader_with_modifiers_yields_modifier_batches(monkeypatch):
    class MockTokenizer:
        def get_bos_token_id(self):
            return 99

        def get_num_modifier_groups(self):
            return 1

        def encode_with_modifiers(self, texts, prepend=None, append=None, num_threads=8):
            assert prepend == 99
            return [([99, 10, 11], [[0], [1], [2]]) for _ in texts]

    def fake_document_batches(split, resume_state_dict, tokenizer_batch_size):
        while True:
            yield ["doc"], (0, 0, 1)

    monkeypatch.setattr("nanochat.dataloader._document_batches", fake_document_batches)

    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        MockTokenizer(),
        B=1,
        T=2,
        split="train",
        device="cpu",
        buffer_size=1,
        with_modifiers=True,
    )
    (inputs, input_mods), (targets, target_mods), state = next(loader)
    assert inputs.tolist() == [[99, 10]]
    assert targets.tolist() == [[10, 11]]
    assert input_mods.tolist() == [[[0], [1]]]
    assert target_mods.tolist() == [[[1], [2]]]
    assert state == {"pq_idx": 0, "rg_idx": 0, "epoch": 1}


def test_dataloader_with_modifiers_requires_tokenizer_support(monkeypatch):
    def fake_document_batches(split, resume_state_dict, tokenizer_batch_size):
        while True:
            yield ["doc"], (0, 0, 1)

    monkeypatch.setattr("nanochat.dataloader._document_batches", fake_document_batches)

    with pytest.raises(ValueError, match="encode_with_modifiers"):
        loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
            object(),
            B=1,
            T=2,
            split="train",
            device="cpu",
            buffer_size=1,
            with_modifiers=True,
        )
        next(loader)
