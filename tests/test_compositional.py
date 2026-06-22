import json

import pytest

from nanochat.cobpe.tokenizer import CompositionalSpec, RustCoBPETokenizer, build_cobpe_metadata
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.token_codec import TokenItem, TokenSequence, stack_sequences


def test_build_cobpe_metadata_is_complete_and_json_serializable():
    payload = build_cobpe_metadata()
    json.dumps(payload)
    spec = CompositionalSpec.from_dict(payload)

    assert spec.group_names == (
        "space_prefix",
        "base_capitalization",
        "determiners",
        "article_capitalization",
        "prepositions",
        "prep_capitalization",
        "prefix_punctuation",
        "suffix_punctuation",
    )
    assert spec.default_modifier == (0,) * len(spec.group_names)
    assert "det_the" in spec.group_value_names["determiners"]
    assert "prep_on" in spec.group_value_names["prepositions"]
    assert "punct_suffix_." in spec.group_value_names["suffix_punctuation"]


def test_stack_sequences_handles_plain_and_modifier_sequences():
    plain_ids, plain_modifiers = stack_sequences([[1, 2], [3]], pad_token_id=0)
    assert plain_ids.tolist() == [[1, 2], [3, 0]]
    assert plain_modifiers is None

    sequences = [TokenSequence([1, 2], [[0], [1]]), TokenSequence([3], [[1]])]
    stacked_ids, modifiers = stack_sequences(sequences, pad_token_id=0, default_modifier=[0])
    assert stacked_ids.tolist() == [[1, 2], [3, 0]]
    assert modifiers.tolist() == [[[0], [1]], [[1], [0]]]


def test_compositional_spec_to_rust_config_exports_runtime_contract():
    spec = CompositionalSpec.from_dict(
        {
            "version": 1,
            "num_modifier_groups": 4,
            "group_names": [
                "space_prefix",
                "determiners",
                "prepositions",
                "suffix_punctuation",
            ],
            "group_value_names": {
                "space_prefix": ["no_space_prefix", "with_space_prefix"],
                "determiners": ["no_determiner", "det_the"],
                "prepositions": ["no_preposition", "prep_on"],
                "suffix_punctuation": ["no_suffix", "punct_suffix_!"],
            },
            "default_modifier": [0, 0, 0, 0],
            "entries": [
                {
                    "token_ids": [1, 2],
                    "base_ids": [10],
                    "modifier_rows": [[1, 1, 2, 0]],
                    "surface": "on the dog",
                }
            ],
        }
    )

    config = spec.to_rust_config()
    assert config["group_names"] == [
        "space_prefix",
        "determiners",
        "prepositions",
        "suffix_punctuation",
    ]
    assert config["token_meta"] == []
    assert config["reverse_entries"] == [
        {
            "token_ids": [1, 2],
            "base_ids": [10],
            "modifier_rows": [[1, 1, 2, 0]],
            "surface": "on the dog",
        }
    ]
    assert config["runtime"]["group_indices"]["space_prefix"] == 0
    assert config["runtime"]["literal_maps"]["determiners"]["the"] == {
        "group_name": "determiners",
        "rel_idx": 1,
    }
    assert config["runtime"]["literal_maps"]["prepositions"]["on"] == {
        "group_name": "prepositions",
        "rel_idx": 1,
    }
    assert config["runtime"]["literal_maps"]["suffix_punctuation"]["!"] == {
        "group_name": "suffix_punctuation",
        "rel_idx": 1,
    }
    assert config["runtime"]["multi_token_first_group_indices"] == [0, 1, 2]
    assert config["runtime"]["attachment_limits"] == {
        "max_prefix_punctuation": 1,
        "max_suffix_punctuation": 1,
    }


class ToyTokenizer:
    def __init__(self):
        self._bos = 99

    def encode_special(self, text):
        if text == "<|bos|>":
            return self._bos
        raise KeyError(text)

    def get_bos_token_id(self):
        return self._bos

    def get_vocab_size(self):
        return 128


class MockBackend:
    def process_text(self, text):
        assert text == "ab"
        return [12], [[2, 0]]

    def process_text_batch(self, texts):
        assert texts == ["ab", "ab"]
        return [([12], [[2, 0]]), ([12], [[2, 0]])]

    def decode_token_with_modifiers(self, token_id, modifier_row):
        assert int(token_id) == 12
        assert modifier_row == [2, 0]
        return "AB"

    def decode_with_modifiers(self, token_ids, modifier_rows):
        assert token_ids == [12]
        assert modifier_rows == [[2, 0]]
        return "AB"

    def utf8_len_with_modifiers_batch(self, token_ids, modifier_rows):
        assert token_ids == [12]
        assert modifier_rows == [[2, 0]]
        return [2]


def _simple_spec():
    return CompositionalSpec.from_dict(
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
    )


def test_rust_cobpe_tokenizer_delegates_to_rust_backend(monkeypatch):
    from nanochat.cobpe import tokenizer as compositional_mod

    monkeypatch.setattr(compositional_mod, "build_rust_backend", lambda spec, tokenizer_dir=None: MockBackend())

    tokenizer = compositional_mod.RustCoBPETokenizer(ToyTokenizer(), _simple_spec())

    single = tokenizer.encode_with_modifiers("ab", prepend=tokenizer.get_bos_token_id())
    batch = tokenizer.encode_with_modifiers(["ab", "ab"], prepend=tokenizer.get_bos_token_id())

    assert single == ([99, 12], [[0, 0], [2, 0]])
    assert batch == [([99, 12], [[0, 0], [2, 0]]), ([99, 12], [[0, 0], [2, 0]])]
    assert tokenizer.decode_token_with_modifiers(12, [2, 0]) == "AB"
    assert tokenizer.decode_with_modifiers([12], [[2, 0]]) == "AB"
    assert tokenizer.utf8_len_with_modifiers_batch([12], [[2, 0]]) == [2]

    seq = tokenizer.encode_sequence("ab", prepend=tokenizer.get_bos_token_id())
    batch_seq = tokenizer.encode_sequences(["ab", "ab"], prepend=tokenizer.get_bos_token_id())
    called_seq = tokenizer("ab", prepend=tokenizer.get_bos_token_id())
    called_batch = tokenizer(["ab", "ab"], prepend=tokenizer.get_bos_token_id())
    assert seq == TokenSequence([99, 12], [[0, 0], [2, 0]])
    assert batch_seq == [seq, seq]
    assert called_seq == seq
    assert called_batch == batch_seq
    assert tokenizer.decode_sequence(TokenSequence([12], [[2, 0]])) == "AB"
    assert tokenizer.empty_sequence() == TokenSequence([], [])
    assert tokenizer.token_item(99) == TokenItem(99, [0, 0])


def test_rust_cobpe_tokenizer_requires_rust_backend():
    with pytest.raises(RuntimeError, match="requires the Rust backend"):
        RustCoBPETokenizer(ToyTokenizer(), _simple_spec())


def test_get_tokenizer_loads_compositional_metadata(tmp_path, monkeypatch):
    from nanochat.cobpe import tokenizer as compositional_mod
    from nanochat import tokenizer as tokenizer_mod

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
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

    monkeypatch.setattr(tokenizer_mod.RustBPETokenizer, "from_directory", classmethod(lambda cls, path: ToyTokenizer()))

    class DummyCommon:
        @staticmethod
        def get_base_dir():
            return str(tmp_path)

    monkeypatch.setattr("nanochat.common.get_base_dir", DummyCommon.get_base_dir)
    monkeypatch.setattr(compositional_mod, "build_rust_backend", lambda spec, tokenizer_dir=None: MockBackend())

    tok = tokenizer_mod.get_tokenizer()
    assert isinstance(tok, RustCoBPETokenizer)
    assert tok.get_num_modifier_groups() == 1


def test_dataloader_with_modifiers_yields_modifier_batches(monkeypatch):
    class MockTokenizer:
        def get_bos_token_id(self):
            return 99

        def get_num_modifier_groups(self):
            return 1

        def encode_with_modifiers(self, texts, prepend=None, append=None, num_threads=8):
            assert prepend == 99
            return [
                ([99, 10, 11], [[0], [1], [2]])
                for _ in texts
            ]

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


def test_dataloader_with_modifiers_keeps_packing_alignment(monkeypatch):
    class MockTokenizer:
        def get_bos_token_id(self):
            return 99

        def get_num_modifier_groups(self):
            return 1

        def encode_with_modifiers(self, texts, prepend=None, append=None, num_threads=8):
            assert prepend == 99
            docs = [
                ([99, 1, 2], [[0], [1], [2]]),
                ([99, 3], [[0], [3]]),
                ([99, 4, 5, 6], [[0], [4], [5], [6]]),
            ]
            return docs[:len(texts)]

    def fake_document_batches(split, resume_state_dict, tokenizer_batch_size):
        while True:
            yield ["a", "b", "c"], (0, 0, 1)

    monkeypatch.setattr("nanochat.dataloader._document_batches", fake_document_batches)

    loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        MockTokenizer(),
        B=1,
        T=4,
        split="train",
        device="cpu",
        buffer_size=3,
        with_modifiers=True,
    )
    (inputs, input_mods), (targets, target_mods), _ = next(loader)
    assert inputs.tolist() == [[99, 4, 5, 6]]
    assert targets.tolist() == [[4, 5, 6, 99]]
    assert input_mods.tolist() == [[[0], [4], [5], [6]]]
    assert target_mods.tolist() == [[[4], [5], [6], [0]]]
