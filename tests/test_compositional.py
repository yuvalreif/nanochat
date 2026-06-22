import json

import pytest

from nanochat.compositional import CompositionalSpec, CompositionalTokenizer, build_cobpe_metadata
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit


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

    modifier = list(spec.default_modifier)
    suffix_idx = spec.group_to_idx["suffix_punctuation"]
    modifier[suffix_idx] = spec.group_value_names["suffix_punctuation"].index("punct_suffix_.")
    assert spec.surface_for_token(10, modifier, lambda _ids: " dog") == "dog."


class ToyTokenizer:
    def __init__(self):
        self._bos = 99
        self._enc = {
            "a": [1],
            "b": [2],
            "ab": [1, 2],
            "cab": [3, 1, 2],
            "canted": [40],
            "dog": [20],
            "search": [82],
            "Search": [80, 81],
            "The cat": [70, 71, 72, 51],
            "\n\n": [30],
        }
        self._dec = {
            1: "a",
            2: "b",
            3: "c",
            10: "A",
            11: "B",
            12: "AB",
            13: "ab",
            20: "dog",
            21: "a",
            22: "b",
            31: "cant",
            32: " ed",
            40: "Canted",
            50: "the",
            51: "cat",
            70: "T",
            71: "he",
            72: " ",
            80: "S",
            81: "earch",
            82: "search",
            30: "\n\n",
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


class SurfaceTokenizer:
    def __init__(self, enc, dec):
        self._enc = {key: list(value) for key, value in enc.items()}
        self._dec = {int(key): value for key, value in dec.items()}

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


class FragmentedDecodeTokenizer:
    def decode(self, ids):
        key = tuple(int(token_id) for token_id in ids)
        if key == (1,):
            return "�"
        if key == (2,):
            return "�"
        if key == (1, 2):
            return "é"
        return "".join(self.decode([token_id]) for token_id in key)


class ByteFragmentTokenizer:
    def __init__(self):
        self._rank_to_bytes = {token_id: bytes([token_id]) for token_id in range(256)}
        self.enc = type("Enc", (), {})()
        self.enc._mergeable_ranks = {
            token_bytes: token_id for token_id, token_bytes in self._rank_to_bytes.items()
        }

    def encode(self, text, prepend=None, append=None, num_threads=8):
        if isinstance(text, list):
            return [self.encode(t, prepend=prepend, append=append, num_threads=num_threads) for t in text]
        ids = list(text.encode("utf-8"))
        if prepend is not None:
            ids = [prepend] + ids
        if append is not None:
            ids = ids + [append]
        return ids

    def decode(self, ids):
        return b"".join(self._rank_to_bytes[int(token_id)] for token_id in ids).decode("utf-8", errors="replace")


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
                        "base_ids": [21, 22],
                        "modifier_rows": [[1, 0], [0, 0]],
                        "surface": "ab",
                    }
                ],
            }
        ),
    )

    decoded = tokenizer.decode_with_modifiers([21, 22], [[1, 0], [0, 0]])
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
                        "base_ids": [13],
                        "modifier_rows": [[1, 1]],
                        "surface": "ab",
                    }
                ],
            }
        ),
    )

    decoded = tokenizer.decode_with_modifiers([13], [[1, 1]])
    assert decoded == " Ab"


def test_compositional_decode_uses_base_surface_instead_of_variant_surface():
    tokenizer = CompositionalTokenizer(
        ToyTokenizer(),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 1,
                "group_names": ["determiners"],
                "group_value_names": {
                    "determiners": ["no_determiner", "det_the"],
                },
                "default_modifier": [0],
                "entries": [
                    {
                        "token_ids": [1, 20],
                        "base_ids": [20],
                        "modifier_rows": [[1]],
                        "surface": "the dog",
                    }
                ],
                "inverse_entries": [
                    {"base_id": 20, "modifier": [1], "surface": "the dog"},
                ],
            }
        ),
    )

    decoded = tokenizer.decode_with_modifiers([20], [[1]])
    assert decoded == "the dog"


def test_compositional_python_path_detaches_raw_function_words_without_duplication():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {
                "The cat sat on the mat.": [1, 2, 3, 2, 4, 2, 5, 2, 6, 2, 7, 8],
            },
            {
                1: "The",
                2: " ",
                3: "cat",
                4: "sat",
                5: "on",
                6: "the",
                7: "mat",
                8: ".",
            },
        ),
        CompositionalSpec.from_dict(build_cobpe_metadata()),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("The cat sat on the mat.")
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "The cat sat on the mat."


def test_compositional_python_path_keeps_punctuation_only_matches_literal():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"[": [10]},
            {10: "["},
        ),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 1,
                "group_names": ["prefix_punctuation"],
                "group_value_names": {
                    "prefix_punctuation": ["no_prefix", "punct_prefix_["],
                },
                "default_modifier": [0],
                "entries": [
                    {
                        "token_ids": [10],
                        "base_ids": [10],
                        "modifier": [1],
                        "surface": "[",
                    }
                ],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("[")
    assert token_ids == [10]
    assert modifier_rows == [[0]]
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "["


def test_compositional_python_path_flushes_unattached_detached_prefixes():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"for your": [1, 2, 3]},
            {1: "for", 2: " ", 3: "your"},
        ),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 3,
                "group_names": ["space_prefix", "determiners", "prepositions"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "determiners": ["no_determiner", "det_your"],
                    "prepositions": ["no_preposition", "prep_for"],
                },
                "default_modifier": [0, 0, 0],
                "entries": [],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("for your")
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "for your"


def test_compositional_python_path_flushes_pending_literal_span_with_leading_space():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {'for "your': [1, 2, 3, 4]},
            {1: "for", 2: " ", 3: '"', 4: "your"},
        ),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 4,
                "group_names": ["space_prefix", "determiners", "prepositions", "prefix_punctuation"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "determiners": ["no_determiner", "det_your"],
                    "prepositions": ["no_preposition", "prep_for"],
                    "prefix_punctuation": ["no_prefix", 'punct_prefix_"'],
                },
                "default_modifier": [0, 0, 0, 0],
                "entries": [],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers('for "your')
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == 'for "your'


def test_compositional_python_path_flushes_pending_preposition_span_with_leading_space():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"found on the": [1, 2, 3, 2, 4]},
            {1: "found", 2: " ", 3: "on", 4: "the"},
        ),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 3,
                "group_names": ["space_prefix", "determiners", "prepositions"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "determiners": ["no_determiner", "det_the"],
                    "prepositions": ["no_preposition", "prep_on"],
                },
                "default_modifier": [0, 0, 0],
                "entries": [],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("found on the")
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "found on the"


def test_compositional_python_path_clears_invalid_bos_space_prefix():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"books": [10]},
            {10: "books", 20: "books"},
        ),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 1,
                "group_names": ["space_prefix"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                },
                "default_modifier": [0],
                "entries": [
                    {
                        "token_ids": [10],
                        "base_ids": [20],
                        "modifier": [1],
                        "surface": "books",
                    }
                ],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("books")
    assert modifier_rows == [[0]]
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "books"


def test_compositional_python_path_clears_contextual_space_after_joiner():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"well-trained": [1, 2, 3]},
            {1: "well", 2: "-", 3: "trained", 11: "well", 13: "trained"},
        ),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 2,
                "group_names": ["space_prefix", "suffix_punctuation"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "suffix_punctuation": ["no_suffix", "punct_suffix_-"],
                },
                "default_modifier": [0, 0],
                "entries": [
                    {
                        "token_ids": [1],
                        "base_ids": [11],
                        "modifier": [0, 0],
                        "surface": "well",
                    },
                    {
                        "token_ids": [3],
                        "base_ids": [13],
                        "modifier": [1, 0],
                        "surface": "trained",
                    },
                ],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("well-trained")
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "well-trained"


def test_compositional_python_path_clears_contextual_cap_after_apostrophe():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"friend's": [1, 2, 3]},
            {1: "friend", 2: "'", 3: "s", 11: "friend", 13: "s"},
        ),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 3,
                "group_names": ["space_prefix", "base_capitalization", "suffix_punctuation"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "base_capitalization": ["no_capitalization", "add_capitalization"],
                    "suffix_punctuation": ["no_suffix", "punct_suffix_'"],
                },
                "default_modifier": [0, 0, 0],
                "entries": [
                    {
                        "token_ids": [1],
                        "base_ids": [11],
                        "modifier": [0, 0, 0],
                        "surface": "friend",
                    },
                    {
                        "token_ids": [3],
                        "base_ids": [13],
                        "modifier": [1, 1, 0],
                        "surface": "s",
                    },
                ],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("friend's")
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "friend's"


def test_compositional_python_path_preserves_literal_space_before_dash():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"a – b": [1, 2, 3, 2, 4]},
            {1: "a", 2: " ", 3: "–", 4: "b"},
        ),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 1,
                "group_names": ["space_prefix"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                },
                "default_modifier": [0],
                "entries": [],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("a – b")
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "a – b"


def test_compositional_decode_preserves_native_surface_for_default_rows():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"unused": [1]},
            {1: "found", 2: " on", 3: " the", 4: " website"},
        ),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 1,
                "group_names": ["space_prefix"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                },
                "default_modifier": [0],
                "entries": [],
            }
        ),
    )

    text = tokenizer.decode_with_modifiers([1, 2, 3, 4], [[0], [0], [0], [0]])
    assert text == "found on the website"


def test_compositional_decode_coalesces_literal_default_runs():
    tokenizer = CompositionalTokenizer(
        FragmentedDecodeTokenizer(),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 1,
                "group_names": ["space_prefix"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                },
                "default_modifier": [0],
                "entries": [],
            }
        ),
    )

    text = tokenizer.decode_with_modifiers([1, 2], [[0], [0]])
    assert text == "é"


def test_compositional_decode_synthesizes_byte_fragment_component_space_prefix():
    tokenizer = CompositionalTokenizer(
        ByteFragmentTokenizer(),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 1,
                "group_names": ["space_prefix"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                },
                "default_modifier": [0],
                "entries": [],
            }
        ),
    )

    text = tokenizer.decode_with_modifiers([195, 169], [[1], [0]])
    assert text == " é"


def test_compositional_decode_synthesizes_byte_fragment_component_suffix():
    tokenizer = CompositionalTokenizer(
        ByteFragmentTokenizer(),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 1,
                "group_names": ["suffix_punctuation"],
                "group_value_names": {
                    "suffix_punctuation": ["no_suffix", "punct_suffix_."],
                },
                "default_modifier": [0],
                "entries": [],
            }
        ),
    )

    text = tokenizer.decode_with_modifiers([195, 169], [[0], [1]])
    assert text == "é."


def test_compositional_decode_does_not_split_byte_component_after_default_token():
    tokenizer = CompositionalTokenizer(
        ByteFragmentTokenizer(),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 1,
                "group_names": ["suffix_punctuation"],
                "group_value_names": {
                    "suffix_punctuation": ["no_suffix", "punct_suffix_."],
                },
                "default_modifier": [0],
                "entries": [],
            }
        ),
    )

    text = tokenizer.decode_with_modifiers([97, 195, 169], [[0], [0], [1]])
    assert text == "aé."


def test_compositional_python_path_attaches_modifiers_to_byte_fragment_component():
    tokenizer = CompositionalTokenizer(
        ByteFragmentTokenizer(),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 2,
                "group_names": ["space_prefix", "suffix_punctuation"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "suffix_punctuation": ["no_suffix", "punct_suffix_."],
                },
                "default_modifier": [0, 0],
                "entries": [],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers(" é.")
    assert token_ids == [195, 169]
    assert modifier_rows == [[1, 0], [0, 1]]
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == " é."


def test_compositional_utf8_len_counts_byte_fragments_without_replacement_chars():
    tokenizer = CompositionalTokenizer(
        ByteFragmentTokenizer(),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 2,
                "group_names": ["space_prefix", "suffix_punctuation"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "suffix_punctuation": ["no_suffix", "punct_suffix_."],
                },
                "default_modifier": [0, 0],
                "entries": [],
            }
        ),
    )

    assert tokenizer.utf8_len_with_modifiers_batch(
        [195, 169, 195, 169],
        [[1, 0], [0, 0], [0, 0], [0, 1]],
    ) == [2, 1, 1, 2]


def test_compositional_python_path_rejects_case_mismatched_entry():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"friend's": [1, 2, 3]},
            {1: "friend", 2: "'", 3: "s", 11: "friend", 13: "S"},
        ),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 3,
                "group_names": ["space_prefix", "base_capitalization", "suffix_punctuation"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "base_capitalization": ["no_capitalization", "add_capitalization"],
                    "suffix_punctuation": ["no_suffix", "punct_suffix_'"],
                },
                "default_modifier": [0, 0, 0],
                "entries": [
                    {
                        "token_ids": [1],
                        "base_ids": [11],
                        "modifier": [0, 0, 0],
                        "surface": "friend",
                    },
                    {
                        "token_ids": [3],
                        "base_ids": [13],
                        "modifier": [0, 1, 0],
                        "surface": "s",
                    },
                ],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("friend's")
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "friend's"


def test_compositional_python_path_keeps_all_caps_preposition_literal():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"ON switch": [1, 2, 3]},
            {1: "ON", 2: " ", 3: "switch"},
        ),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 3,
                "group_names": ["space_prefix", "prepositions", "prep_capitalization"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "prepositions": ["no_preposition", "prep_on"],
                    "prep_capitalization": ["no_prep_cap", "add_prep_cap"],
                },
                "default_modifier": [0, 0, 0],
                "entries": [],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("ON switch")
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "ON switch"


def test_compositional_python_path_keeps_mixed_case_surface_literal():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"polyA": [1]},
            {1: "polyA", 11: "polya"},
        ),
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
                        "token_ids": [1],
                        "base_ids": [11],
                        "modifier": [0, 1],
                        "surface": "polyA",
                    },
                ],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("polyA")
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "polyA"


def test_compositional_python_path_does_not_carry_space_prefix_across_newline():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"and \n\nDynamic": [1, 2, 3, 4]},
            {1: "and", 2: " ", 3: "\n\n", 4: "Dynamic", 14: "dynamic"},
        ),
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
                        "token_ids": [4],
                        "base_ids": [14],
                        "modifier": [0, 1],
                        "surface": "Dynamic",
                    },
                ],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("and \n\nDynamic")
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "and \n\nDynamic"


def test_compositional_python_path_does_not_collapse_double_space_before_span():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"method.  On similar": [1, 2, 3, 4]},
            {1: "method.", 2: " ", 3: " On", 4: " similar"},
        ),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 3,
                "group_names": ["space_prefix", "prepositions", "prep_capitalization"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "prepositions": ["no_preposition", "prep_on"],
                    "prep_capitalization": ["no_prep_cap", "add_prep_cap"],
                },
                "default_modifier": [0, 0, 0],
                "entries": [],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("method.  On similar")
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "method.  On similar"


def test_compositional_python_path_does_not_attach_prefix_punctuation_across_space():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"- Scientists": [1, 2, 3]},
            {1: "-", 2: " ", 3: "Scientists", 13: "scientists"},
        ),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 3,
                "group_names": ["space_prefix", "base_capitalization", "prefix_punctuation"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "base_capitalization": ["no_capitalization", "add_capitalization"],
                    "prefix_punctuation": ["no_prefix", "punct_prefix_-"],
                },
                "default_modifier": [0, 0, 0],
                "entries": [
                    {
                        "token_ids": [3],
                        "base_ids": [13],
                        "modifier": [0, 1, 0],
                        "surface": "Scientists",
                    },
                ],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("- Scientists")
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "- Scientists"


def test_compositional_python_path_clears_internal_space_prefix_for_multi_token_match():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"com.github": [1]},
            {1: "com.github", 11: "com.", 12: "github"},
        ),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 1,
                "group_names": ["space_prefix"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                },
                "default_modifier": [0],
                "entries": [
                    {
                        "token_ids": [1],
                        "base_ids": [11, 12],
                        "modifier_rows": [[0], [1]],
                        "surface": "com.github",
                    },
                ],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("com.github")
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "com.github"


def test_compositional_python_path_keeps_suffix_punctuation_chain():
    tokenizer = CompositionalTokenizer(
        SurfaceTokenizer(
            {"name']": [1, 2, 3]},
            {1: "name", 2: "'", 3: "]"},
        ),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 2,
                "group_names": ["space_prefix", "suffix_punctuation"],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "suffix_punctuation": ["no_suffix", "punct_suffix_'", "punct_suffix_]"],
                },
                "default_modifier": [0, 0],
                "entries": [],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("name']")
    assert tokenizer.decode_with_modifiers(token_ids, modifier_rows) == "name']"


def test_compositional_decode_uses_stored_surface_for_multi_token_reverse_matches():
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
                        "token_ids": [40],
                        "base_ids": [31, 32],
                        "modifier_rows": [[0, 1], [0, 0]],
                        "surface": "canted",
                    }
                ],
            }
        ),
    )

    decoded = tokenizer.decode_with_modifiers([31, 32], [[0, 1], [0, 0]])
    assert decoded == "Canted"


def test_compositional_decode_preserves_whitespace_only_tokens():
    tokenizer = CompositionalTokenizer(
        ToyTokenizer(),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 1,
                "default_modifier": [0],
                "entries": [],
            }
        ),
    )

    decoded = tokenizer.decode_with_modifiers([30], [[0]])
    assert decoded == "\n\n"


def test_compositional_encode_detaches_split_prefix_determiner():
    tokenizer = CompositionalTokenizer(
        ToyTokenizer(),
        CompositionalSpec.from_dict(
            {
                "version": 1,
                "num_modifier_groups": 4,
                "group_names": [
                    "space_prefix",
                    "base_capitalization",
                    "determiners",
                    "article_capitalization",
                ],
                "group_value_names": {
                    "space_prefix": ["no_space_prefix", "with_space_prefix"],
                    "base_capitalization": ["no_capitalization", "add_capitalization"],
                    "determiners": ["no_determiner", "det_the"],
                    "article_capitalization": ["no_capitalization", "add_capitalization"],
                },
                "default_modifier": [0, 0, 0, 0],
                "entries": [
                    {
                        "token_ids": [70, 71],
                        "base_ids": [50],
                        "modifier_rows": [[0, 1, 0, 0]],
                        "surface": "the",
                    }
                ],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("The cat")
    assert token_ids == [51]
    assert modifier_rows == [[0, 0, 1, 1]]


def test_compositional_encode_applies_titlecase_fallback():
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
                "entries": [],
            }
        ),
    )

    token_ids, modifier_rows = tokenizer.encode_with_modifiers("Search")
    assert token_ids == [82]
    assert modifier_rows == [[0, 1]]


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
