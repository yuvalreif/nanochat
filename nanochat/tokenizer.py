"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""

import os
import re
import copy
from functools import lru_cache

from nanochat.compositional import CompositionalSpec, CompositionalTokenizer

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
NORMALIZED_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|\p{L}+|\p{N}{1,2}|[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
WORD_PATTERN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
PUNCT_TO_WORD_BOUNDARY_PATTERN = re.compile(r"([^\w\s])(?=\w)")
WORD_TO_PUNCT_BOUNDARY_PATTERN = re.compile(r"(?<=\w)([^\w\s])")

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def lowercase_standard_capitalization(text: str) -> str:
    def _replace(match: re.Match) -> str:
        word = match.group(0)
        if len(word) >= 2 and word[0].isupper() and word[1:].islower():
            return word.lower()
        return word

    return WORD_PATTERN.sub(_replace, text)


def separate_affix_punctuation(text: str) -> str:
    text = PUNCT_TO_WORD_BOUNDARY_PATTERN.sub(r"\1 ", text)
    text = WORD_TO_PUNCT_BOUNDARY_PATTERN.sub(r" \1", text)
    return text


def apply_space_cap_normalization(text: str) -> str:
    return lowercase_standard_capitalization(separate_affix_punctuation(text))


def _token_bytes_to_string(token_bytes: bytes) -> str:
    return token_bytes.decode("latin-1")


def _extract_vocab_and_merges(mergeable_ranks: dict[bytes, int]) -> tuple[dict[str, int], list[tuple[str, str]]]:
    vocab: dict[str, int] = {}
    merges_local: list[tuple[bytes, bytes, int]] = []
    for token, rank in mergeable_ranks.items():
        vocab[_token_bytes_to_string(token)] = rank
        if len(token) == 1:
            continue
        local: list[tuple[bytes, bytes, int]] = []
        for idx in range(1, len(token)):
            left = token[:idx]
            right = token[idx:]
            if left in mergeable_ranks and right in mergeable_ranks and (left + right) in mergeable_ranks:
                local.append((left, right, rank))
        local = sorted(local, key=lambda x: (mergeable_ranks[x[0]], mergeable_ranks[x[1]]))
        merges_local.extend(local)
    merges_local = sorted(merges_local, key=lambda x: x[2])
    merges = [
        (_token_bytes_to_string(left), _token_bytes_to_string(right))
        for (left, right, _) in merges_local
    ]
    return vocab, merges


def build_backend_tokenizer(
    *,
    mergeable_ranks: dict[bytes, int],
    pattern: str,
    special_tokens_in_id_order: list[str],
) -> HFTokenizer:
    vocab, merges = _extract_vocab_and_merges(mergeable_ranks)
    tokenizer = HFTokenizer(BPE(vocab, merges, fuse_unk=False))
    if hasattr(tokenizer.model, "ignore_merges"):
        tokenizer.model.ignore_merges = True
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(pattern), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(special_tokens_in_id_order)
    return tokenizer

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        # num_threads is ignored (only used by the nanochat Tokenizer for parallel encoding)
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        # Different HuggingFace models use different BOS tokens and there is little consistency
        # 1) attempt to find a <|bos|> token
        bos = self.encode_special("<|bos|>")
        # 2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3) if these fail, it's better to crash than to silently return None
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")


def detect_redundant_token_ids(tokenizer) -> dict[str, list[int]]:
    vocab_size = tokenizer.get_vocab_size()
    special_ids = {int(tokenizer.encode_special(tok)) for tok in tokenizer.get_special_tokens()}
    id_to_surface = {}
    surface_to_ids = {}
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        surface = tokenizer.decode([token_id])
        if not surface:
            continue
        id_to_surface[token_id] = surface
        surface_to_ids.setdefault(surface, []).append(token_id)

    def _first_alpha_index(text: str) -> int:
        for idx, ch in enumerate(text):
            if ch.isalpha():
                return idx
        return -1

    def _is_standard_titlecase(text: str, alpha_idx: int) -> bool:
        alpha_chars = [ch for ch in text[alpha_idx:] if ch.isalpha()]
        if len(alpha_chars) < 2:
            return False
        return alpha_chars[0].isupper() and all(ch.islower() for ch in alpha_chars[1:])

    space_prefixed = []
    capitalized = []
    for token_id, surface in id_to_surface.items():
        if len(surface) >= 2 and surface.startswith(" ") and not surface.startswith("  "):
            counterpart = surface[1:]
            if counterpart and counterpart in surface_to_ids:
                space_prefixed.append(int(token_id))
        alpha_idx = _first_alpha_index(surface)
        if alpha_idx >= 0 and _is_standard_titlecase(surface, alpha_idx):
            counterpart = surface[:alpha_idx] + surface[alpha_idx].lower() + surface[alpha_idx + 1 :]
            if counterpart in surface_to_ids:
                capitalized.append(int(token_id))
    return {
        "space_prefixed_with_unspaced_counterpart": sorted(set(space_prefixed)),
        "capitalized_with_lowercase_counterpart": sorted(set(capitalized)),
    }

# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo
import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens, # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def train_from_iterator_normalized_space_cap(cls, text_iterator, vocab_size):
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=NORMALIZED_SPLIT_PATTERN)
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe_normalized_space_cap",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        # yes this is confusing because this token is almost always PREPENDED to the beginning of the document
        # it most often is used to signal the start of a new sequence to the LLM during inference etc.
        # so in nanoChat we always use "<|bos|>" short for "beginning of sequence", but historically it is often called "<|endoftext|>".
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text can be either a string or a list of strings

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: slightly inefficient here? :( hmm
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: same
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        # save the encoding object to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")
        mergeable_ranks = getattr(self.enc, "_mergeable_ranks", None)
        pattern = getattr(self.enc, "_pat_str", None)
        special_tokens_map = getattr(self.enc, "_special_tokens", None)
        if isinstance(mergeable_ranks, dict) and pattern and isinstance(special_tokens_map, dict):
            special_tokens_in_id_order = [t for t, _ in sorted(special_tokens_map.items(), key=lambda kv: kv[1])]
            backend = build_backend_tokenizer(
                mergeable_ranks=mergeable_ranks,
                pattern=pattern,
                special_tokens_in_id_order=special_tokens_in_id_order,
            )
            tokenizer_json = os.path.join(tokenizer_dir, "tokenizer.json")
            backend.save(tokenizer_json)
            print(f"Saved tokenizer backend to {tokenizer_json}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids


def finalize_tokenizer_vocab(
    rust_tokenizer: RustBPETokenizer,
    *,
    target_vocab_size: int,
    redundant_token_ids: list[int],
) -> tuple[RustBPETokenizer, dict]:
    enc = rust_tokenizer.enc
    mergeable_ranks = getattr(enc, "_mergeable_ranks", None)
    pattern = getattr(enc, "_pat_str", None)
    special_tokens_map = getattr(enc, "_special_tokens", None)
    if not isinstance(mergeable_ranks, dict) or not pattern or not isinstance(special_tokens_map, dict):
        raise ValueError("Unexpected tokenizer internals while finalizing tokenizer vocab.")

    num_special = int(len(special_tokens_map))
    target_mergeable_size = int(target_vocab_size) - num_special
    if target_mergeable_size < 256:
        raise ValueError(f"target vocab size {target_vocab_size} leaves fewer than 256 mergeables")

    existing_mergeable_size = int(len(mergeable_ranks))
    if target_mergeable_size > existing_mergeable_size:
        raise ValueError(
            f"target mergeable size {target_mergeable_size} exceeds existing {existing_mergeable_size}"
        )

    redundant_mergeable = sorted(
        int(tid) for tid in redundant_token_ids
        if 0 <= int(tid) < existing_mergeable_size
    )
    redundant_set = set(redundant_mergeable)
    keep = set(range(min(256, existing_mergeable_size)))
    for tid in range(existing_mergeable_size):
        if tid in keep or tid in redundant_set:
            continue
        keep.add(int(tid))
        if len(keep) >= target_mergeable_size:
            break
    fallback_added = 0
    if len(keep) < target_mergeable_size:
        for tid in redundant_mergeable:
            if tid in keep:
                continue
            keep.add(int(tid))
            fallback_added += 1
            if len(keep) >= target_mergeable_size:
                break
    if len(keep) != target_mergeable_size:
        raise RuntimeError(
            f"failed selecting target mergeables: got {len(keep)} expected {target_mergeable_size}"
        )

    rank_to_token = {int(rank): token_bytes for token_bytes, rank in mergeable_ranks.items()}
    selected_old_ranks = sorted(keep)
    new_mergeable_ranks = {
        rank_to_token[old_rank]: new_rank
        for new_rank, old_rank in enumerate(selected_old_ranks)
    }
    specials_in_order = [t for t, _ in sorted(special_tokens_map.items(), key=lambda kv: kv[1])]
    new_specials = {tok: len(new_mergeable_ranks) + i for i, tok in enumerate(specials_in_order)}
    finalized_enc = tiktoken.Encoding(
        name=getattr(enc, "name", "rustbpe"),
        pat_str=pattern,
        mergeable_ranks=new_mergeable_ranks,
        special_tokens=new_specials,
    )
    stats = {
        "target_vocab_size": int(target_vocab_size),
        "existing_vocab_size": int(existing_mergeable_size + num_special),
        "target_mergeable_size": int(target_mergeable_size),
        "existing_mergeable_size": int(existing_mergeable_size),
        "redundant_candidates": int(len(redundant_mergeable)),
        "retained_redundant_mergeables": int(sum(1 for x in selected_old_ranks if x in redundant_set)),
        "dropped_redundant_mergeables": int(sum(1 for x in redundant_mergeable if x not in keep)),
        "fallback_added_mergeables": int(fallback_added),
    }
    return RustBPETokenizer(finalized_enc, "<|bos|>"), stats

# -----------------------------------------------------------------------------
# nanochat-specific convenience functions

def get_tokenizer():
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    metadata_path = os.path.join(tokenizer_dir, "compositional.json")
    if os.path.exists(metadata_path):
        spec = CompositionalSpec.from_path(metadata_path)
        return CompositionalTokenizer(tokenizer, spec, tokenizer_dir=tokenizer_dir)
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return tokenizer

def get_token_bytes(device="cpu"):
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
