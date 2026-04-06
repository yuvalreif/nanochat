import torch

from nanochat.core_eval import batch_sequences_mc, evaluate_example


class CoreCompositionalTokenizer:
    def __init__(self):
        self._bos = 99
        self._mapping = {
            "q A": ([1, 2], [[0], [1]]),
            "q B": ([1, 2], [[0], [2]]),
            "ctx": ([3], [[0]]),
            "ctx A": ([3, 2], [[0], [3]]),
        }

    def has_compositional_mode(self):
        return True

    def get_bos_token_id(self):
        return self._bos

    def get_default_modifier(self):
        return [0]

    def encode_with_modifiers(self, text, prepend=None, append=None, num_threads=8):
        if isinstance(text, list):
            return [
                self.encode_with_modifiers(t, prepend=prepend, append=append, num_threads=num_threads)
                for t in text
            ]
        token_ids, modifier_rows = self._mapping[text]
        token_ids = list(token_ids)
        modifier_rows = [list(row) for row in modifier_rows]
        if prepend is not None:
            token_ids.insert(0, int(prepend))
            modifier_rows.insert(0, [0])
        if append is not None:
            token_ids.append(int(append))
            modifier_rows.append([0])
        return token_ids, modifier_rows


class CoreCompositionalModel:
    def __init__(self):
        self.max_seq_len = None

    def __call__(self, input_ids, modifier_ids=None, return_hidden=False):
        batch_size, seq_len = input_ids.shape
        logits = torch.zeros(batch_size, seq_len, 128)
        logits[:, :, 99] = -1.0
        logits[input_ids == 99, 1] = 5.0
        logits[input_ids == 1, 2] = 5.0
        logits[input_ids == 3, 2] = 5.0
        next_modifier = torch.zeros(batch_size, seq_len, 1)
        if modifier_ids is not None:
            next_modifier = torch.roll(modifier_ids[..., :1].float(), shifts=-1, dims=1)
        if return_hidden:
            return logits, next_modifier
        return logits

    def get_modifier_logits(self, hidden, token_ids):
        batch_size, seq_len = token_ids.shape
        logits = torch.zeros(batch_size, seq_len, 4)
        target_group = hidden[..., 0].long().clamp(min=0, max=3)
        logits.scatter_(-1, target_group.unsqueeze(-1), 5.0)
        return [logits]


def test_batch_sequences_mc_uses_modifier_rows_for_prefix_matching():
    tokenizer = CoreCompositionalTokenizer()
    tokens, modifier_rows, start_idxs, end_idxs = batch_sequences_mc(tokenizer, ["q A", "q B"])

    assert tokens == [[99, 1, 2], [99, 1, 2]]
    assert modifier_rows == [[[0], [0], [1]], [[0], [0], [2]]]
    assert start_idxs == [2, 2]
    assert end_idxs == [3, 3]


def test_evaluate_example_supports_compositional_core_for_mc_and_lm():
    model = CoreCompositionalModel()
    tokenizer = CoreCompositionalTokenizer()

    mc_item = {
        "query": "q",
        "choices": ["A", "B"],
        "gold": 0,
    }
    mc_task = {
        "task_type": "multiple_choice",
        "num_fewshot": 0,
        "continuation_delimiter": " ",
    }
    assert evaluate_example(0, model, tokenizer, [mc_item], "cpu", mc_task) is True

    lm_item = {
        "context": "ctx",
        "continuation": "A",
        "gold": 0,
    }
    lm_task = {
        "task_type": "language_modeling",
        "num_fewshot": 0,
        "continuation_delimiter": " ",
    }
    assert evaluate_example(0, model, tokenizer, [lm_item], "cpu", lm_task) is True
