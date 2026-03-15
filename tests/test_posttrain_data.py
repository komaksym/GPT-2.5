import torch

from posttrain import data as posttrain_data


class DummyTokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}


class DummySplit:
    column_names = ["instruction", "context", "response", "category"]


class DummyDatasetDict(dict):
    def map(self, fn, batched, fn_kwargs, remove_columns):
        self.map_args = {
            "fn": fn,
            "batched": batched,
            "fn_kwargs": fn_kwargs,
            "remove_columns": remove_columns,
        }
        return self


class DummyRawDataset:
    def class_encode_column(self, column):
        self.class_encoded = column
        return self

    def train_test_split(self, test_size, stratify_by_column, seed):
        self.split_args = {
            "test_size": test_size,
            "stratify_by_column": stratify_by_column,
            "seed": seed,
        }
        return DummyDatasetDict(train=DummySplit(), test=DummySplit())


def test_format_prompt_without_context():
    prompt = posttrain_data.format_prompt("  Summarize this.  ", "")

    assert prompt == "Instruction:\nSummarize this.\nResponse:\n"


def test_format_prompt_with_context():
    prompt = posttrain_data.format_prompt("  Summarize this.  ", "  With notes. ")

    assert prompt == "Instruction:\nSummarize this.\nContext:\nWith notes.\nResponse:\n"


def test_tokenize_masks_prompt_and_appends_eos(monkeypatch):
    monkeypatch.setattr(posttrain_data.GPTConfig, "context_length", 10_000)
    tokenizer = DummyTokenizer()

    tokenized = posttrain_data.tokenize(
        {
            "instruction": ["Hi"],
            "context": [""],
            "response": ["OK"],
        },
        tokenizer,
    )

    prompt_ids = tokenizer(
        posttrain_data.format_prompt("Hi", ""), add_special_tokens=False
    )["input_ids"]
    response_ids = tokenizer("OK", add_special_tokens=False)["input_ids"]

    assert tokenized["input_ids"] == [
        prompt_ids + response_ids + [tokenizer.eos_token_id]
    ]
    assert tokenized["labels"] == [
        [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]
    ]
    assert tokenized["attention_mask"] == [[1] * len(tokenized["input_ids"][0])]


def test_tokenize_skips_sequences_longer_than_context(monkeypatch):
    monkeypatch.setattr(posttrain_data.GPTConfig, "context_length", 5)

    tokenized = posttrain_data.tokenize(
        {
            "instruction": ["This is too long"],
            "context": [""],
            "response": ["response"],
        },
        DummyTokenizer(),
    )

    assert tokenized == {"input_ids": [], "labels": [], "attention_mask": []}


def test_load_instruction_dataset_builds_stratified_split(monkeypatch):
    raw_dataset = DummyRawDataset()
    load_calls = []
    tokenizer = DummyTokenizer()

    def fake_load_dataset(name, split):
        load_calls.append((name, split))
        return raw_dataset

    monkeypatch.setattr(posttrain_data, "load_dataset", fake_load_dataset)

    dataset = posttrain_data._load_instruction_dataset(
        tokenizer,
        dataset_id="org/custom-dataset",
        split="validation",
    )

    assert load_calls == [("org/custom-dataset", "validation")]
    assert raw_dataset.class_encoded == "category"
    assert raw_dataset.split_args == {
        "test_size": 0.1,
        "stratify_by_column": "category",
        "seed": 42,
    }
    assert dataset.map_args == {
        "fn": posttrain_data.tokenize,
        "batched": True,
        "fn_kwargs": {"tokenizer": tokenizer},
        "remove_columns": DummySplit.column_names,
    }


def test_custom_collator_pads_to_longest_sequence():
    batch = posttrain_data.CustomCollatorWithPadding(tokenizer=DummyTokenizer())(
        [
            {"input_ids": [1, 2], "labels": [3, 4], "attention_mask": [1, 1]},
            {"input_ids": [5], "labels": [6], "attention_mask": [1]},
        ]
    )

    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2], [5, 0]]))
    assert torch.equal(batch["labels"], torch.tensor([[3, 4], [6, -100]]))
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1], [1, 0]]))
