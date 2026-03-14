from types import SimpleNamespace

import numpy as np
import torch

from posttrain import tune


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
    prompt = tune.format_prompt("  Summarize this.  ", "")

    assert prompt == "Instruction:\nSummarize this.\nResponse:\n"


def test_format_prompt_with_context():
    prompt = tune.format_prompt("  Summarize this.  ", "  With notes. ")

    assert prompt == "Instruction:\nSummarize this.\nContext:\nWith notes.\nResponse:\n"


def test_tokenize_masks_prompt_and_appends_eos(monkeypatch):
    monkeypatch.setattr(tune.GPTConfig, "context_length", 10_000)
    tokenizer = DummyTokenizer()

    tokenized = tune.tokenize(
        {
            "instruction": ["Hi"],
            "context": [""],
            "response": ["OK"],
        },
        tokenizer,
    )

    prompt_ids = tokenizer(tune.format_prompt("Hi", ""), add_special_tokens=False)["input_ids"]
    response_ids = tokenizer("OK", add_special_tokens=False)["input_ids"]

    assert tokenized["input_ids"] == [prompt_ids + response_ids + [tokenizer.eos_token_id]]
    assert tokenized["labels"] == [[-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]]
    assert tokenized["attention_mask"] == [[1] * len(tokenized["input_ids"][0])]


def test_tokenize_skips_sequences_longer_than_context(monkeypatch):
    monkeypatch.setattr(tune.GPTConfig, "context_length", 5)

    tokenized = tune.tokenize(
        {
            "instruction": ["This is too long"],
            "context": [""],
            "response": ["response"],
        },
        DummyTokenizer(),
    )

    assert tokenized == {"input_ids": [], "labels": [], "attention_mask": []}


def test_hf_wrapper_restores_tied_embedding_hooks(monkeypatch):
    class FakeEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(1, 1))

    class FakeLinear(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight

    class FakeTransformerLM(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.emb = FakeEmbedding()
            self.linear = FakeLinear(self.emb.weight)

        def forward(self, input_ids, attention_mask=None):
            batch, seq_len = input_ids.shape
            return torch.zeros(batch, seq_len, 1), None

    monkeypatch.setattr(tune, "TransformerLM", FakeTransformerLM)

    model = tune.HFTransformerLM(
        tune.MyConfig(vocab_size=1, context_length=1, num_layers=1, num_heads=1, d_model=1, d_ff=64, device="cpu")
    )

    assert model._tied_weights_keys == {"model.linear.weight": "model.emb.weight"}
    assert model.get_input_embeddings() is model.model.emb
    assert model.get_output_embeddings() is model.model.linear
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight


def test_load_pretraining_model_uses_override_paths(monkeypatch):
    snapshot_calls = []
    load_calls = []
    base_model = object()

    monkeypatch.setattr(tune, "_build_base_model", lambda: base_model)
    monkeypatch.setattr(tune, "snapshot_download", lambda *args, **kwargs: snapshot_calls.append((args, kwargs)))
    monkeypatch.setattr(tune, "load_checkpoint", lambda path, model: load_calls.append((path, model)))

    loaded = tune._load_pretraining_model(
        repo_id="repo/custom",
        checkpoint_pattern="weights/*",
        checkpoint_path="custom/checkpoint",
        local_dir="custom-local",
    )

    assert loaded is base_model
    assert snapshot_calls == [
        (
            ("repo/custom",),
            {
                "allow_patterns": "weights/*",
                "repo_type": "model",
                "local_dir": "custom-local",
            },
        )
    ]
    assert load_calls == [("custom/checkpoint", base_model)]


def test_load_instruction_dataset_builds_stratified_split(monkeypatch):
    raw_dataset = DummyRawDataset()
    load_calls = []
    tokenizer = DummyTokenizer()

    def fake_load_dataset(name, split):
        load_calls.append((name, split))
        return raw_dataset

    monkeypatch.setattr(tune, "load_dataset", fake_load_dataset)

    dataset = tune._load_instruction_dataset(
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
        "fn": tune.tokenize,
        "batched": True,
        "fn_kwargs": {"tokenizer": tokenizer},
        "remove_columns": DummySplit.column_names,
    }


def test_custom_collator_pads_to_longest_sequence():
    batch = tune.CustomCollatorWithPadding(tokenizer=DummyTokenizer())(
        [
            {"input_ids": [1, 2], "labels": [3, 4], "attention_mask": [1, 1]},
            {"input_ids": [5], "labels": [6], "attention_mask": [1]},
        ]
    )

    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2], [5, 0]]))
    assert torch.equal(batch["labels"], torch.tensor([[3, 4], [6, -100]]))
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1], [1, 0]]))


def test_compute_metrics_returns_perplexity():
    metrics = tune.compute_metrics(SimpleNamespace(losses=np.array([0.0, np.log(4.0)])))

    assert metrics == {"perplexity": 2.0}


def test_inference_test_uses_pretraining_checkpoint(monkeypatch, capsys):
    captured = {}
    generate_calls = []

    class FakeHFModel:
        def __init__(self):
            self.device = None

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    fake_hf_model = FakeHFModel()

    def fake_load_pretraining_model(**kwargs):
        captured["pretraining"] = kwargs
        return "base-model"

    def fake_build_hf_model(base_model):
        captured["base_model"] = base_model
        return fake_hf_model

    monkeypatch.setattr(tune, "_load_pretraining_model", fake_load_pretraining_model)
    monkeypatch.setattr(tune, "_build_hf_model", fake_build_hf_model)
    monkeypatch.setattr(
        tune,
        "generate",
        lambda **kwargs: generate_calls.append(kwargs) or ["seq-a", "seq-b"],
    )

    tune.inference_test(
        prompt="prompt",
        pretraining_checkpoint=True,
        pretraining_repo_id="repo/custom",
        pretraining_checkpoint_pattern="weights/*",
        pretraining_checkpoint_path="custom/checkpoint",
        pretraining_local_dir="custom-local",
    )

    assert captured["pretraining"] == {
        "repo_id": "repo/custom",
        "checkpoint_pattern": "weights/*",
        "checkpoint_path": "custom/checkpoint",
        "local_dir": "custom-local",
    }
    assert captured["base_model"] == "base-model"
    assert fake_hf_model.device == tune.GPTConfig.device
    assert generate_calls == [
        {
            "prompt": "prompt",
            "max_tokens": 50,
            "context_length": tune.GPTConfig.context_length,
            "batch_size": 5,
            "model": fake_hf_model,
            "temp": 0.9,
            "top_p": 0.8,
            "device": fake_hf_model.device,
        }
    ]
    assert capsys.readouterr().out == "seq-a\nseq-b\n"


def test_inference_test_uses_saved_posttraining_checkpoint(monkeypatch):
    generated = []
    loaded_paths = []

    class FakeHFModel:
        def __init__(self):
            self.device = None

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    fake_model = FakeHFModel()

    monkeypatch.setattr(
        tune.HFTransformerLM,
        "from_pretrained",
        lambda path: loaded_paths.append(path) or fake_model,
    )
    monkeypatch.setattr(tune, "generate", lambda **kwargs: generated.append(kwargs) or [])

    tune.inference_test(
        prompt="prompt",
        pretraining_checkpoint=False,
        posttraining_checkpoint_path="custom/posttrain",
    )

    assert loaded_paths == ["custom/posttrain"]
    assert generated[0]["model"] is fake_model
    assert generated[0]["device"] == tune.GPTConfig.device


def test_main_builds_trainer_and_saves_model(monkeypatch):
    captured = {}
    tokenized_dataset = DummyDatasetDict(train="train-split", test="test-split")
    trainer_calls = {}
    wandb_calls = []

    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token = None

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeTrainer:
        def __init__(self, **kwargs):
            trainer_calls["kwargs"] = kwargs

        def train(self):
            trainer_calls["trained"] = True

        def save_model(self, path):
            trainer_calls["save_path"] = path

    def fake_load_pretraining_model(**kwargs):
        captured["pretraining"] = kwargs
        return "base-model"

    def fake_load_instruction_dataset(tokenizer, **kwargs):
        captured["dataset"] = {"tokenizer": tokenizer, **kwargs}
        return tokenized_dataset

    monkeypatch.setattr(tune.wandb, "init", lambda project: wandb_calls.append(project))
    monkeypatch.setattr(tune, "_load_pretraining_model", fake_load_pretraining_model)
    monkeypatch.setattr(tune, "_build_hf_model", lambda base_model: f"wrapped:{base_model}")
    monkeypatch.setattr(tune.AutoTokenizer, "from_pretrained", lambda name: FakeTokenizer())
    monkeypatch.setattr(tune, "_load_instruction_dataset", fake_load_instruction_dataset)
    monkeypatch.setattr(tune, "CustomCollatorWithPadding", lambda tokenizer: ("collator", tokenizer))
    monkeypatch.setattr(tune, "TrainingArguments", FakeTrainingArguments)
    monkeypatch.setattr(tune, "Trainer", FakeTrainer)

    tune.main(
        pretraining_repo_id="repo/custom",
        pretraining_checkpoint_pattern="weights/*",
        pretraining_checkpoint_path="custom/checkpoint",
        pretraining_local_dir="custom-local",
        posttraining_checkpoint_path="custom/posttrain",
        dataset_id="org/custom-dataset",
        dataset_split="validation",
        batch_size=4,
    )

    assert captured["pretraining"] == {
        "repo_id": "repo/custom",
        "checkpoint_pattern": "weights/*",
        "checkpoint_path": "custom/checkpoint",
        "local_dir": "custom-local",
    }
    assert captured["dataset"]["dataset_id"] == "org/custom-dataset"
    assert captured["dataset"]["split"] == "validation"
    assert wandb_calls == ["gpt-2.5"]
    assert trainer_calls["kwargs"]["model"] == "wrapped:base-model"
    assert trainer_calls["kwargs"]["train_dataset"] == "train-split"
    assert trainer_calls["kwargs"]["eval_dataset"] == "test-split"
    assert trainer_calls["kwargs"]["data_collator"][0] == "collator"
    assert trainer_calls["kwargs"]["compute_metrics"] is tune.compute_metrics
    args = trainer_calls["kwargs"]["args"]
    assert args.eval_strategy == "epoch"
    assert args.include_for_metrics == ["loss"]
    assert args.logging_steps == 100
    assert args.report_to == "wandb"
    assert args.per_device_train_batch_size == 4
    assert args.per_device_eval_batch_size == 4
    assert args.prediction_loss_only is True
    assert args.num_train_epochs == 3
    assert args.learning_rate == 1e-5
    assert trainer_calls["trained"] is True
    assert trainer_calls["save_path"] == "custom/posttrain"
