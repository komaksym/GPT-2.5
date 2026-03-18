from types import SimpleNamespace

import numpy as np
import pytest
import torch

from post_train import data as posttrain_data
from post_train import model as posttrain_model
from post_train import tune


class DummyDatasetDict(dict):
    pass


def test_tune_reexports_model_and_data_symbols():
    assert tune.MyConfig is posttrain_model.MyConfig
    assert tune.HFTransformerLM is posttrain_model.HFTransformerLM
    assert tune._build_base_model is posttrain_model._build_base_model
    assert tune._load_pretraining_model is posttrain_model._load_pretraining_model
    assert tune._build_hf_model is posttrain_model._build_hf_model
    assert tune.format_prompt is posttrain_data.format_prompt
    assert tune.pad_sample is posttrain_data.pad_sample
    assert tune.CustomCollatorWithPadding is posttrain_data.CustomCollatorWithPadding
    assert tune._load_instruction_dataset is posttrain_data._load_instruction_dataset


def test_compute_metrics_returns_perplexity():
    metrics = tune.compute_metrics(SimpleNamespace(losses=np.array([0.0, np.log(4.0)])))

    assert metrics == {"perplexity": 2.0}


def test_get_training_dtype_uses_fp16_when_bf16_is_unavailable(monkeypatch):
    monkeypatch.setattr(tune.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tune.torch.cuda, "is_bf16_supported", lambda: False)

    assert tune.get_training_dtype() is torch.float16
    assert tune.get_trainer_precision_kwargs(torch.float16) == {"fp16": True}


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
            "encoder": tune.ENCODER,
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
    monkeypatch.setattr(
        tune, "generate", lambda **kwargs: generated.append(kwargs) or []
    )

    tune.inference_test(
        prompt="prompt",
        pretraining_checkpoint=False,
        posttraining_checkpoint_path="custom/posttrain",
    )

    assert loaded_paths == ["custom/posttrain"]
    assert generated[0]["model"] is fake_model
    assert generated[0]["device"] == tune.GPTConfig.device


def test_main_builds_sft_trainer_and_enables_packed_flash_attention(monkeypatch):
    captured = {}
    tokenized_dataset = DummyDatasetDict(train="train-split", test="test-split")
    trainer_calls = {}
    wandb_calls = []

    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token = None

        def __len__(self):
            return 7

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(dtype=None, _attn_implementation="sdpa")
            self.attn_implementation_calls = []
            self.resized_to = None

        def set_attn_implementation(self, attn_implementation):
            self.attn_implementation_calls.append(attn_implementation)
            self.config._attn_implementation = attn_implementation

        def resize_token_embeddings(self, size):
            self.resized_to = size

    class FakeSFTConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeSFTTrainer:
        def __init__(self, **kwargs):
            trainer_calls["kwargs"] = kwargs

        def train(self):
            trainer_calls["trained"] = True

        def save_model(self, path):
            trainer_calls["save_path"] = path

    fake_model = FakeModel()

    def fake_load_pretraining_model(**kwargs):
        captured["pretraining"] = kwargs
        return "base-model"

    monkeypatch.setattr(tune.wandb, "init", lambda project: wandb_calls.append(project))
    monkeypatch.setattr(tune, "_load_pretraining_model", fake_load_pretraining_model)
    monkeypatch.setattr(tune, "_build_hf_model", lambda base_model: fake_model)
    monkeypatch.setattr(tune, "get_tokenizer", lambda tokenizer_path="gpt2": FakeTokenizer())
    monkeypatch.setattr(tune, "load_dataset", lambda *args, **kwargs: tokenized_dataset)
    monkeypatch.setattr(tune, "SFTConfig", FakeSFTConfig)
    monkeypatch.setattr(tune, "SFTTrainer", FakeSFTTrainer)
    monkeypatch.setattr(tune, "is_flash_attn_2_installed", lambda: True)
    monkeypatch.setattr(tune.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tune.torch.cuda, "is_bf16_supported", lambda: True)

    tune.main(
        pretraining_repo_id="repo/custom",
        pretraining_checkpoint_pattern="weights/*",
        pretraining_checkpoint_path="custom/checkpoint",
        pretraining_local_dir="custom-local",
        posttraining_checkpoint_path="custom/posttrain",
        dataset_id="org/custom-dataset",
        batch_size=4,
    )

    assert captured["pretraining"] == {
        "repo_id": "repo/custom",
        "checkpoint_pattern": "weights/*",
        "checkpoint_path": "custom/checkpoint",
        "local_dir": "custom-local",
    }
    assert wandb_calls == ["gpt-2.5"]
    assert fake_model.attn_implementation_calls == ["flash_attention_2"]
    assert fake_model.config.dtype is torch.bfloat16
    assert fake_model.resized_to == 7
    assert trainer_calls["kwargs"]["model"] is fake_model
    assert trainer_calls["kwargs"]["train_dataset"] == "train-split"
    assert trainer_calls["kwargs"]["eval_dataset"] == "test-split"
    assert trainer_calls["kwargs"]["compute_metrics"] is tune.compute_metrics
    args = trainer_calls["kwargs"]["args"]
    assert args.eval_strategy == "epoch"
    assert args.include_for_metrics == ["loss"]
    assert args.logging_steps == 100
    assert args.report_to == "wandb"
    assert args.gradient_checkpointing is False
    assert args.per_device_train_batch_size == 4
    assert args.per_device_eval_batch_size == 4
    assert args.prediction_loss_only is True
    assert args.num_train_epochs == 3
    assert args.learning_rate == 1e-5
    assert args.assistant_only_loss is True
    assert args.packing is tune.PACKING
    assert args.bf16 is True
    assert trainer_calls["trained"] is True
    assert trainer_calls["save_path"] == "custom/posttrain"


def test_main_falls_back_to_sdpa_when_flash_attn_is_unavailable(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(dtype=None, _attn_implementation="sdpa")
            self.attn_implementation_calls = []
            self.resized_to = None

        def set_attn_implementation(self, attn_implementation):
            self.attn_implementation_calls.append(attn_implementation)
            self.config._attn_implementation = attn_implementation

        def resize_token_embeddings(self, size):
            self.resized_to = size

    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token = None

        def __len__(self):
            return 7

    class FakeSFTConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeSFTTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def train(self):
            return None

        def save_model(self, path):
            return None

    fake_model = FakeModel()
    tokenized_dataset = DummyDatasetDict(train="train-split", test="test-split")

    monkeypatch.setattr(tune.wandb, "init", lambda project: None)
    monkeypatch.setattr(tune, "_load_pretraining_model", lambda **kwargs: "base-model")
    monkeypatch.setattr(tune, "_build_hf_model", lambda base_model: fake_model)
    monkeypatch.setattr(tune, "get_tokenizer", lambda tokenizer_path="gpt2": FakeTokenizer())
    monkeypatch.setattr(tune, "load_dataset", lambda *args, **kwargs: tokenized_dataset)
    monkeypatch.setattr(tune, "SFTConfig", FakeSFTConfig)
    monkeypatch.setattr(tune, "SFTTrainer", FakeSFTTrainer)
    monkeypatch.setattr(tune, "is_flash_attn_2_installed", lambda: False)
    monkeypatch.setattr(tune.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tune.torch.cuda, "is_bf16_supported", lambda: True)

    with pytest.warns(UserWarning, match="falling back to SDPA"):
        tune.main()

    assert fake_model.attn_implementation_calls == []
    assert fake_model.config._attn_implementation == "sdpa"


def test_main_disables_wandb_reporting_when_init_fails(monkeypatch):
    trainer_calls = {}

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(dtype=None, _attn_implementation="sdpa")
            self.attn_implementation_calls = []
            self.resized_to = None

        def set_attn_implementation(self, attn_implementation):
            self.attn_implementation_calls.append(attn_implementation)
            self.config._attn_implementation = attn_implementation

        def resize_token_embeddings(self, size):
            self.resized_to = size

    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token = None

        def __len__(self):
            return 7

    class FakeSFTConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeSFTTrainer:
        def __init__(self, **kwargs):
            trainer_calls["kwargs"] = kwargs

        def train(self):
            trainer_calls["trained"] = True

        def save_model(self, path):
            trainer_calls["save_path"] = path

    fake_model = FakeModel()
    tokenized_dataset = DummyDatasetDict(train="train-split", test="test-split")

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(
        tune.wandb,
        "init",
        lambda project: (_ for _ in ()).throw(tune.CommError("user is not logged in")),
    )
    monkeypatch.setattr(tune, "_load_pretraining_model", lambda **kwargs: "base-model")
    monkeypatch.setattr(tune, "_build_hf_model", lambda base_model: fake_model)
    monkeypatch.setattr(tune, "get_tokenizer", lambda tokenizer_path="gpt2": FakeTokenizer())
    monkeypatch.setattr(tune, "load_dataset", lambda *args, **kwargs: tokenized_dataset)
    monkeypatch.setattr(tune, "SFTConfig", FakeSFTConfig)
    monkeypatch.setattr(tune, "SFTTrainer", FakeSFTTrainer)
    monkeypatch.setattr(tune, "is_flash_attn_2_installed", lambda: True)
    monkeypatch.setattr(tune.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tune.torch.cuda, "is_bf16_supported", lambda: True)

    with pytest.warns(UserWarning, match="W&B initialization failed"):
        tune.main()

    assert trainer_calls["kwargs"]["args"].report_to == "none"
    assert trainer_calls["trained"] is True
