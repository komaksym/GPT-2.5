from types import SimpleNamespace

import numpy as np
import pytest
import torch

from post_train import tune


class DummyDatasetDict(dict):
    pass


def test_compute_metrics_returns_perplexity():
    """Convert mean loss into perplexity."""
    metrics = tune.compute_metrics(SimpleNamespace(losses=np.array([0.0, np.log(4.0)])))

    assert metrics == {"perplexity": 2.0}


def test_get_training_dtype_uses_fp16_when_bf16_is_unavailable(monkeypatch):
    """Prefer fp16 on CUDA when bf16 is not supported."""
    monkeypatch.setattr(tune.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tune.torch.cuda, "is_bf16_supported", lambda: False)

    assert tune.get_training_dtype() is torch.float16
    assert tune.get_trainer_precision_kwargs(torch.float16) == {"fp16": True}


def test_get_tokenizer_sets_pad_token_and_chat_template(monkeypatch):
    """Install the expected special tokens and chat template."""

    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token = None
        chat_template = None

    fake_tokenizer = FakeTokenizer()
    calls = {}

    monkeypatch.setattr(
        tune.AutoTokenizer,
        "from_pretrained",
        lambda tokenizer_path, extra_special_tokens=None: (
            calls.update(
                {
                    "tokenizer_path": tokenizer_path,
                    "extra_special_tokens": extra_special_tokens,
                }
            )
            or fake_tokenizer
        ),
    )

    tokenizer = tune.get_tokenizer(tokenizer_path="gpt2")

    assert tokenizer is fake_tokenizer
    assert calls == {
        "tokenizer_path": "gpt2",
        "extra_special_tokens": {
            "user_token": "<|user|>",
            "assistant_token": "<|assistant|>",
            "system_token": "<|system|>",
        },
    }
    assert tokenizer.pad_token == tokenizer.eos_token
    assert "<|assistant|>" in tokenizer.chat_template
    assert "{% generation %}" in tokenizer.chat_template


def test_configure_packed_attention_enables_flash_attention_when_available(
    monkeypatch,
):
    """Enable FlashAttention 2 for packed training when available."""
    calls = []

    class FakeModel:
        def set_attn_implementation(self, implementation):
            """Record the requested attention backend."""
            calls.append(implementation)

    monkeypatch.setattr(tune.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tune, "is_flash_attn_2_installed", lambda: True)
    monkeypatch.setattr(tune.torch.backends.cuda, "flash_sdp_enabled", lambda: True)

    tune.configure_packed_attention(FakeModel())

    assert calls == ["flash_attention_2"]


def test_configure_packed_attention_warns_when_flash_attention_is_unavailable(
    monkeypatch,
):
    """Warn and avoid enabling flash attention when unavailable."""

    class FakeModel:
        def set_attn_implementation(self, implementation):
            """Fail fast if the code tries to enable flash attention."""
            raise AssertionError("flash attention should not be enabled")

    monkeypatch.setattr(tune.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tune, "is_flash_attn_2_installed", lambda: False)

    with pytest.warns(UserWarning, match="falling back to SDPA"):
        tune.configure_packed_attention(FakeModel())


def test_main_builds_sft_trainer_for_current_api(monkeypatch):
    """Build the SFT trainer with the expected tokenizer, model, and args."""
    load_calls = []
    dataset_calls = []
    trainer_calls = {}
    wandb_calls = []
    tokenized_dataset = DummyDatasetDict(train="train-split", test="test-split")

    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token = None

        def __len__(self):
            """Report the resized tokenizer length."""
            return 7

    class FakeModel:
        def __init__(self):
            """Track mutable state updated by the training setup."""
            self.config = SimpleNamespace(dtype=None, _attn_implementation="sdpa")
            self.attn_implementation_calls = []
            self.resized_to = None
            self.tied = False

        def set_attn_implementation(self, attn_implementation):
            """Record the requested attention backend."""
            self.attn_implementation_calls.append(attn_implementation)
            self.config._attn_implementation = attn_implementation

        def resize_token_embeddings(self, size):
            """Record the embedding resize request."""
            self.resized_to = size

        def tie_weights(self):
            """Record that tie_weights was called."""
            self.tied = True

    class FakeSFTConfig:
        def __init__(self, **kwargs):
            """Store trainer arguments for later assertions."""
            self.__dict__.update(kwargs)

    class FakeSFTTrainer:
        def __init__(self, **kwargs):
            """Capture trainer construction kwargs."""
            trainer_calls["kwargs"] = kwargs

        def train(self):
            """Record that training was launched."""
            trainer_calls["trained"] = True

    fake_model = FakeModel()

    monkeypatch.setattr(tune.wandb, "init", lambda project: wandb_calls.append(project))
    monkeypatch.setattr(
        tune.HFTransformerLM,
        "from_pretrained",
        lambda repo_id, subfolder=None: (
            load_calls.append((repo_id, subfolder)) or fake_model
        ),
    )
    monkeypatch.setattr(
        tune, "get_tokenizer", lambda tokenizer_path="gpt2": FakeTokenizer()
    )
    monkeypatch.setattr(
        tune,
        "load_dataset",
        lambda dataset_id, split=None: (
            dataset_calls.append((dataset_id, split)) or tokenized_dataset
        ),
    )
    monkeypatch.setattr(tune, "SFTConfig", FakeSFTConfig)
    monkeypatch.setattr(tune, "SFTTrainer", FakeSFTTrainer)
    monkeypatch.setattr(tune, "is_flash_attn_2_installed", lambda: True)
    monkeypatch.setattr(tune.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tune.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(tune.torch.backends.cuda, "flash_sdp_enabled", lambda: True)

    tune.main(
        repo_id="repo/custom",
        checkpoint_subfolder="custom/subfolder",
        dataset_id="org/custom-dataset",
        dataset_split=None,
        batch_size=4,
    )

    assert load_calls == [("repo/custom", "custom/subfolder")]
    assert dataset_calls == [("org/custom-dataset", None)]
    assert wandb_calls == ["gpt-2.5"]
    assert fake_model.attn_implementation_calls == ["flash_attention_2"]
    assert fake_model.config.dtype is torch.bfloat16
    assert fake_model.resized_to == 7
    assert fake_model.tied is True
    assert trainer_calls["kwargs"]["model"] is fake_model
    assert trainer_calls["kwargs"]["train_dataset"] == "train-split"
    assert trainer_calls["kwargs"]["eval_dataset"] == "test-split"
    assert trainer_calls["kwargs"]["compute_metrics"] is tune.compute_metrics
    args = trainer_calls["kwargs"]["args"]
    assert args.eval_strategy == "epoch"
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


def test_main_disables_wandb_reporting_when_init_fails(monkeypatch):
    """Disable W&B reporting when initialization raises a connection error."""
    trainer_calls = {}
    tokenized_dataset = DummyDatasetDict(train="train-split", test="test-split")

    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token = None

        def __len__(self):
            """Report the resized tokenizer length."""
            return 7

    class FakeModel:
        def __init__(self):
            """Track mutable state updated by the training setup."""
            self.config = SimpleNamespace(dtype=None, _attn_implementation="sdpa")
            self.resized_to = None
            self.tied = False

        def set_attn_implementation(self, attn_implementation):
            """Record the requested attention backend."""
            self.config._attn_implementation = attn_implementation

        def resize_token_embeddings(self, size):
            """Record the embedding resize request."""
            self.resized_to = size

        def tie_weights(self):
            """Record that tie_weights was called."""
            self.tied = True

    class FakeSFTConfig:
        def __init__(self, **kwargs):
            """Store trainer arguments for later assertions."""
            self.__dict__.update(kwargs)

    class FakeSFTTrainer:
        def __init__(self, **kwargs):
            """Capture trainer construction kwargs."""
            trainer_calls["kwargs"] = kwargs

        def train(self):
            """Record that training was launched."""
            trainer_calls["trained"] = True

    fake_model = FakeModel()

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(
        tune.wandb,
        "init",
        lambda project: (_ for _ in ()).throw(tune.CommError("user is not logged in")),
    )
    monkeypatch.setattr(
        tune.HFTransformerLM,
        "from_pretrained",
        lambda repo_id, subfolder=None: fake_model,
    )
    monkeypatch.setattr(
        tune, "get_tokenizer", lambda tokenizer_path="gpt2": FakeTokenizer()
    )
    monkeypatch.setattr(
        tune, "load_dataset", lambda dataset_id, split=None: tokenized_dataset
    )
    monkeypatch.setattr(tune, "SFTConfig", FakeSFTConfig)
    monkeypatch.setattr(tune, "SFTTrainer", FakeSFTTrainer)
    monkeypatch.setattr(tune.torch.cuda, "is_available", lambda: False)

    with pytest.warns(UserWarning, match="W&B initialization failed"):
        tune.main()

    assert trainer_calls["kwargs"]["args"].report_to == "none"
    assert trainer_calls["trained"] is True
