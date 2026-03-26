import json
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from my_gpt_model.configuration_gpt25 import MyConfig
from my_gpt_model import modeling_gpt25


MODEL_SOURCE_DIR = Path(__file__).resolve().parents[1] / "my_gpt_model"


def _write_local_model_repo(tmp_path: Path) -> Path:
    """Create a minimal local HF repo for remote-code loading tests."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    config = MyConfig(
        vocab_size=32,
        context_length=8,
        num_layers=1,
        num_heads=2,
        d_model=8,
        d_ff=64,
    )
    model = modeling_gpt25.GPT25ForCausalLM(config)
    model.save_pretrained(repo_dir, safe_serialization=True)

    config_path = repo_dir / "config.json"
    config_data = json.loads(config_path.read_text())
    config_data["architectures"] = ["GPT25ForCausalLM"]
    config_data["auto_map"] = {
        "AutoConfig": "configuration_gpt25.MyConfig",
        "AutoModel": "modeling_gpt25.GPT25Model",
        "AutoModelForCausalLM": "modeling_gpt25.GPT25ForCausalLM",
    }
    config_path.write_text(json.dumps(config_data, indent=2) + "\n")

    for filename in ("__init__.py", "configuration_gpt25.py", "modeling_gpt25.py"):
        shutil.copy(MODEL_SOURCE_DIR / filename, repo_dir / filename)

    return repo_dir


def test_auto_classes_load_from_pretrained(tmp_path):
    """Verify the auto classes load the custom GPT-2.5 package correctly."""
    repo_dir = _write_local_model_repo(tmp_path)

    config = AutoConfig.from_pretrained(repo_dir, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(repo_dir, trust_remote_code=True)
    causal_model = AutoModelForCausalLM.from_pretrained(
        repo_dir, trust_remote_code=True
    )

    assert type(config).__name__ == "MyConfig"
    assert config.hidden_size == config.d_model == 8
    assert config.num_attention_heads == config.num_heads == 2
    assert config.max_position_embeddings == config.context_length == 8
    assert type(base_model).__name__ == "GPT25Model"
    assert type(causal_model).__name__ == "GPT25ForCausalLM"
    assert (
        causal_model.get_output_embeddings().weight
        is causal_model.get_input_embeddings().weight
    )


def test_attention_backend_kwargs_flow_to_attention(monkeypatch):
    """Ensure extra kwargs reach the registered attention backend."""
    captured = {}

    def fake_attention_interface(
        module,
        query,
        key,
        value,
        attention_mask=None,
        scaling=None,
        **kwargs,
    ):
        """Record attention kwargs and return a passthrough-style output."""
        captured.update(kwargs)
        return value.transpose(1, 2).contiguous(), None

    monkeypatch.setattr(
        modeling_gpt25.ALL_ATTENTION_FUNCTIONS,
        "get_interface",
        lambda *args, **kwargs: fake_attention_interface,
    )

    model = modeling_gpt25.GPT25Model(
        MyConfig(
            vocab_size=32,
            context_length=8,
            num_layers=1,
            num_heads=2,
            d_model=8,
            d_ff=64,
        )
    )
    outputs = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        cache_position=torch.tensor([0, 1, 2]),
        custom_flag="seen",
    )

    assert outputs.last_hidden_state.shape == (1, 3, 8)
    assert torch.equal(captured["cache_position"], torch.tensor([0, 1, 2]))
    assert captured["custom_flag"] == "seen"


def test_hf_transformer_lm_alias_is_preserved():
    """Keep the legacy HFTransformerLM alias pointing at the causal LM class."""
    assert modeling_gpt25.HFTransformerLM is modeling_gpt25.GPT25ForCausalLM


def test_transformer_lm_forward_projects_hidden_states():
    """Allow direct TransformerLM.forward calls to produce logits."""
    model = modeling_gpt25.TransformerLM(
        vocab_size=32,
        context_length=8,
        num_layers=1,
        d_model=8,
        num_heads=2,
        d_ff=64,
    )

    logits, loss = model(input_ids=torch.tensor([[1, 2, 3]]))

    assert logits.shape == (1, 3, 32)
    assert loss is None


def test_causal_lm_forward_returns_past_key_values():
    """Ensure cached forward passes return preallocated KV caches."""
    model = modeling_gpt25.GPT25ForCausalLM(
        MyConfig(
            vocab_size=32,
            context_length=8,
            num_layers=1,
            num_heads=2,
            d_model=8,
            d_ff=64,
        )
    )
    outputs = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)

    assert outputs.past_key_values is not None
    assert len(outputs.past_key_values) == 1
    key_cache, value_cache, cache_length = outputs.past_key_values[0]
    assert key_cache.shape == (1, 2, 8, 4)
    assert value_cache.shape == (1, 2, 8, 4)
    assert cache_length == 3


def test_causal_lm_forward_honors_cache_capacity():
    """Respect an explicit cache_capacity during the prefill pass."""
    model = modeling_gpt25.GPT25ForCausalLM(
        MyConfig(
            vocab_size=32,
            context_length=8,
            num_layers=1,
            num_heads=2,
            d_model=8,
            d_ff=64,
        )
    )
    outputs = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        use_cache=True,
        cache_capacity=4,
    )

    assert outputs.past_key_values is not None
    key_cache, value_cache, cache_length = outputs.past_key_values[0]
    assert key_cache.shape == (1, 2, 4, 4)
    assert value_cache.shape == (1, 2, 4, 4)
    assert cache_length == 3


def test_causal_lm_cached_decode_matches_full_decode():
    """Match cached decode logits against a full forward pass."""
    model = modeling_gpt25.GPT25ForCausalLM(
        MyConfig(
            vocab_size=32,
            context_length=8,
            num_layers=1,
            num_heads=2,
            d_model=8,
            d_ff=64,
        )
    )
    model.eval()

    prefill_outputs = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)
    cached_outputs = model(
        input_ids=torch.tensor([[4]]),
        past_key_values=prefill_outputs.past_key_values,
        cache_position=torch.tensor([3]),
        use_cache=True,
    )
    full_outputs = model(input_ids=torch.tensor([[1, 2, 3, 4]]), use_cache=False)

    assert cached_outputs.past_key_values is not None
    assert cached_outputs.past_key_values[0][0].shape[-2] == 8
    assert cached_outputs.past_key_values[0][2] == 4
    assert torch.allclose(
        cached_outputs.logits[:, -1, :],
        full_outputs.logits[:, -1, :],
        atol=1e-5,
        rtol=1e-4,
    )
