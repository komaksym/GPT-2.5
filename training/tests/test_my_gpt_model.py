import json
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from my_gpt_model.configuration_gpt25 import MyConfig
from my_gpt_model import modeling_gpt25


MODEL_SOURCE_DIR = Path(__file__).resolve().parents[1] / "my_gpt_model"


def _write_local_model_repo(tmp_path: Path) -> Path:
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
    assert modeling_gpt25.HFTransformerLM is modeling_gpt25.GPT25ForCausalLM
