import torch

from pre_train.model import build_attention_mask
from post_train import model as posttrain_model


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
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.emb = FakeEmbedding()
            self.linear = FakeLinear(self.emb.weight)
            self.attn_implementations = []

        def set_attn_implementation(self, attn_implementation):
            self.attn_implementations.append(attn_implementation)

        def forward(self, input_ids, attention_mask=None, position_ids=None):
            batch, seq_len = input_ids.shape
            return torch.zeros(batch, seq_len, 1), None

    monkeypatch.setattr(posttrain_model, "TransformerLM", FakeTransformerLM)

    model = posttrain_model.HFTransformerLM(
        posttrain_model.MyConfig(
            vocab_size=1,
            context_length=1,
            num_layers=1,
            num_heads=1,
            d_model=1,
            d_ff=64,
            device="cpu",
        )
    )

    assert model._tied_weights_keys == {"model.linear.weight": "model.emb.weight"}
    assert model.get_input_embeddings() is model.model.emb
    assert model.get_output_embeddings() is model.model.linear
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight
    assert model.model.attn_implementations == [model.config._attn_implementation]


def test_hf_wrapper_forwards_position_ids(monkeypatch):
    captured = {}

    class FakeEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4))

    class FakeLinear(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight

    class FakeTransformerLM(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.emb = FakeEmbedding()
            self.linear = FakeLinear(self.emb.weight)

        def set_attn_implementation(self, attn_implementation):
            captured.setdefault("attn_implementations", []).append(attn_implementation)

        def forward(self, input_ids, attention_mask=None, position_ids=None):
            captured["input_ids"] = input_ids
            captured["attention_mask"] = attention_mask
            captured["position_ids"] = position_ids
            batch, seq_len = input_ids.shape
            return torch.zeros(batch, seq_len, 4), None

    monkeypatch.setattr(posttrain_model, "TransformerLM", FakeTransformerLM)

    model = posttrain_model.HFTransformerLM(
        posttrain_model.MyConfig(
            vocab_size=4,
            context_length=8,
            num_layers=1,
            num_heads=1,
            d_model=4,
            d_ff=64,
            device="cpu",
        )
    )

    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 1]])
    position_ids = torch.tensor([[0, 1, 0]])
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )

    assert captured["attn_implementations"] == [
        model.config._attn_implementation,
        model.config._attn_implementation,
    ]
    assert torch.equal(captured["input_ids"], input_ids)
    assert torch.equal(captured["attention_mask"], attention_mask)
    assert torch.equal(captured["position_ids"], position_ids)
    assert outputs.logits.shape == (1, 3, 4)


def test_build_attention_mask_preserves_causal_attention_within_each_packed_sequence():
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])

    mask = build_attention_mask(
        seq_len=position_ids.size(1),
        device=position_ids.device,
        position_ids=position_ids,
    )

    assert mask.dtype == torch.bool
    assert mask[0, 0, 2, 0].item() is True
    assert mask[0, 0, 4, 3].item() is True


def test_build_attention_mask_blocks_cross_sequence_attention():
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])

    mask = build_attention_mask(
        seq_len=position_ids.size(1),
        device=position_ids.device,
        position_ids=position_ids,
    )

    assert mask[0, 0, 3, 3].item() is True
    assert mask[0, 0, 3, 2].item() is False
    assert mask[0, 0, 4, 1].item() is False


def test_hf_wrapper_resizes_token_embeddings_and_reties_output_head():
    model = posttrain_model.HFTransformerLM(
        posttrain_model.MyConfig(
            vocab_size=4,
            context_length=8,
            num_layers=1,
            num_heads=1,
            d_model=8,
            d_ff=64,
            device="cpu",
        )
    )

    old_weight = model.get_input_embeddings().weight.detach().clone()

    resized = model.resize_token_embeddings(6)

    assert type(resized) is type(model.get_input_embeddings())
    assert resized.weight.shape == (6, 8)
    torch.testing.assert_close(resized.weight[:4], old_weight)
    torch.testing.assert_close(resized.weight[4:], old_weight.mean(dim=0).expand(2, -1))
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight
    assert model.config.vocab_size == 6
    assert model.vocab_size == 6


def test_hf_wrapper_set_attn_implementation_syncs_runtime(monkeypatch):
    class FakeEmbedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(1, 1))

    class FakeLinear(torch.nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight

    class FakeTransformerLM(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.emb = FakeEmbedding()
            self.linear = FakeLinear(self.emb.weight)
            self.attn_implementations = []

        def set_attn_implementation(self, attn_implementation):
            self.attn_implementations.append(attn_implementation)

        def forward(self, input_ids, attention_mask=None, position_ids=None):
            batch, seq_len = input_ids.shape
            return torch.zeros(batch, seq_len, 1), None

    monkeypatch.setattr(posttrain_model, "TransformerLM", FakeTransformerLM)

    model = posttrain_model.HFTransformerLM(
        posttrain_model.MyConfig(
            vocab_size=1,
            context_length=1,
            num_layers=1,
            num_heads=1,
            d_model=1,
            d_ff=64,
            device="cpu",
        )
    )
    monkeypatch.setattr(
        posttrain_model.HFTransformerLM,
        "_check_and_adjust_attn_implementation",
        lambda self, attn_implementation, is_init_check=False: attn_implementation,
    )

    model.set_attn_implementation("flash_attention_2")

    assert model.config._attn_implementation == "flash_attention_2"
    assert model.model.attn_implementations[-1] == "flash_attention_2"


def test_load_pretraining_model_uses_override_paths(monkeypatch):
    snapshot_calls = []
    load_calls = []
    base_model = object()

    monkeypatch.setattr(posttrain_model, "_build_base_model", lambda: base_model)
    monkeypatch.setattr(
        posttrain_model,
        "snapshot_download",
        lambda *args, **kwargs: snapshot_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        posttrain_model,
        "load_checkpoint",
        lambda path, model: load_calls.append((path, model)),
    )

    loaded = posttrain_model._load_pretraining_model(
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
