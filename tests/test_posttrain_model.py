import torch

from posttrain import model as posttrain_model


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
