from types import SimpleNamespace

import torch

from post_train import chat as posttrain_chat


def test_run_inference_loads_checkpoint_and_starts_chat(monkeypatch):
    captured = {}

    class FakeModel:
        def __init__(self):
            self.tied = False
            self.device = None

        def tie_weights(self):
            self.tied = True

        def to(self, device):
            self.device = device
            return self

    fake_model = FakeModel()
    fake_tokenizer = SimpleNamespace(name="tokenizer")

    def fake_from_pretrained(repo_id, subfolder=None):
        captured["load"] = (repo_id, subfolder)
        return fake_model

    monkeypatch.setattr(
        posttrain_chat.HFTransformerLM,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(posttrain_chat, "get_tokenizer", lambda: fake_tokenizer)
    monkeypatch.setattr(
        posttrain_chat,
        "chat",
        lambda **kwargs: captured.setdefault("chat", kwargs),
    )

    device = torch.device("cpu")
    posttrain_chat.run_inference(
        repo_id="repo/custom",
        repo_id_subfolder="custom/subfolder",
        context_length=64,
        max_new_tokens=32,
        temp=0.7,
        top_p=0.85,
        device=device,
    )

    assert captured["load"] == ("repo/custom", "custom/subfolder")
    assert fake_model.tied is True
    assert fake_model.device == device
    assert captured["chat"] == {
        "model": fake_model,
        "tokenizer": fake_tokenizer,
        "context_length": 64,
        "max_new_tokens": 32,
        "temp": 0.7,
        "top_p": 0.85,
        "device": device,
    }
