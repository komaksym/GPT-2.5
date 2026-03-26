from contextlib import nullcontext
from types import SimpleNamespace

import torch

from post_train import chat as posttrain_chat


def test_run_inference_loads_root_repo_and_starts_chat(monkeypatch):
    """Verify run_inference wires together the model, tokenizer, and chat loop."""
    captured = {}

    class FakeModel:
        def __init__(self):
            """Initialize mutable fields used by the test."""
            self.tied = False
            self.device = None

        def tie_weights(self):
            """Record that tie_weights was called."""
            self.tied = True

        def to(self, device):
            """Record the device transfer and behave like nn.Module.to."""
            self.device = device
            return self

    fake_model = FakeModel()
    fake_tokenizer = SimpleNamespace(name="tokenizer")

    def fake_model_from_pretrained(repo_id):
        """Record the repo id used to load the model."""
        captured["model_load"] = repo_id
        return fake_model

    def fake_tokenizer_from_pretrained(repo_id, trust_remote_code=False):
        """Record the tokenizer load arguments."""
        captured["tokenizer_load"] = (repo_id, trust_remote_code)
        return fake_tokenizer

    monkeypatch.setattr(
        posttrain_chat.HFTransformerLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr(
        posttrain_chat.AutoTokenizer,
        "from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        posttrain_chat,
        "chat",
        lambda **kwargs: captured.setdefault("chat", kwargs),
    )

    device = torch.device("cpu")
    posttrain_chat.run_inference(
        repo_id="repo/custom",
        context_length=64,
        max_new_tokens=32,
        temp=0.7,
        top_p=0.85,
        device=device,
    )

    assert captured["model_load"] == "repo/custom"
    assert captured["tokenizer_load"] == ("repo/custom", True)
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


def test_generate_returns_only_response_text(monkeypatch):
    """Ensure generation trims the assistant stop token from the response."""
    sampled_tokens = iter((torch.tensor([[5]]), torch.tensor([[6]])))

    class FakeTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            """Return a fixed prompt tensor for the test conversation."""
            return torch.tensor([[1, 2, 3]])

        def decode(self, token_ids):
            """Decode the sampled tokens into deterministic strings."""
            if token_ids == [5]:
                return "hello"
            if token_ids == [6]:
                return "<|assistant|>"
            return ""

    class FakeModel:
        def __call__(self, inputs):
            """Return logits with the expected generation shape."""
            vocab_size = 8
            logits = torch.zeros(inputs.shape[0], inputs.shape[1], vocab_size)
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(posttrain_chat, "softmax", lambda x, dim, temp=1.0: x)
    monkeypatch.setattr(
        posttrain_chat,
        "top_p_sampling",
        lambda probs, p=0.8: next(sampled_tokens),
    )
    monkeypatch.setattr(
        posttrain_chat, "_autocast_context", lambda device: nullcontext()
    )

    response = posttrain_chat.generate(
        context=[{"role": "user", "content": "Hello"}],
        max_new_tokens=4,
        context_length=16,
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        device=torch.device("cpu"),
    )

    assert response == "hello"
