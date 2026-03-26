from types import SimpleNamespace

import torch

from serving import inference


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(max_position_embeddings=64)
        self.eval_called = False
        self.to_device = None

    def eval(self):
        self.eval_called = True

    def to(self, device):
        self.to_device = device


def test_load_inference_resources_uses_cuda_inference_dtype(monkeypatch):
    captured = {}
    fake_model = FakeModel()
    fake_tokenizer = object()

    monkeypatch.setattr(inference, "get_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        inference.AutoTokenizer,
        "from_pretrained",
        lambda repo_id, trust_remote_code=True: fake_tokenizer,
    )

    def fake_model_from_pretrained(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured["kwargs"] = kwargs
        return fake_model

    monkeypatch.setattr(
        inference.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )

    resources = inference.load_inference_resources("repo/example")

    assert resources.model is fake_model
    assert resources.tokenizer is fake_tokenizer
    assert resources.device == torch.device("cuda")
    assert captured == {
        "repo_id": "repo/example",
        "kwargs": {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        },
    }
    assert fake_model.eval_called is True
    assert fake_model.to_device == torch.device("cuda")


def test_load_inference_resources_omits_torch_dtype_off_cuda(monkeypatch):
    captured = {}
    fake_model = FakeModel()

    monkeypatch.setattr(inference, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        inference.AutoTokenizer,
        "from_pretrained",
        lambda repo_id, trust_remote_code=True: object(),
    )

    def fake_model_from_pretrained(repo_id, **kwargs):
        captured["kwargs"] = kwargs
        return fake_model

    monkeypatch.setattr(
        inference.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_from_pretrained,
    )

    resources = inference.load_inference_resources("repo/example")

    assert captured["kwargs"] == {"trust_remote_code": True}
    assert resources.model is fake_model
