from types import SimpleNamespace

import torch

from serving import inference


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(max_position_embeddings=64)
        self.eval_called = False
        self.to_device = None
        self.attn_implementations = []

    def eval(self):
        self.eval_called = True

    def to(self, device):
        self.to_device = device

    def set_attn_implementation(self, attn_implementation):
        self.attn_implementations.append(attn_implementation)


class FakeTokenizer:
    eos_token_id = None

    def get_vocab(self):
        return {}


def test_load_inference_resources_uses_cuda_inference_dtype(monkeypatch):
    captured = {}
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()

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
    assert resources.stop_token_ids == set()
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
        lambda repo_id, trust_remote_code=True: FakeTokenizer(),
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
    assert resources.stop_token_ids == set()


def test_configure_attention_backend_prefers_flash_attention_on_cuda():
    fake_model = FakeModel()

    inference.configure_attention_backend(fake_model, torch.device("cuda"))

    assert fake_model.attn_implementations == ["flash_attention_2"]


def test_configure_attention_backend_falls_back_to_sdpa():
    fake_model = FakeModel()

    def flaky_set_attn_implementation(attn_implementation):
        fake_model.attn_implementations.append(attn_implementation)
        if attn_implementation == "flash_attention_2":
            raise RuntimeError("flash unavailable")

    fake_model.set_attn_implementation = flaky_set_attn_implementation

    inference.configure_attention_backend(fake_model, torch.device("cuda"))

    assert fake_model.attn_implementations == ["flash_attention_2", "sdpa"]


def test_generate_response_uses_cached_stop_token_ids(monkeypatch):
    class FakeDecodeTokenizer(FakeTokenizer):
        def decode(self, token_ids):
            return "".join(str(token_id) for token_id in token_ids)

    class FakeOutputs:
        def __init__(self):
            self.logits = torch.zeros(1, 1, 8)

    class FakeModel:
        def __call__(self, input_ids):
            return FakeOutputs()

    resources = inference.InferenceResources(
        model=FakeModel(),
        tokenizer=FakeDecodeTokenizer(),
        device=torch.device("cpu"),
        context_length=8,
        stop_token_ids={5},
    )

    monkeypatch.setattr(
        inference,
        "_prepare_inputs",
        lambda tokenizer, messages, device: torch.tensor([[1, 2, 3]]),
    )
    monkeypatch.setattr(
        inference, "_top_p_sample", lambda logits, temp, top_p: torch.tensor([[5]])
    )
    monkeypatch.setattr(
        inference,
        "_stop_token_ids",
        lambda tokenizer: (_ for _ in ()).throw(AssertionError),
    )

    response = inference.generate_response(
        messages=[{"role": "user", "content": "hello"}],
        resources=resources,
        max_new_tokens=2,
    )

    assert response == ""
