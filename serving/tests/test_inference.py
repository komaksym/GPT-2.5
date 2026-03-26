from types import SimpleNamespace

import torch

from serving import inference


class FakeModel:
    def __init__(self):
        """Initialize runtime state captured during inference tests."""
        self.config = SimpleNamespace(max_position_embeddings=64)
        self.eval_called = False
        self.to_device = None
        self.attn_implementations = []
        self.actions = []

    def eval(self):
        """Record that eval mode was requested."""
        self.eval_called = True
        self.actions.append(("eval", None))

    def to(self, device):
        """Record the device transfer request."""
        self.to_device = device
        self.actions.append(("to", device))

    def set_attn_implementation(self, attn_implementation):
        """Record the configured attention backend."""
        self.attn_implementations.append(attn_implementation)
        self.actions.append(("attn", attn_implementation))


class FakeTokenizer:
    eos_token_id = None

    def get_vocab(self):
        """Return an empty vocab for stop-token tests."""
        return {}


def test_get_torch_compile_mode_defaults_to_disabled(monkeypatch):
    """Leave torch.compile disabled unless the flag or env enables it."""
    monkeypatch.delenv(inference.TORCH_COMPILE_ENV, raising=False)

    assert inference.get_torch_compile_mode() is None


def test_get_torch_compile_mode_uses_explicit_mode_when_enabled():
    """Prefer the caller-supplied compile mode when compilation is enabled."""
    assert (
        inference.get_torch_compile_mode(
            use_torch_compile=True, torch_compile_mode="max-autotune"
        )
        == "max-autotune"
    )


def test_load_inference_resources_applies_torch_compile(monkeypatch):
    """Wrap the model with torch.compile when the compile flag is enabled."""
    fake_model = FakeModel()
    fake_tokenizer = FakeTokenizer()
    compiled_model = object()

    monkeypatch.setattr(inference, "get_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        inference.AutoTokenizer,
        "from_pretrained",
        lambda repo_id, trust_remote_code=True: fake_tokenizer,
    )
    monkeypatch.setattr(
        inference.AutoModelForCausalLM,
        "from_pretrained",
        lambda repo_id, **kwargs: fake_model,
    )

    def fake_compile(model, mode, dynamic):
        assert model is fake_model
        assert mode == "max-autotune-no-cudagraphs"
        assert dynamic is True
        return compiled_model

    monkeypatch.setattr(torch, "compile", fake_compile)

    resources = inference.load_inference_resources(
        "repo/example",
        use_torch_compile=True,
    )

    assert resources.model is compiled_model
    assert resources.torch_compile_mode == "max-autotune-no-cudagraphs"


def test_mark_compile_step_begin_is_noop_when_compile_is_disabled(monkeypatch):
    """Skip cudagraph step markers when torch.compile is not active."""
    calls = []
    monkeypatch.setattr(
        torch.compiler,
        "cudagraph_mark_step_begin",
        lambda: calls.append("step"),
    )

    resources = inference.InferenceResources(
        model=object(),
        tokenizer=FakeTokenizer(),
        device=torch.device("cpu"),
        inference_dtype=None,
        attention_backend="sdpa",
        context_length=8,
        stop_token_ids=set(),
    )

    inference._mark_compile_step_begin(resources)

    assert calls == []


def test_mark_compile_step_begin_marks_compiled_steps(monkeypatch):
    """Mark cudagraph step boundaries when torch.compile is enabled."""
    calls = []
    monkeypatch.setattr(
        torch.compiler,
        "cudagraph_mark_step_begin",
        lambda: calls.append("step"),
    )

    resources = inference.InferenceResources(
        model=object(),
        tokenizer=FakeTokenizer(),
        device=torch.device("cuda"),
        inference_dtype=torch.bfloat16,
        attention_backend="sdpa",
        context_length=8,
        stop_token_ids=set(),
        torch_compile_mode="max-autotune-no-cudagraphs",
    )

    inference._mark_compile_step_begin(resources)

    assert calls == ["step"]


def test_load_inference_resources_uses_cuda_inference_dtype(monkeypatch):
    """Use CUDA-specific dtype selection when loading inference resources."""
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
        """Capture model loading kwargs and return the fake model."""
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
    assert resources.inference_dtype == torch.bfloat16
    assert resources.attention_backend == "sdpa"
    assert resources.torch_compile_mode is None
    assert resources.stop_token_ids == set()
    assert captured == {
        "repo_id": "repo/example",
        "kwargs": {
            "trust_remote_code": True,
            "dtype": torch.bfloat16,
        },
    }
    assert fake_model.eval_called is True
    assert fake_model.to_device == torch.device("cuda")
    assert fake_model.actions[:2] == [
        ("to", torch.device("cuda")),
        ("attn", "sdpa"),
    ]


def test_load_inference_resources_omits_torch_dtype_off_cuda(monkeypatch):
    """Avoid passing a dtype override when running off CUDA."""
    captured = {}
    fake_model = FakeModel()

    monkeypatch.setattr(inference, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        inference.AutoTokenizer,
        "from_pretrained",
        lambda repo_id, trust_remote_code=True: FakeTokenizer(),
    )

    def fake_model_from_pretrained(repo_id, **kwargs):
        """Capture model loading kwargs and return the fake model."""
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
    assert resources.inference_dtype is None
    assert resources.attention_backend == "sdpa"
    assert resources.torch_compile_mode is None
    assert resources.stop_token_ids == set()


def test_configure_attention_backend_prefers_sdpa_on_cuda():
    """Prefer SDPA by default for the serving benchmark configuration."""
    fake_model = FakeModel()

    attention_backend = inference.configure_attention_backend(
        fake_model, torch.device("cuda")
    )

    assert attention_backend == "sdpa"
    assert fake_model.attn_implementations == ["sdpa"]


def test_configure_attention_backend_respects_env_override(monkeypatch):
    """Honor the environment override for the attention backend."""
    fake_model = FakeModel()
    monkeypatch.setenv(inference.ATTENTION_BACKEND_ENV, "flash_attention_2")

    attention_backend = inference.configure_attention_backend(
        fake_model, torch.device("cuda")
    )

    assert attention_backend == "flash_attention_2"
    assert fake_model.attn_implementations == ["flash_attention_2"]


def test_configure_attention_backend_falls_back_to_sdpa(monkeypatch):
    """Fall back to SDPA when the requested backend cannot be enabled."""
    fake_model = FakeModel()

    def flaky_set_attn_implementation(attn_implementation):
        """Raise for flash attention while recording attempted backends."""
        fake_model.attn_implementations.append(attn_implementation)
        if attn_implementation == "flash_attention_2":
            raise RuntimeError("flash unavailable")

    fake_model.set_attn_implementation = flaky_set_attn_implementation
    monkeypatch.setenv(inference.ATTENTION_BACKEND_ENV, "flash_attention_2")

    attention_backend = inference.configure_attention_backend(
        fake_model, torch.device("cuda")
    )

    assert attention_backend == "sdpa"
    assert fake_model.attn_implementations == ["flash_attention_2", "sdpa"]


def test_generate_response_uses_cached_stop_token_ids(monkeypatch):
    """Stop generation using the cached stop ids instead of recomputing them."""

    class FakeDecodeTokenizer(FakeTokenizer):
        def decode(self, token_ids):
            """Decode token ids into a deterministic string for assertions."""
            return "".join(str(token_id) for token_id in token_ids)

    class FakeOutputs:
        def __init__(self):
            """Expose logits and a cache slot like a HF output object."""
            self.logits = torch.zeros(1, 1, 8)
            self.past_key_values = None

    class FakeModel:
        def __call__(
            self,
            input_ids,
            use_cache=None,
            cache_position=None,
            cache_capacity=None,
            past_key_values=None,
        ):
            """Return logits without producing a KV cache."""
            return FakeOutputs()

    resources = inference.InferenceResources(
        model=FakeModel(),
        tokenizer=FakeDecodeTokenizer(),
        device=torch.device("cpu"),
        inference_dtype=None,
        attention_backend="sdpa",
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


def test_generate_response_uses_past_key_values_for_cached_decode(monkeypatch):
    """Reuse past_key_values for token-by-token cached decoding."""

    class FakeDecodeTokenizer(FakeTokenizer):
        def decode(self, token_ids):
            """Decode token ids into a deterministic string for assertions."""
            return "".join(str(token_id) for token_id in token_ids)

    class FakeOutputs:
        def __init__(self, token_count, past_key_values):
            """Expose logits and supplied cache contents."""
            self.logits = torch.zeros(1, token_count, 8)
            self.past_key_values = past_key_values

    class FakeModel:
        def __init__(self):
            """Track decode-time calls for later assertions."""
            self.config = SimpleNamespace(use_cache=True)
            self.calls = []

        def __call__(
            self,
            input_ids,
            use_cache=None,
            cache_position=None,
            cache_capacity=None,
            past_key_values=None,
        ):
            """Record the cached decode inputs and return a new fake cache."""
            self.calls.append(
                {
                    "shape": tuple(input_ids.shape),
                    "use_cache": use_cache,
                    "cache_position": cache_position.tolist(),
                    "cache_capacity": cache_capacity,
                    "past_key_values": past_key_values,
                }
            )
            return FakeOutputs(input_ids.shape[-1], f"cache-{len(self.calls)}")

    model = FakeModel()
    resources = inference.InferenceResources(
        model=model,
        tokenizer=FakeDecodeTokenizer(),
        device=torch.device("cpu"),
        inference_dtype=None,
        attention_backend="sdpa",
        context_length=8,
        stop_token_ids={5},
    )
    sampled_tokens = iter((6, 7, 5))

    monkeypatch.setattr(
        inference,
        "_prepare_inputs",
        lambda tokenizer, messages, device: torch.tensor([[1, 2, 3]]),
    )
    monkeypatch.setattr(
        inference,
        "_top_p_sample",
        lambda logits, temp, top_p: torch.tensor([[next(sampled_tokens)]]),
    )

    response = inference.generate_response(
        messages=[{"role": "user", "content": "hello"}],
        resources=resources,
        max_new_tokens=3,
    )

    assert response == "67"
    assert model.calls == [
        {
            "shape": (1, 3),
            "use_cache": True,
            "cache_position": [0, 1, 2],
            "cache_capacity": 6,
            "past_key_values": None,
        },
        {
            "shape": (1, 1),
            "use_cache": True,
            "cache_position": [3],
            "cache_capacity": None,
            "past_key_values": "cache-1",
        },
        {
            "shape": (1, 1),
            "use_cache": True,
            "cache_position": [4],
            "cache_capacity": None,
            "past_key_values": "cache-2",
        },
    ]


def test_generate_response_stops_at_context_boundary(monkeypatch):
    """Stop decoding once generation reaches the configured context limit."""

    class FakeDecodeTokenizer(FakeTokenizer):
        def decode(self, token_ids):
            """Decode token ids into a deterministic string for assertions."""
            return "".join(str(token_id) for token_id in token_ids)

    class FakeOutputs:
        def __init__(self, token_count, past_key_values):
            """Expose logits and supplied cache contents."""
            self.logits = torch.zeros(1, token_count, 8)
            self.past_key_values = past_key_values

    class FakeModel:
        def __init__(self):
            """Track decode-time calls for later assertions."""
            self.config = SimpleNamespace(use_cache=True)
            self.calls = []

        def __call__(
            self,
            input_ids,
            use_cache=None,
            cache_position=None,
            cache_capacity=None,
            past_key_values=None,
        ):
            """Record the decode input shape and return a new fake cache."""
            self.calls.append(tuple(input_ids.shape))
            return FakeOutputs(input_ids.shape[-1], f"cache-{len(self.calls)}")

    model = FakeModel()
    resources = inference.InferenceResources(
        model=model,
        tokenizer=FakeDecodeTokenizer(),
        device=torch.device("cpu"),
        inference_dtype=None,
        attention_backend="sdpa",
        context_length=3,
        stop_token_ids=set(),
    )

    monkeypatch.setattr(
        inference,
        "_prepare_inputs",
        lambda tokenizer, messages, device: torch.tensor([[1, 2, 3]]),
    )
    monkeypatch.setattr(
        inference, "_top_p_sample", lambda logits, temp, top_p: torch.tensor([[6]])
    )

    response = inference.generate_response(
        messages=[{"role": "user", "content": "hello"}],
        resources=resources,
        max_new_tokens=2,
    )

    assert response == "6"
    assert model.calls == [(1, 3)]
