from contextlib import nullcontext
from dataclasses import dataclass
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

DEFAULT_MODEL_REPO_ID = "itskoma/MyGPT"
MODEL_REPO_ID_ENV = "MODEL_REPO_ID"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMP = 0.9
DEFAULT_TOP_P = 0.8
DEFAULT_ATTENTION_BACKEND = "sdpa"
ATTENTION_BACKEND_ENV = "INFERENCE_ATTENTION_BACKEND"
# Cached decoding mutates and reuses KV tensors across steps, which does not
# play well with TorchInductor cudagraph capture on this model path.
DEFAULT_TORCH_COMPILE_MODE = "max-autotune-no-cudagraphs"
TORCH_COMPILE_ENV = "INFERENCE_USE_TORCH_COMPILE"
TORCH_COMPILE_MODE_ENV = "INFERENCE_TORCH_COMPILE_MODE"
STOP_TOKENS = ("<|endoftext|>", "<|user|>", "<|system|>", "<|assistant|>")
TRUST_REMOTE_CODE = True


@dataclass(slots=True)
class InferenceResources:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    inference_dtype: torch.dtype | None
    attention_backend: str | None
    context_length: int
    stop_token_ids: set[int]
    torch_compile_mode: str | None = None


def get_model_repo_id() -> str:
    """Return the model repo configured for serving."""
    return os.environ.get(MODEL_REPO_ID_ENV, DEFAULT_MODEL_REPO_ID)


def get_device() -> torch.device:
    """Pick the best available inference device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_inference_dtype(device: torch.device) -> torch.dtype | None:
    """Choose an inference dtype for the selected device."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _env_flag_is_enabled(name: str, default: bool = False) -> bool:
    """Parse a boolean flag from the environment."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def get_torch_compile_mode(
    use_torch_compile: bool | None = None,
    torch_compile_mode: str | None = None,
) -> str | None:
    """Resolve the torch.compile mode, or return None when disabled."""
    compile_enabled = (
        _env_flag_is_enabled(TORCH_COMPILE_ENV)
        if use_torch_compile is None
        else use_torch_compile
    )
    if not compile_enabled:
        return None

    resolved_mode = torch_compile_mode or os.environ.get(
        TORCH_COMPILE_MODE_ENV, DEFAULT_TORCH_COMPILE_MODE
    )
    return resolved_mode or DEFAULT_TORCH_COMPILE_MODE


def configure_attention_backend(
    model: PreTrainedModel, _device: torch.device
) -> str | None:
    """Set the preferred attention backend when the model exposes that hook."""
    if not hasattr(model, "set_attn_implementation"):
        return getattr(model.config, "_attn_implementation", None)

    # Our serving benchmarks for the 124M model were consistently faster with
    # SDPA than Flash Attention, even on CUDA. Keep an env override so we can
    # still force a different backend when experimenting.
    preferred_backend = os.environ.get(ATTENTION_BACKEND_ENV, DEFAULT_ATTENTION_BACKEND)
    try:
        model.set_attn_implementation(preferred_backend)
        return preferred_backend
    except (RuntimeError, ValueError):
        if preferred_backend != DEFAULT_ATTENTION_BACKEND:
            model.set_attn_implementation(DEFAULT_ATTENTION_BACKEND)
            return DEFAULT_ATTENTION_BACKEND
    return None


def format_dtype(dtype: torch.dtype | None) -> str | None:
    """Convert a torch dtype into a JSON-friendly string."""
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


def get_context_length(model: PreTrainedModel) -> int:
    """Infer the model context length from common config fields."""
    for attr in ("context_length", "max_position_embeddings", "n_positions"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Unable to determine model context length")


def maybe_compile_model(
    model: PreTrainedModel,
    device: torch.device,
    torch_compile_mode: str | None,
) -> PreTrainedModel:
    """Optionally wrap the model in torch.compile for inference."""
    if torch_compile_mode is None:
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is unavailable in this PyTorch build.")
    if device.type == "mps":
        raise RuntimeError("torch.compile is not supported for MPS inference here.")

    return torch.compile(model, mode=torch_compile_mode, dynamic=True)


def _load_tokenizer(repo_id: str) -> PreTrainedTokenizerBase:
    """Load the serving tokenizer with the repo's custom code enabled."""
    return AutoTokenizer.from_pretrained(repo_id, trust_remote_code=TRUST_REMOTE_CODE)


def _model_load_kwargs(inference_dtype: torch.dtype | None) -> dict[str, object]:
    """Build the kwargs used to load the causal LM for inference."""
    model_kwargs: dict[str, object] = {"trust_remote_code": TRUST_REMOTE_CODE}
    if inference_dtype is not None:
        model_kwargs["dtype"] = inference_dtype
    return model_kwargs


def _trim_to_context_length(
    input_ids: torch.Tensor, context_length: int
) -> torch.Tensor:
    """Clamp the prompt to the model's maximum supported context window."""
    if input_ids.shape[-1] <= context_length:
        return input_ids
    return input_ids[:, -context_length:]


def _decode_response_tokens(
    tokenizer: PreTrainedTokenizerBase, token_ids: list[int]
) -> str:
    """Decode generated token ids into a stripped response string."""
    return tokenizer.decode(token_ids).strip()


def load_inference_resources(
    repo_id: str | None = None,
    *,
    use_torch_compile: bool | None = None,
    torch_compile_mode: str | None = None,
) -> InferenceResources:
    """Load the tokenizer, model, and runtime metadata needed for serving."""
    resolved_repo_id = repo_id or get_model_repo_id()
    device = get_device()
    inference_dtype = get_inference_dtype(device)
    resolved_torch_compile_mode = get_torch_compile_mode(
        use_torch_compile=use_torch_compile,
        torch_compile_mode=torch_compile_mode,
    )
    tokenizer = _load_tokenizer(resolved_repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        resolved_repo_id,
        **_model_load_kwargs(inference_dtype),
    )
    model.to(device)
    attention_backend = configure_attention_backend(model, device)
    model.eval()
    context_length = get_context_length(model)
    model = maybe_compile_model(model, device, resolved_torch_compile_mode)
    return InferenceResources(
        model=model,
        tokenizer=tokenizer,
        device=device,
        inference_dtype=inference_dtype,
        attention_backend=attention_backend,
        context_length=context_length,
        stop_token_ids=_stop_token_ids(tokenizer),
        torch_compile_mode=resolved_torch_compile_mode,
    )


def _autocast_context(device: torch.device):
    """Return the autocast context used during generation."""
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _stop_token_ids(tokenizer: PreTrainedTokenizerBase) -> set[int]:
    """Collect token ids that should terminate chat generation."""
    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))

    vocab = tokenizer.get_vocab()
    for token in STOP_TOKENS:
        token_id = vocab.get(token)
        if token_id is not None:
            stop_ids.add(int(token_id))
    return stop_ids


def _top_p_sample(logits: torch.Tensor, temp: float, top_p: float) -> torch.Tensor:
    """Sample one token from the logits using temperature and nucleus sampling."""
    probs = torch.softmax(logits / temp, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    nucleus_probs = sorted_probs.clone()
    nucleus_probs[sorted_indices_to_remove] = 0.0
    nucleus_probs = nucleus_probs / nucleus_probs.sum(dim=-1, keepdim=True)

    sampled_sorted_index = torch.multinomial(nucleus_probs, 1)
    return torch.gather(sorted_indices, -1, sampled_sorted_index)


def _prepare_inputs(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    device: torch.device,
) -> torch.Tensor:
    """Render chat messages into a device-local input tensor."""
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = rendered.input_ids if hasattr(rendered, "input_ids") else rendered
    return input_ids.to(device=device)


def _mark_compile_step_begin(resources: InferenceResources) -> None:
    """Start a new cudagraph step when using torch.compile inference."""
    if resources.torch_compile_mode is None:
        return
    compiler = getattr(torch, "compiler", None)
    if compiler is None or not hasattr(compiler, "cudagraph_mark_step_begin"):
        return
    compiler.cudagraph_mark_step_begin()


def _model_supports_kv_cache(model: PreTrainedModel) -> bool:
    """Report whether the loaded model advertises KV-cache support."""
    return bool(getattr(getattr(model, "config", None), "use_cache", False))


def _generate_response_without_cache(
    input_ids: torch.Tensor,
    resources: InferenceResources,
    max_new_tokens: int,
    temp: float,
    top_p: float,
) -> str:
    """Generate a response for models that do not expose KV caching."""
    response_tokens: list[int] = []
    current_length = input_ids.shape[-1]
    buffer_length = min(resources.context_length, current_length + max_new_tokens)
    input_buffer = input_ids.new_empty((input_ids.shape[0], buffer_length))
    input_buffer[:, :current_length] = input_ids

    for _ in range(max_new_tokens):
        _mark_compile_step_begin(resources)
        with _autocast_context(resources.device):
            logits = resources.model(input_ids=input_buffer[:, :current_length]).logits[
                :, -1, :
            ]

        next_token = _top_p_sample(logits, temp=temp, top_p=top_p)
        next_token_id = int(next_token.item())
        if next_token_id in resources.stop_token_ids:
            break

        response_tokens.append(next_token_id)
        if current_length < buffer_length:
            input_buffer[:, current_length] = next_token[:, 0]
            current_length += 1
            continue

        # Once the buffer is full, keep a sliding context window instead of
        # reallocating a new tensor every decoding step.
        input_buffer[:, :-1] = input_buffer[:, 1:].clone()
        input_buffer[:, -1] = next_token[:, 0]

    return _decode_response_tokens(resources.tokenizer, response_tokens)


def _next_token_logits(
    input_ids: torch.Tensor,
    resources: InferenceResources,
    cache_position: torch.Tensor,
    cache_capacity: int | None = None,
    past_key_values=None,
):
    """Run one cached forward pass and return the next-token logits."""
    model_kwargs = {
        "input_ids": input_ids,
        "use_cache": True,
        "cache_position": cache_position,
    }
    if cache_capacity is not None:
        model_kwargs["cache_capacity"] = cache_capacity
    if past_key_values is not None:
        model_kwargs["past_key_values"] = past_key_values

    _mark_compile_step_begin(resources)
    with _autocast_context(resources.device):
        outputs = resources.model(**model_kwargs)

    return outputs.logits[:, -1, :], getattr(outputs, "past_key_values", None)


@torch.inference_mode()
def generate_response(
    messages: list[dict[str, str]],
    resources: InferenceResources,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temp: float = DEFAULT_TEMP,
    top_p: float = DEFAULT_TOP_P,
) -> str:
    """Generate a chat response with cached decode when the model supports it."""
    input_ids = _trim_to_context_length(
        _prepare_inputs(resources.tokenizer, messages, resources.device),
        resources.context_length,
    )

    if not _model_supports_kv_cache(resources.model):
        return _generate_response_without_cache(
            input_ids,
            resources,
            max_new_tokens=max_new_tokens,
            temp=temp,
            top_p=top_p,
        )

    # Prefill the cache with the full prompt once, then decode token by token.
    cache_capacity = min(resources.context_length, input_ids.shape[-1] + max_new_tokens)
    cache_position = torch.arange(input_ids.shape[-1], device=resources.device)
    logits, past_key_values = _next_token_logits(
        input_ids,
        resources,
        cache_position=cache_position,
        cache_capacity=cache_capacity,
    )
    if past_key_values is None:
        return _generate_response_without_cache(
            input_ids,
            resources,
            max_new_tokens=max_new_tokens,
            temp=temp,
            top_p=top_p,
        )

    response_tokens: list[int] = []
    cache_length = input_ids.shape[-1]

    for _ in range(max_new_tokens):
        next_token = _top_p_sample(logits, temp=temp, top_p=top_p)
        next_token_id = int(next_token.item())
        if next_token_id in resources.stop_token_ids:
            break

        response_tokens.append(next_token_id)
        if cache_length >= resources.context_length:
            break

        logits, past_key_values = _next_token_logits(
            next_token,
            resources,
            cache_position=torch.tensor([cache_length], device=resources.device),
            past_key_values=past_key_values,
        )
        cache_length += 1

    return _decode_response_tokens(resources.tokenizer, response_tokens)
