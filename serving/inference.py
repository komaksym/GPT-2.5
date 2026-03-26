from contextlib import nullcontext
from dataclasses import dataclass
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

DEFAULT_MODEL_REPO_ID = "itskoma/MyGPT"
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMP = 0.9
DEFAULT_TOP_P = 0.8
STOP_TOKENS = ("<|endoftext|>", "<|user|>", "<|system|>", "<|assistant|>")


@dataclass(slots=True)
class InferenceResources:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    inference_dtype: torch.dtype | None
    attention_backend: str | None
    context_length: int
    stop_token_ids: set[int]


def get_model_repo_id() -> str:
    return os.environ.get("MODEL_REPO_ID", DEFAULT_MODEL_REPO_ID)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_inference_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def configure_attention_backend(
    model: PreTrainedModel, device: torch.device
) -> str | None:
    if not hasattr(model, "set_attn_implementation"):
        return getattr(model.config, "_attn_implementation", None)

    preferred_backend = "flash_attention_2" if device.type == "cuda" else "sdpa"
    try:
        model.set_attn_implementation(preferred_backend)
        return preferred_backend
    except (RuntimeError, ValueError):
        if preferred_backend != "sdpa":
            model.set_attn_implementation("sdpa")
            return "sdpa"
    return None


def format_dtype(dtype: torch.dtype | None) -> str | None:
    if dtype is None:
        return None
    return str(dtype).removeprefix("torch.")


def get_context_length(model: PreTrainedModel) -> int:
    for attr in ("context_length", "max_position_embeddings", "n_positions"):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError("Unable to determine model context length")


def load_inference_resources(repo_id: str | None = None) -> InferenceResources:
    resolved_repo_id = repo_id or get_model_repo_id()
    device = get_device()
    inference_dtype = get_inference_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_repo_id,
        trust_remote_code=True,
    )
    model_kwargs = {"trust_remote_code": True}
    if inference_dtype is not None:
        model_kwargs["torch_dtype"] = inference_dtype
    model = AutoModelForCausalLM.from_pretrained(resolved_repo_id, **model_kwargs)
    model.to(device)
    attention_backend = configure_attention_backend(model, device)
    model.eval()
    return InferenceResources(
        model=model,
        tokenizer=tokenizer,
        device=device,
        inference_dtype=inference_dtype,
        attention_backend=attention_backend,
        context_length=get_context_length(model),
        stop_token_ids=_stop_token_ids(tokenizer),
    )


def _autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _stop_token_ids(tokenizer: PreTrainedTokenizerBase) -> set[int]:
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
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = rendered.input_ids if hasattr(rendered, "input_ids") else rendered
    return input_ids.to(device=device)


@torch.inference_mode()
def generate_response(
    messages: list[dict[str, str]],
    resources: InferenceResources,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temp: float = DEFAULT_TEMP,
    top_p: float = DEFAULT_TOP_P,
) -> str:
    input_ids = _prepare_inputs(resources.tokenizer, messages, resources.device)
    response_tokens: list[int] = []

    for _ in range(max_new_tokens):
        if input_ids.shape[-1] > resources.context_length:
            input_ids = input_ids[:, -resources.context_length :]

        with _autocast_context(resources.device):
            logits = resources.model(input_ids=input_ids).logits[:, -1, :]

        next_token = _top_p_sample(logits, temp=temp, top_p=top_p)
        next_token_id = int(next_token.item())
        if next_token_id in resources.stop_token_ids:
            break

        input_ids = torch.cat((input_ids, next_token), dim=1)
        response_tokens.append(next_token_id)

    return resources.tokenizer.decode(response_tokens).strip()
