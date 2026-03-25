from contextlib import nullcontext
import sys

from pre_train.model import GPTConfig, softmax, top_p_sampling
import torch
from post_train.model import HFTransformerLM
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

DEFAULT_REPO_ID = "itskoma/MyGPT"
STOP_WORDS = {"<|endoftext|>", "<|user|>", "<|system|>", "<|assistant|>"}


def _autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def load_tokenizer(repo_id: str) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)


@torch.inference_mode()
def generate(
    context: list[dict],
    max_tokens: int | None = None,
    context_length: int = GPTConfig.context_length,
    model: HFTransformerLM | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
    temp: float = 0.9,
    top_p: float = 0.8,
    device: torch.device | None = None,
    max_new_tokens: int | None = None,
) -> str:
    """
    Main generation loop for the posttraining LLM.
    context: chat context (previous user prompts and responses)
    max_tokens: number of tokens to generate per sequence
    context_length: maximum window size the model can handle
    model: the transformer model
    temp: softmax temperature
    top_p: nucleus sampling threshold
    """
    if model is None:
        raise ValueError("model must be provided")
    if tokenizer is None:
        raise ValueError("tokenizer must be provided")

    token_limit = max_tokens if max_tokens is not None else max_new_tokens
    if token_limit is None:
        raise ValueError("Either max_tokens or max_new_tokens must be provided")

    inputs = tokenizer.apply_chat_template(
        context, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if hasattr(inputs, "input_ids"):
        inputs = inputs.input_ids
    inputs = inputs.to(device=device)

    response_tokens = []
    for _ in range(token_limit):
        if inputs.shape[-1] > context_length:
            inputs = inputs[:, -context_length:]
        with _autocast_context(device):
            logits = model(inputs).logits
        probs = softmax(logits[:, -1, :], dim=-1, temp=temp)
        next_token = top_p_sampling(probs, p=top_p)
        decoded_next_token = tokenizer.decode([next_token.item()])
        if decoded_next_token in STOP_WORDS:
            break
        inputs = torch.cat((inputs, next_token), dim=1)
        response_tokens.append(next_token.item())

    answer = tokenizer.decode(response_tokens)
    return answer.strip()


def chat(
    model,
    tokenizer: PreTrainedTokenizerBase | None = None,
    context_length: int = 1024,
    max_new_tokens: int = 128,
    temp: float = 0.9,
    top_p: float = 0.8,
    device: torch.device | None = None,
) -> None:
    """Accept either the HF wrapper or the raw model and delegate generation."""
    assert max_new_tokens < context_length, (
        "Number of new generated tokens should be < context_length"
    )

    model.eval()
    waiting_for_response_schema = "\n" + "-" * 30 + "Responding..." + "-" * 30 + "\n"
    stop_word = "e"
    context = []

    while True:
        print("#" * 20, f"Ask anything. To end, type {stop_word}", "#" * 20)
        sys.stdout.write("\nPROMPT: ")
        user_input = input()
        if user_input == stop_word:
            break
        print(waiting_for_response_schema)
        context.append({"content": user_input, "role": "user"})
        response = generate(
            context=context,
            max_new_tokens=max_new_tokens,
            context_length=context_length,
            model=model,
            tokenizer=tokenizer,
            temp=temp,
            top_p=top_p,
            device=device,
        )
        print("RESPONSE: ", response, end="\n" * 2)
        context.append({"content": response, "role": "assistant"})


def run_inference(
    repo_id: str = DEFAULT_REPO_ID,
    tokenizer: PreTrainedTokenizerBase | None = None,
    context_length: int = GPTConfig.context_length,
    max_new_tokens: int = 128,
    temp: float = 0.9,
    top_p: float = 0.8,
    device: torch.device | None = None,
):
    model = HFTransformerLM.from_pretrained(repo_id)

    device = GPTConfig.device if device is None else device
    model.tie_weights()
    model.to(device)

    tokenizer = tokenizer or load_tokenizer(repo_id)

    chat(
        model=model,
        tokenizer=tokenizer,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
        temp=temp,
        top_p=top_p,
        device=device,
    )


if __name__ == "__main__":
    run_inference(max_new_tokens=256)
