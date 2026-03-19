from pre_train.model import GPTConfig, softmax, top_p_sampling
from transformers import PythonBackend
import sys
from huggingface_hub import snapshot_download
import torch
from post_train.model import HFTransformerLM
from post_train.model import (
    DEFAULT_REPO_ID,
    DEFAULT_CHECKPOINT_LOCAL_DIR,
)
from post_train.tune import (
    DEFAULT_POSTTRAINING_CHECKPOINT_PATTERN,
    DEFAULT_POSTTRAINING_CHECKPOINT_PATH,
)


@torch.inference_mode()
def generate(
    context: list[dict],
    max_tokens: int | None = None,
    context_length: int = GPTConfig.context_length,
    model: HFTransformerLM | None = None,
    tokenizer: PythonBackend = None,
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
    batch_size: number of responses to generate
    model: the transformer model
    temp: softmax temperature
    top_p: nucleus sampling threshold
    """
    if model is None:
        raise ValueError("model must be provided")

    token_limit = max_tokens if max_tokens is not None else max_new_tokens
    if token_limit is None:
        raise ValueError("Either max_tokens or max_new_tokens must be provided")

    stop_words = ["<|endoftext|>\n", "<|user|>\n", "<|system|>\n"]

    response_tokens = []
    inputs = tokenizer.apply_chat_template(
        context, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

    for _ in range(token_limit):
        if inputs.shape[-1] > context_length:
            inputs = inputs[:, -context_length:]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(inputs).logits
        probs = softmax(logits[:, -1, :], dim=-1, temp=temp)
        next_token = top_p_sampling(probs, p=top_p)
        inputs = torch.cat((inputs, next_token), dim=1)
        response_tokens.append(next_token.item())
        if tokenizer.decode([next_token.item()]) in stop_words:
            break

    updated_context = tokenizer.decode(response_tokens)
    last_answer = updated_context.split("<|assistant|>")[-1]
    return last_answer


def chat(
    model,
    tokenizer: PythonBackend = None,
    context_length: int = 1024,
    max_new_tokens: int = 128,
    temp: float = 0.9,
    top_p: float = 0.8,
    device: torch.device = None,
) -> list[str]:
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
        sys.stdout.write("PROMPT: ")
        user_input = input()
        if user_input == stop_word:
            break
        print(waiting_for_response_schema)
        context.append({"content": user_input, "role": "user"})
        response = generate(
            prompt=context,
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
    checkpoint_pattern: str = DEFAULT_POSTTRAINING_CHECKPOINT_PATTERN,
    checkpoint_path: str = DEFAULT_POSTTRAINING_CHECKPOINT_PATH,
    local_dir: str = DEFAULT_CHECKPOINT_LOCAL_DIR,
    tokenizer: PythonBackend = None,
    context_length=GPTConfig.context_length,
    max_new_tokens: int = 128,
    temp: float = 0.9,
    top_p: float = 0.8,
    device: torch.device | None = None,
):
    assert "posttraining" in checkpoint_path, (
        "checkpoint_path is not path to a POST-training checkpoint"
    )

    snapshot_download(
        repo_id,
        allow_patterns=checkpoint_pattern,
        repo_type="model",
        local_dir=local_dir,
    )
    model = HFTransformerLM.from_pretrained(checkpoint_path)

    device = GPTConfig.device if device is None else device
    model.tie_weights()
    model.to(device)

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
    run_inference(
        checkpoint_pattern=DEFAULT_POSTTRAINING_CHECKPOINT_PATTERN,
        checkpoint_path=DEFAULT_POSTTRAINING_CHECKPOINT_PATH,
    )
