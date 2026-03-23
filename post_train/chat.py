from pre_train.model import GPTConfig, softmax, top_p_sampling
from transformers import PythonBackend
import sys
import torch
from post_train.model import (
    HFTransformerLM,
    DEFAULT_REPO_ID,
)
from post_train.tune import (
    get_tokenizer,
)

DEFAULT_POSTTRAINING_SUBFOLDER = "posttraining_checkpoint"


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

    stop_words = ["<|endoftext|>", "<|user|>", "<|system|>"]
    stop_reason = ""

    response_tokens = []
    inputs = tokenizer.apply_chat_template(
        context, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if len(inputs) > 1:
        inputs = inputs.input_ids
    inputs = inputs.to(device=device)

    for _ in range(token_limit):
        if inputs.shape[-1] > context_length:
            inputs = inputs[:, -context_length:]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(inputs).logits
        probs = softmax(logits[:, -1, :], dim=-1, temp=temp)
        next_token = top_p_sampling(probs, p=top_p)
        decoded_next_tok = tokenizer.decode([next_token.item()])
        if decoded_next_tok in stop_words:
            stop_reason = f"STOPWORD: {decoded_next_tok}."
            break
        inputs = torch.cat((inputs, next_token), dim=1)
        response_tokens.append(next_token.item())

    if not stop_reason:
        stop_reason = "MAX TOKEN LIMIT EXCEEDED."

    ctx_before_response = tokenizer.decode(inputs)[0].split("<|assistant|>")[:-1]
    print(f"CONTEXT SEEN BY THE MODEL:\n {ctx_before_response}\n")
    answer = tokenizer.decode(response_tokens)
    return answer.strip(), stop_reason


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
        sys.stdout.write("\nPROMPT: ")
        user_input = input()
        # user_input = "Hello, how are you?" # OVERRIDE (REMOVE LATER)
        if user_input == stop_word:
            break
        print(waiting_for_response_schema)
        context.append({"content": user_input, "role": "user"})
        response, stop_reason = generate(
            context=context,
            max_new_tokens=max_new_tokens,
            context_length=context_length,
            model=model,
            tokenizer=tokenizer,
            temp=temp,
            top_p=top_p,
            device=device,
        )
        print("RESPONSE: ", response, f"\nSTOP REASON: {stop_reason}", end="\n" * 2)
        context.append({"content": response, "role": "assistant"})


def run_inference(
    repo_id: str = DEFAULT_REPO_ID,
    repo_id_subfolder: str = DEFAULT_POSTTRAINING_SUBFOLDER,
    tokenizer: PythonBackend = None,
    context_length=GPTConfig.context_length,
    max_new_tokens: int = 128,
    temp: float = 0.9,
    top_p: float = 0.8,
    device: torch.device | None = None,
):
    model = HFTransformerLM.from_pretrained(repo_id, subfolder=repo_id_subfolder)

    device = GPTConfig.device if device is None else device
    model.tie_weights()
    model.to(device)
    # model = torch.compile(model, mode="default")

    tokenizer = get_tokenizer()

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
        repo_id_subfolder=DEFAULT_POSTTRAINING_SUBFOLDER,
        max_new_tokens=256,
    )
