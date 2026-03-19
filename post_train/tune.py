from importlib import import_module
from importlib.util import find_spec
import os
import warnings

import numpy as np
import torch
import wandb
from post_train.model import (
    DEFAULT_PRETRAINING_CHECKPOINT_PATH,
    DEFAULT_PRETRAINING_CHECKPOINT_PATTERN,
    DEFAULT_CHECKPOINT_LOCAL_DIR,
    DEFAULT_REPO_ID,
    HFTransformerLM,
    MyConfig,
    _build_base_model,
    _build_hf_model,
    _load_pretraining_model,
)
from pre_train.model import GPTConfig, softmax, top_p_sampling
from transformers import AutoTokenizer, PythonBackend
from datasets import load_dataset
from huggingface_hub import snapshot_download
from trl import SFTTrainer, SFTConfig
from wandb.errors import CommError, UsageError
import sys


DEFAULT_POSTTRAINING_REPO_ID = "itskoma/GPT2.5"
DEFAULT_POSTTRAINING_CHECKPOINT_PATTERN = "posttraining_checkpoint/*"
DEFAULT_POSTTRAINING_CHECKPOINT_PATH = "checkpoints/posttraining_checkpoint/"

DEFAULT_DATASET_ID = "HuggingFaceTB/smol-smoltalk"
DEFAULT_BATCH_SIZE = 16
PACKING = True
__all__ = [
    "DEFAULT_REPO_ID",
    "DEFAULT_PRETRAINING_CHECKPOINT_PATTERN",
    "DEFAULT_PRETRAINING_CHECKPOINT_PATH",
    "DEFAULT_CHECKPOINT_LOCAL_DIR",
    "DEFAULT_DATASET_ID",
    "DEFAULT_BATCH_SIZE",
    "MyConfig",
    "HFTransformerLM",
    "_build_base_model",
    "_load_pretraining_model",
    "_build_hf_model",
    "compute_metrics",
    "get_trainer_precision_kwargs",
    "get_training_dtype",
    "chat",
    "inference_test",
    "is_flash_attn_2_installed",
    "main",
]


def compute_metrics(eval_pred):
    mean_loss = float(np.mean(eval_pred.losses))
    return {"perplexity": np.exp(mean_loss)}


def is_flash_attn_2_installed() -> bool:
    if find_spec("flash_attn") is None:
        return False
    try:
        import_module("flash_attn")
    except Exception:
        return False
    return True


def get_training_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_trainer_precision_kwargs(dtype: torch.dtype) -> dict[str, bool]:
    if dtype is torch.bfloat16:
        return {"bf16": True}
    if dtype is torch.float16:
        return {"fp16": True}
    return {}


def configure_packed_attention(model) -> None:
    if not PACKING:
        return

    if torch.cuda.is_available() and is_flash_attn_2_installed():
        assert torch.backends.cuda.flash_sdp_enabled()
        model.set_attn_implementation("flash_attention_2")
        return

    warnings.warn(
        "flash_attn is unavailable for packed training; falling back to SDPA.",
        stacklevel=2,
    )


@torch.inference_mode()
def generate(
    prompt: str,
    max_tokens: int | None = None,
    context_length: int = GPTConfig.context_length,
    batch_size: int = 1,
    model: HFTransformerLM | None = None,
    tokenizer: PythonBackend = None,
    temp: float = 0.9,
    top_p: float = 0.8,
    device: torch.device | None = None,
    max_new_tokens: int | None = None,
) -> list[str]:
    """
    Main generation loop for the posttraining LLM.
    prompt: starting text
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
    responses = []

    for _ in range(batch_size):
        response_tokens = []
        inputs = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt"
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

        responses.append(tokenizer.decode(response_tokens) + "\n")

    return responses


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
    context = ""

    while True:
        print("#" * 20, f"Ask anything. To end, type {stop_word}", "#" * 20)
        sys.stdout.write("PROMPT: ")
        user_input = input()
        if user_input == stop_word:
            break
        print(waiting_for_response_schema)
        prompt = f"<|user|>\n{user_input}\nResponse:\n"
        context += prompt
        response = generate(
            prompt=context,
            max_new_tokens=max_new_tokens,
            context_length=context_length,
            batch_size=1,
            model=model,
            tokenizer=tokenizer,
            temp=temp,
            top_p=top_p,
            device=device,
        )[0]
        print("RESPONSE: ", response, end="\n" * 2)
        context += response


def _print_sequences(sequences: list[str]) -> None:
    for sequence in sequences:
        print(sequence, end="" if sequence.endswith("\n") else "\n")


def get_tokenizer(tokenizer_path="gpt2"):
    extra_special_tokens = ["<|user|>", "<|assistant|>", "<|system|>"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, extra_special_tokens=extra_special_tokens
    )
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|' + message['role'] + '|>\\n' }}"
        "{% if message['role'] == 'assistant' %}"
        "{% generation %}"
        "{{ message['content'] + eos_token + '\\n' }}"
        "{% endgeneration %}"
        "{% else %}"
        "{{ message['content'] + eos_token + '\\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|assistant|>\\n' }}"
        "{% endif %}"
    )
    return tokenizer


def inference_test(
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
    model.eval()

    chat(
        model=model,
        tokenizer=tokenizer,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
        temp=temp,
        top_p=top_p,
        device=device,
    )


def main(
    repo_id: str = DEFAULT_REPO_ID,
    pretraining_checkpoint_pattern: str = DEFAULT_PRETRAINING_CHECKPOINT_PATTERN,
    pretraining_checkpoint_path: str = DEFAULT_PRETRAINING_CHECKPOINT_PATH,
    pretraining_local_dir: str = DEFAULT_CHECKPOINT_LOCAL_DIR,
    posttraining_checkpoint_path: str = DEFAULT_POSTTRAINING_CHECKPOINT_PATH,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_split: str = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Fine-tune the pretrained base model on the configured instruction dataset."""
    rank = int(os.environ.get("RANK", "0"))
    report_to = "none"
    if rank == 0:
        try:
            wandb.init(project="gpt-2.5")
            report_to = "wandb"
        except (CommError, UsageError) as exc:
            warnings.warn(
                f"W&B initialization failed; continuing without remote logging: {exc}",
                stacklevel=2,
            )
    training_dtype = get_training_dtype()

    base_model = _load_pretraining_model(
        repo_id=repo_id,
        checkpoint_pattern=pretraining_checkpoint_pattern,
        checkpoint_path=pretraining_checkpoint_path,
        local_dir=pretraining_local_dir,
    )
    model = _build_hf_model(base_model)
    model.config.dtype = training_dtype
    configure_packed_attention(model)

    tokenizer = get_tokenizer(tokenizer_path="gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model.tie_weights()

    dataset = load_dataset(dataset_id, dataset_split=dataset_split)

    trainer_args = SFTConfig(
        eval_strategy="epoch",
        gradient_checkpointing=False,
        include_for_metrics=["loss"],
        logging_steps=100,
        save_steps=1000,
        report_to=report_to,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        prediction_loss_only=True,
        num_train_epochs=3,
        learning_rate=1e-5,
        assistant_only_loss=True,
        packing=PACKING,
        **get_trainer_precision_kwargs(training_dtype),
    )

    trainer = SFTTrainer(
        model=model,
        args=trainer_args,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(posttraining_checkpoint_path)


if __name__ == "__main__":
    # main()
    inference_test(
        pretraining_checkpoint=False,
        checkpoint_pattern=DEFAULT_POSTTRAINING_CHECKPOINT_PATTERN,
        checkpoint_path=DEFAULT_POSTTRAINING_CHECKPOINT_PATH,
    )
