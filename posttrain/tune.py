import numpy as np
import torch
import wandb
from posttrain.data import (
    DEFAULT_DATASET_ID,
    CustomCollatorWithPadding,
    _load_instruction_dataset,
    format_prompt,
    pad_sample,
    tokenize,
)
from posttrain.model import (
    DEFAULT_PRETRAINING_CHECKPOINT_PATH,
    DEFAULT_PRETRAINING_CHECKPOINT_PATTERN,
    DEFAULT_PRETRAINING_LOCAL_DIR,
    DEFAULT_PRETRAINING_REPO_ID,
    HFTransformerLM,
    MyConfig,
    _build_base_model,
    _build_hf_model,
    _load_pretraining_model,
)
from pretrain.model import GPTConfig, softmax, top_p_sampling
from transformers import AutoTokenizer, Trainer, TrainingArguments
import tiktoken


DEFAULT_POSTTRAINING_CHECKPOINT_PATH = "checkpoints/posttraining_checkpoint"
DEFAULT_BATCH_SIZE = 8
ENCODER = tiktoken.get_encoding("gpt2")
__all__ = [
    "DEFAULT_PRETRAINING_REPO_ID",
    "DEFAULT_PRETRAINING_CHECKPOINT_PATTERN",
    "DEFAULT_PRETRAINING_CHECKPOINT_PATH",
    "DEFAULT_PRETRAINING_LOCAL_DIR",
    "DEFAULT_POSTTRAINING_CHECKPOINT_PATH",
    "DEFAULT_DATASET_ID",
    "DEFAULT_BATCH_SIZE",
    "ENCODER",
    "MyConfig",
    "HFTransformerLM",
    "_build_base_model",
    "_load_pretraining_model",
    "_build_hf_model",
    "format_prompt",
    "tokenize",
    "pad_sample",
    "CustomCollatorWithPadding",
    "_load_instruction_dataset",
    "compute_metrics",
    "generate",
    "chat",
    "inference_test",
    "main",
]


def compute_metrics(eval_pred):
    mean_loss = float(np.mean(eval_pred.losses))
    return {"perplexity": np.exp(mean_loss)}


@torch.inference_mode()
def generate(
    prompt: str,
    max_tokens: int | None = None,
    context_length: int = GPTConfig.context_length,
    batch_size: int = 1,
    model: HFTransformerLM | None = None,
    encoder=ENCODER,
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

    stop_words = ["<|endoftext|>", "Instruction:\n"]
    responses = []

    for _ in range(batch_size):
        response_tokens = []
        inputs = torch.tensor(encoder.encode(prompt), device=device).unsqueeze(0)
        for _ in range(token_limit):
            if inputs.shape[-1] > context_length:
                inputs = inputs[:, -context_length:]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(inputs)
            probs = softmax(logits[:, -1, :], dim=-1, temp=temp)
            next_token = top_p_sampling(probs, p=top_p)
            inputs = torch.cat((inputs, next_token), dim=1)
            response_tokens.append(next_token.item())
            if encoder.decode([next_token.item()]) in stop_words:
                break

        responses.append(encoder.decode(response_tokens) + "\n")

    return responses


def chat(
    model,
    encoder=ENCODER,
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
    waiting_for_response_schema = "\n-----------------Responding...-----------------\n"
    stop_word = "e"
    context = ""

    while True:
        user_input = input(f"Ask anything. To end, type {stop_word}")
        if user_input == stop_word:
            break
        print(user_input + waiting_for_response_schema)
        prompt = f"Instruction:\n{user_input}\nResponse:\n"
        context += prompt
        response = generate(
            prompt=context,
            max_new_tokens=max_new_tokens,
            context_length=context_length,
            batch_size=1,
            model=model,
            encoder=encoder,
            temp=temp,
            top_p=top_p,
            device=device,
        )[0]
        print(response, end="\n" * 2)
        context += response


def _print_sequences(sequences: list[str]) -> None:
    for sequence in sequences:
        print(sequence, end="" if sequence.endswith("\n") else "\n")


def inference_test(
    prompt: str | None = None,
    pretraining_checkpoint=True,
    pretraining_repo_id: str = DEFAULT_PRETRAINING_REPO_ID,
    pretraining_checkpoint_pattern: str = DEFAULT_PRETRAINING_CHECKPOINT_PATTERN,
    pretraining_checkpoint_path: str = DEFAULT_PRETRAINING_CHECKPOINT_PATH,
    pretraining_local_dir: str = DEFAULT_PRETRAINING_LOCAL_DIR,
    posttraining_checkpoint_path: str = DEFAULT_POSTTRAINING_CHECKPOINT_PATH,
    encoder=ENCODER,
    context_length=GPTConfig.context_length,
    max_tokens: int = 50,
    batch_size: int = 5,
    max_new_tokens: int = 128,
    temp: float = 0.9,
    top_p: float = 0.8,
    device: torch.device | None = None,
):
    """Run a quick generation smoke test from either the base or posttrained model."""
    if pretraining_checkpoint:
        model = _build_hf_model(
            _load_pretraining_model(
                repo_id=pretraining_repo_id,
                checkpoint_pattern=pretraining_checkpoint_pattern,
                checkpoint_path=pretraining_checkpoint_path,
                local_dir=pretraining_local_dir,
            )
        )
    else:
        model = HFTransformerLM.from_pretrained(posttraining_checkpoint_path)

    device = GPTConfig.device if device is None else device
    model.to(device)
    model.eval()

    if prompt is not None:
        responses = generate(
            prompt=prompt,
            max_tokens=max_tokens,
            context_length=context_length,
            batch_size=batch_size,
            model=model,
            encoder=encoder,
            temp=temp,
            top_p=top_p,
            device=device,
        )
        _print_sequences(responses)
        return

    chat(
        model,
        encoder,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
        temp=temp,
        top_p=top_p,
        device=device,
    )


def main(
    pretraining_repo_id: str = DEFAULT_PRETRAINING_REPO_ID,
    pretraining_checkpoint_pattern: str = DEFAULT_PRETRAINING_CHECKPOINT_PATTERN,
    pretraining_checkpoint_path: str = DEFAULT_PRETRAINING_CHECKPOINT_PATH,
    pretraining_local_dir: str = DEFAULT_PRETRAINING_LOCAL_DIR,
    posttraining_checkpoint_path: str = DEFAULT_POSTTRAINING_CHECKPOINT_PATH,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_split: str = "train",
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Fine-tune the pretrained base model on the configured instruction dataset."""
    wandb.init(project="gpt-2.5")

    base_model = _load_pretraining_model(
        repo_id=pretraining_repo_id,
        checkpoint_pattern=pretraining_checkpoint_pattern,
        checkpoint_path=pretraining_checkpoint_path,
        local_dir=pretraining_local_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = _load_instruction_dataset(
        tokenizer,
        dataset_id=dataset_id,
        split=dataset_split,
    )
    data_collator = CustomCollatorWithPadding(tokenizer)
    model = _build_hf_model(base_model)

    training_args = TrainingArguments(
        eval_strategy="epoch",
        include_for_metrics=["loss"],
        logging_steps=100,
        report_to="wandb",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        prediction_loss_only=True,
        num_train_epochs=3,
        learning_rate=1e-5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(posttraining_checkpoint_path)


if __name__ == "__main__":
    main()
