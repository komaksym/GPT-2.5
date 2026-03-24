from importlib import import_module
from importlib.util import find_spec
import os
import warnings

import numpy as np
import torch
import wandb
from post_train.model import (
    DEFAULT_REPO_ID,
    DEFAULT_CHECKPOINT_SUBFOLDER,
    HFTransformerLM
)
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from wandb.errors import CommError, UsageError


DEFAULT_POSTTRAINING_CHECKPOINT_PATTERN = "posttraining_checkpoint/*"

DEFAULT_DATASET_ID = "HuggingFaceTB/smol-smoltalk"
DEFAULT_BATCH_SIZE = 8
PACKING = True


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


def main(
    repo_id: str = DEFAULT_REPO_ID,
    checkpoint_subfolder: str = DEFAULT_CHECKPOINT_SUBFOLDER,
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

    model = HFTransformerLM.from_pretrained(repo_id, subfolder=checkpoint_subfolder)
    model.config.dtype = training_dtype
    configure_packed_attention(model)

    tokenizer = get_tokenizer(tokenizer_path="gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model.tie_weights()

    dataset = load_dataset(dataset_id, split=dataset_split)

    trainer_args = SFTConfig(
        eval_strategy="epoch",
        gradient_checkpointing=False,
        logging_steps=100,
        save_steps=1000,
        report_to=report_to,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        learning_rate=1e-5,
        assistant_only_loss=True,
        prediction_loss_only=True,
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

   
if __name__ == "__main__":
    main()
