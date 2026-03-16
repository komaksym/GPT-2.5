from dataclasses import dataclass
from typing import Any

import torch
from datasets import load_dataset
from pretrain.model import GPTConfig
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

DEFAULT_DATASET_ID = "HuggingFaceTB/smol-smoltalk"


def format_prompt(instruction, context):
    instruction = instruction.strip()
    context = context.strip() if context else ""

    if context:
        return f"Instruction:\n{instruction}\nContext:\n{context}\nResponse:\n"
    return f"Instruction:\n{instruction}\nResponse:\n"


def tokenize(examples, tokenizer):
    inputs = []
    targets = []
    attention_masks = []

    for instruction, context, response in zip(
        examples["instruction"], examples["context"], examples["response"]
    ):
        prompt = format_prompt(instruction, context)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]

        if len(input_ids) > GPTConfig.context_length:
            continue

        inputs.append(input_ids)
        targets.append(
            [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]
        )
        attention_masks.append([1] * len(input_ids))

    return {"input_ids": inputs, "labels": targets, "attention_mask": attention_masks}


def pad_sample(sample, max_length, tokenizer):
    pad_amount = max_length - len(sample["input_ids"])
    return {
        "input_ids": sample["input_ids"] + [tokenizer.pad_token_id] * pad_amount,
        "labels": sample["labels"] + [-100] * pad_amount,
        "attention_mask": sample["attention_mask"] + [0] * pad_amount,
    }


@dataclass
class CustomCollatorWithPadding(DataCollatorWithPadding):
    tokenizer: PreTrainedTokenizerBase
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def pad_batch(self, batch):
        max_length = max(len(sample["input_ids"]) for sample in batch)
        # Keep label padding aligned with the model's ignore_index=-100 loss masking.
        padded_batch = [
            pad_sample(sample, max_length, self.tokenizer) for sample in batch
        ]
        return {
            "input_ids": torch.tensor([sample["input_ids"] for sample in padded_batch]),
            "labels": torch.tensor([sample["labels"] for sample in padded_batch]),
            "attention_mask": torch.tensor(
                [sample["attention_mask"] for sample in padded_batch]
            ),
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        return self.pad_batch(features)


def _load_instruction_dataset(
    tokenizer,
    dataset_id: str = DEFAULT_DATASET_ID,
    split: str = "train",
):
    """Load, stratify, and tokenize the instruction-tuning dataset."""
    dataset = load_dataset(dataset_id, split=split)
    dataset = dataset.class_encode_column("category").train_test_split(
        test_size=0.1,
        stratify_by_column="category",
        seed=42,
    )
    return dataset.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset["train"].column_names,
    )
