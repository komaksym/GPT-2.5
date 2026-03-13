from pretrain.model import (
    load_checkpoint,
    TransformerLM,
    GPTConfig,
    top_p_sampling,
    softmax,
)
import tiktoken
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PretrainedConfig,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)
from transformers.modeling_outputs import CausalLMOutput
from transformers.utils import PaddingStrategy
from huggingface_hub import snapshot_download
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from dataclasses import dataclass
from typing import Any


def format_prompt(instruction, context):
    instruction = instruction.strip()
    context = context.strip() if context else ""

    if context:
        prompt = f"Instruction:\n{instruction}\nContext:\n{context}\nResponse:\n"
    else:
        prompt = f"Instruction:\n{instruction}\nResponse:\n"
    return prompt


def tokenize(examples, tokenizer):
    inputs = []
    targets = []
    attention_masks = []

    for response, context, instruction in zip(
        examples["output"], examples["input"], examples["instruction"]
    ):
        prompt = format_prompt(instruction, context)

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + response_ids + [tokenizer.eos_token_id]
        labels = [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)

        if len(input_ids) > GPTConfig.context_length:
            continue

        inputs.append(input_ids)
        targets.append(labels)
        attention_masks.append(attention_mask)

    return {"input_ids": inputs, "labels": targets, "attention_mask": attention_masks}


def pad_sample(sample, max_length, tokenizer):
    pad_amount = max_length - len(sample["input_ids"])
    return {
        "input_ids": sample["input_ids"] + [tokenizer.pad_token_id] * pad_amount,
        "labels": sample["labels"] + [-100] * pad_amount,
        "attention_mask": sample["attention_mask"] + [0] * pad_amount,
    }


def pad_dataset(dataset, tokenizer):
    for split in dataset:
        max_length = max(len(sample) for sample in dataset[split]["input_ids"])
        dataset[split] = dataset[split].map(
            pad_sample, fn_kwargs={"max_length": max_length, "tokenizer": tokenizer}
        )
    return dataset


class MyConfig(PretrainedConfig):
    model_type = "gpt2.5"

    def __init__(
        self,
        vocab_size=GPTConfig.vocab_size,
        context_length=GPTConfig.context_length,
        num_layers=GPTConfig.num_layers,
        num_heads=GPTConfig.num_heads,
        d_model=GPTConfig.d_model,
        d_ff=GPTConfig.d_ff,
        theta=GPTConfig.theta,
        device=str(GPTConfig.device),
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.theta = float(theta)
        self.device = device if isinstance(device, str) else str(device)


class HFTransformerLM(PreTrainedModel):
    config_class = MyConfig
    _tied_weights_keys = {"model.linear.weight": "model.emb.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = TransformerLM(
            config.vocab_size,
            config.context_length,
            config.num_layers,
            config.d_model,
            config.num_heads,
            config.d_ff,
            config.theta,
            config.device,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.emb

    def get_output_embeddings(self):
        return self.model.linear

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        logits, _ = self.model(input_ids, attention_mask=attention_mask)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(loss=loss, logits=logits)


def compute_metrics(eval_pred):
    mean_loss = float(np.mean(eval_pred.losses))
    return {"perplexity": np.exp(mean_loss)}


@torch.inference_mode()
def generate(
    prompt: str,
    max_tokens: int,
    context_length: int,
    batch_size: int,
    model: nn.Module,
    temp: float,
    top_p: float,
    device: torch.device,
) -> list[str]:
    """
    Main generation loop for the LLM.
    prompt: starting text
    max_tokens: number of tokens to generate per sequence
    context_length: maximum window size the model can handle
    batch_size: number of sequences to generate
    model: the transformer model
    temp: softmax temperature
    top_p: nucleus sampling threshold
    """
    enc = tiktoken.get_encoding("gpt2")
    sentences = []
    model.eval()

    for i in range(batch_size):
        # Encode prompt and move to device
        inputs = torch.tensor(enc.encode(prompt), device=device).unsqueeze(0)
        for _ in range(max_tokens):
            # Generate next token logits using autocast for speed/memory efficiency
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                output = model(inputs)
                logits = output.logits
            # Apply temperature and top-p sampling on the last token's logits
            probs = softmax(logits[:, -1, :], dim=-1, temp=temp)
            next_token = top_p_sampling(probs, p=top_p)
            # Append token to sequence
            inputs = torch.cat((inputs, next_token), dim=1)
            # Stop if the end-of-text special token is generated
            if enc.decode([next_token.item()]) == "<|endoftext|>":
                break
            # Truncate sequence if it exceeds the model's maximum context length
            if inputs.shape[-1] > context_length:
                inputs = inputs[:, -context_length:]

        # Record the final generated sequence
        sentences.append(
            f"\nGenerated sequence №{i + 1}:\n" + enc.decode(inputs[0].tolist()) + "\n"
        )
    return sentences


def inference_test(prompt=None, pretraining_checkpoint=True):
    if pretraining_checkpoint:
        base_model = TransformerLM(
            GPTConfig.vocab_size,
            GPTConfig.context_length,
            GPTConfig.num_layers,
            GPTConfig.d_model,
            GPTConfig.num_heads,
            GPTConfig.d_ff,
            GPTConfig.theta,
            GPTConfig.device,
        )

        # Download pretraining checkpoint
        snapshot_download(
            "itskoma/GPT2.5",
            allow_patterns="pretraining_checkpoint/*",
            repo_type="model",
            local_dir="checkpoints",
        )
        # Load state dict to the model
        load_checkpoint("checkpoints/pretraining_checkpoint/", base_model)

        # Model
        model = HFTransformerLM(MyConfig())
        model.model.load_state_dict(base_model.state_dict())
    else:
        model = HFTransformerLM.from_pretrained("checkpoints/posttraining_checkpoint")

    device = GPTConfig.device
    model.to(device)

    seqs = generate(
        prompt=prompt,
        max_tokens=50,
        context_length=GPTConfig.context_length,
        batch_size=5,
        model=model,
        temp=0.9,
        top_p=0.8,
        device=model.device,
    )

    for s in seqs:
        print(s)


@dataclass
class CustomCollatorWithPadding(DataCollatorWithPadding):
    tokenizer: PreTrainedTokenizerBase
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def pad_batch(self, batch):
        max_length = max(len(sample["input_ids"]) for sample in batch)

        for sample in batch:
            pad_amount = max_length - len(sample["input_ids"])
            sample["input_ids"] = (
                sample["input_ids"] + [self.tokenizer.pad_token_id] * pad_amount
            )
            sample["labels"] = sample["labels"] + [-100] * pad_amount
            sample["attention_mask"] = sample["attention_mask"] + [0] * pad_amount

        return {
            "input_ids": torch.tensor([sample["input_ids"] for sample in batch]),
            "labels": torch.tensor([sample["labels"] for sample in batch]),
            "attention_mask": torch.tensor(
                [sample["attention_mask"] for sample in batch]
            ),
        }

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # Pad input_ids
        return self.pad_batch(features)


def main():
    # Track with wandb
    wandb.init(project="gpt-2.5")

    base_model = TransformerLM(
        GPTConfig.vocab_size,
        GPTConfig.context_length,
        GPTConfig.num_layers,
        GPTConfig.d_model,
        GPTConfig.num_heads,
        GPTConfig.d_ff,
        GPTConfig.theta,
        GPTConfig.device,
    )

    # Download pretraining checkpoint
    snapshot_download(
        "itskoma/GPT2.5",
        allow_patterns="pretraining_checkpoint/*",
        repo_type="model",
        local_dir="checkpoints",
    )
    # Load state dict to the model
    load_checkpoint("checkpoints/pretraining_checkpoint/", base_model)
    # Load the dataset for instruction tuning
    # dataset = load_dataset("Cleanlab/databricks-dolly-15k-cleaned", split="train")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    # Perform stratified split
    # dataset = dataset.class_encode_column("category").train_test_split(
    # test_size=0.1,
    # stratify_by_column='category',
    # seed=42)

    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(
        tokenize,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset["train"].column_names,
    )
    # Pad dataset
    # dataset = pad_dataset(dataset, tokenizer)
    data_collator = CustomCollatorWithPadding(tokenizer)

    # Slice for faster testing iteration
    # dataset["train"] = dataset["train"].select(range(10))
    # dataset["test"] = dataset["test"].select(range(10))

    # Model
    config = MyConfig()
    model = HFTransformerLM(config)
    model.model.load_state_dict(base_model.state_dict())
    model.tie_weights()
    BATCH_SIZE = 4

    training_args = TrainingArguments(
        eval_strategy="epoch",
        include_for_metrics=["loss"],
        logging_steps=100,
        report_to="wandb",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        prediction_loss_only=True,
        num_train_epochs=3,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        # lr_scheduler_type="cosine",
        # warmup_steps=0.05,
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
    trainer.save_model("checkpoints/posttraining_checkpoint")


if __name__ == "__main__":
    # inference_test(prompt="The capital of Germany is ", pretraining_checkpoint=True)
    main()
    # inference_test(prompt="2+2 is ", pretraining_checkpoint=False)
