from pretrain.model import load_checkpoint, TransformerLM, GPTConfig, generate
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, GPT2Config
from transformers.modeling_outputs import CausalLMOutput
from huggingface_hub import snapshot_download
import evaluate
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import wandb


def format_prompt(instruction, context):
    instruction = instruction.strip()
    context = context.strip() if context else ""

    if context:
        prompt = (
            "Instruction:\n"
            f"{instruction}\n"
            "Context:\n"
            f"{context}\n"
            "Response:\n"
        )
    else:
        prompt = (
            "Instruction:\n"
            f"{instruction}\n"
            "Response:\n"
        )
    return prompt
    

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
        labels = [-100] * len(prompt_ids) + response_ids + [tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)

        if len(input_ids) > GPTConfig.context_length:
            continue

        inputs.append(input_ids)
        targets.append(labels)
        attention_masks.append(attention_mask)

    return {"input_ids": inputs, "labels": targets, "attention_mask": attention_masks}


def pad_sample(sample, max_length, tokenizer):
    pad_amount = max_length - len(sample['input_ids'])
    return {
        "input_ids": sample['input_ids'] + [tokenizer.pad_token_id] * pad_amount,
        "labels": sample['labels'] + [-100] * pad_amount,
        "attention_mask": sample['attention_mask'] + [0] * pad_amount
    }


def pad_dataset(dataset, tokenizer):
    for split in dataset:
        max_length = max(len(sample) for sample in dataset[split]['input_ids'])    
        dataset[split] = dataset[split].map(
            pad_sample,
            fn_kwargs={"max_length": max_length, "tokenizer": tokenizer}
        )
    return dataset


class HFTransformerLM(nn.Module):
    config_class = GPT2Config

    def __init__(self, base_model, config):
        super().__init__()
        self.config = config
        self.base_model = base_model
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        logits, _ = self.base_model(input_ids, attention_mask=attention_mask)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        return CausalLMOutput(loss=loss, logits=logits)


def compute_metrics(eval_pred):
    mean_loss = float(np.mean(eval_pred.losses))
    return {"perplexity": np.exp(mean_loss)}


if __name__ == "__main__":
    # Track with wandb
    wandb.init(project="gpt-2.5")

    base_model = TransformerLM(GPTConfig.vocab_size, GPTConfig.context_length, GPTConfig.num_layers,
                               GPTConfig.d_model, GPTConfig.num_heads, GPTConfig.d_ff,
                               GPTConfig.theta, GPTConfig.device)

    # Download pretraining checkpoint
    checkpoint = snapshot_download("itskoma/GPT2.5", allow_patterns="pretraining_checkpoint/*", 
                                   repo_type="model", local_dir="checkpoints")
    # Load state dict to the model
    load_checkpoint("checkpoints/pretraining_checkpoint/", base_model)
    # Load the dataset for instruction tuning
    dataset = load_dataset("Cleanlab/databricks-dolly-15k-cleaned", split="train")
    # Perform stratified split
    dataset = dataset.class_encode_column("category").train_test_split(
        test_size=0.1,
        stratify_by_column='category',
        seed=42)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer},
                          remove_columns = dataset['train'].column_names)
    # Pad dataset
    dataset = pad_dataset(dataset, tokenizer)

    # Slice for faster testing iteration
    #dataset["train"] = dataset["train"].select(range(10))
    #dataset["test"] = dataset["test"].select(range(10))

    # Model
    config = GPT2Config()
    model = HFTransformerLM(base_model, config)
    BATCH_SIZE = 5

    training_args = TrainingArguments(
        output_dir = "checkpoints/posttraining/",
        eval_strategy="epoch",
        include_for_metrics=["loss"],
        logging_steps=100,
        report_to="wandb",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        project="gpt-2.5",
        hub_model_id = "itskoma/GPT2.5"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()

    breakpoint()
    seqs = generate(
        prompt="The capital of France is ", 
        max_tokens=50, context_length=GPTConfig.context_length,
        batch_size=5, model=base_model, temp=0.9, top_p=0.8, 
        device=base_model.device
        )
    for s in seqs:
        print(s)
