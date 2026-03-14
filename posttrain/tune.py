from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from huggingface_hub import snapshot_download
from pretrain.model import (
    GPTConfig,
    TransformerLM,
    load_checkpoint,
    softmax,
    top_p_sampling,
)
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
)
from transformers.modeling_outputs import CausalLMOutput
from transformers.utils import PaddingStrategy
import tiktoken


DEFAULT_PRETRAINING_REPO_ID = "itskoma/GPT2.5"
DEFAULT_PRETRAINING_CHECKPOINT_PATTERN = "pretraining_checkpoint/*"
DEFAULT_PRETRAINING_CHECKPOINT_PATH = "checkpoints/pretraining_checkpoint/"
DEFAULT_PRETRAINING_LOCAL_DIR = "checkpoints"
DEFAULT_POSTTRAINING_CHECKPOINT_PATH = "checkpoints/posttraining_checkpoint"
DEFAULT_DATASET_ID = "Cleanlab/databricks-dolly-15k-cleaned"
DEFAULT_BATCH_SIZE = 8
ENCODER = tiktoken.get_encoding("gpt2")


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


def pad_dataset(dataset, tokenizer):
    for split in dataset:
        max_length = max(len(sample) for sample in dataset[split]["input_ids"])
        dataset[split] = dataset[split].map(
            pad_sample,
            fn_kwargs={"max_length": max_length, "tokenizer": tokenizer},
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
        **kwargs,
    ):
        super().__init__(**kwargs)
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
        """Expose TransformerLM through the minimal HF interface used by Trainer."""
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


def _build_base_model():
    """Construct the raw pretrained TransformerLM with the project defaults."""
    return TransformerLM(
        GPTConfig.vocab_size,
        GPTConfig.context_length,
        GPTConfig.num_layers,
        GPTConfig.d_model,
        GPTConfig.num_heads,
        GPTConfig.d_ff,
        GPTConfig.theta,
        GPTConfig.device,
    )


def _load_pretraining_model(
    repo_id: str = DEFAULT_PRETRAINING_REPO_ID,
    checkpoint_pattern: str = DEFAULT_PRETRAINING_CHECKPOINT_PATTERN,
    checkpoint_path: str = DEFAULT_PRETRAINING_CHECKPOINT_PATH,
    local_dir: str = DEFAULT_PRETRAINING_LOCAL_DIR,
):
    """Download and load a base checkpoint into a fresh TransformerLM."""
    base_model = _build_base_model()
    snapshot_download(
        repo_id,
        allow_patterns=checkpoint_pattern,
        repo_type="model",
        local_dir=local_dir,
    )
    load_checkpoint(checkpoint_path, base_model)
    return base_model


def _build_hf_model(base_model):
    """Wrap a loaded base model for HF Trainer while keeping tied weights exposed."""
    model = HFTransformerLM(MyConfig())
    model.model.load_state_dict(base_model.state_dict())
    return model


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


@torch.inference_mode()
def generate(
    prompt: str,
    max_new_tokens: int,
    context_length: int,
    model: HFTransformerLM,
    encoder=ENCODER,
    temp: float = 0.9,
    top_p: float = 0.8,
    device: torch.device = None,
) -> list[str]:
    """
    Main generation loop for the posttraining LLM.
    prompt: starting text
    max_new_tokens: number of tokens to generate per sequence
    context_length: maximum window size the model can handle
    model: the transformer model
    temp: softmax temperature
    top_p: nucleus sampling threshold
    """
    stop_words = ["<|endoftext|>", "Instruction:\n"]
    response = []

    # Encode prompt and move to device
    inputs = torch.tensor(encoder.encode(prompt), device=device).unsqueeze(0)
    for _ in range(max_new_tokens):
        # Truncate sequence if it exceeds the model's maximum context length
        if inputs.shape[-1] > context_length:
            inputs = inputs[:, -context_length:]
        # Generate next token logits using autocast for speed/memory efficiency
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(inputs)
        # Apply temperature and top-p sampling on the last token's logits
        probs = softmax(logits[:, -1, :], dim=-1, temp=temp)
        next_token = top_p_sampling(probs, p=top_p)
        # Append token to sequence
        inputs = torch.cat((inputs, next_token), dim=1)
        response.append(next_token.item())
        # Stop if the end-of-text special token is generated
        if encoder.decode([next_token.item()]) in stop_words:
            break

    return encoder.decode(response) + "\n"


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
    keep_running = True
    context = ""

    while keep_running:
        # Capture user input
        user_input = input(f"Ask anything. To end, type {stop_word}")
        if user_input == stop_word:
            break
        # Printing input
        print(user_input + waiting_for_response_schema)
        # Build a prompt by following a schema
        prompt = f"Instruction:\n{user_input}\nResponse:\n"
        # Add the prompt to the conversation context
        context += prompt
        # Generate response to the context + prompt and update the context
        response = generate(
            context, max_new_tokens, context_length, model, encoder, temp, top_p, device
        )
        print(response, end="\n" * 2)
        # Update context by appending response
        context += response


def inference_test(
    pretraining_checkpoint=True,
    pretraining_repo_id: str = DEFAULT_PRETRAINING_REPO_ID,
    pretraining_checkpoint_pattern: str = DEFAULT_PRETRAINING_CHECKPOINT_PATTERN,
    pretraining_checkpoint_path: str = DEFAULT_PRETRAINING_CHECKPOINT_PATH,
    pretraining_local_dir: str = DEFAULT_PRETRAINING_LOCAL_DIR,
    posttraining_checkpoint_path: str = DEFAULT_POSTTRAINING_CHECKPOINT_PATH,
    encoder=ENCODER,
    context_length=GPTConfig.context_length,
    max_new_tokens: int = 128,
    temp: float = 0.9,
    top_p: float = 0.8,
    device=None,
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

    device = GPTConfig.device
    model.to(device)

    chat(
        model,
        encoder,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
        temp=temp,
        top_p=top_p,
        device=device,
    )


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
