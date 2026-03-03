from pretrain.model import load_checkpoint, TransformerLM, GPTConfig
from datasets import load_dataset
from transformers import AutoTokenizer 
from huggingface_hub import snapshot_download


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
            input_ids = input_ids[-GPTConfig.context_length:]
            labels = labels[-GPTConfig.context_length:]
            attention_mask = attention_mask[-GPTConfig.context_length:]

        inputs.append(input_ids)
        targets.append(labels)
        attention_masks.append(attention_mask)

    return {"inputs": inputs, "targets": targets, "attention_mask": attention_masks}


if __name__ == "__main__":
    base_model = TransformerLM(GPTConfig.vocab_size, GPTConfig.context_length, GPTConfig.num_layers,
                               GPTConfig.d_model, GPTConfig.num_heads, GPTConfig.d_ff,
                               GPTConfig.theta, GPTConfig.device)

    # Download pretraining checkpoint
    checkpoint = snapshot_download("itskoma/GPT2.5", allow_patterns="pretraining_checkpoint/*", 
                                   repo_type="model", local_dir="checkpoints")
    # Load state dict to the model
    load_checkpoint("checkpoints/pretraining_checkpoint/", base_model)

    # Load the dataset for instruction tuning
    dataset = load_dataset("Cleanlab/databricks-dolly-15k-cleaned")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    dataset = dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer},
                          remove_columns = dataset['train'].column_names)
    breakpoint()
