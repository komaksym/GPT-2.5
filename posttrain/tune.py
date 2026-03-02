from pretrain.model import load_checkpoint, TransformerLM, GPTConfig
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config
from huggingface_hub import snapshot_download


def tokenize(example, tokenizer):
    if example.get("context"):
        prompt = (
            "Instruction:\n"
            f"{example["instruction"]}\n"
            "Context:\n"
            f"{example["context"]}\n"
            "Response:\n"
        )
    else:
        prompt = (
            "Instruction:\n"
            f"{example["instruction"]}\n"
            "Response:\n"
        )
    target = example["response"]

    # Tokenize
    prompt_ids = tokenizer(prompt, add_special_tokens=False)
    target_ids = tokenizer(target, add_special_tokens=False)

    # Combine prompt + answer into a single prompt
    full_prompt = prompt_ids + target_ids + tokenizer.eos

    # Make sure it doesn't go over the context length  
    if len(full_prompt) > GPT2Config.context_length:
        full_prompt = full_prompt[GPT2Config.context_length:]

    return prompt_ids
    #return {"prompts": [prompt_ids], "targets": [target_ids]}


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

    dataset = dataset.map(tokenize, batched=True, fn_kwargs={"tokenizer": tokenizer})
    breakpoint()
