from pretrain.model import load_checkpoint, TransformerLM, AdamW, is_distributed, TransformerBlock, generate
#from pretrain.train import VOCAB_SIZE, setup
#from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
#from torch.distributed.fsdp import MixedPrecision
#from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch
from huggingface_hub import hf_hub_download
import importlib
import sys
#import os
#import functools
#import torch.distributed as dist


context_length = 1024
num_layers = 12
vocab_size = 50304
d_model = 768
num_heads = 12
d_ff = 2048
theta = 10000
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a_max = 18e-4 


if __name__ == "__main__":
    repo_id = "mikeawilliams/gpt2"

    model_code_path = hf_hub_download(repo_id=repo_id, filename="model.py")

    # Import the model module
    spec = importlib.util.spec_from_file_location("model", model_code_path)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules["model"] = model_module
    spec.loader.exec_module(model_module)


    checkpoint_path = hf_hub_download(repo_id=repo_id, filename="model_19072.pt")
    state_dict = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    model = TransformerLM(vocab_size, context_length, num_layers, 
                          d_model, num_heads, d_ff, 10000, device)

    # Load state dict
    model.load_state_dict(state_dict["model"])

    # Test
    generated_sqs = generate(
        "Once upon a time,",
        max_tokens=50,
        context_length=context_length,
        batch_size=5,
        model=model,
        temp=0.8,
        top_p=0.9,
        device=device,
    )

    for seq in generated_sqs:
        # Add to the wandb table
        print(seq)