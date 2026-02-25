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
num_layers = 2
vocab_size = 50257
d_model = 768
num_heads = 2
d_ff = 2048
theta = 10000
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a_max = 18e-4 


if __name__ == "__main__":
    model = TransformerLM(vocab_size, context_length, num_layers, 
                          d_model, num_heads, d_ff, 10000, device)
 

    # Load state dict
    load_checkpoint("checkpoints/final_checkpoint", model)
    breakpoint()