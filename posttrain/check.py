from pretrain.model import load_checkpoint, TransformerLM, AdamW, is_distributed, TransformerBlock
from pretrain.train import VOCAB_SIZE, setup
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch
import os
import functools
import torch.distributed as dist


context_length = 1024
num_layers = 12
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
    setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FSDP
    setup()

    local_rank, my_auto_wrap_policy, mp_policy = None, None, None
    rank = dist.get_rank()

    if is_distributed():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        my_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerBlock},
        )

        bf16_ready = (
            torch.version.cuda and torch.cuda.is_bf16_supported() and dist.is_nccl_available()
        )

        bfSixteen = MixedPrecision(
            param_dtype=torch.bfloat16,
            # Gradient communication precision.
            reduce_dtype=torch.bfloat16,
            # Buffer precision.
            buffer_dtype=torch.bfloat16,
        )

        if bf16_ready:
            mp_policy = bfSixteen
        else:
            mp_policy = None

    model = TransformerLM(
        VOCAB_SIZE, context_length, num_layers, d_model, num_heads, d_ff, theta, device=device
    )

     # If in distributed mode
    if rank is not None and transformer_auto_wrap_policy and mp_policy:
        model = FSDP(
            model.to(rank),
            auto_wrap_policy=transformer_auto_wrap_policy,
            mixed_precision=mp_policy,
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
        )

    optimizer = AdamW(model.parameters(), a_max, betas, eps, weight_decay)
    rank = 0
    rank = dist.get_rank()
    checkpoint_path = "checkpoint/final_checkpoint.pt"

    it = load_checkpoint(checkpoint_path, model, optimizer, rank)
    breakpoint()