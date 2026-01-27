from model.model import *
import argparse
import numpy as np
import torch
import os
import wandb
import tiktoken
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import MixedPrecision, FullStateDictConfig, StateDictType
from torch.distributed.checkpoint.state_dict import get_state_dict
import functools


temp_path = "checkpoints/mid_training_checkpoint.pt"
final_path = "checkpoints/final_checkpoint.pt"

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def is_distributed():
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup():
    if not is_distributed():
        print("Running in Single-GPU/CPU mode")
        return

    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def run_evaluation(dataset_loader, model, context_length, device, run, rank, iteration):
    model.eval()

    with torch.no_grad():
        # Run validation
        val_inputs, val_labels = dataset_loader.next_batch()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, val_loss = model(val_inputs, val_labels)

        if rank == 0:
            print(f"step {iteration+1}, val loss: {val_loss.item()}")
            # Log loss in wandb
            run.log({"val_loss": val_loss.item()})

        # Run generation
        generated_sqs = generate("Once upon a time,", 20, context_length, model, 
                temp=0.8, top_p=0.9, device=device)

        # Print generated sentences
        if rank == 0:
            for seq in generated_sqs:
                print(seq)

    model.train()


def training_together(train_set, val_set, batch_size, grad_accum_steps, context_length, num_layers, 
                      d_model, num_heads, d_ff, theta, train_steps, lr, betas, eps, weight_decay,
                      device, rank, autowrap_policy, mp_policy, checkpoint=None):


    # Dataset loaders
    train_set_loader = DataLoader(train_set, batch_size, device)
    val_set_loader = DataLoader(val_set, batch_size, device)


    model = TransformerLM(50257, context_length, num_layers,
                          d_model, num_heads, d_ff, theta, device=device)
    
    # Wandb init
    run = None # For global scope
    if rank == 0:
        run = wandb.init(project="gpt-2.5")
        config = run.config
        run.watch(model)

    # If in distributed mode
    if rank is not None and autowrap_policy and mp_policy:
        model = FSDP(model.to(rank), 
                    auto_wrap_policy=autowrap_policy,
                    mixed_precision=mp_policy,
                    device_id=torch.cuda.current_device(),
                    sync_module_states=True)
                 
    # Warch model with wandb
    optimizer = AdamW(model.parameters(), lr, betas, eps, weight_decay)
    i = 0

    # Events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Check if checkpoint exists
    if rank == 0:
        if checkpoint is not None:
            # If it does, load it and keep training from the checkpoint
            i = load_checkpoint(checkpoint, model, optimizer)
            print("Continuing training from checkpoint!")
        else:
            print("Training from scratch!")
    
    # Start training
    while i < train_steps:
        start_event.record()
        # Update params once accumulated gradients
        loss_accum = 0.0
        for _ in range(grad_accum_steps):
            inputs, labels = train_set_loader.next_batch()

            # Predictions
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(inputs, labels)
            # Normalize the loss
            loss = loss / grad_accum_steps
            loss_accum += loss.detach().item()
            # Compute gradients
            loss.backward()
        # Step optimizer
        optimizer.step()
        # Zero grads
        optimizer.zero_grad()
        end_event.record()
        # Wait for the GPU to reach end_event
        torch.cuda.synchronize()
        step_time_ms = start_event.elapsed_time(end_event)
        tokens_per_sec = (batch_size * context_length) / (step_time_ms / 1000)
        # Coordinated logging
        if rank == 0:
            print(f"step {i+1}, loss: {loss_accum:.3f}, dt: {step_time_ms:.3f}, tok/s: {tokens_per_sec:.3f}")
            # Log loss in wandb
            run.log({"loss": loss_accum})


        # Save checkpoint and run validation every x steps
        if i >= 100 and i % 100 == 0:
            # FSDP way of saving a checkpoint
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                model, StateDictType.FULL_STATE_DICT, save_policy
            ):
                model_state_dict, optim_state_dict = get_state_dict(model, optimizer)
                #cpu_state = model.state_dict()
            if rank == 0:
                print("Saving a checkpoint...")
                save_checkpoint(model_state_dict, optim_state_dict, i, temp_path)
                print("Saved a mid-training checkpoint!")

            # Run evaluation
            run_evaluation(val_set_loader, model, context_length,
                           device, run, rank, i)

        # If about to finish training, delete the mid training checkpoint
        # And save the full training checkpoint
        elif i == train_steps - 1:
            if rank == 0:
                # Delete the mid training checkpoint
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    print("Removed mid-training checkpoint!")
                # Create a final checkpoint
                save_checkpoint(model, optimizer, i, final_path)
                print("Saved final checkpoint!")

        # Next training step
        i += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_accum_steps", type=int)
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--theta", type=float)
    parser.add_argument("--train_steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--weight_decay", type=float)

    # Parse args from CLI
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FSDP
    setup()

    local_rank, my_auto_wrap_policy, mp_policy = None, None, None
    if is_distributed():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        my_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy, 
            transformer_layer_cls={TransformerBlock},
        )

        bf16_ready = (
            torch.version.cuda
            and torch.cuda.is_bf16_supported()
            and dist.is_nccl_available()
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

    # Load the data
    train_data = np.load("ts_train_set_gpt2tok.npy", mmap_mode='r')
    train_set = data_loading(dataset=train_data, batch_size=1000000, \
                        context_length=args.context_length, device=device)

    val_data = np.load("ts_valid_set_gpt2tok.npy", mmap_mode='r')
    val_set = data_loading(dataset=val_data, batch_size=100000, \
                        context_length=args.context_length, device=device)

    # Start training
    training_together(train_set, val_set, args.batch_size, args.grad_accum_steps, args.context_length,
                      args.num_layers, args.d_model, args.num_heads, args.d_ff, 
                      args.theta, args.train_steps, args.lr, (args.beta1, args.beta2),
                      args.eps, args.weight_decay, device, local_rank, my_auto_wrap_policy, mp_policy)
    
    cleanup()


if __name__ == "__main__":
    main()