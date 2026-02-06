from train import training_together, VOCAB_SIZE, TRAINING_SET_DATA_CREATION_BATCH_SIZE, \
                                VAL_SET_DATA_CREATION_BATCH_SIZE, run_evaluation, cleanup, setup, is_distributed
import argparse
import functools
import os
import warnings

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
import tqdm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import wandb
from hellaswag import HellaSwagLoader, compute_hellaswag
from model.model import (
    AdamW,
    DataLoader,
    TransformerBlock,
    TransformerLM,
    generate,
    gradient_clipping,
    learning_rate_schedule,
    load_checkpoint,
    save_checkpoint,
)


temp_path = "checkpoints/mid_training_checkpoint.pt"
final_path = "checkpoints/final_checkpoint.pt"

def train(
    train_set_loader,
    val_set_loader,
    batch_size,
    grad_accum_steps,
    context_length,
    num_layers,
    d_model,
    num_heads,
    d_ff,
    theta,
    train_steps,
    a_max,
    betas,
    eps,
    weight_decay,
    device,
    rank,
    autowrap_policy,
    mp_policy,
    checkpoint=None,
):

    model = TransformerLM(
        VOCAB_SIZE, context_length, num_layers, d_model, num_heads, d_ff, theta, device=device
    )

    # Torch compile the model
    model.compile()

    # Wandb init
    run = None  # For global scope
    pbar = None
    master_rank = True if rank == 0 else False
    if master_rank:
        run = wandb.init(project="gpt-2.5")
        config = run.config
        config.batch_size = batch_size
        config.grad_accum_steps = grad_accum_steps
        config.context_length = context_length
        config.num_layers = num_layers
        config.d_model = d_model
        config.num_heads = num_heads
        config.d_ff = d_ff
        config.theta = theta
        config.train_steps = train_steps
        config.lr_max = a_max
        config.betas = betas
        config.eps = eps
        config.weight_decay = weight_decay
        config.training_set_data_creation_batch_size = TRAINING_SET_DATA_CREATION_BATCH_SIZE
        config.val_set_data_creation_batch_size = VAL_SET_DATA_CREATION_BATCH_SIZE
        config.training_sampled_with_replacement = (
            False if "train_set_loader" and "val_set_loader" in locals() else True
        )
        run.watch(model)
        master_table = wandb.Table(columns=["step", "prediction"], log_mode="INCREMENTAL")

    # If in distributed mode
    if rank is not None and autowrap_policy and mp_policy:
        model = FSDP(
            model.to(rank),
            auto_wrap_policy=autowrap_policy,
            mixed_precision=mp_policy,
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
        )

    # HellaSwag evaluation
    hellaswag_loader = HellaSwagLoader(batch_size, context_length, tiktoken.get_encoding("gpt2"))

    # Warch model with wandb
    optimizer = AdamW(model.parameters(), a_max, betas, eps, weight_decay)
    last_checkpoint_loss = float("inf")
    i = 0

    # Events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Check if checkpoint exists
    if checkpoint is not None and os.path.exists(checkpoint):
        # If it does, load it and keep training from the checkpoint
        i = load_checkpoint(checkpoint, model, optimizer, rank)
        if master_rank:
            print(f"Continuing training from checkpoint at iteration {i}!")
    else:
        if master_rank:
            print("Training from scratch!")
        pbar = tqdm.tqdm(range(train_steps), colour="blue")

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
        # Grad clipping
        if rank is not None:
            norm = model.clip_grad_norm_(1.0)
        else:
            norm = gradient_clipping(model.parameters(), 1.0)
        # Learning rate scheduler
        lr = learning_rate_schedule(i, a_max, 0.1 * a_max, 0.05 * train_steps, train_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # Step optimizer
        optimizer.step()
        # Zero grads
        optimizer.zero_grad()
        end_event.record()
        # Wait for the GPU to reach end_event
        torch.cuda.synchronize()
        step_time_ms = start_event.elapsed_time(end_event)
        tokens_per_sec = (batch_size * context_length) / (step_time_ms / 1000)
        perplexity = np.exp(loss_accum)
        # Coordinated logging
        if master_rank:
            print(
                f"step {i + 1}, loss: {loss_accum:.3f}, perp: {perplexity:.3f}, norm: {norm:.3f}, dt: {step_time_ms:.3f}, tok/s: {tokens_per_sec:.3f}"
            )
            # Log loss in wandb
            run.log({"loss": loss_accum, "perplexity": perplexity, "norm": norm, "lr": lr})
            # Increment pbar
            pbar.update(1)

        # Run evaluation
        if i % 100 == 0:
            # Wandb table for tracking generated sequences
            generated_seqs = run_evaluation(
                val_set_loader, model, context_length, device, run, rank, i
            )

            # Run HellaSwag
            hs_inputs, hs_labels, completion_mask = hellaswag_loader.next_batch()
            hs_score = compute_hellaswag(model, hs_inputs, hs_labels, completion_mask)
            if master_rank:
                # Populate the wandb table
                for seq in generated_seqs:
                    # Add to the wandb table
                    master_table.add_data(i, seq)
                    print(seq)

                # print to console
                print(f"HellaSwag results: {hs_score:.4f}")

                # Log to wandb
                run.log({"generated_sequences": master_table, "HellaSwag score": hs_score})

        # Save checkpoint
        if i >= 500 and i % 500 == 0:
            # Save a new checkpoint only if cur_loss < last_loss
            if loss_accum < last_checkpoint_loss:
                if master_rank:
                    print("Saving a checkpoint...")
                # Create a folder
                folder_name = temp_path.split("/")[0]
                os.makedirs(folder_name, exist_ok=True)
                save_checkpoint(model, optimizer, i, temp_path, rank)
                # Update last checkpoint loss
                last_checkpoint_loss = loss_accum
                if master_rank:
                    print("Checkpoint was saved.")

        # If about to finish training, delete the mid training checkpoint
        # And save the full training checkpoint
        if i == train_steps - 1:
            if loss_accum < last_checkpoint_loss:
                # Delete the mid training checkpoint
                if master_rank:
                    # Create a folder
                    folder_name = final_path.split("/")[0]
                    os.makedirs(folder_name, exist_ok=True)
                    # Create a final checkpoint
                    save_checkpoint(model, optimizer, i, final_path, rank)
                    print("Saved final checkpoint!")
        # Next training step
        i += 1
    # Close progress bar, since the training is finished
    if master_rank:
        pbar.close()

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

    # Load the data
    train_set_loader = DataLoader("fineweb_train.bin", args.batch_size, args.context_length)
    val_set_loader = DataLoader("fineweb_test.bin", args.batch_size, args.context_length)

    # Start training
    train(
        train_set_loader,
        val_set_loader,
        args.batch_size,
        args.grad_accum_steps,
        args.context_length,
        args.num_layers,
        args.d_model,
        args.num_heads,
        args.d_ff,
        args.theta,
        args.train_steps,
        args.lr,
        (args.beta1, args.beta2),
        args.eps,
        args.weight_decay,
        device,
        local_rank,
        my_auto_wrap_policy,
        mp_policy,
    )

    dist.barrier()
    cleanup()

if __name__ == "__main__":
    # 2: Define the search space
    sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}
    
    # 3: Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="gpt-2.5")

    wandb.agent(sweep_id, function=main, count=10)



