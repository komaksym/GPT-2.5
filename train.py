from model.model import *
import argparse
import numpy as np
import torch
import os
import wandb
import tiktoken
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from deepeval.benchmarks.tasks import HellaSwagTask
import functools
import warnings
import tqdm
from hellaswag import MyGPT
from deepeval.benchmarks import HellaSwag, LAMBADA


warnings.filterwarnings("ignore")

temp_path = "checkpoints/mid_training_checkpoint.pt"
final_path = "checkpoints/final_checkpoint.pt"

VOCAB_SIZE = 50257
TRAINING_SET_DATA_CREATION_BATCH_SIZE = 1000000
VAL_SET_DATA_CREATION_BATCH_SIZE = 100000

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
    master_rank = True if rank == 0 else False

    with torch.no_grad():
        # Run validation
        val_inputs, val_labels = dataset_loader.next_batch()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, val_loss = model(val_inputs, val_labels)

        if master_rank:
            print(f"step {iteration+1}, val loss: {val_loss.item()}")
            # Log loss in wandb
            run.log({"val_loss": val_loss.item()})

        # Run generation
        generated_sqs = generate("Once upon a time,", max_tokens=50, context_length=context_length, 
                                batch_size=5, model=model, temp=0.8, top_p=0.9, device=device)

    model.train()
    return generated_sqs


def training_together(train_set_loader, val_set_loader, batch_size, grad_accum_steps, context_length, num_layers, 
                      d_model, num_heads, d_ff, theta, train_steps, a_max, betas, eps, weight_decay,
                      device, rank, autowrap_policy, mp_policy, checkpoint=None):

    model = TransformerLM(VOCAB_SIZE, context_length, num_layers,
                          d_model, num_heads, d_ff, theta, device=device)
    
    # Torch compile the model
    model.compile()

    # Wandb init
    run = None # For global scope
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
        config.training_sampled_with_replacement = False if 'train_set_loader' and \
                                                        'val_set_loader' in locals() else True
        run.watch(model)
        master_table = wandb.Table(columns=["step", "prediction"], log_mode="INCREMENTAL")


    # If in distributed mode
    if rank is not None and autowrap_policy and mp_policy:
        model = FSDP(model.to(rank), 
                    auto_wrap_policy=autowrap_policy,
                    mixed_precision=mp_policy,
                    device_id=torch.cuda.current_device(),
                    sync_module_states=True)
                 
    # HellaSwag
    my_model = MyGPT(model=model, tokenizer=tiktoken.get_encoding("gpt2"), device=device)
    hellaswag_benchmark = HellaSwag(tasks=[HellaSwagTask.TRIMMING_BRANCHES_OR_HEDGES, HellaSwagTask.BATON_TWIRLING])

    # Warch model with wandb
    optimizer = AdamW(model.parameters(), a_max, betas, eps, weight_decay)
    last_checkpoint_loss = float('inf')
    i = 0

    # Events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # Check if checkpoint exists
    if master_rank:
        if checkpoint is not None:
            # If it does, load it and keep training from the checkpoint
            i = load_checkpoint(checkpoint, model, optimizer, rank)
            print("Continuing training from checkpoint!")
        else:
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
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            print(f"step {i+1}, loss: {loss_accum:.3f}, perp: {perplexity:.3f}, norm: {norm:.3f}, dt: {step_time_ms:.3f}, tok/s: {tokens_per_sec:.3f}")
            # Log loss in wandb
            run.log({"loss": loss_accum, "perplexity": perplexity, "norm": norm, "lr": lr})
            # Increment pbar
            pbar.update(1)

        # Run evaluation
        if i % 10 == 0:
            # Wandb table for tracking generated sequences
            generated_seqs = run_evaluation(val_set_loader, model, context_length,
                           device, run, rank, i)
            
            # Run HellaSwag
            hellaswag_results = hellaswag_benchmark.evaluate(my_model, batch_size=5)
            if master_rank: 
                # Populate the wandb table
                for idx, seq in enumerate(generated_seqs):
                    # Add to the wandb table
                    master_table.add_data(i, seq)
                    print(seq)

                # print to console
                print(f"HellaSwag results: {hellaswag_results}")

                # Log to wandb
                run.log({"HellaSwag score": hellaswag_results})
                run.log({"generated_sequences": master_table}) 

        # Save checkpoint
        elif i >= 500 and i % 500 == 0:
            # Save a new checkpoint only if cur_loss < last_loss
            if loss_accum < last_checkpoint_loss:
                if master_rank:
                    print("Saving a checkpoint...")
                # Create a folder
                folder_name = temp_path.split("/")[0]
                os.makedirs(folder_name, exist_ok=True)
                save_checkpoint(model, optimizer, i, temp_path, rank, loss_accum)
                if master_rank:
                    print("Checkpoint was saved.")

        # If about to finish training, delete the mid training checkpoint
        # And save the full training checkpoint
        elif i == train_steps - 1:
            if loss_accum < last_checkpoint_loss:
                # Delete the mid training checkpoint
                if master_rank:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        print("Removed mid-training checkpoint!")
                    # Create a final checkpoint
                    save_checkpoint(model, optimizer, i, final_path, rank, loss_accum)
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
    train_set_loader = DataLoader("ts_v2_train.bin", args.batch_size, args.context_length)
    val_set_loader = DataLoader("ts_v2_valid.bin", args.batch_size, args.context_length)

    # Start training
    training_together(train_set_loader, val_set_loader, args.batch_size, args.grad_accum_steps, args.context_length,
                      args.num_layers, args.d_model, args.num_heads, args.d_ff, 
                      args.theta, args.train_steps, args.lr, (args.beta1, args.beta2),
                      args.eps, args.weight_decay, device, local_rank, my_auto_wrap_policy, mp_policy)
    
    dist.barrier()
    cleanup()


if __name__ == "__main__":
    main()


"""
To run the script:
WANDB_MODE=disabled uv run torchrun --nproc_per_node 1 train.py --batch_size 1 --grad_accum_steps 1 --context_length 1024 --num_layers 12 --d_model 768 --num_heads 12 --d_ff 2048 --theta 10000 --train_steps 20000 --lr 6e-4 --beta1 0.9 --beta2 0.95 --eps 1e-8 --weight_decay 0.1

"""