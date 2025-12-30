from model.model import *
import argparse
import numpy as np
import torch
import os
import wandb

temp_path = "checkpoints/mid_training_checkpoint.pt"
final_path = "checkpoints/final_checkpoint.pt"


def training_together(train_set, val_set, batch_size, vocab_size, context_length, num_layers, 
                      d_model, num_heads, d_ff, theta, train_steps, 
                      lr, betas, eps, weight_decay, device):


    # Wandb init
    run = wandb.init(project="gpt-2.5")
    config = run.config

    model = TransformerLM(vocab_size, context_length, num_layers,
                          d_model, num_heads, d_ff, theta, device=device)
    # Warch model with wandb
    run.watch(model)
    optimizer = AdamW(model.parameters(), lr, betas, eps, weight_decay)
    i = 0

    # Check if checkpoint exists
    if os.path.exists(temp_path):
        # If it does, load it and keep training from the checkpoint
        i = load_checkpoint(temp_path, model, optimizer)
        print("Continuing training from checkpoint!")
    else:
        print("Training from scratch!")
    
    while i < train_steps:
        inputs, labels = sample_data(train_set, batch_size, device)
        # Zero grads
        optimizer.zero_grad()
        # Predictions
        logits, loss = model(inputs, labels)
        # Compute gradients
        loss.backward()
        # Step optimizer
        optimizer.step()
        print(f"step {i+1}, loss: {loss.item()}")
        # Log loss in wandb
        run.log({"loss": loss.item()})


        # Save checkpoint and run validation every x steps
        if i > 10 and i % 10 == 0:
            save_checkpoint(model, optimizer, i, temp_path)
            print("Saved a mid-training checkpoint!")

            val_inputs, val_labels = sample_data(val_set, batch_size, device)
            _, val_loss = model(val_inputs, val_labels)
            print(f"step {i+1}, val loss: {val_loss.item()}")
            # Log loss in wandb
            run.log({"val_loss": val_loss.item()})

        # If about to finish training, delete the mid training checkpoint
        # And save the full training checkpoint
        elif i == train_steps - 1:
            # Delete the mid training checkpoint
            if os.path.exists(temp_path):
                os.remove(temp_path)
                print("Removed mid-training checkpoint!")
            # Create a final checkpoint
            save_checkpoint(model, optimizer, i, final_path)
            print("Saved final checkpoint!")

        i += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--batch_size", type=int)
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

    # Load the data
    train_data = np.load("ts_train_set.npy", mmap_mode='r')
    train_set = data_loading(dataset=train_data, batch_size=100000, \
                        context_length=args.context_length, device=device)

    val_data = np.load("ts_valid_set.npy", mmap_mode='r')
    val_set = data_loading(dataset=val_data, batch_size=10000, \
                        context_length=args.context_length, device=device)

    # Start training
    training_together(train_set, val_set, args.batch_size, args.vocab_size, args.context_length,
                      args.num_layers, args.d_model, args.num_heads, args.d_ff, 
                      args.theta, args.train_steps, args.lr, (args.beta1, args.beta2),
                      args.eps, args.weight_decay, device)
    


if __name__ == "__main__":
    main()