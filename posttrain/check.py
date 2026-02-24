from pretrain.model import load_checkpoint, TransformerLM, AdamW
from pretrain.train import VOCAB_SIZE
import torch

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

    model = TransformerLM(
        VOCAB_SIZE, context_length, num_layers, d_model, num_heads, d_ff, theta, device=device
    )
    optimizer = AdamW(model.parameters(), a_max, betas, eps, weight_decay)
    rank = 0
    checkpoint_path = "checkpoint/final_checkpoint.pt"

    it = load_checkpoint(checkpoint_path, model, optimizer, rank)
    breakpoint()