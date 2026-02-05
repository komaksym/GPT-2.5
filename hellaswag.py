from datasets import load_dataset

import torch
from model.model import TransformerLM, softmax
from train import VOCAB_SIZE
import numpy as np


def calculate_score(model, inputs, labels, completion_mask):
    """
    inputs: (B_flat, T) where B_flat = num_examples * 4
    labels: (num_examples) containing indices [0, 3, 1, ...]
    completion_mask: (B_flat, T) 
                     1 for Completion tokens
                     0 for Context tokens AND Padding tokens
    """
    model.eval()
    with torch.no_grad():
        # 1. Forward Pass
        logits, _ = model(inputs)
        
        # 2. Align Logits (Predict Next Token)
        # logits: [A, B, C] -> Predicts [B, C, D]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs[..., 1:].contiguous()
        shift_mask = completion_mask[..., 1:].contiguous() # Align mask too!
        
        # 3. Calculate Log Probs
        # Use Gather to pick the exact log-prob of the target token
        log_probs = softmax(shift_logits, dim=-1, is_log=True)
        
        # gather expects index to have same dim as input, so unsqueeze -1
        target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # 4. Apply Masking (The Safe Way)
        # We want to sum ONLY the completion tokens.
        # We zero out the log-probs of Context and Padding.
        # (Mathematically, adding 0.0 in log-space is "adding nothing to the sum")
        masked_log_probs = target_log_probs * shift_mask
        
        # 5. Sum and Normalize
        # sum(dim=-1) sums across the sequence length
        sum_log_probs = masked_log_probs.sum(dim=-1)
        
        # Count actual tokens in the completion to normalize
        # Avoid division by zero with clamp
        num_completion_tokens = shift_mask.sum(dim=-1).clamp(min=1e-9)
        
        # Average Log Probability (Higher is better, closer to 0)
        scores = sum_log_probs / num_completion_tokens
        
        # 6. Reshape and Compare
        # We assume B_flat is num_examples * 4
        num_examples = labels.size(0)
        scores = scores.view(num_examples, -1) # Reshape to [Batch, Options]
        
        predictions = torch.argmax(scores, dim=-1) # Pick index of highest prob
        
        # Calculate accuracy
        acc = (predictions == labels).float().mean().item()
        
        model.train()
        return acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerLM(
        VOCAB_SIZE,
        context_length=1024,
        num_layers=12,
        d_model=768,
        num_heads=12,
        d_ff=2048,
        theta=10000,
        device=device
    )

    dl = DataLoader(5, 1024)
    #inputs = torch.randint(high=10, size=(6, 7))
    #mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0],
                         #[0, 0, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0],
                         #[0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 0]])
    #labels = torch.tensor([2, 0])
    #print(calculate_score(model, inputs, labels, mask))


