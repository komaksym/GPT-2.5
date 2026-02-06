import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

from model.model import softmax


class HellaSwagLoader:
    def __init__(self, B, T, tokenizer):
        self.B = B
        self.T = T
        self.dataset = load_dataset("Rowan/hellaswag", split="validation")
        self.tokenizer = tokenizer
        self.cur_shard_pos = 0
        self.n_examples = self.dataset.num_rows
        self.eos_token = self.tokenizer._special_tokens["<|endoftext|>"]

    def next_batch(self):
        # Pluck out the context batch from the dataset
        context = self.dataset[self.cur_shard_pos : self.cur_shard_pos + self.B][
            "ctx"
        ]  # list (5, arbitrary)
        endings = self.dataset[self.cur_shard_pos : self.cur_shard_pos + self.B][
            "endings"
        ]  # list (5, 4, arbitrary)
        labels = self.dataset[self.cur_shard_pos : self.cur_shard_pos + self.B][
            "label"
        ]  # list (5, 1)

        # Tokenizer the context and endings
        context_ids = self.tokenizer.encode_batch(context)  # list (5, arbitrary)
        endings_ids = [self.tokenizer.encode_batch(e) for e in endings]  # list (5, 4, arbitrary)

        # Populate contexts with endings
        all_inputs = []
        all_masks = []

        for i in range(self.B):
            for j in range(len(endings_ids[i])):
                # Join and truncate
                full_ids = context_ids[i] + endings_ids[i][j]
                if len(full_ids) > self.T:
                    full_ids = full_ids[-self.T :]

                # Identify the completion tokens
                completion_len = len(endings_ids[i][j])

                # Create a mask of 0s, then fill the end with 1s
                mask = torch.zeros(len(full_ids), dtype=torch.int)
                # Ensure we don't exceed the actual truncated length
                mask_start = max(0, len(full_ids) - completion_len)
                mask[mask_start:] = 1

                all_inputs.append(torch.tensor(full_ids))
                all_masks.append(mask)

        # Synchronized padding
        inputs_padded = pad_sequence(all_inputs, batch_first=True, padding_value=self.eos_token)
        completion_mask = pad_sequence(all_masks, batch_first=True, padding_value=0)

        # Convert labels from string to ints
        labels = torch.tensor([int(label) for label in labels])

        # Advance pointer
        self.cur_shard_pos += self.B

        # Reset if we hit the end
        if self.cur_shard_pos + self.B > self.n_examples:
            self.cur_shard_pos = 0

        return inputs_padded, labels, completion_mask


def compute_hellaswag(model, inputs, labels, completion_mask):
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
        shift_mask = completion_mask[..., 1:].contiguous()  # Align mask too!

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
        scores = scores.view(num_examples, -1)  # Reshape to [Batch, Options]

        predictions = torch.argmax(scores, dim=-1)  # Pick index of highest prob

        # Calculate accuracy
        acc = (predictions == labels).float().mean().item()

        model.train()
        return acc
