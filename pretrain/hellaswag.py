import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

from typing import Any
import torch.nn as nn
from .model import softmax


class HellaSwagLoader:
    def __init__(self, B: int, T: int, tokenizer: Any):
        """
        Initializes the HellaSwag evaluation loader.
        B: Batch size
        T: Maximum sequence length
        tokenizer: Tokenizer instance with encode_batch and _special_tokens
        """
        self.B = B
        self.T = T
        self.dataset = load_dataset("Rowan/hellaswag", split="validation")
        self.tokenizer = tokenizer
        self.cur_shard_pos = 0
        self.n_examples = int(self.dataset.num_rows)
        self.eos_token = int(self.tokenizer._special_tokens["<|endoftext|>"])

    def _create_batch(
        self, start_idx: int, end_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = self.dataset[start_idx:end_idx]
        context = batch["ctx"]
        endings = batch["endings"]
        labels = batch["label"]
        batch_size = len(context)

        # Tokenize the context and endings
        context_ids = self.tokenizer.encode_batch(context)
        endings_ids = [self.tokenizer.encode_batch(example_endings) for example_endings in endings]

        # Populate contexts with endings
        all_inputs = []
        all_masks = []

        for i in range(batch_size):
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
        labels = torch.tensor([int(label) for label in labels], dtype=torch.long)

        return inputs_padded, labels, completion_mask

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the next batch of padded inputs, ground truth labels, and completion masks.
        Returns:
            inputs_padded: (B * 4, T)
            labels: (B,)
            completion_mask: (B * 4, T)
        """
        if self.cur_shard_pos >= self.n_examples:
            self.cur_shard_pos = 0

        end_idx = min(self.cur_shard_pos + self.B, self.n_examples)
        batch = self._create_batch(self.cur_shard_pos, end_idx)

        # Advance pointer
        self.cur_shard_pos = 0 if end_idx >= self.n_examples else end_idx

        return batch

    def iter_batches(self):
        for start_idx in range(0, self.n_examples, self.B):
            end_idx = min(start_idx + self.B, self.n_examples)
            yield self._create_batch(start_idx, end_idx)


@torch.inference_mode()
def compute_hellaswag_stats(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    completion_mask: torch.Tensor,
    device: torch.device,
) -> tuple[int, int]:
    """
    inputs: (B_flat, T) where B_flat = num_examples * 4
    labels: (num_examples) containing indices [0, 3, 1, ...]
    completion_mask: (B_flat, T)
                     1 for Completion tokens
                     0 for Context tokens AND Padding tokens
    """
    labels = labels.to(device)
    inputs = inputs.to(device)
    completion_mask = completion_mask.to(device)

    # 1. Forward Pass
    logits, _ = model(inputs)

    # 2. Align Logits (Predict Next Token)
    # logits: [A, B, C] -> Predicts [B, C, D]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs[..., 1:].contiguous()
    shift_mask = completion_mask[..., 1:].contiguous()

    # 3. Calculate Log Probs
    log_probs = softmax(shift_logits, dim=-1, is_log=True)

    # gather expects index to have same dim as input, so unsqueeze -1
    target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    # 4. Apply Masking (The Safe Way)
    masked_log_probs = target_log_probs * shift_mask

    # 5. Sum and Normalize
    # sum(dim=-1) sums across the sequence length
    sum_log_probs = masked_log_probs.sum(dim=-1)

    # Count actual tokens in the completion to normalize
    num_completion_tokens = shift_mask.sum(dim=-1).clamp(min=1e-9)

    # Average Log Probability (Higher is better, closer to 0)
    scores = sum_log_probs / num_completion_tokens

    # 6. Reshape and Compare
    # We assume B_flat is num_examples * 4
    num_examples = labels.size(0)
    scores = scores.view(num_examples, -1)  # Reshape to [Batch, Options]

    predictions = torch.argmax(scores, dim=-1)  # Pick index of highest prob

    correct = int((predictions == labels).sum().item())
    total = int(labels.numel())
    return correct, total


@torch.inference_mode()
def compute_hellaswag(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    completion_mask: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    correct, total = compute_hellaswag_stats(model, inputs, labels, completion_mask, device)
    model.train()
    return correct / total if total else 0.0


@torch.inference_mode()
def evaluate_hellaswag(
    model: nn.Module,
    loader: HellaSwagLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_correct = 0
    total_examples = 0

    for inputs, labels, completion_mask in loader.iter_batches():
        correct, batch_examples = compute_hellaswag_stats(
            model, inputs, labels, completion_mask, device
        )
        total_correct += correct
        total_examples += batch_examples

    model.train()
    return total_correct / total_examples if total_examples else 0.0
