import math
import os
import typing
from collections.abc import Callable
from typing import Optional

import numpy as np
import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class Linear(nn.Module):
    """
    Standard linear layer with Xavier initialization, but without bias (standard for some LLM architectures).
    Weights are truncated to stay within 3 standard deviations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        # Specify Xavier initialization: std = sqrt(2 / (fan_in + fan_out))
        mean, std = 0, np.sqrt(2 / (in_features + out_features))
        # Init the params from the normal distribution with said mean and std
        param = torch.normal(
            mean=mean, std=std, size=(out_features, in_features), dtype=dtype, device=device
        )
        # Truncate to avoid outlier weights that can destabilize training
        nn.init.trunc_normal_(param, a=-3 * std, b=3 * std)
        # Init the weight via the nn.Parameter
        self.weight = nn.Parameter(data=param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        # Returns: (..., out_features)
        return x @ self.weight.T


class Embedding(nn.Module):
    """
    Standard embedding layer initialized from a normal distribution.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        # Specify mean for the param initialization
        mean, std = 0, 1
        # Init the params from the normal distribution with said mean and std
        param = torch.normal(
            mean=mean, std=std, size=(num_embeddings, embedding_dim), dtype=dtype, device=device
        )
        # Truncate to stay within predictable bounds
        nn.init.trunc_normal_(param, a=-3, b=3)
        # Init the embedding via the nn.Parameter
        self.weight = nn.Parameter(data=param)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T)
        # Returns: (B, T, embedding_dim)
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Standardized in Llama and other modern transformer architectures for stability.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        # Init the learnable gain parameter (initialized to all ones)
        param = torch.tensor([1.0] * d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(data=param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        # Upcast the dtype to float32 to avoid overflowing/precision loss when squaring the input
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Calculate RMS = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(1 / self.d_model * reduce(x**2, "B T C -> B T 1", "sum") + self.eps)
        # Normalize: x / RMS * gain_parameter
        rmsnorm = x / rms * self.weight
        # Downcast back to the initial dtype (e.g., bfloat16 or float16)
        return rmsnorm.to(dtype=in_dtype)


def SiLU(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation function"""

    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function (SiLU(xW) * (xV))W2.
    Commonly used in the feed-forward network of modern transformers.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        # Init d_ff dimension (typically 8/3 * d_model in Llama)
        self.d_ff = (8 / 3) * d_model if not d_ff else d_ff
        assert self.d_ff % 64 == 0, "The dimensionality of the feedforward is not a multiple of 64"

        # Specify Xavier initialization for weights
        mean, std = 0, np.sqrt(2 / (d_model + d_ff))

        # w1 and w3 are for the gated part; w2 is for the projection back to d_model
        self.w1 = nn.Parameter(
            data=torch.normal(mean, std, (d_ff, d_model), device=device, dtype=dtype)
        )
        self.w3 = nn.Parameter(
            data=torch.normal(mean, std, (d_ff, d_model), device=device, dtype=dtype)
        )
        self.w2 = nn.Parameter(
            data=torch.normal(mean, std, (d_model, d_ff), device=device, dtype=dtype)
        )

        # Truncate weights for stability
        nn.init.trunc_normal_(self.w1, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.w3, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.w2, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        # result = (SiLU(x @ w1^T) * (x @ w3^T)) @ w2^T
        result = (SiLU(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T
        return result


class RoPE(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    Rotates pairs of dimensions in the embedding space to encode relative positional information.
    Ref: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even (pairs of dimensions).")
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.d_half = d_k // 2
        self.max_seq_len = int(max_seq_len)
        self.device = device if device is not None else torch.device("cpu")

        # Compute frequencies: freq_i = theta^(-2i/d)
        j = torch.arange(self.d_half, dtype=torch.float32, device=self.device)
        inv_freq = self.theta ** (-2.0 * j / float(self.d_k))  # (d_half,)

        # Compute angles for each position: angle = pos * freq
        pos = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.device).unsqueeze(
            1
        )  # (max_seq_len, 1)
        angles = pos * inv_freq.unsqueeze(0)  # (max_seq_len, d_half)

        # Precompute cos and sin buffers for the entire context window
        self.register_buffer("cos", torch.cos(angles))  # (max_seq_len, d_half)
        self.register_buffer("sin", torch.sin(angles))  # (max_seq_len, d_half)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Applies rotation to the input tensor x based on token_positions.
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        """
        if x.shape[-1] != self.d_k:
            raise ValueError(f"Last dim of x must be d_k={self.d_k}, got {x.shape[-1]}.")
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError(
                "token_positions must have same seq_len in its last dim as x's sequence dimension."
            )

        # Retrieve precomputed cos/sin for the specific positions
        token_positions = token_positions.long().to(self.cos.device)
        cos_pos = self.cos[token_positions].to(self.device)  # (..., seq_len, d_half)
        sin_pos = self.sin[token_positions].to(self.device)  # (..., seq_len, d_half)

        # Split x into even and odd indices for rotation
        x_even = x[..., 0::2]  # (..., seq_len, d_half)
        x_odd = x[..., 1::2]  # (..., seq_len, d_half)

        # Apply rotation matrix: [cos -sin; sin cos] * [even; odd]
        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd = x_even * sin_pos + x_odd * cos_pos

        # Recombine into the original d_k shape
        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)  # (..., seq_len, d_half, 2)
        new_shape = list(x.shape[:-2]) + [x.shape[-2], self.d_k]
        x_rot = x_rot.view(*new_shape)
        return x_rot


def softmax(
    x: torch.Tensor, dim: int, is_log: bool = False, temp: float = 1.0
) -> torch.Tensor:
    """
    Numerically stable Softmax or Log-Softmax implementation.
    Uses the LogSumExp trick to prevent overflow.
    """
    # Scale x by temperature (T=1 is standard, higher T = smoother distribution)
    x_scaled = x / temp

    # LogSumExp trick: log(sum(exp(x_i))) = m + log(sum(exp(x_i - m))) where m = max(x)
    m = torch.max(x_scaled, dim=dim, keepdim=True).values
    log_sum_exp = m + torch.log(torch.sum(torch.exp(x_scaled - m), dim=dim, keepdim=True))

    # log_softmax(x) = x - log_sum_exp(x)
    log_probs = x_scaled - log_sum_exp

    if is_log:
        return log_probs
    else:
        return torch.exp(log_probs)


def scaled_dot_prod_attn(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    d_k = Q.shape[-1]

    # Calculate pre softmax
    logits = torch.einsum("b...qd,b...kd->b...qk", Q, K) / math.sqrt(d_k)

    # Apply the mask if exists
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))

    # Compute probs
    scores = softmax(logits, dim=-1)
    return torch.einsum("b...qk,b...kd->b...qd", scores, V)


class MultiheadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) layer with optional RoPE positional encoding.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: Optional[float] = None,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        assert d_model % num_heads == 0, "num heads should be a power of 2"

        self.d_k = self.d_v = d_model // num_heads
        self.num_heads = num_heads

        # Projections for Q, K, V and Output
        self.Wq = Linear(d_model, d_model, device=device)
        self.Wk = Linear(d_model, d_model, device=device)
        self.Wv = Linear(d_model, d_model, device=device)
        self.Wo = Linear(d_model, d_model, device=device)

        # Optional Rotary Positional Embedding
        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # x: (B, T, d_model)
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Split Q, K, V into heads: (B, T, d_model) -> (B, num_heads, T, d_k)
        Q = rearrange(Q, "b t (h d) -> b h t d", h=self.num_heads)
        K = rearrange(K, "b t (h d) -> b h t d", h=self.num_heads)
        V = rearrange(V, "b t (h d) -> b h t d", h=self.num_heads)

        seq_len = x.shape[-2]

        # Apply RoPE if initialized
        if self.rope:
            # Rotate Q and K
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Compute Scaled Dot Product Attention with causal mask
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # Concatenate heads: (B, num_heads, T, d_k) -> (B, T, d_model)
        out = rearrange(out, "b h t d -> b t (h d)")
        # Final linear projection
        return self.Wo(out)


class TransformerBlock(nn.Module):
    """
    Single Transformer Block consisting of Pre-Norm, MHSA, and Pre-Norm, SwiGLU FFN.
    Uses residual connections.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: Optional[float] = None,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.norm_att = RMSNorm(d_model, device=device)
        self.norm_ff = RMSNorm(d_model, device=device)
        self.mhsa = MultiheadSelfAttention(d_model, num_heads, theta, max_seq_len, device)
        self.ffn = SwiGLU(d_model, d_ff, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm + Self-Attention + Residual connection
        attn = x + self.mhsa(self.norm_att(x))
        # Pre-Norm + Feed-Forward + Residual connection
        ffwd = attn + self.ffn(self.norm_ff(attn))
        return ffwd


class TransformerLM(nn.Module):
    """
    Standard Decoder-Only Transformer Language Model.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.device = device

        self.emb = Embedding(vocab_size, d_model, device=device)
        self.tblocks = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device=device)
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(d_model, device=device)
        self.linear = Linear(d_model, vocab_size, device=device)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Input: x (token IDs) shape (B, T)
        Targets: Optional ground truth token IDs for loss calculation.
        Returns: (logits, loss)
        """
        # (B, T) -> (B, T, d_model)
        emb = self.emb(x)

        # Pass embedding through N transformer blocks
        for tblock in self.tblocks:
            emb = tblock(emb)

        # Final RMS Normalization
        normed = self.norm(emb)

        # Pass through linear head to get logits: (B, T, d_model) -> (B, T, vocab_size)
        logits = self.linear(normed)

        loss = None
        if targets is not None:
            # Calculate cross entropy loss if targets are provided
            loss = cross_entropy_loss(logits, targets)

        return logits, loss


def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Standard Cross Entropy Loss using numerically stable Log-Softmax.
    inputs: (B, T, C) - Logits
    targets: (B, T) - Ground truth indices
    """
    B, T, C = inputs.shape

    # Use stable log-probs: log_probs[b, t, c]
    log_probs = softmax(inputs, dim=-1, is_log=True)

    # Gather the log-probabilities assigned to the target classes
    b = torch.arange(B, device=inputs.device).unsqueeze(1)
    t = torch.arange(T, device=inputs.device).unsqueeze(0)

    log_probs_correct = log_probs[b, t, targets]

    # Negative Log Likelihood (NLL) Mean
    loss = -torch.mean(log_probs_correct)
    return loss


class AdamW(torch.optim.Optimizer):
    """
    AdamW Optimizer (Adam with Decoupled Weight Decay).
    Implements the standard Adam update but applies weight decay directly to the parameters.
    """

    def __init__(
        self,
        params: typing.Iterable[nn.Parameter],
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "epsilon": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """Standard optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["epsilon"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Initialize optimizer state (time t, momentum m, variance v)
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                t = state.get("t") + 1
                state["t"] = t

                grad = p.grad.data
                m = state["m"]
                v = state["v"]

                # Update biased first moment estimate: m = beta1*m + (1-beta1)*grad
                m = m * beta1 + (1 - beta1) * grad
                # Update biased second raw moment estimate: v = beta2*v + (1-beta2)*grad^2
                v = v * beta2 + (1 - beta2) * grad**2

                state["m"] = m
                state["v"] = v

                # Correct biases for m and v:
                # alpha_t = lr * sqrt(1-beta2^t) / (1-beta1^t)
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # Compute denominator (epsilon for numerical stability)
                denom = v.sqrt() + eps

                # Apply Adam parameter update
                p.data -= alpha_t * m / denom

                # Apply Weight Decay (Decoupled from the gradient update)
                if weight_decay != 0:
                    p.data -= lr * weight_decay * p.data

        return loss


def learning_rate_schedule(
    t: int, a_max: float, a_min: float, T_w: int, T_c: int
) -> float:
    """
    Cosine Learning Rate Schedule with Warmup.
    t: current iteration
    a_max: peak learning rate
    a_min: final learning rate
    T_w: warmup steps
    T_c: total steps for annealing
    """
    # 1. Linear Warmup: Increase LR linearly from 0 to a_max
    if t < T_w:
        a_t = t / T_w * a_max
    # 2. Cosine Annealing: Decay LR from a_max down to a_min
    elif T_w <= t <= T_c:
        a_t = a_min + 1 / 2 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (a_max - a_min)
    # 3. Post Annealing: Keep LR at a_min
    else:
        a_t = a_min
    return a_t


def gradient_clipping(
    params: typing.Iterable[nn.Parameter], max_l2_norm: float
) -> None:
    """
    Global Gradient Clipping to prevent exploding gradients.
    Rescales gradients if their total L2 norm exceeds max_l2_norm.
    """
    eps = 1e-6

    # 1. Compute total global L2 norm across all parameters
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += torch.sum(p.grad**2)

    total_norm = torch.sqrt(total_sq)

    # 2. If under threshold, no scaling needed
    if total_norm < max_l2_norm:
        return

    # 3. Compute scale factor (scale = threshold / current_norm)
    scale = max_l2_norm / (total_norm + eps)

    # 4. Apply scaling factor to every gradient tensor in-place
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale)


def to_cpu(obj: typing.Any) -> typing.Any:
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    else:
        return obj


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    rank: int,
) -> None:
    """
    Save a distributed checkpoint for an FSDP-wrapped model.
    Only rank 0 performs the actual I/O, but all ranks participate in state-dict gathering.
    """

    # FSDP policy: Offload the full state dict to CPU to avoid OOM on the master rank
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        # get_state_dict retrieves the unsharded state from all ranks
        model_state_dict, optim_state_dict = get_state_dict(model, optimizer)

    if rank == 0:
        print("Saving a checkpoint...")
        # Package states into a single dictionary
        checkpoint = {
            "model_state": model_state_dict,
            "optimizer_state": optim_state_dict,
            "iteration_state": iteration,
        }
        # Save to disk
        torch.save(checkpoint, out)
        print("Saved a mid-training checkpoint!")


def load_checkpoint(
    checkpoint_path: str | os.PathLike,
    fsdp_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rank: int,
) -> int:
    """
    Load an FSDP-distributed checkpoint.
    Broadcasts state from rank 0 to all other shards.
    """
    load_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, load_cfg):
        checkpoint = None
        if rank == 0:
            # Rank 0 loads the file into memory
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Broadcast iteration counter to all ranks so they stay in sync
        iteration_val = checkpoint["iteration_state"] if rank == 0 else 0
        iteration = torch.tensor(iteration_val).to(rank)
        dist.broadcast(iteration, src=0)

        # Retrieve state dicts (rank 0 has the data, others have None initially)
        model_state = checkpoint["model_state"] if rank == 0 else None
        optim_state = checkpoint["optimizer_state"] if rank == 0 else None

        # set_state_dict handles the communication to load the full state into shards
        set_state_dict(
            fsdp_model,
            optimizer,
            model_state_dict=model_state,
            optim_state_dict=optim_state,
        )

    return iteration.item()


def generate(
    prompt: str,
    max_tokens: int,
    context_length: int,
    batch_size: int,
    model: nn.Module,
    temp: float,
    top_p: float,
    device: torch.device,
) -> list[str]:
    """
    Main generation loop for the LLM.
    prompt: starting text
    max_tokens: number of tokens to generate per sequence
    context_length: maximum window size the model can handle
    batch_size: number of sequences to generate
    model: the transformer model
    temp: softmax temperature
    top_p: nucleus sampling threshold
    """
    enc = tiktoken.get_encoding("gpt2")
    sentences = []

    for i in range(batch_size):
        # Encode prompt and move to device
        inputs = torch.tensor(enc.encode(prompt), device=device).unsqueeze(0)
        for _ in range(max_tokens):
            # Generate next token logits using autocast for speed/memory efficiency
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(inputs)
            # Apply temperature and top-p sampling on the last token's logits
            probs = softmax(logits[:, -1, :], dim=-1, temp=temp)
            next_token = top_p_sampling(probs, p=top_p)
            # Append token to sequence
            inputs = torch.cat((inputs, next_token), dim=1)
            # Stop if the end-of-text special token is generated
            if enc.decode([next_token.item()]) == "<|endoftext|>":
                break
            # Truncate sequence if it exceeds the model's maximum context length
            if inputs.shape[-1] > context_length:
                inputs = inputs[:, -context_length:]

        # Record the final generated sequence
        sentences.append(
            f"\nGenerated sequence №{i + 1}:\n" + enc.decode(inputs[0].tolist()) + "\n"
        )
    return sentences


def top_p_sampling(probs: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    probs: [Batch Size, Vocab Size] - The raw probabilities (already softmaxed)
    p: float - The cumulative probability threshold (e.g., 0.9)
    """

    # 1. Sort probabilities in descending order
    # sorted_probs: [Batch, Vocab], sorted_indices: [Batch, Vocab]
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # 2. Compute cumulative sum
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 3. Remove tokens with cumulative probability above the threshold (p)
    # We want to KEEP the first token that crosses the threshold, so we shift the mask right by 1.
    # Logic: If cumsum[i] > p, then token[i] is usually excluded.
    # But we want the set to sum to AT LEAST p.
    sorted_indices_to_remove = cumulative_probs > p

    # Shift the mask to the right to ensure we keep the first token that crossed the threshold
    # ...[..., 1:] removes the last column, zero-padding at the start shifts it right.
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0  # Always keep the most probable token

    # 4. Zero out the probabilities of removed tokens
    # We clone to avoid modifying gradients if this were part of a training loop
    nucleus_probs = sorted_probs.clone()
    nucleus_probs[sorted_indices_to_remove] = 0.0

    # 5. Renormalize the remaining probabilities
    # (Optional but good practice: ensure they sum to 1 exactly)
    nucleus_probs = nucleus_probs / nucleus_probs.sum(dim=-1, keepdim=True)

    # 6. Sample from the modified distribution
    # sampled_sorted_index: [Batch, 1] (Indices relative to the SORTED array)
    sampled_sorted_index = torch.multinomial(nucleus_probs, 1)

    # 7. Gather the original indices
    # We map the index from "sorted space" back to "vocabulary space"
    original_indices = torch.gather(sorted_indices, -1, sampled_sorted_index)

    return original_indices


class DataLoader:
    """
    Efficient DataLoader that uses np.memmap to stream tokens directly from disk.
    Avoids loading the entire dataset into RAM.
    """

    def __init__(self, filename: str | os.PathLike, B: int, T: int) -> None:
        self.B = B
        self.T = T
        # Memory-map the binary file for efficient reading
        self.dataset = np.memmap(filename, dtype=np.uint16, mode="r")
        self.cur_shard_pos = 0
        self.n_tokens = len(self.dataset)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the next (Inputs, Targets) batch of tokens."""
        B, T = self.B, self.T

        # Slice the dataset. Targets are shifted by 1 relative to inputs.
        buf = self.dataset[self.cur_shard_pos : self.cur_shard_pos + B * T + 1]

        # Convert to torch tensor (uint16 -> int64)
        buf_torch = torch.from_numpy(buf.astype(np.int64))

        # Inputs include everything except the last token
        x = buf_torch[:-1].view(B, T)
        # Targets include everything except the first token
        y = buf_torch[1:].view(B, T)

        # Advance pointer for the next call
        self.cur_shard_pos += B * T

        # Loop back to the start if we reach the end of the dataset
        if self.cur_shard_pos + (B * T + 1) > self.n_tokens:
            self.cur_shard_pos = 0

        return x, y
