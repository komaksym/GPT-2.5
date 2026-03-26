import math
import os
import typing
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import tiktoken
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

GPT_INIT_STD = 0.02


def is_distributed() -> bool:
    """Checks if the script is running in a distributed environment (e.g., via torchrun)."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


class Linear(nn.Linear):
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
        """Initialize a bias-free projection with GPT-style weight init."""
        super().__init__(
            in_features,
            out_features,
            bias=False,
            device=device,
            dtype=dtype,
        )
        nn.init.normal_(self.weight, mean=0.0, std=GPT_INIT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project the input across its last dimension."""
        # x: (..., in_features)
        # Returns: (..., out_features)
        return super().forward(x)


class Embedding(nn.Embedding):
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
        """Initialize token embeddings with GPT-style normal weights."""
        super().__init__(
            num_embeddings,
            embedding_dim,
            device=device,
            dtype=dtype,
        )
        nn.init.normal_(self.weight, mean=0.0, std=GPT_INIT_STD)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Look up embeddings for a batch of token ids."""
        # token_ids: (B, T)
        # Returns: (B, T, embedding_dim)
        return super().forward(token_ids)


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
        """Initialize RMSNorm with a learnable gain vector."""
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        # Init the learnable gain parameter (initialized to all ones)
        param = torch.tensor([1.0] * d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(data=param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize hidden states by their root-mean-square magnitude."""
        # x: (B, T, C)
        # Upcast the dtype to float32 to avoid overflowing/precision loss when squaring the input
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Calculate RMS = sqrt(mean(x^2) + eps)
        rms = torch.sqrt(
            1 / self.d_model * reduce(x**2, "B T C -> B T 1", "sum") + self.eps
        )
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
        """Initialize the gated feed-forward projection weights."""
        super().__init__()
        # Init d_ff dimension (typically 8/3 * d_model in Llama)
        self.d_ff = (8 / 3) * d_model if not d_ff else d_ff
        assert self.d_ff % 64 == 0, (
            "The dimensionality of the feedforward is not a multiple of 64"
        )
        # w1 and w3 are for the gated part; w2 is for the projection back to d_model
        self.w1 = nn.Parameter(
            data=torch.empty((d_ff, d_model), device=device, dtype=dtype)
        )
        self.w3 = nn.Parameter(
            data=torch.empty((d_ff, d_model), device=device, dtype=dtype)
        )
        self.w2 = nn.Parameter(
            data=torch.empty((d_model, d_ff), device=device, dtype=dtype)
        )

        nn.init.normal_(self.w1, mean=0.0, std=GPT_INIT_STD)
        nn.init.normal_(self.w3, mean=0.0, std=GPT_INIT_STD)
        nn.init.normal_(self.w2, mean=0.0, std=GPT_INIT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU feed-forward transformation."""
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
        """Precompute rotary frequencies for the configured context length."""
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
        pos = torch.arange(
            self.max_seq_len, dtype=torch.float32, device=self.device
        ).unsqueeze(1)  # (max_seq_len, 1)
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
            raise ValueError(
                f"Last dim of x must be d_k={self.d_k}, got {x.shape[-1]}."
            )
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError(
                "token_positions must have same seq_len in its last dim as x's sequence dimension."
            )

        # Retrieve precomputed cos/sin for the specific positions
        token_positions = token_positions.long().to(x.device)
        cos_pos = self.cos[token_positions].to(x.device)  # (..., seq_len, d_half)
        sin_pos = self.sin[token_positions].to(x.device)  # (..., seq_len, d_half)

        # Split x into even and odd indices for rotation
        x_even = x[..., 0::2]  # (..., seq_len, d_half)
        x_odd = x[..., 1::2]  # (..., seq_len, d_half)

        # Apply rotation matrix: [cos -sin; sin cos] * [even; odd]
        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd = x_even * sin_pos + x_odd * cos_pos

        # Recombine into the original d_k shape
        x_rot = torch.stack(
            [x_rot_even, x_rot_odd], dim=-1
        )  # (..., seq_len, d_half, 2)
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
    log_sum_exp = m + torch.log(
        torch.sum(torch.exp(x_scaled - m), dim=dim, keepdim=True)
    )

    # log_softmax(x) = x - log_sum_exp(x)
    log_probs = x_scaled - log_sum_exp

    if is_log:
        return log_probs
    else:
        return torch.exp(log_probs)


def scaled_dot_prod_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention with an optional boolean mask."""
    d_k = Q.shape[-1]

    # Calculate pre softmax
    logits = torch.einsum("b...qd,b...kd->b...qk", Q, K) / math.sqrt(d_k)

    # Apply the mask if exists
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))

    # Compute probs
    scores = softmax(logits, dim=-1)
    return torch.einsum("b...qk,b...kd->b...qd", scores, V)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback attention kernel compatible with HF attention hooks."""
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights.masked_fill(
            ~attention_mask, torch.finfo(attn_weights.dtype).min
        )

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def positions_are_packed(position_ids: torch.Tensor | None) -> bool:
    """Detect packed sequences by looking for non-monotonic position ids."""
    if position_ids is None or position_ids.size(-1) <= 1:
        return False
    return bool((position_ids[..., 1:] <= position_ids[..., :-1]).any())


def build_attention_mask(
    seq_len: int,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Build a causal mask that also respects packed sequence boundaries."""
    has_packed_boundaries = positions_are_packed(position_ids)
    if attention_mask is None and not has_packed_boundaries:
        return None

    causal_mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()
    combined_mask = causal_mask.unsqueeze(0)

    if has_packed_boundaries:
        # Packed samples reset position ids back to zero, so segment ids keep
        # attention constrained to tokens from the same packed sequence.
        segment_ids = (position_ids == 0).to(torch.int64).cumsum(dim=-1) - 1
        same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
        combined_mask = combined_mask & same_segment

    if attention_mask is not None:
        key_padding_mask = attention_mask[:, None, :].bool()
        combined_mask = combined_mask & key_padding_mask

    return combined_mask.unsqueeze(1)


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
        """Initialize self-attention projections and optional RoPE."""
        super().__init__()

        assert d_model % num_heads == 0, "num heads should be a power of 2"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.is_causal = True
        self.config = SimpleNamespace(_attn_implementation="sdpa")

        # Projections for Q, K, V and Output
        self.Wq = Linear(d_model, d_model, device=device)
        self.Wk = Linear(d_model, d_model, device=device)
        self.Wv = Linear(d_model, d_model, device=device)
        self.Wo = Linear(d_model, d_model, device=device)

        # Optional Rotary Positional Embedding
        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device)

    @property
    def attn_implementation(self) -> str:
        """Return the currently configured attention backend."""
        return self.config._attn_implementation

    def set_attn_implementation(self, attn_implementation: str) -> None:
        """Store the attention backend to use for runtime calls."""
        self.config._attn_implementation = attn_implementation.removeprefix("paged|")

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run self-attention over the input sequence."""
        # x: (B, T, d_model)
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # Time dim
        T = Q.shape[1]

        # Split Q, K, V into heads: (B, T, d_model) -> (B, num_heads, T, d_k)
        Q = rearrange(Q, "b t (h d) -> b h t d", h=self.num_heads)
        K = rearrange(K, "b t (h d) -> b h t d", h=self.num_heads)
        V = rearrange(V, "b t (h d) -> b h t d", h=self.num_heads)

        seq_len = x.shape[-2]

        # Apply RoPE if initialized
        if self.rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        attention_impl = self.attn_implementation
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attention_impl, eager_attention_forward
        )
        packed_mask = build_attention_mask(
            T,
            x.device,
            attention_mask=attention_mask,
            position_ids=token_positions,
        )

        if attention_impl == "flash_attention_2":
            out, _ = attention_interface(
                self,
                Q,
                K,
                V,
                attention_mask=attention_mask,
                scaling=self.d_k**-0.5,
                position_ids=token_positions,
            )
        else:
            out, _ = attention_interface(
                self,
                Q,
                K,
                V,
                attention_mask=packed_mask,
                scaling=self.d_k**-0.5,
            )

        # Concatenate heads: (B, T, num_heads, d_k) -> (B, T, d_model)
        out = rearrange(out, "b t h d -> b t (h d)")
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
        """Initialize one pre-norm attention-plus-FFN transformer block."""
        super().__init__()

        self.norm_att = RMSNorm(d_model, device=device)
        self.norm_ff = RMSNorm(d_model, device=device)
        self.mhsa = MultiheadSelfAttention(
            d_model, num_heads, theta, max_seq_len, device
        )
        self.ffn = SwiGLU(d_model, d_ff, device=device)

    def set_attn_implementation(self, attn_implementation: str) -> None:
        """Propagate the selected attention backend to the attention layer."""
        self.mhsa.set_attn_implementation(attn_implementation)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention, feed-forward layers, and residual connections."""
        # Pre-Norm + Self-Attention + Residual connection
        attn = x + self.mhsa(
            self.norm_att(x),
            token_positions=position_ids,
            attention_mask=attention_mask,
        )
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
        """Construct the decoder-only Transformer stack and tied LM head."""
        super().__init__()

        self.device = device

        self.emb = Embedding(vocab_size, d_model, device=device)
        self.tblocks = nn.ModuleList(
            TransformerBlock(
                d_model, num_heads, d_ff, theta, context_length, device=device
            )
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(d_model, device=device)
        self.linear = Linear(d_model, vocab_size, device=device)

        # Tie output head to token embeddings
        self.linear.weight = self.emb.weight

    def set_attn_implementation(self, attn_implementation: str) -> None:
        """Set the attention backend on every transformer block."""
        for tblock in self.tblocks:
            tblock.set_attn_implementation(attn_implementation)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Input: x (token IDs) shape (B, T)
        Targets: Optional ground truth token IDs for loss calculation.
        Returns: (logits, loss)
        """
        # (B, T) -> (B, T, d_model)
        emb = self.emb(x)

        # Pass embedding through N transformer blocks
        for tblock in self.tblocks:
            emb = tblock(emb, attention_mask=attention_mask, position_ids=position_ids)

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
        """Initialize optimizer state and hyperparameters for AdamW."""
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "epsilon": eps,
            "weight_decay": weight_decay,
        }
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
        a_t = a_min + 1 / 2 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (
            a_max - a_min
        )
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
    """Recursively move tensors nested in Python containers onto the CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    else:
        return obj


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None, iteration=None):
        """Capture model, optimizer, and iteration state for checkpointing."""
        self.model = model
        self.optimizer = optimizer
        self.iteration = iteration

    def state_dict(self):
        """Return a checkpoint-ready application state dictionary."""
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer
        )
        return {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "iteration": self.iteration,
        }

    def load_state_dict(self, state_dict):
        """Restore model, optimizer, and iteration state from a checkpoint."""
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )
        self.iteration = state_dict["iteration"]


def fsdp_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out_path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    """
    Save a distributed checkpoint for an FSDP-wrapped model.
    Only rank 0 performs the actual I/O, but all ranks participate in state-dict gathering.
    """

    state = {
        "app": AppState(
            model,
            optimizer,
            iteration,
        )
    }
    dcp.save(state, checkpoint_id=out_path)


def fsdp_load_checkpoint(
    checkpoint_path: str | os.PathLike,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Load an FSDP-distributed checkpoint.
    Broadcasts state from rank 0 to all other shards.
    """

    app = AppState(model, optimizer)
    state = {"app": app}
    dcp.load(state_dict=state, checkpoint_id=checkpoint_path)
    return int(app.iteration)


def load_checkpoint(
    checkpoint_path: str | os.PathLike,
    model: nn.Module,
) -> int:
    """
    Load a distributed checkpoint into a non-distributed model (not FSDP).
    """
    state = {
        "app": {
            "model": model.state_dict(),
        }
    }

    dcp.load(state_dict=state, checkpoint_id=checkpoint_path)
    model.load_state_dict(state["app"]["model"])


@torch.inference_mode()
def generate(
    prompt: str,
    max_tokens: int,
    context_length: int,
    batch_size: int,
    model: nn.Module,
    temp: float = 0.9,
    top_p: float = 0.8,
    device: torch.device = None,
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
            # Truncate sequence if it exceeds the model's maximum context length
            if inputs.shape[-1] > context_length:
                inputs = inputs[:, -context_length:]
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

    def __init__(
        self, filename: str | os.PathLike, B: int, T: int, rank=0, world_size=1
    ) -> None:
        """Open the memmapped dataset and initialize the shard cursor."""
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
        # Memory-map the binary file for efficient reading
        self.dataset = np.memmap(filename, dtype=np.uint16, mode="r")
        self.n_tokens = len(self.dataset)

        self.span = B * T
        self.local_pos = self.span * self.rank

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the next (Inputs, Targets) batch of tokens."""

        if self.local_pos + self.span + 1 > self.n_tokens:
            self.local_pos = self.span * self.rank

        start_pos = self.local_pos
        end_pos = start_pos + self.span + 1

        # Slice the dataset. Targets are shifted by 1 relative to inputs.
        buf = self.dataset[start_pos:end_pos]

        # Convert to torch tensor (uint16 -> int64)
        buf_torch = torch.from_numpy(buf.astype(np.int64))

        # Inputs include everything except the last token
        x = buf_torch[:-1].view(self.B, self.T)
        # Targets include everything except the first token
        y = buf_torch[1:].view(self.B, self.T)

        # Advance pointer for the next call
        self.local_pos += self.span * self.world_size

        return x, y


@dataclass
class GPTConfig:
    context_length = 1024
    num_layers = 12
    vocab_size = 50257
    d_model = 768
    num_heads = 12
    d_ff = 2048
    theta = 10000
    betas = (0.9, 0.95)
    eps = 1e-8
    weight_decay = 0.1
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    a_max = 6e-4
