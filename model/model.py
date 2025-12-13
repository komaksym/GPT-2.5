import numpy as np
import torch
import torch.nn as nn
import math
from einops import einsum, rearrange, reduce


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # Specify mean for the param initialization
        mean, std = 0, np.sqrt(2 / (in_features + out_features))
        # Init the params from the normal distribution with said mean and std
        param = torch.normal(
            mean=mean, std=std, size=(out_features, in_features), dtype=dtype, device=device
        )
        # Truncate
        nn.init.trunc_normal_(param, a=-3 * std, b=3 * std)
        # Init the weight via the nn.Parameter
        self.weight = nn.Parameter(data=param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # Specify mean for the param initialization
        mean, std = 0, 1
        # Init the params from the normal distribution with said mean and std
        param = torch.normal(
            mean=mean, std=std, size=(num_embeddings, embedding_dim), dtype=dtype, device=device
        )
        # Truncate
        nn.init.trunc_normal_(param, a=-3, b=3)
        # Init the embedding via the nn.Parameter
        self.weight = nn.Parameter(data=param)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        # Init the learnable gain parameter
        param = torch.tensor([1.0] * d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(data=param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast the dtype to float32 to avoid overflowing when squaring the input in rms
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Calculate RMS
        rms = np.sqrt(1 / self.d_model * reduce(x**2, "B T C -> B T 1", "sum") + self.eps)
        # Normalize
        rmsnorm = x / rms * self.weight
        # Downcast back to the initial dtype
        return rmsnorm.to(dtype=in_dtype)


def SiLU(x):
    """SiLU activation function"""

    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SiLU activation function + gating mechanism = SwiGLU"""

    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()
        # Init d_ff dimension
        self.d_ff = (8 / 3) * d_model if not d_ff else d_ff
        assert self.d_ff % 64 == 0, "The dimensionality of the feedforward is not a multiple of 64"

        # init params
        self.w1 = nn.Parameter(data=torch.randn((d_ff, d_model), device=device, dtype=dtype))
        self.w3 = nn.Parameter(data=torch.randn((d_ff, d_model), device=device, dtype=dtype))
        self.w2 = nn.Parameter(data=torch.randn((d_model, d_ff), device=device, dtype=dtype))

    def forward(self, x):
        # Run swiglu
        result = (SiLU(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T
        return result


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even (pairs of dimensions).")
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.d_half = d_k // 2
        self.max_seq_len = int(max_seq_len)
        self.device = device if device is not None else torch.device("cpu")

        j = torch.arange(self.d_half, dtype=torch.float32, device=self.device)
        inv_freq = self.theta ** (-2.0 * j / float(self.d_k))  # (d_half,)
        pos = torch.arange(self.max_seq_len, dtype=torch.float32, device=self.device).unsqueeze(
            1
        )  # (max_seq_len,1)
        angles = pos * inv_freq.unsqueeze(0)  # (max_seq_len, d_half)

        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(f"Last dim of x must be d_k={self.d_k}, got {x.shape[-1]}.")
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError(
                "token_positions must have same seq_len in its last dim as x's sequence dimension."
            )

        token_positions = token_positions.long().to(self.cos.device)
        cos_pos = self.cos[token_positions]  # (..., seq_len, d_half)
        sin_pos = self.sin[token_positions]

        x_even = x[..., 0::2]  # (..., seq_len, d_half)
        x_odd = x[..., 1::2]  # (..., seq_len, d_half)

        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd = x_even * sin_pos + x_odd * cos_pos

        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)  # (..., seq_len, d_half, 2)
        new_shape = list(x.shape[:-2]) + [x.shape[-2], self.d_k]
        x_rot = x_rot.view(*new_shape)
        return x_rot


def softmax(x, dim):
    # Subtract the max to avoid overflow
    x = x - torch.max(x, dim=dim).values.unsqueeze(-1)
    # Calculate softmax
    return torch.exp(x) / torch.sum(torch.exp(x), dim=dim).unsqueeze(-1)


def scaled_dot_prod_attn(Q, K, V, mask=None):
    d_k = Q.shape[-1]

    # Calculate pre softmax
    logits = torch.einsum("b...qd,b...kd->b...qk", Q, K) / math.sqrt(d_k)

    # Apply the mask if exists
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))

    # Compute probs
    scores = softmax(logits, dim=-1)
    return torch.einsum("b...qk,b...kd->b...qd", scores, V)
