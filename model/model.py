import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
import math
import torch.nn as nn
from einops import einsum, rearrange, reduce
from collections.abc import Callable, Iterable
from typing import Optional
import numpy.typing as npt
import typing
import os


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
        rms = torch.sqrt(1 / self.d_model * reduce(x**2, "B T C -> B T 1", "sum") + self.eps)
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
        cos_pos = self.cos[token_positions].to(self.device)  # (..., seq_len, d_half)
        sin_pos = self.sin[token_positions].to(self.device)

        x_even = x[..., 0::2]  # (..., seq_len, d_half)
        x_odd = x[..., 1::2]  # (..., seq_len, d_half)

        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd = x_even * sin_pos + x_odd * cos_pos

        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)  # (..., seq_len, d_half, 2)
        new_shape = list(x.shape[:-2]) + [x.shape[-2], self.d_k]
        x_rot = x_rot.view(*new_shape)
        return x_rot


def softmax(x, dim, is_log=False, temp=1):
    # Scale x by temperature
    x_scaled = x / temp

    # LogSumExp trick for numerical stability:
    m = torch.max(x_scaled, dim=dim, keepdim=True).values
    log_sum_exp = m + torch.log(torch.sum(torch.exp(x_scaled - m), dim=dim, keepdim=True))

    # log_softmax = inputs - log_sum_exp
    log_probs = x_scaled - log_sum_exp

    if is_log:
        return log_probs
    else:
        return torch.exp(log_probs)


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


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None, device=None):
        super().__init__()

        assert d_model % num_heads == 0, "num heads should be a power of 2"

        self.d_k = self.d_v = d_model // num_heads
        self.num_heads = num_heads

        self.Wq = nn.Parameter(torch.randn(d_model, d_model, device=device))
        self.Wk = nn.Parameter(torch.randn(d_model, d_model, device=device))
        self.Wv = nn.Parameter(torch.randn(d_model, d_model, device=device))
        self.Wo = nn.Parameter(torch.randn(d_model, d_model, device=device))

        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device)

    def forward(self, x, token_positions=None):
        Q = x @ self.Wq.T
        K = x @ self.Wk.T
        V = x @ self.Wv.T

        # Split Q, K, V into heads
        Q = rearrange(Q, "b t (h d) -> b h t d", h=self.num_heads)
        K = rearrange(K, "b t (h d) -> b h t d", h=self.num_heads)
        V = rearrange(V, "b t (h d) -> b h t d", h=self.num_heads)

        seq_len = x.shape[-2]

        # Apply RoPE
        if self.rope:
            # Rotate Q and K
            if token_positions is None:
                token_positions = torch.arange(seq_len).unsqueeze(0)

            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        # Concat heads
        out = rearrange(out, "b h t d -> b t (h d)")
        return torch.einsum("hd,btd->bth", self.Wo, out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta=None, max_seq_len=None, device=None):
        super().__init__()

        self.norm_att = RMSNorm(d_model, device=device)
        self.norm_ff = RMSNorm(d_model, device=device)
        self.mhsa = MultiheadSelfAttention(d_model, num_heads, theta, max_seq_len, device)
        self.ffn = SwiGLU(d_model, d_ff, device=device)

    def forward(self, x):
        # Attention part of the block
        attn = x + self.mhsa(self.norm_att(x))
        # Position-wise feed-forward part of the block
        ffwd = attn + self.ffn(self.norm_ff(attn))
        return ffwd


class TransformerLM(nn.Module):
    def __init__(
        self, vocab_size, context_length, num_layers,
        d_model, num_heads, d_ff, theta=None, device=None
    ):
        super().__init__()

        self.emb = Embedding(vocab_size, d_model, device=device)
        self.tblocks = [
            TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device=device)
            for _ in range(num_layers)
        ]
        self.norm = RMSNorm(d_model, device=device)
        self.linear = Linear(d_model, vocab_size, device=device)

    def forward(self, x, targets=None):
        emb = self.emb(x)
        # Pass embedding through transformer blocks
        for tblock in self.tblocks:
            emb = tblock(emb)
        # Norm the transformer block output
        normed = self.norm(emb)
        # Pass through linear
        logits = self.linear(normed)
        # Calculate loss
        loss = None
        if targets is not None:
            loss = cross_entropy_loss(logits, targets)
        return logits, loss


def cross_entropy_loss(inputs, targets):
    B, T, C = inputs.shape

    # LogSumExp trick for numerical stability:
    log_probs = softmax(inputs, dim=-1, is_log=True)
    
    # Pick out the log probs for the correct classes
    b = torch.arange(B).unsqueeze(1)
    t = torch.arange(T).unsqueeze(0)
    log_probs_correct = log_probs[b, t, targets]
    
    # Mean negative log likelihood
    loss = -torch.mean(log_probs_correct)
    return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "epsilon": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
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
                
                # State initialization
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                t = state.get("t") + 1
                state["t"] = t
                
                grad = p.grad.data
                m = state["m"]
                v = state["v"]
                
                # Update moments
                m = m * beta1 + (1 - beta1) * grad
                v = v * beta2 + (1 - beta2) * grad**2

                state["m"] = m
                state["v"] = v
                
                # Bias correction
                alpha_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                
                # Update parameters
                denom = v.sqrt() + eps
                
                # Apply updates
                p.data -= alpha_t * m / denom
                # Apply weight decay
                p.data -= lr * weight_decay * p.data
        
        return loss


def learning_rate_schedule(t, a_max, a_min, T_w, T_c):
    # Warmup
    if t < T_w:
        a_t = t / T_w * a_max
    # Cosine annealing
    elif T_w <= t <= T_c:
        a_t = a_min + 1/2 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (a_max - a_min)
    # Post annealing
    else:
        a_t = a_min
    return a_t


def gradient_clipping(params, max_l2_norm):
    eps = 1e-6

    # 1. compute total global norm
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += torch.sum(p.grad ** 2)
    
    total_norm = torch.sqrt(total_sq)

    # 2. if we're under threshold, we're done
    if total_norm < max_l2_norm:
        return
    
    # 3. compute scale factor
    scale = max_l2_norm / (total_norm + eps)

    # 4. apply scaling to every grad tensor
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale)


def data_loading(dataset: npt.NDArray, batch_size: int,
                 context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # Preallocate x's and y's
    inputs = torch.zeros(batch_size, context_length, device=device, dtype=torch.long)
    targets = torch.zeros(batch_size, context_length, device=device, dtype=torch.long)

    # Number of data points
    n = dataset.shape[0]

    # Generate the random sample starting points of size batch_size
    random_samples = torch.randint(0, n - context_length, (batch_size, ))

    # Allocate the samples
    for batch, i in enumerate(random_samples):
        inputs[batch] = torch.tensor(dataset[i : i + context_length])
        targets[batch] = torch.tensor(dataset[i + 1 : i + 1 + context_length])
    
    return (inputs, targets)


def to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    else:
        return obj


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):

    # Join dicts into a single checkpoint dict
    checkpoint = {"model_state": model} | \
                 {"optimizer_state": optimizer} | \
                 {"iteration_state": iteration}
    # Save
    torch.save(checkpoint, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    # Load the checkpoint dict
    checkpoint = torch.load(src)

    # Extract dicts from the checkpoint and load state dicts
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    # Return iteration number
    return checkpoint["iteration_state"]['iteration']


def sample_data(dataset, batch_size, device):
    # Generate the random sample starting points of size batch_size
    n = dataset[0].shape[0]

    random_batch_idx = torch.randint(0, n - batch_size, (1, ))

    return (dataset[0][random_batch_idx:random_batch_idx+batch_size].to(device=device),
            dataset[1][random_batch_idx:random_batch_idx+batch_size].to(device=device))


def generate(prompt, max_tokens, context_length, model, temp, top_p, device):
    enc = tiktoken.get_encoding("o200k_base")
    sentences = []

    for i in range(5):
        inputs = torch.tensor(enc.encode(prompt), device=device).unsqueeze(0)
        for _ in range(max_tokens):
            # Generate next token
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits, _ = model(inputs)
            # Apply softmax with temperature
            probs = softmax(logits[:, -1, :], dim=-1, temp=temp)
            # Use top p sampling for next token
            next_token = top_p_sampling(probs, p=top_p, device=device)
            # Concatenate the token to the inputs tensor
            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=1)
            # If generated endoftext = end subsequent generation
            if enc.decode(next_token.tolist()) == "<|endoftext|>":
                break
            # If the input is larger than the context length, 
            # Use only the last context length amount of tokens
            if inputs.shape[-1] > context_length:
                inputs = inputs[-context_length:]

        # Print output
        sentences.append(f"\nGenerated sequence №{i+1}:\n" + enc.decode(inputs[0].tolist()) + "\n")
    return sentences


def top_p_sampling(probs, p, device):
    # Flatten the first dimension
    probs = torch.flatten(probs)
    # Sort probabilities
    sorted_probs, indices = torch.sort(probs, descending=True)

    # Calculate the cumsum
    cumsum = torch.cumsum(sorted_probs, dim=0)
    # Get the first index of an element where cumsum > p
    idx = torch.argmax((cumsum > p).int()).item()
    # Get the nucleus
    nucleus = sorted_probs[:idx+1]
    # Randomly sample a token now
    token = torch.multinomial(nucleus, 1)
    # Return the token by looking up in indices
    return torch.tensor([indices[token]], device=device)


class DataLoader:
    def __init__(self, filename, B, T):
        self.B = B
        self.T = T
        self.dataset = np.load(filename, mmap_mode='r')
        self.cur_shard_pos = 0
        self.n_tokens = len(self.dataset)


    def next_batch(self):
        B, T = self.B, self.T
        
        # Calculate the slice
        buf = self.dataset[self.cur_shard_pos : self.cur_shard_pos + B * T + 1]

        # Convert to torch and move to GPU only now
        buf_torch = torch.from_numpy(buf.astype(np.int64))

        x = buf_torch[:-1].view(B, T) # Inputs
        y = buf_torch[1:].view(B, T)

        # Advance pointer
        self.cur_shard_pos += B * T

        # Reset if we hit the end
        if self.cur_shard_pos + (B * T + 1) > self.n_tokens:
            self.cur_shard_pos = 0

        return x, y

