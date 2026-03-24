import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import logging

from transformers import PreTrainedModel
from .configuration_gpt25 import MyConfig
from transformers.modeling_outputs import CausalLMOutput


logger = logging.getLogger(__name__)

GPT_INIT_STD = 0.02


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
        super().__init__(
            in_features,
            out_features,
            bias=False,
            device=device,
            dtype=dtype,
        )
        nn.init.normal_(self.weight, mean=0.0, std=GPT_INIT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        super().__init__(
            num_embeddings,
            embedding_dim,
            device=device,
            dtype=dtype,
        )
        nn.init.normal_(self.weight, mean=0.0, std=GPT_INIT_STD)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
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
        # w1 and w3 are for the gated part; w2 is for the projection back to d_model
        self.w1 = nn.Parameter(data=torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w3 = nn.Parameter(data=torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2 = nn.Parameter(data=torch.empty((d_model, d_ff), device=device, dtype=dtype))

        nn.init.normal_(self.w1, mean=0.0, std=GPT_INIT_STD)
        nn.init.normal_(self.w3, mean=0.0, std=GPT_INIT_STD)
        nn.init.normal_(self.w2, mean=0.0, std=GPT_INIT_STD)

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
        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1)  # (..., seq_len, d_half, 2)
        new_shape = list(x.shape[:-2]) + [x.shape[-2], self.d_k]
        x_rot = x_rot.view(*new_shape)
        return x_rot


def softmax(x: torch.Tensor, dim: int, is_log: bool = False, temp: float = 1.0) -> torch.Tensor:
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
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights.masked_fill(
            ~attention_mask, torch.finfo(attn_weights.dtype).min
        )

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()
    return attn_output, attn_weights


def positions_are_packed(position_ids: torch.Tensor | None) -> bool:
    if position_ids is None or position_ids.size(-1) <= 1:
        return False
    return bool((position_ids[..., 1:] <= position_ids[..., :-1]).any())


def build_attention_mask(
    seq_len: int,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor | None:
    has_packed_boundaries = positions_are_packed(position_ids)
    if attention_mask is None and not has_packed_boundaries:
        return None

    causal_mask = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()
    combined_mask = causal_mask.unsqueeze(0)

    if has_packed_boundaries:
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
        return self.config._attn_implementation

    def set_attn_implementation(self, attn_implementation: str) -> None:
        self.config._attn_implementation = attn_implementation.removeprefix("paged|")

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        super().__init__()

        self.norm_att = RMSNorm(d_model, device=device)
        self.norm_ff = RMSNorm(d_model, device=device)
        self.mhsa = MultiheadSelfAttention(d_model, num_heads, theta, max_seq_len, device)
        self.ffn = SwiGLU(d_model, d_ff, device=device)

    def set_attn_implementation(self, attn_implementation: str) -> None:
        self.mhsa.set_attn_implementation(attn_implementation)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        super().__init__()

        self.device = device

        self.emb = Embedding(vocab_size, d_model, device=device)
        self.tblocks = nn.ModuleList(
            TransformerBlock(d_model, num_heads, d_ff, theta, context_length, device=device)
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(d_model, device=device)
        self.linear = Linear(d_model, vocab_size, device=device)

        # Tie output head to token embeddings
        self.linear.weight = self.emb.weight

    def set_attn_implementation(self, attn_implementation: str) -> None:
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


class HFTransformerLM(PreTrainedModel):
    config_class = MyConfig
    _tied_weights_keys = {"model.linear.weight": "model.emb.weight"}
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config):
        """Expose TransformerLM through the minimal HF interface used by Trainer."""
        super().__init__(config)
        self.model = TransformerLM(
            config.vocab_size,
            config.context_length,
            config.num_layers,
            config.d_model,
            config.num_heads,
            config.d_ff,
            config.theta,
            config.device,
        )
        self._sync_attn_implementation()
        self.post_init()

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False
    ) -> str:
        try:
            return super()._check_and_adjust_attn_implementation(
                attn_implementation, is_init_check=is_init_check
            )
        except Exception as exc:
            if attn_implementation is None:
                raise

            is_paged = attn_implementation.startswith("paged|")
            requested_implementation = attn_implementation.removeprefix("paged|")
            if requested_implementation not in {
                "flash_attention_2",
                "flash_attention_3",
            }:
                raise

            fallback_implementation = "paged|sdpa" if is_paged else "sdpa"
            logger.warning(
                "FlashAttention requested but unavailable (%s). Falling back to %s.",
                exc,
                fallback_implementation,
            )
            return super()._check_and_adjust_attn_implementation(
                fallback_implementation, is_init_check=is_init_check
            )

    def _get_runtime_attn_implementation(self) -> str:
        return self.config._attn_implementation.removeprefix("paged|")

    def _sync_attn_implementation(self) -> None:
        self.model.set_attn_implementation(self._get_runtime_attn_implementation())

    def get_input_embeddings(self):
        return self.model.emb

    def get_output_embeddings(self):
        return self.model.linear

    def set_input_embeddings(self, new_embeddings):
        self.model.emb = new_embeddings

    def set_output_embeddings(self, new_embeddings):
        self.model.linear = new_embeddings

    def set_attn_implementation(self, attn_implementation: str | dict):
        super().set_attn_implementation(attn_implementation)
        self._sync_attn_implementation()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        **kwargs,
    ):
        self._sync_attn_implementation()
        logits, _ = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(loss=loss, logits=logits)