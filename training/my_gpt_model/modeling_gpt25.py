import math
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import logging

from .configuration_gpt25 import MyConfig


logger = logging.getLogger(__name__)

GPT_INIT_STD = 0.02
PastKeyValue = tuple[torch.Tensor, torch.Tensor, int]
PastKeyValues = tuple[PastKeyValue, ...]


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
        cos_pos = self.cos[token_positions].to(
            device=x.device, dtype=x.dtype
        )  # (..., seq_len, d_half)
        sin_pos = self.sin[token_positions].to(
            device=x.device, dtype=x.dtype
        )  # (..., seq_len, d_half)

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
    query_len: int,
    key_len: int,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
    query_position_ids: torch.Tensor | None = None,
    key_position_ids: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Build a causal mask that also respects packed sequence boundaries."""
    if attention_mask is not None and attention_mask.shape[-1] != key_len:
        raise ValueError("attention_mask must match the key sequence length.")

    if key_position_ids is None and query_len == key_len:
        key_position_ids = query_position_ids

    has_packed_boundaries = positions_are_packed(key_position_ids)
    if attention_mask is None and query_len == key_len and not has_packed_boundaries:
        return None

    if query_position_ids is None:
        query_start = key_len - query_len
        query_position_ids = torch.arange(
            query_start, key_len, device=device
        ).unsqueeze(0)
    else:
        query_position_ids = query_position_ids.to(device)

    if key_position_ids is None:
        key_position_ids = torch.arange(key_len, device=device).unsqueeze(0)
    else:
        key_position_ids = key_position_ids.to(device)

    combined_mask = key_position_ids[:, None, :] <= query_position_ids[:, :, None]

    if has_packed_boundaries:
        # Packed samples reset position ids back to zero, so segment ids keep
        # attention constrained to tokens from the same packed sequence.
        query_segment_ids = (query_position_ids == 0).to(torch.int64).cumsum(dim=-1) - 1
        key_segment_ids = (key_position_ids == 0).to(torch.int64).cumsum(dim=-1) - 1
        same_segment = query_segment_ids[:, :, None] == key_segment_ids[:, None, :]
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
        self.max_seq_len = max_seq_len
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
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, PastKeyValue | None]:
        """Run self-attention and optionally update the KV cache."""
        # x: (B, T, d_model)
        cache_capacity = kwargs.pop("cache_capacity", None)
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
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        if token_positions.ndim == 1:
            token_positions = token_positions.unsqueeze(0)
        if past_key_value is not None and positions_are_packed(token_positions):
            raise ValueError("KV caching does not support packed position_ids.")

        if self.rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        key_positions = token_positions
        present_key_value = None
        if use_cache:
            if self.max_seq_len is None:
                raise ValueError("KV caching requires max_seq_len to be configured.")

            if past_key_value is None:
                requested_cache_capacity = (
                    self.max_seq_len if cache_capacity is None else int(cache_capacity)
                )
                if requested_cache_capacity < T:
                    raise ValueError(
                        "KV cache capacity must fit the provided input length."
                    )
                if requested_cache_capacity > self.max_seq_len:
                    raise ValueError(
                        "KV cache capacity exceeded configured max_seq_len."
                    )
                key_cache = K.new_empty(
                    K.shape[0], self.num_heads, requested_cache_capacity, self.d_k
                )
                value_cache = V.new_empty(
                    V.shape[0], self.num_heads, requested_cache_capacity, self.d_k
                )
                cache_length = 0
            else:
                key_cache, value_cache, cache_length = past_key_value

            next_cache_length = cache_length + T
            cache_capacity = key_cache.shape[-2]
            if next_cache_length > cache_capacity:
                # Grow the cache geometrically to avoid reallocating every
                # decoding step while still respecting the context limit.
                new_cache_capacity = min(
                    self.max_seq_len,
                    max(next_cache_length, min(cache_capacity * 2, self.max_seq_len)),
                )
                if new_cache_capacity < next_cache_length:
                    raise ValueError("KV cache length exceeded configured max_seq_len.")
                expanded_key_cache = K.new_empty(
                    K.shape[0], self.num_heads, new_cache_capacity, self.d_k
                )
                expanded_value_cache = V.new_empty(
                    V.shape[0], self.num_heads, new_cache_capacity, self.d_k
                )
                if cache_length:
                    expanded_key_cache[:, :, :cache_length, :] = key_cache[
                        :, :, :cache_length, :
                    ]
                    expanded_value_cache[:, :, :cache_length, :] = value_cache[
                        :, :, :cache_length, :
                    ]
                key_cache = expanded_key_cache
                value_cache = expanded_value_cache
                cache_capacity = new_cache_capacity

            if next_cache_length > self.max_seq_len:
                raise ValueError("KV cache length exceeded configured max_seq_len.")

            key_cache[:, :, cache_length:next_cache_length, :] = K
            value_cache[:, :, cache_length:next_cache_length, :] = V

            K = key_cache[:, :, :next_cache_length, :]
            V = value_cache[:, :, :next_cache_length, :]
            key_positions = torch.arange(next_cache_length, device=x.device).unsqueeze(
                0
            )
            present_key_value = (key_cache, value_cache, next_cache_length)
        elif past_key_value is not None:
            past_k, past_v, _ = past_key_value
            K = torch.cat((past_k, K), dim=-2)
            V = torch.cat((past_v, V), dim=-2)
            key_positions = torch.arange(K.shape[-2], device=x.device).unsqueeze(0)

        attention_impl = self.attn_implementation
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            attention_impl, eager_attention_forward
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
                **kwargs,
            )
        else:
            packed_mask = None
            # Single-token decode with strictly increasing positions is already
            # causal, so we can skip materializing an explicit mask.
            skip_decode_mask = (
                attention_mask is None
                and T == 1
                and not positions_are_packed(key_positions)
                and torch.equal(token_positions[..., -1], key_positions[..., -1])
            )
            if not skip_decode_mask:
                packed_mask = build_attention_mask(
                    query_len=T,
                    key_len=K.shape[-2],
                    device=x.device,
                    attention_mask=attention_mask,
                    query_position_ids=token_positions,
                    key_position_ids=key_positions,
                )
            out, _ = attention_interface(
                self,
                Q,
                K,
                V,
                attention_mask=packed_mask,
                scaling=self.d_k**-0.5,
                **kwargs,
            )

        # Concatenate heads: (B, T, num_heads, d_k) -> (B, T, d_model)
        out = rearrange(out, "b t h d -> b t (h d)")
        # Final linear projection
        return self.Wo(out), present_key_value


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
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, PastKeyValue | None]:
        """Apply attention, feed-forward layers, and residual connections."""
        # Pre-Norm + Self-Attention + Residual connection
        attn_output, present_key_value = self.mhsa(
            self.norm_att(x),
            token_positions=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )
        attn = x + attn_output
        # Pre-Norm + Feed-Forward + Residual connection
        ffwd = attn + self.ffn(self.norm_ff(attn))
        return ffwd, present_key_value


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

    def forward_hidden_states(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, PastKeyValues | None]:
        """Return normalized hidden states and optional next-step KV caches."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids or inputs_embeds.")

        emb = self.emb(input_ids) if inputs_embeds is None else inputs_embeds

        if past_key_values is not None and len(past_key_values) != len(self.tblocks):
            raise ValueError(
                "past_key_values must provide one cache entry per transformer block."
            )

        next_past_key_values: list[PastKeyValue] = []
        layer_past_key_values = (
            past_key_values
            if past_key_values is not None
            else (None,) * len(self.tblocks)
        )

        for tblock, layer_past_key_value in zip(self.tblocks, layer_past_key_values):
            emb, present_key_value = tblock(
                emb,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past_key_value,
                use_cache=use_cache,
                **kwargs,
            )
            if present_key_value is not None:
                next_past_key_values.append(present_key_value)

        return self.norm(emb), tuple(next_past_key_values) if use_cache else None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        targets: Optional[torch.Tensor] = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return logits and optional training loss for a token batch."""
        hidden_states, _ = self.forward_hidden_states(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        logits = self.linear(hidden_states)

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


class GPT25Model(PreTrainedModel):
    config_class = MyConfig
    base_model_prefix = "model"
    _tied_weights_keys = {"model.linear.weight": "model.emb.weight"}
    _supports_flash_attn = True
    _supports_attention_backend = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config):
        """Base GPT-2.5 model exposed through the HF AutoModel contract."""
        super().__init__(config)
        self.model = TransformerLM(
            config.vocab_size,
            config.context_length,
            config.num_layers,
            config.d_model,
            config.num_heads,
            config.d_ff,
            config.theta,
        )
        self._sync_attn_implementation()
        self.post_init()

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False
    ) -> str:
        """Fall back to SDPA when flash attention is unavailable at runtime."""
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
        """Return the backend name understood by the inner TransformerLM."""
        attn_implementation = (
            getattr(self.config, "_attn_implementation", None) or "sdpa"
        )
        return attn_implementation.removeprefix("paged|")

    def _sync_attn_implementation(self) -> None:
        """Push the configured attention backend into the wrapped model."""
        self.model.set_attn_implementation(self._get_runtime_attn_implementation())

    def get_input_embeddings(self):
        """Expose the token embedding layer through the HF API."""
        return self.model.emb

    def get_output_embeddings(self):
        """Expose the output projection through the HF API."""
        return self.model.linear

    def set_input_embeddings(self, new_embeddings):
        """Replace the wrapped token embeddings."""
        self.model.emb = new_embeddings

    def set_output_embeddings(self, new_embeddings):
        """Replace the wrapped output projection."""
        self.model.linear = new_embeddings

    def set_attn_implementation(self, attn_implementation: str | dict):
        """Update the configured attention backend and sync it to runtime."""
        super().set_attn_implementation(attn_implementation)
        self._sync_attn_implementation()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        return_dict=None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        """Return hidden states, optionally alongside updated KV caches."""
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        cache_position = kwargs.get("cache_position")
        if position_ids is None and cache_position is not None:
            position_ids = (
                cache_position.unsqueeze(0)
                if cache_position.ndim == 1
                else cache_position
            )
        use_cache = self.config.use_cache if use_cache is None else use_cache
        hidden_states, next_past_key_values = self.model.forward_hidden_states(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        if return_dict is False:
            outputs = (hidden_states,)
            if use_cache:
                outputs += (next_past_key_values,)
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_past_key_values,
        )


class GPT25ForCausalLM(GPT25Model):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        return_dict=None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ):
        """Return causal LM logits, optional loss, and optional KV caches."""
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        cache_position = kwargs.get("cache_position")
        if position_ids is None and cache_position is not None:
            position_ids = (
                cache_position.unsqueeze(0)
                if cache_position.ndim == 1
                else cache_position
            )
        use_cache = self.config.use_cache if use_cache is None else use_cache
        hidden_states, next_past_key_values = self.model.forward_hidden_states(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        logits = self.model.linear(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if return_dict is False:
            if loss is None:
                outputs = (logits,)
            else:
                outputs = (loss, logits)
            if use_cache:
                outputs += (next_past_key_values,)
            return outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_past_key_values,
        )


HFTransformerLM = GPT25ForCausalLM
