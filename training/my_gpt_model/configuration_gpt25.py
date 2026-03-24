from dataclasses import dataclass

import torch
from transformers import PretrainedConfig


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


class MyConfig(PretrainedConfig):
    model_type = "gpt2.5"

    def __init__(
        self,
        vocab_size=GPTConfig.vocab_size,
        context_length=None,
        max_position_embeddings=None,
        num_layers=None,
        num_hidden_layers=None,
        num_heads=None,
        num_attention_heads=None,
        num_key_value_heads=None,
        d_model=None,
        hidden_size=None,
        d_ff=None,
        intermediate_size=None,
        theta=GPTConfig.theta,
        device=None,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)

        self.vocab_size = int(vocab_size)
        self.context_length = int(
            context_length
            if context_length is not None
            else max_position_embeddings
            if max_position_embeddings is not None
            else GPTConfig.context_length
        )
        self.max_position_embeddings = self.context_length

        self.num_layers = int(
            num_layers
            if num_layers is not None
            else num_hidden_layers
            if num_hidden_layers is not None
            else GPTConfig.num_layers
        )
        self.num_hidden_layers = self.num_layers

        self.num_heads = int(
            num_heads
            if num_heads is not None
            else num_attention_heads
            if num_attention_heads is not None
            else GPTConfig.num_heads
        )
        self.num_attention_heads = self.num_heads

        self.d_model = int(
            d_model
            if d_model is not None
            else hidden_size
            if hidden_size is not None
            else GPTConfig.d_model
        )
        self.hidden_size = self.d_model

        self.d_ff = int(
            d_ff
            if d_ff is not None
            else intermediate_size
            if intermediate_size is not None
            else GPTConfig.d_ff
        )
        self.intermediate_size = self.d_ff

        self.theta = float(theta)
        self.num_key_value_heads = int(
            num_key_value_heads
            if num_key_value_heads is not None
            else self.num_attention_heads
        )

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads for GPT-2.5."
            )
        self.head_dim = self.hidden_size // self.num_attention_heads

        # Retain the legacy field for compatibility with older checkpoints, but
        # runtime code must not rely on config-driven device placement.
        self.device = None if device is None else str(device)


if __name__ == "__main__":
    myconf = MyConfig()
    myconf.save_pretrained("myconf")
