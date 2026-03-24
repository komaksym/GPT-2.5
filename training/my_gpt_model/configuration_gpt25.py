from transformers import PretrainedConfig
import torch
from dataclasses import dataclass


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
        context_length=GPTConfig.context_length,
        num_layers=GPTConfig.num_layers,
        num_heads=GPTConfig.num_heads,
        d_model=GPTConfig.d_model,
        d_ff=GPTConfig.d_ff,
        theta=GPTConfig.theta,
        device=str(GPTConfig.device),
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.theta = float(theta)
        self.device = device if isinstance(device, str) else str(device)
    

if __name__ == "__main__":
    myconf = MyConfig()
    myconf.save_pretrained("myconf")