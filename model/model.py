import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # Specify mean for the param initialization
        mean, std = 0, np.sqrt(2/(in_features+out_features))
        # Init the params from the normal distribution with said mean and std
        param = torch.normal(mean=mean, std=std, size=(out_features, in_features),
                             dtype=dtype, device=device)
        # Truncate
        nn.init.trunc_normal_(param, a =-3 * std, b = 3 * std)
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
        param = torch.normal(mean=mean, std=std, size=(num_embeddings, embedding_dim),
                             dtype=dtype, device=device)
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