import numpy as np
import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # Init the params from the normal distribution with said mean and std
        param = torch.normal(
            mean=0, std=np.sqrt(2 / (in_features + out_features)), size=(out_features, in_features)
        )
        # Truncate
        nn.init.trunc_normal_(param, a=-3, b=3)
        # Init the weight via the nn.Parameter
        self.weight = nn.Parameter(data=param)

    def forward(self, x):
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # Specify mean for the param initialization
        mean, std = 0, 1
        # Init the params from the normal distribution with said mean and std
        param = torch.normal(mean=mean, std=std, size=(num_embeddings, embedding_dim))
        # Truncate
        nn.init.trunc_normal_(param, a=-3, b=3)
        # Init the embedding via the nn.Parameter
        self.weight = nn.Parameter(data=param).to(dtype=dtype, device=device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Lookup
        return self.weight[token_ids]
