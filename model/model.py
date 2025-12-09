import torch
import torch.nn as nn
import numpy as np
from einops import einsum, rearrange

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # Init the params from the normal distribution with said mean and std
        param = torch.normal(mean=0, std=np.sqrt(2/(in_features+out_features)), size=(out_features, in_features))
        # Truncate
        nn.init.trunc_normal_(param, a=-3, b=3)
        # Init the weight via the nn.Parameter
        self.weight = nn.Parameter(data=param)

    def forward(self, x):
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")