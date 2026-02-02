import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.rms_norm_eps = config.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(config.head_dim))

    def forward(self, x: torch.tensor):
        rms = x.pow(2).mean(dim=-1, keepdim=True) + self.rms_norm_eps
        x = x / torch.sqrt(rms)
        return x * self.weight
