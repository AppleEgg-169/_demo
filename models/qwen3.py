import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.modules):
    def __init__(
        self,
        config,
    ):
        super.__init__()
        self.config = config

        self.q_proj = None

    def forward(
        self,
        positions,
        hidden_states,
        kv_cache,
    ):
        pass
