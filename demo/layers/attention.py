import torch
from torch import nn
from transformers import Qwen3Config


class Attention(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
    ):
        super().__init__()
        self.num_head = config.num_head
        self.head_dim = config.head_dim
        self.scale = config.scale
        self.num_kv_head = config.num_kv_head
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(
        self,
        q: torch.Tensor,  # [B,Tq,Hq,D]
        k: torch.Tensor,  # [B,Tk,Hkv,D]
        v: torch.Tensor,  # [B,Tk,Hkv,D]
    ) -> torch.Tensor:
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, (
            f"expected q/k/v 4D, got {q.shape}, {k.shape}, {v.shape}"
        )
        B, Tq, Hq, D = q.shape
        Bk, Tk, Hkv, Dk = k.shape
        Bv, Tv, Hvv, Dv = v.shape
        causal = True
        assert B == Bk == Bv and Tk == Tv and Hkv == Hvv and D == Dk == Dv

        if Hkv != Hq:
            assert Hq % Hkv == 0, f"GQA requires Hq % Hkv == 0, got Hq={Hq}, Hkv={Hkv}"
            repeat = Hq // Hkv
            k = k.repeat_interleave(repeat, dim=2)  # [B,Tk,Hq,D]
            v = v.repeat_interleave(repeat, dim=2)  # [B,Tk,Hq,D]

        qh = q.permute(0, 2, 1, 3)  # [B,H,Tq,D]
        kh = k.permute(0, 2, 1, 3)  # [B,H,Tk,D]
        vh = v.permute(0, 2, 1, 3)  # [B,H,Tk,D]

        scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale  # [B,H,Tq,Tk]

        if causal:
            causal_mask = torch.ones(
                (Tq, Tk), device=scores.device, dtype=torch.bool
            ).tril()
            scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)  # [B,H,Tq,Tk]
        out = torch.matmul(probs, vh)  # [B,H,Tq,D]

        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, Tq, Hq * D)
        return out
