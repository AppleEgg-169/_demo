import torch
from torch import nn
import torch.nn.functional as F

from demo.layers.attention import Attention
from demo.layers.layernorm import RMSNorm
from demo.layers.rotary_embedding import get_rope
from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_hidden_layers: int = 16
    hidden_size: int = 4096
    num_head: int = 32
    num_kv_head: int = 8
    head_dim: int = 128
    intermediate_size: int = 8192
    rms_norm_eps: float = 1e-6
    vocab_size: int = 16384
    rope_theta: float = 1e6
    max_position_embeddings: int = 32768
    scale: float = 1.0


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.num_head = config.num_head
        self.num_kv_head = config.num_kv_head

        self.q_proj = nn.Linear(self.hidden_size, self.num_head * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_head * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_head * self.head_dim)
        self.o_proj = nn.Linear(self.num_head * self.head_dim, self.hidden_size)

        self.attn = Attention(config)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.q_norm = RMSNorm(config)
        self.k_norm = RMSNorm(config)

    def forward(
        self,
        positions,
        hidden_states,
    ):
        B, T, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(
            B, T, self.num_head, self.head_dim
        )  # [B,T,Hq,D]
        k = self.k_proj(hidden_states).view(
            B, T, self.num_kv_head, self.head_dim
        )  # [B,T,Hkv,D]
        v = self.v_proj(hidden_states).view(
            B, T, self.num_kv_head, self.head_dim
        )  # [B,T,Hkv,D]

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        out = self.attn(q, k, v)
        output = self.o_proj(out)
        return output


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(
            config.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(config.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.tensor):
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class DecoderLayers(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.attn = Qwen3Attention(config=config)

        self.mlp = Qwen3MLP(config=config)

        self.input_norm = RMSNorm(config)
        self.post_attn_norm = RMSNorm(config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ):
        if residual is None:
            hidden_states, residual = self.input_norm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_norm(hidden_states, residual)

        hidden_states = self.attn(positions, hidden_states)

        hidden_states, residual = self.post_attn_norm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3Model(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.embed = 1
        self.layers = nn.ModuleList(
            [DecoderLayers(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config)

    def forward(
        self,
        input_ids,
    ):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.model = Qwen3Model(config)
        # self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        config,
    ):
        pass


if __name__ == "__main__":
    config = ModelConfig()
    config.scale = config.head_dim**-0.5  # 补充 scale

    # 模拟输入
    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    # 2. 实例化模型
    # 注意：你需要确保 RMSNorm 和 RotaryEmbedding 已正确定义
    model = Qwen3Attention(config).eval()

    try:
        with torch.no_grad():
            output = model(positions, hidden_states)

        # 3. 验证输出形状
        expected_shape = (batch_size, seq_len, config.hidden_size)
        assert output.shape == expected_shape, (
            f"形状错误: 期望 {expected_shape}, 得到 {output.shape}"
        )
        print("✅ 形状验证通过!")

        # 4. 验证因果性 (简易测试)
        # 如果是因果的，改变输入的最后一个词，不应影响输出的前面几个词
        hs2 = hidden_states.clone()
        hs2[:, -1, :] += 1.0  # 只改变最后一个词
        out1 = model(positions, hidden_states)
        out2 = model(positions, hs2)

        # 比较前 seq_len-1 个位置是否一致
        diff = torch.abs(out1[:, :-1, :] - out2[:, :-1, :]).max()
        assert diff < 1e-6, f"因果性验证失败! 差异: {diff}"
        print("✅ 因果掩码验证通过!")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()
