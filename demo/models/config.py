from dataclasses import dataclass


@dataclass
class ModelConfig:
    num_hidden_layers: int = 16
    hidden_size: int = 4096
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 8192
    rms_norm_eps: float = 1e-6
    vocab_size: int = 16384
    rope_theta: float = 1e6
    max_position_embeddings: int = 32768
