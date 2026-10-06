"""Dimensions shared by the small Qwen-style language-model exercises."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StudentLanguageConfig:
    # A deliberately small learning vocabulary. A tokenizer will later map text
    # strings into IDs in [0, vocab_size).
    vocab_size: int = 128

    # This must match StudentMergerConfig.language_hidden_size so visual and
    # text embeddings can eventually share one decoder sequence.
    hidden_size: int = 96

    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    intermediate_size: int = 256
    num_layers: int = 2
    max_sequence_length: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "max_sequence_length": self.max_sequence_length,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for text RoPE")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def query_heads_per_kv_head(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads
