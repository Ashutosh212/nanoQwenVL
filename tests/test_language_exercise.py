from __future__ import annotations

import pytest
import torch

from qwen25_scratch.language_steps.causal_mask import DEMO_SEQUENCE_LENGTH
from qwen25_scratch.language_steps.config import StudentLanguageConfig
from qwen25_scratch.language_steps.language_mlp import (
    make_demo_hidden_states as make_mlp_demo_hidden_states,
)
from qwen25_scratch.language_steps.multi_head_attention import (
    MultiHeadCausalSelfAttention,
    make_demo_hidden_states,
)
from qwen25_scratch.language_steps.text_rope import DEMO_TEXT_LENGTH
from qwen25_scratch.language_steps.token_embedding import make_demo_token_ids


def test_language_config_preserves_planned_dimensions() -> None:
    config = StudentLanguageConfig()
    assert config.hidden_size == 96
    assert config.head_dim == 24
    assert config.query_heads_per_kv_head == 2


def test_language_config_rejects_invalid_gqa_head_counts() -> None:
    with pytest.raises(
        ValueError,
        match="num_attention_heads must be divisible by num_key_value_heads",
    ):
        StudentLanguageConfig(num_attention_heads=4, num_key_value_heads=3)


def test_token_embedding_demo_inputs() -> None:
    token_ids = make_demo_token_ids()
    assert token_ids.shape == (2, 6)
    assert token_ids.dtype == torch.long
    assert token_ids[0, 2].item() == 11
    assert token_ids[0, 3].item() == 11
    assert token_ids[1, 4].item() == 11
    assert DEMO_SEQUENCE_LENGTH == 5
    assert DEMO_TEXT_LENGTH == 5


def test_multi_head_attention_demo_inputs() -> None:
    hidden_states = make_demo_hidden_states()
    assert hidden_states.shape == (2, 5, 96)
    assert hidden_states.dtype == torch.float32


def test_completed_multi_head_attention() -> None:
    torch.manual_seed(7)
    config = StudentLanguageConfig()
    attention = MultiHeadCausalSelfAttention(config)
    hidden_states = make_demo_hidden_states()
    result = attention(hidden_states)

    assert result.hidden_states.shape == (2, 5, 96)
    assert result.probabilities.shape == (2, 4, 5, 5)
    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert torch.count_nonzero(result.probabilities[..., future]) == 0


def test_language_mlp_demo_inputs() -> None:
    hidden_states = make_mlp_demo_hidden_states()
    assert hidden_states.shape == (2, 5, 96)
    assert hidden_states.dtype == torch.float32
