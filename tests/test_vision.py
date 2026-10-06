from __future__ import annotations

import pytest
import torch

from qwen25_scratch.vision import (
    Axial2DRotaryEmbedding,
    RMSNorm,
    VisionConfig,
    VisionEncoder,
    VisionPatchEmbedding,
    apply_2d_rope,
)


def tiny_config(**overrides: object) -> VisionConfig:
    values: dict[str, object] = {
        "in_channels": 3,
        "patch_size": 2,
        "temporal_patch_size": 2,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_heads": 2,
        "depth": 2,
    }
    values.update(overrides)
    return VisionConfig(**values)


def test_config_rejects_head_dimension_incompatible_with_2d_rope() -> None:
    with pytest.raises(ValueError, match="head_dim must be divisible by 4"):
        tiny_config(hidden_size=12, num_heads=2)


def test_rms_norm_matches_its_equation() -> None:
    layer = RMSNorm(3, eps=1e-6)
    values = torch.tensor([[1.0, 2.0, 3.0]])
    actual = layer(values)
    expected = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual.mean(-1), torch.zeros(1))


def test_still_image_is_repeated_for_temporal_patch_projection() -> None:
    patcher = VisionPatchEmbedding(tiny_config(hidden_size=16))
    torch.nn.init.ones_(patcher.projection.weight)
    image = torch.ones(1, 3, 4, 6)
    output = patcher(image)
    assert output.tokens.shape == (1, 6, 16)
    assert output.grid_thw == (1, 2, 3)
    assert output.positions.tolist() == [
        [0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]
    ]
    # C * temporal_patch * patch_h * patch_w = 3 * 2 * 2 * 2 = 24.
    torch.testing.assert_close(output.tokens, torch.full_like(output.tokens, 24.0))


def test_patch_embedding_rejects_non_divisible_image_size() -> None:
    patcher = VisionPatchEmbedding(tiny_config())
    with pytest.raises(ValueError, match="height=5 must be divisible"):
        patcher(torch.randn(1, 3, 5, 6))


def test_2d_rope_origin_is_identity_and_preserves_vector_norms() -> None:
    rope = Axial2DRotaryEmbedding(head_dim=8)
    positions = torch.tensor([[0, 0], [1, 2]])
    queries = torch.randn(1, 2, 2, 8)
    keys = torch.randn(1, 2, 2, 8)
    cos, sin = rope(positions, dtype=queries.dtype)
    rotated_queries, rotated_keys = apply_2d_rope(queries, keys, cos, sin)
    torch.testing.assert_close(rotated_queries[:, :, 0], queries[:, :, 0])
    torch.testing.assert_close(rotated_keys[:, :, 0], keys[:, :, 0])
    torch.testing.assert_close(
        rotated_queries.norm(dim=-1), queries.norm(dim=-1), rtol=1e-5, atol=1e-6
    )


def test_encoder_shapes_attention_probabilities_and_gradients() -> None:
    torch.manual_seed(3)
    model = VisionEncoder(tiny_config())
    images = torch.randn(2, 3, 4, 6)
    output = model(images, output_attentions=True)
    output.last_hidden_state.square().mean().backward()
    assert output.last_hidden_state.shape == (2, 6, 16)
    assert output.grid_thw == (1, 2, 3)
    assert output.attentions is not None
    assert len(output.attentions) == 2
    assert output.attentions[0].shape == (2, 2, 6, 6)
    torch.testing.assert_close(
        output.attentions[0].sum(dim=-1),
        torch.ones(2, 2, 6),
        rtol=1e-5,
        atol=1e-6,
    )
    gradient = model.patch_embedding.projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
