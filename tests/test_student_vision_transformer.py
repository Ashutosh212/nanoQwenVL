from __future__ import annotations

import pytest
import torch

from qwen25_scratch.vision_steps.config import StudentVisionConfig
from qwen25_scratch.vision_steps.vision_transformer import (
    StudentVisionClassifier,
    StudentVisionTransformer,
)


def tiny_config(**overrides: object) -> StudentVisionConfig:
    values: dict[str, object] = {
        "image_height": 28,
        "image_width": 42,
        "in_channels": 3,
        "patch_size": 14,
        "hidden_size": 64,
        "num_heads": 4,
        "intermediate_size": 96,
        "depth": 2,
    }
    values.update(overrides)
    return StudentVisionConfig(**values)


def test_full_student_encoder_shapes_attention_and_gradients() -> None:
    torch.manual_seed(3)
    model = StudentVisionTransformer(tiny_config())
    images = torch.randn(2, 3, 28, 42)

    output = model(images, output_attentions=True)
    assert output.last_hidden_state.shape == (2, 6, 64)
    assert output.flattened_patches.shape == (2, 6, 588)
    assert output.grid_size == (2, 3)
    assert output.positions.tolist() == [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
    ]
    assert output.positions.device == images.device
    assert output.attentions is not None
    assert len(output.attentions) == 2
    assert output.attentions[0].shape == (2, 4, 6, 6)
    torch.testing.assert_close(
        output.attentions[0].sum(dim=-1),
        torch.ones(2, 4, 6),
        rtol=1e-5,
        atol=1e-6,
    )

    output.last_hidden_state.square().mean().backward()
    gradients = (
        model.patch_embedding.projection.weight.grad,
        model.blocks[0].attention.qkv.weight.grad,
        model.blocks[0].mlp.gate_projection.weight.grad,
    )
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_classifier_maps_pooled_tokens_to_ten_logits() -> None:
    model = StudentVisionClassifier(tiny_config(), num_classes=10)
    output = model(torch.randn(3, 3, 28, 42))
    assert output.vision_output.last_hidden_state.shape == (3, 6, 64)
    assert output.pooled_tokens.shape == (3, 64)
    assert output.logits.shape == (3, 10)


def test_encoder_rejects_image_size_not_divisible_by_patch_size() -> None:
    model = StudentVisionTransformer(tiny_config())
    with pytest.raises(ValueError, match="must both be divisible"):
        model(torch.randn(1, 3, 29, 42))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_mixed_precision_forward_backward() -> None:
    device = torch.device("cuda")
    model = StudentVisionClassifier(tiny_config(), num_classes=10).to(device)
    images = torch.randn(2, 3, 28, 42, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(images)
        loss = output.logits.square().mean()
    loss.backward()

    assert output.logits.device.type == "cuda"
    assert output.vision_output.positions.device.type == "cuda"
    assert model.encoder.blocks[0].attention.qkv.weight.grad is not None
