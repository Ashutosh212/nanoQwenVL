from __future__ import annotations

import torch
from PIL import Image

from qwen25_scratch.tracing import describe_tensor


def test_image_patch_projection_has_gradients() -> None:
    image = Image.new("RGB", (28, 28), color=(10, 20, 30))
    image_tensor = (
        torch.tensor(list(image.getdata()), dtype=torch.float32)
        .reshape(28, 28, 3)
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )
    projection = torch.nn.Conv2d(3, 16, kernel_size=14, stride=14, bias=False)

    patch_grid = projection(image_tensor)
    patch_tokens = patch_grid.flatten(2).transpose(1, 2)
    patch_tokens.mean().backward()

    assert patch_tokens.shape == (1, 4, 16)
    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()


def test_tensor_description_exposes_movement_details() -> None:
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    description = describe_tensor("example", tensor)

    assert "shape=(2, 3)" in description
    assert "dtype=torch.float32" in description
    assert "device=cpu" in description
    assert "stride=(3, 1)" in description

