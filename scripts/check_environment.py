#!/usr/bin/env python
"""Verify dependencies and trace a tiny image-to-patch backward pass."""

from __future__ import annotations

import platform

import numpy as np
import PIL
import torch
import torchvision
from torch import nn

from qwen25_scratch.tracing import trace_tensor


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    torch.manual_seed(7)
    device = select_device()

    print("Dependency report")
    print(f"  Python:      {platform.python_version()}")
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  TorchVision: {torchvision.__version__}")
    print(f"  NumPy:       {np.__version__}")
    print(f"  Pillow:      {PIL.__version__}")
    print(f"  CUDA build:  {torch.version.cuda}")
    print(f"  Device:      {device}")

    batch, channels, height, width = 2, 3, 28, 28
    patch_size, hidden_size = 14, 32
    images = torch.randn(batch, channels, height, width, device=device)
    patch_projection = nn.Conv2d(
        channels,
        hidden_size,
        kernel_size=patch_size,
        stride=patch_size,
        bias=False,
        device=device,
    )

    trace_tensor("images [B,C,H,W]", images)
    patch_grid = patch_projection(images)
    trace_tensor("patch_grid [B,D,H/P,W/P]", patch_grid)
    patch_tokens = patch_grid.flatten(2).transpose(1, 2)
    trace_tensor("patch_tokens [B,N,D]", patch_tokens)

    loss = patch_tokens.square().mean()
    trace_tensor("loss", loss)
    loss.backward()
    trace_tensor("patch_projection.weight.grad", patch_projection.weight.grad)
    print("Smoke test passed: image tensors reached a differentiable token sequence.")


if __name__ == "__main__":
    main()

