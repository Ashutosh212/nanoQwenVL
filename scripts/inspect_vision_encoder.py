#!/usr/bin/env python
"""Run a tiny vision encoder and print every important tensor movement."""

from __future__ import annotations

import torch

from qwen25_scratch.tracing import trace_tensor
from qwen25_scratch.vision import VisionConfig, VisionEncoder


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    torch.manual_seed(7)
    config = VisionConfig(
        patch_size=14,
        temporal_patch_size=2,
        hidden_size=64,
        intermediate_size=192,
        num_heads=4,
        depth=2,
        trace=True,
    )
    model = VisionEncoder(config)
    images = torch.randn(1, 3, 28, 42)

    print("Mini Qwen2.5-VL vision encoder")
    print(f"config: {config}")
    print(f"trainable parameters: {parameter_count(model):,}")
    print("attention: softmax(Q @ K^T / sqrt(head_dim)) @ V")
    print("RMSNorm: x / sqrt(mean(x^2) + eps) * learned_scale")

    output = model(images, output_attentions=True)
    trace_tensor("vision.final_tokens [B,N,D]", output.last_hidden_state)
    print(f"patch grid (T,H,W): {output.grid_thw}")

    loss = output.last_hidden_state.square().mean()
    trace_tensor("vision.loss", loss)
    loss.backward()
    trace_tensor(
        "vision.patch_embedding.weight.grad",
        model.patch_embedding.projection.weight.grad,
    )
    print("Backward pass complete.")


if __name__ == "__main__":
    main()
