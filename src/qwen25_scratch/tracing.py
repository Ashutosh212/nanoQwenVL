"""Tensor tracing helpers used while learning and debugging the model."""

from __future__ import annotations

import torch


def describe_tensor(name: str, tensor: torch.Tensor, *, values: int = 8) -> str:
    """Return a compact description without changing the autograd graph."""
    flat = tensor.detach().flatten()
    preview = flat[:values].to(device="cpu", dtype=torch.float32).tolist()
    grad_state = "tracks_grad" if tensor.requires_grad else "no_grad"
    return (
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
        f"device={tensor.device} stride={tensor.stride()} {grad_state} "
        f"values={preview}"
    )


def trace_tensor(name: str, tensor: torch.Tensor, *, values: int = 8) -> None:
    """Print shape, storage layout, device, autograd state, and sample values."""
    print(describe_tensor(name, tensor, values=values))

