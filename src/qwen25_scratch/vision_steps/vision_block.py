"""A complete pre-normalized vision transformer block.

This file integrates exercises 3, 5, and 6 without copying their math.

Run independently:
    python -m qwen25_scratch.vision_steps.vision_block
"""

import torch
from torch import nn

from qwen25_scratch.tracing import trace_tensor
from qwen25_scratch.vision_steps.config import StudentVisionConfig
from qwen25_scratch.vision_steps.rms_norm import StudentRMSNorm
from qwen25_scratch.vision_steps.self_attention import VisionSelfAttention
from qwen25_scratch.vision_steps.swiglu import StudentSwiGLU


class StudentVisionBlock(nn.Module):
    """Pre-norm attention and SwiGLU, each followed by a residual addition.

    Tensor flow:
        x [B,N,D]
          -> RMSNorm -> attention -> add x
          -> RMSNorm -> SwiGLU    -> add residual
          -> output [B,N,D]
    """

    def __init__(self, config: StudentVisionConfig, *, trace: bool = False) -> None:
        super().__init__()
        self.trace = trace
        self.attention_norm = StudentRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.attention = VisionSelfAttention(config)
        self.mlp_norm = StudentRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = StudentSwiGLU(config, trace=trace)

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return block output [B,N,D] and attention [B,A,N,N]."""
        attention_input = self.attention_norm(hidden_states)
        attention_output, probabilities = self.attention(
            attention_input, positions
        )
        after_attention = hidden_states + attention_output

        mlp_input = self.mlp_norm(after_attention)
        mlp_output = self.mlp(mlp_input)
        output = after_attention + mlp_output

        if self.trace:
            trace_tensor("block.attention_input [B,N,D]", attention_input)
            trace_tensor("block.attention_output [B,N,D]", attention_output)
            trace_tensor("block.after_attention [B,N,D]", after_attention)
            trace_tensor("block.mlp_input [B,N,D]", mlp_input)
            trace_tensor("block.mlp_output [B,N,D]", mlp_output)
            trace_tensor("block.output [B,N,D]", output)
        return output, probabilities


def demo() -> None:
    torch.manual_seed(7)
    config = StudentVisionConfig()
    hidden_states = torch.randn(1, 6, config.hidden_size)
    positions = torch.tensor(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]],
        dtype=torch.long,
    )

    print("INTEGRATION 1 — COMPLETE VISION BLOCK")
    print("input: [1,6,64], positions: [6,2]")
    output, probabilities = StudentVisionBlock(config, trace=True)(
        hidden_states, positions
    )
    assert output.shape == (1, 6, 64)
    assert probabilities.shape == (1, 4, 6, 6)
    output.square().mean().backward()
    trace_tensor("block output [B,N,D]", output)
    print("PASS: residual block shapes and backward pass are correct.")


if __name__ == "__main__":
    demo()
