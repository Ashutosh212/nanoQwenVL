"""Merger exercise 2: project grouped vision tokens into the LLM width.

Run independently after completing spatial_merge.py:
    python -m qwen25_scratch.merger_steps.merger_projection
"""

import torch
import torch.nn.functional as F
from torch import nn

from qwen25_scratch.merger_steps.config import StudentMergerConfig
from qwen25_scratch.tracing import trace_tensor
from qwen25_scratch.vision_steps.common import ExerciseIncomplete, print_incomplete


class MergerProjection(nn.Module):
    """MLP: grouped width -> grouped width -> language-model width."""

    def __init__(self, config: StudentMergerConfig) -> None:
        super().__init__()
        self.config = config

        # TODO A — create the learned layers:
        #   1. input_projection: Linear(grouped_hidden_size, grouped_hidden_size)
        #   2. output_projection: Linear(grouped_hidden_size, language_hidden_size)
        #
        # For the default learning model these are Linear(256,256) and
        # Linear(256,96). GELU has no learned parameters, so you may call
        # torch.nn.functional.gelu directly in forward.

        self.input_projection = nn.Linear(config.grouped_hidden_size, config.grouped_hidden_size)
        self.output_projection = nn.Linear(config.grouped_hidden_size, config.language_hidden_size)


    def forward(self, grouped_tokens: torch.Tensor) -> torch.Tensor:
        """Map [B,N_merged,256] to [B,N_merged,96].

        TODO B — implement and trace these operations:
          1. Validate grouped_tokens is a rank-3 [B,N,D] tensor.
          2. Validate D == self.config.grouped_hidden_size.
          3. hidden = input_projection(grouped_tokens) -> [B,N,256].
          4. activated = GELU(hidden) -> [B,N,256].
          5. output = output_projection(activated) -> [B,N,96].
          6. Trace grouped_tokens, hidden, activated, and output.
          7. Return output.

        GELU equation (exact form used by F.gelu):
            GELU(x) = x * Phi(x)
        where Phi is the standard normal cumulative distribution function.
        """

        if grouped_tokens.ndim != 3:
            raise ValueError(f"grouped_tokens must be rank-3, got {grouped_tokens.ndim}")
        if grouped_tokens.shape[2] != self.config.grouped_hidden_size:
            raise ValueError(
                f"Expected grouped_tokens.shape[2]={self.config.grouped_hidden_size}, "
                f"got {grouped_tokens.shape[2]}"
            )
        
        hidden = self.input_projection(grouped_tokens)
        activated = F.gelu(hidden)
        output = self.output_projection(activated)
        return output

        trace_tensor("projection input [B,N,256]", grouped_tokens)
        trace_tensor("first linear [B,N,256]", hidden)
        trace_tensor("after GELU [B,N,256]", activated)
        trace_tensor("projection output [B,N,96]", output)
        
        raise ExerciseIncomplete("Implement MergerProjection.forward")


def make_demo_grouped_tokens() -> torch.Tensor:
    """Create three deterministic grouped tokens with width 256."""
    values = torch.linspace(-1.0, 1.0, steps=3 * 256)
    return values.reshape(1, 3, 256)


def demo() -> None:
    torch.manual_seed(7)
    config = StudentMergerConfig()
    grouped_tokens = make_demo_grouped_tokens()

    print("MERGER EXERCISE 2 — LEARNED PROJECTION")
    print("expected tensor flow:")
    print("  [1,3,256] -> Linear(256,256) -> GELU -> Linear(256,96)")
    trace_tensor("grouped tokens [B,N_merged,256]", grouped_tokens)

    try:
        projection = MergerProjection(config)
        output = projection(grouped_tokens)
    except ExerciseIncomplete as error:
        print_incomplete(error, "merger_steps/merger_projection.py")
        print("YOUR TURN: complete TODO A in __init__, then TODO B in forward.")
        return

    assert output.shape == (1, 3, config.language_hidden_size)
    expected = projection.output_projection(
        F.gelu(projection.input_projection(grouped_tokens))
    )
    torch.testing.assert_close(output, expected)

    output.square().mean().backward()
    for name, parameter in projection.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"

    trace_tensor("projected visual tokens [B,N_merged,96]", output)
    print("PASS: projection equation, output shape, and gradients are correct.")


if __name__ == "__main__":
    demo()
