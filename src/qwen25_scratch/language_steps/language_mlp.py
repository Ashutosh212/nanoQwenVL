"""Language exercise 5: Qwen-style SwiGLU feed-forward network.

Run independently:
    python -m qwen25_scratch.language_steps.language_mlp

The class and validator are provided; you implement the tensor operations.
"""

import torch
from torch import nn

from qwen25_scratch.language_steps.config import StudentLanguageConfig
from qwen25_scratch.tracing import trace_tensor
from qwen25_scratch.vision_steps.common import ExerciseIncomplete, print_incomplete


DEMO_BATCH_SIZE = 2
DEMO_SEQUENCE_LENGTH = 5


class LanguageSwiGLU(nn.Module):
    """SwiGLU(x) = down(SiLU(gate(x)) * up(x))."""

    def __init__(
        self,
        config: StudentLanguageConfig,
        *,
        trace: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.trace = trace

        # TODO A — create three bias-free learned projections:
        #   D = config.hidden_size = 96
        #   M = config.intermediate_size = 256
        #
        #   gate_projection: Linear(D, M, bias=False)
        #   up_projection:   Linear(D, M, bias=False)
        #   down_projection: Linear(M, D, bias=False)
        #
        # Use these exact attribute names so the validator can inspect them.

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map [B,L,D] -> two [B,L,M] paths -> [B,L,D].

        TODO B — implement one operation at a time:
          1. Validate hidden_states is [B,L,D] and D matches the config.
          2. gate_values = gate_projection(hidden_states) -> [B,L,M].
          3. activated_gate = SiLU(gate_values) -> [B,L,M].
             SiLU(z) = z * sigmoid(z).
          4. up_values = up_projection(hidden_states) -> [B,L,M].
          5. gated_values = activated_gate * up_values -> [B,L,M].
             This multiplication is elementwise, not matrix multiplication.
          6. output = down_projection(gated_values) -> [B,L,D].
          7. When self.trace is true, trace all five intermediate tensors.
          8. Return output.
        """

        raise ExerciseIncomplete("Implement the language SwiGLU MLP")


def make_demo_hidden_states() -> torch.Tensor:
    """Return deterministic [B,L,D] inputs for this exercise."""

    generator = torch.Generator().manual_seed(11)
    config = StudentLanguageConfig()
    return torch.randn(
        DEMO_BATCH_SIZE,
        DEMO_SEQUENCE_LENGTH,
        config.hidden_size,
        generator=generator,
    )


def demo() -> None:
    config = StudentLanguageConfig()
    hidden_states = make_demo_hidden_states().requires_grad_(True)

    print("LANGUAGE EXERCISE 5 — SwiGLU MLP")
    print("input:          [B,L,D] = [2,5,96]")
    print("gate/up paths:  [B,L,M] = [2,5,256]")
    print("output:         [B,L,D] = [2,5,96]")
    print("equation: down(SiLU(gate(x)) * up(x))")
    trace_tensor("hidden states [B,L,D]", hidden_states)

    try:
        mlp = LanguageSwiGLU(config, trace=True)
        output = mlp(hidden_states)
    except ExerciseIncomplete as error:
        print_incomplete(error, "language_steps/language_mlp.py")
        print("YOUR TURN: complete TODO A first, then implement TODO B in order.")
        return

    assert output.shape == hidden_states.shape

    expected = mlp.down_projection(
        torch.nn.functional.silu(mlp.gate_projection(hidden_states))
        * mlp.up_projection(hidden_states)
    )
    torch.testing.assert_close(output, expected)

    output.square().mean().backward()
    assert hidden_states.grad is not None
    for name in ("gate_projection", "up_projection", "down_projection"):
        layer = getattr(mlp, name)
        assert layer.weight.grad is not None, f"no gradient reached {name}"

    trace_tensor("output [B,L,D]", output)
    print("PASS: SwiGLU equation, output shape, and gradients are correct.")


if __name__ == "__main__":
    demo()
