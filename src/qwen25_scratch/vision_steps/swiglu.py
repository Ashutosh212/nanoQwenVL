"""Exercise 6: implement the gated vision feed-forward network.

Run independently:
    python -m qwen25_scratch.vision_steps.swiglu
"""

import torch
from torch import nn

from qwen25_scratch.tracing import trace_tensor
from qwen25_scratch.vision_steps.common import ExerciseIncomplete, print_incomplete
from qwen25_scratch.vision_steps.config import StudentVisionConfig

class StudentSwiGLU(nn.Module):
    """SwiGLU(x) = down(SiLU(gate(x)) * up(x))."""

    def __init__(self, config: StudentVisionConfig, *, trace: bool = False) -> None:
        super().__init__()
        self.trace = trace
        self.gate_projection = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_projection = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_projection = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map [B,N,D] -> [B,N,M] -> [B,N,D].

        Pseudocode:
          1. gate_values = gate_projection(x) -> [B,N,M].
          2. activated_gate = SiLU(gate_values).
          3. up_values = up_projection(x) -> [B,N,M].
          4. gated = activated_gate * up_values (elementwise).
          5. output = down_projection(gated) -> [B,N,D].
        Print every intermediate tensor in your first implementation.
        """
        gate_values = self.gate_projection(hidden_states)
        activated_gate = torch.nn.functional.silu(gate_values)
        up_values = self.up_projection(hidden_states)
        gated_values = activated_gate * up_values
        output = self.down_projection(gated_values)

        if self.trace:
            trace_tensor("SwiGLU gate projection [B,N,M]", gate_values)
            trace_tensor("SwiGLU activated gate [B,N,M]", activated_gate)
            trace_tensor("SwiGLU up projection [B,N,M]", up_values)
            trace_tensor("SwiGLU gated values [B,N,M]", gated_values)
            trace_tensor("SwiGLU down projection [B,N,D]", output)

        return output

def demo() -> None:
    torch.manual_seed(7)
    config = StudentVisionConfig()
    hidden_states = torch.randn(1, 6, config.hidden_size)
    print("EXERCISE 6 — SwiGLU")
    print("expected: [1,6,64] -> two [1,6,192] paths -> [1,6,64]")
    trace_tensor("input [B,N,D]", hidden_states)
    try:
        output = StudentSwiGLU(config, trace=True)(hidden_states)
    except ExerciseIncomplete as error:
        print_incomplete(error, "vision_steps/swiglu.py")
        return

    assert output.shape == hidden_states.shape
    output.square().mean().backward()
    trace_tensor("output [B,N,D]", output)
    print("PASS: output shape and backward pass are correct.")


if __name__ == "__main__":
    demo()
