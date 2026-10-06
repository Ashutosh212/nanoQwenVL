"""Exercise 3: implement RMSNorm from its scalar equation.

Run independently:
    python -m qwen25_scratch.vision_steps.rms_norm
"""

import torch
from torch import nn

from qwen25_scratch.tracing import trace_tensor
from qwen25_scratch.vision_steps.common import ExerciseIncomplete, print_incomplete

class StudentRMSNorm(nn.Module):
    """RMSNorm(x) = weight * x / sqrt(mean(x^2) + epsilon)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize only the final D axis and preserve the input dtype.

        Pseudocode:
          1. Save input dtype; convert a working copy to float32.
          2. square -> [B,N,D].
          3. mean(dim=-1, keepdim=True) -> [B,N,1].
          4. reciprocal square root of variance + epsilon.
          5. multiply x by reciprocal RMS.
          6. cast back to input dtype and multiply learned weight [D].
        """

        hidden_states_dtype = hidden_states.dtype
        hidden_states_float = hidden_states.float()

        rms = torch.sqrt(torch.mean(hidden_states_float ** 2, dim=-1, keepdim=True) + self.eps)
        normalized = hidden_states_float / rms
        output = normalized * self.weight.to(hidden_states_float.device)
        return output.to(hidden_states_dtype)
        
        raise ExerciseIncomplete("Implement StudentRMSNorm.forward")


def demo() -> None:
    values = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [-2.0, -1.0, 1.0, 2.0]]]
    )
    print("EXERCISE 3 — RMSNORM")
    print("equation: y = weight * x / sqrt(mean(x^2) + epsilon)")
    trace_tensor("input [B,N,D]", values)
    try:
        output = StudentRMSNorm(hidden_size=4)(values)
    except ExerciseIncomplete as error:
        print_incomplete(error, "vision_steps/rms_norm.py")
        return

    expected = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(output, expected)
    trace_tensor("normalized [B,N,D]", output)
    print("output feature means (not forced to zero):", output.mean(-1))
    print("output RMS (approximately one):", output.square().mean(-1).sqrt())
    print("PASS: implementation matches the RMSNorm equation.")


if __name__ == "__main__":
    demo()
