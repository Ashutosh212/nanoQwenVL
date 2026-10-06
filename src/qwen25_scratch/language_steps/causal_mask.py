"""Language exercise 2: construct a causal self-attention mask.

Run independently:
    python -m qwen25_scratch.language_steps.causal_mask
"""

from dataclasses import dataclass

import torch
from torch import nn

from qwen25_scratch.vision_steps.common import ExerciseIncomplete, print_incomplete


DEMO_SEQUENCE_LENGTH = 5


@dataclass
class CausalMaskOutput:
    allowed: torch.Tensor  # bool [L,L]; True means attention is permitted
    additive: torch.Tensor  # floating [1,1,L,L]; 0 allowed, -inf blocked


class CausalAttentionMask(nn.Module):
    """Prevent each query position from seeing future key positions."""

    def forward(
        self,
        sequence_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CausalMaskOutput:
        """Return boolean [L,L] and broadcastable additive [1,1,L,L] masks.

        Rows are query positions and columns are key positions.

        For L=5, the boolean mask must be:
            [[1,0,0,0,0],
             [1,1,0,0,0],
             [1,1,1,0,0],
             [1,1,1,1,0],
             [1,1,1,1,1]]

        Pseudocode:
          1. Validate sequence_length is positive.
          2. Validate dtype is a floating-point dtype.
          3. positions = arange(L, device=device) -> [L].
          4. query_positions = positions[:, None] -> [L,1].
          5. key_positions = positions[None, :] -> [1,L].
          6. allowed = key_positions <= query_positions -> bool [L,L].
          7. Create an all-zero [L,L] tensor on the requested device/dtype.
          8. Fill positions where allowed is False with negative infinity.
          9. Add batch and head axes -> additive [1,1,L,L].
         10. Return CausalMaskOutput(allowed, additive).

        Later attention uses:
            masked_scores = scores + additive
            probabilities = softmax(masked_scores, dim=-1)

        The singleton batch/head axes broadcast over scores [B,A,L,L].
        """

        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        if not torch.is_floating_point(torch.empty((), dtype=dtype)):
            raise ValueError(f"dtype must be a floating-point dtype, got {dtype}")
        
        positions = torch.arange(sequence_length, device=device)
        query_positions = positions[:, None]
        key_positions = positions[None, :]

        # print(f"query_positions shape: {query_positions.shape}, key_positions shape: {key_positions.shape}")
        # print(f"query_positions: {query_positions}, key_positions: {key_positions}")

        allowed = key_positions <= query_positions

        additive = torch.zeros((sequence_length, sequence_length), device=device, dtype=dtype)
        additive[~allowed] = float('-inf')
        additive = additive.unsqueeze(0).unsqueeze(0)  # [1,1,L,L]
        additive = additive.to(dtype=dtype)


        return CausalMaskOutput(allowed=allowed, additive= additive)

def demo() -> None:
    torch.manual_seed(7)
    length = DEMO_SEQUENCE_LENGTH
    device = torch.device("cpu")
    dtype = torch.float32

    print("LANGUAGE EXERCISE 2 — CAUSAL ATTENTION MASK")
    print("rows=query positions, columns=key positions")
    print("expected boolean shape: [5,5]")
    print("expected additive shape: [1,1,5,5]")

    try:
        output = CausalAttentionMask()(
            length,
            device=device,
            dtype=dtype,
        )
    except ExerciseIncomplete as error:
        print_incomplete(error, "language_steps/causal_mask.py")
        print("YOUR TURN: implement the ten pseudocode steps in forward.")
        return

    expected_allowed = torch.tril(
        torch.ones(length, length, dtype=torch.bool, device=device)
    )
    assert output.allowed.shape == (length, length)
    assert output.allowed.dtype == torch.bool
    assert output.allowed.device == device
    torch.testing.assert_close(output.allowed, expected_allowed)

    assert output.additive.shape == (1, 1, length, length)
    assert output.additive.dtype == dtype
    assert output.additive.device == device
    additive_2d = output.additive[0, 0]
    torch.testing.assert_close(
        additive_2d[expected_allowed],
        torch.zeros_like(additive_2d[expected_allowed]),
    )
    assert torch.isneginf(additive_2d[~expected_allowed]).all()

    # Demonstrate the actual reason for the additive mask.
    scores = torch.randn(2, 4, length, length)
    masked_scores = scores + output.additive
    probabilities = torch.softmax(masked_scores, dim=-1)
    future = (~expected_allowed)[None, None, :, :].expand_as(probabilities)
    torch.testing.assert_close(
        probabilities[future],
        torch.zeros_like(probabilities[future]),
    )
    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(2, 4, length),
        rtol=1e-5,
        atol=1e-6,
    )

    print("allowed mask (1=visible, 0=future):")
    print(output.allowed.to(torch.int64))
    print("additive mask:")
    print(output.additive[0, 0])
    print("attention probabilities for batch 0, head 0:")
    print(probabilities[0, 0])
    print("PASS: future probabilities are zero and every row sums to one.")


if __name__ == "__main__":
    demo()
