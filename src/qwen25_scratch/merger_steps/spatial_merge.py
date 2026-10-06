"""Merger exercise 1: manually group neighboring 2x2 vision tokens.

This exercise contains the inputs, expected ordering, and validation, but leaves
the tensor movement for you to implement.

Run independently:
    python -m qwen25_scratch.merger_steps.spatial_merge
"""

from dataclasses import dataclass

import torch
from torch import nn

from qwen25_scratch.tracing import trace_tensor
from qwen25_scratch.vision_steps.common import ExerciseIncomplete, print_incomplete


@dataclass
class SpatialMergeOutput:
    grouped_tokens: torch.Tensor  # [B,(Gh/m)*(Gw/m),m*m*D]
    merged_grid_size: tuple[int, int]


class SpatialPatchGrouper(nn.Module):
    """Rearrange row-major patch tokens into non-overlapping spatial groups."""

    def __init__(self, merge_size: int = 2) -> None:
        super().__init__()
        if merge_size <= 0:
            raise ValueError("merge_size must be positive")
        self.merge_size = merge_size

    def forward(
        self, tokens: torch.Tensor, grid_size: tuple[int, int]
    ) -> SpatialMergeOutput:
        """Group [B,Gh*Gw,D] into [B,(Gh/m)*(Gw/m),m*m*D].

        Pseudocode—write each operation yourself:

          1. Read B, N, D and unpack grid_size as Gh, Gw.
          2. Validate N == Gh*Gw.
          3. Validate Gh and Gw are divisible by merge_size m.
          4. Reshape row-major tokens:
                 [B, Gh*Gw, D] -> [B, Gh, Gw, D]
          5. Split both spatial axes into groups:
                 [B, Gh, Gw, D]
                 -> [B, Gh/m, m, Gw/m, m, D]
          6. Permute so the two group-grid axes come before the two local axes:
                 [B, Gh/m, m, Gw/m, m, D]
                 -> [B, Gh/m, Gw/m, m, m, D]
          7. Call contiguous(), then flatten the two local axes and D:
                 -> [B, (Gh/m)*(Gw/m), m*m*D]
          8. Return SpatialMergeOutput(grouped, (Gh/m, Gw/m)).

        For a 4x6 grid with m=2, the first group must contain token IDs
        [0, 1, 6, 7]. A simple reshape without the permutation is wrong.
        """

        B, N, D = tokens.shape
        Gh, Gw = grid_size
        m = self.merge_size

        if N != Gh * Gw:
            raise ValueError(f"Expected N={Gh*Gw}, got N={N}")

        if Gh % m != 0 or Gw % m != 0:
            raise ValueError(f"Grid size ({Gh},{Gw}) not divisible by merge_size {m}")
        
        tokens_reshaped = tokens.reshape(B, Gh, Gw, D)
        tokens_split = tokens_reshaped.reshape(B, Gh // m, m, Gw // m, m, D)
        tokens_permuted = tokens_split.permute(0, 1, 3, 2, 4, 5).contiguous()
        grouped_tokens = tokens_permuted.reshape(B, (Gh // m) * (Gw // m), m * m * D)
        return SpatialMergeOutput(grouped_tokens, (Gh // m, Gw // m))

        raise ExerciseIncomplete("Implement SpatialPatchGrouper.forward")


def make_demo_tokens() -> tuple[torch.Tensor, tuple[int, int]]:
    """Create a visible 4x6 token grid with two features per token."""
    grid_height, grid_width = 4, 6
    token_ids = torch.arange(grid_height * grid_width, dtype=torch.float32)
    tokens = torch.stack((token_ids, token_ids + 0.5), dim=-1).unsqueeze(0)
    return tokens, (grid_height, grid_width)


def demo() -> None:
    tokens, grid_size = make_demo_tokens()
    grid_height, grid_width = grid_size

    print("MERGER EXERCISE 1 — SPATIAL 2x2 TOKEN GROUPING")
    print("input tokens: [B,N,D] = [1,24,2]")
    print("input spatial grid: Gh=4, Gw=6")
    print("merge size: m=2")
    print("expected output: [1,6,8], merged grid=(2,3)")
    print("token IDs in their original row-major grid:")
    print(tokens[0, :, 0].reshape(grid_height, grid_width).to(torch.long))
    print("expected group token IDs:")
    expected_group_ids = torch.tensor(
        [
            [0, 1, 6, 7],
            [2, 3, 8, 9],
            [4, 5, 10, 11],
            [12, 13, 18, 19],
            [14, 15, 20, 21],
            [16, 17, 22, 23],
        ]
    )
    print(expected_group_ids)
    trace_tensor("input tokens [B,N,D]", tokens)

    try:
        output = SpatialPatchGrouper(merge_size=2)(tokens, grid_size)
    except ExerciseIncomplete as error:
        print_incomplete(error, "merger_steps/spatial_merge.py")
        print("YOUR TURN: implement only the eight pseudocode steps above.")
        return

    assert output.grouped_tokens.shape == (1, 6, 8)
    assert output.merged_grid_size == (2, 3)

    # Every original token has D=2 features. Selecting every second value from
    # each flattened group recovers its four integer token IDs.
    actual_group_ids = output.grouped_tokens[0, :, 0::2].to(torch.long)
    torch.testing.assert_close(actual_group_ids, expected_group_ids)
    trace_tensor("grouped tokens [B,N_merged,4D]", output.grouped_tokens)
    print("first merged vector:", output.grouped_tokens[0, 0])
    print("PASS: every 2x2 neighborhood is grouped in the correct order.")


if __name__ == "__main__":
    demo()
