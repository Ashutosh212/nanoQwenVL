"""Exercise 1: move image pixels into 14x14 patch tokens by hand.

Run independently:
    python -m qwen25_scratch.vision_steps.patch_embedding
"""

from dataclasses import dataclass

import torch
from torch import nn

from qwen25_scratch.tracing import trace_tensor
from qwen25_scratch.vision_steps.common import ExerciseIncomplete, print_incomplete
from qwen25_scratch.vision_steps.config import StudentVisionConfig


@dataclass
class PatchOutput:
    flattened_patches: torch.Tensor  # [B,N,C*P*P]
    tokens: torch.Tensor  # [B,N,D]
    grid_size: tuple[int, int]


class ManualPatchEmbedding(nn.Module):
    """Use reshape, permute and indexing—not Conv/Unfold/einops."""

    def __init__(self, config: StudentVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Linear(
            config.patch_vector_size, config.hidden_size, bias=False
        )

    def forward(self, images: torch.Tensor) -> PatchOutput:
        """Transform [B,C,H,W] into raw patches and [B,N,D] tokens.

        Pseudocode:
          1. Read B,C,H,W and validate C, H%P, W%P.
          2. Gh=H//P; Gw=W//P.
          3. reshape [B,C,H,W] -> [B,C,Gh,P,Gw,P].
          4. permute -> [B,Gh,Gw,C,P,P].
          5. Print stride before and after contiguous().
          6. reshape -> [B,Gh*Gw,C*P*P] = [B,4500,588].
          7. projection -> [B,4500,64].
          8. Return PatchOutput(patches, tokens, (Gh,Gw)).
        """

        B, C, H, W = images.shape
        P = self.config.patch_size
        Gh = H // P
        Gw = W // P

        images_reshaped = images.reshape(B, C, Gh, P, Gw, P)
        images_permuted = images_reshaped.permute(0, 2, 4, 1, 3, 5).contiguous()
        flattened_patches = images_permuted.reshape(B, Gh * Gw, C * P * P)
        tokens = self.projection(flattened_patches)
        return PatchOutput(flattened_patches, tokens, (Gh, Gw))
        raise ExerciseIncomplete("Implement ManualPatchEmbedding.forward")


def demo() -> None:
    torch.manual_seed(7)
    config = StudentVisionConfig()
    images = torch.randn(
        1, config.in_channels, config.image_height, config.image_width
    )
    print("EXERCISE 1 — MANUAL PATCH EMBEDDING")
    print("expected: [1,3,700,1260] -> [1,4500,588] -> [1,4500,64]")
    print("grid: 700/14=50 rows, 1260/14=90 columns")
    trace_tensor("input images [B,C,H,W]", images)

    try:
        output = ManualPatchEmbedding(config)(images)
    except ExerciseIncomplete as error:
        print_incomplete(error, "vision_steps/patch_embedding.py")
        return

    assert output.grid_size == (50, 90)
    assert output.flattened_patches.shape == (1, 4500, 588)
    assert output.tokens.shape == (1, 4500, 64)

    # Shape alone cannot detect a wrong permutation, so verify patch ordering.
    expected_first = images[0, :, 0:14, 0:14].reshape(-1)
    expected_second = images[0, :, 0:14, 14:28].reshape(-1)
    expected_last = images[0, :, 686:700, 1246:1260].reshape(-1)
    torch.testing.assert_close(output.flattened_patches[0, 0], expected_first)
    torch.testing.assert_close(output.flattened_patches[0, 1], expected_second)
    torch.testing.assert_close(output.flattened_patches[0, -1], expected_last)
    trace_tensor("raw patches [B,N,588]", output.flattened_patches)
    trace_tensor("projected tokens [B,N,64]", output.tokens)
    print("PASS: shapes and row-major pixel ordering are correct.")


if __name__ == "__main__":
    demo()
