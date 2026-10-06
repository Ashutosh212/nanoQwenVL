"""Merger exercise 3: assemble the complete vision-language merger.

This exercise imports the components you already completed:
    StudentRMSNorm -> SpatialPatchGrouper -> MergerProjection

Run independently:
    python -m qwen25_scratch.merger_steps.vision_language_merger
"""

from dataclasses import dataclass

import torch
from torch import nn

from qwen25_scratch.merger_steps.config import StudentMergerConfig
from qwen25_scratch.merger_steps.merger_projection import MergerProjection
from qwen25_scratch.merger_steps.spatial_merge import SpatialPatchGrouper
from qwen25_scratch.tracing import trace_tensor
from qwen25_scratch.vision_steps.common import ExerciseIncomplete, print_incomplete
from qwen25_scratch.vision_steps.config import StudentVisionConfig
from qwen25_scratch.vision_steps.rms_norm import StudentRMSNorm
from qwen25_scratch.vision_steps.vision_transformer import StudentVisionTransformer


@dataclass
class VisionLanguageMergerOutput:
    """Keep intermediate tensors visible while learning the integration."""

    visual_tokens: torch.Tensor  # [B,N_merged,D_language]
    normalized_tokens: torch.Tensor  # [B,N,D_vision]
    grouped_tokens: torch.Tensor  # [B,N_merged,4*D_vision]
    merged_grid_size: tuple[int, int]


class VisionLanguageMerger(nn.Module):
    """Normalize ViT tokens, group 2x2 neighbors, and project to LLM width."""

    def __init__(self, config: StudentMergerConfig) -> None:
        super().__init__()
        self.config = config

        # TODO A — import and construct, do not reimplement:
        #   1. self.norm = StudentRMSNorm(vision_hidden_size)
        #   2. self.grouper = SpatialPatchGrouper(spatial_merge_size)
        #   3. self.projection = MergerProjection(config)

        self.norm = StudentRMSNorm(self.config.vision_hidden_size)
        self.grouper = SpatialPatchGrouper(self.config.spatial_merge_size)
        self.projection = MergerProjection(self.config)

    def forward(
        self,
        vision_tokens: torch.Tensor,
        grid_size: tuple[int, int],
    ) -> VisionLanguageMergerOutput:
        """Map ViT [B,N,64] into visual LLM tokens [B,N/4,96].

        TODO B — implement the integration:
          1. Validate vision_tokens is rank 3: [B,N,D_vision].
          2. Validate its final width equals config.vision_hidden_size.
          3. normalized = self.norm(vision_tokens) -> [B,N,64].
          4. grouped_output = self.grouper(normalized, grid_size).
          5. grouped = grouped_output.grouped_tokens -> [B,N/4,256].
          6. visual = self.projection(grouped) -> [B,N/4,96].
          7. Trace vision_tokens, normalized, grouped, and visual.
          8. Return VisionLanguageMergerOutput(
                 visual,
                 normalized,
                 grouped,
                 grouped_output.merged_grid_size,
             ).
        """

        if vision_tokens.ndim != 3:
            raise ValueError(f"vision_tokens must be rank-3, got {vision_tokens.ndim}")
        if vision_tokens.shape[2] != self.config.vision_hidden_size:
            raise ValueError(
                f"Expected vision_tokens.shape[2]={self.config.vision_hidden_size}, "
                f"got {vision_tokens.shape[2]}"
            )
        
        vision_tokens = self.norm(vision_tokens)
        grouped_output = self.grouper(vision_tokens, grid_size)
        grouped = grouped_output.grouped_tokens
        visual = self.projection(grouped)
        trace_tensor("normalized vision tokens [B,N,D_vision]", vision_tokens)
        trace_tensor("grouped tokens [B,N_merged,256]", grouped)
        trace_tensor("visual tokens [B,N_merged,96]", visual)
        return VisionLanguageMergerOutput(
            visual_tokens=visual,
            normalized_tokens=vision_tokens,
            grouped_tokens=grouped,
            merged_grid_size=grouped_output.merged_grid_size,
        )
        raise ExerciseIncomplete("Implement VisionLanguageMerger.forward")


def make_demo_image() -> torch.Tensor:
    """One deterministic 56x56 RGB image gives a 4x4 patch grid."""
    generator = torch.Generator().manual_seed(7)
    return torch.randn(1, 3, 56, 56, generator=generator)


def demo() -> None:
    torch.manual_seed(7)
    vision_config = StudentVisionConfig(
        image_height=56,
        image_width=56,
        patch_size=14,
        hidden_size=64,
        num_heads=4,
        intermediate_size=192,
        depth=2,
    )
    merger_config = StudentMergerConfig(
        vision_hidden_size=64,
        language_hidden_size=96,
        spatial_merge_size=2,
    )
    image = make_demo_image()
    vision_transformer = StudentVisionTransformer(vision_config)

    print("MERGER EXERCISE 3 — COMPLETE VISION-LANGUAGE MERGER")
    print("expected tensor flow:")
    print("  image [1,3,56,56]")
    print("  -> ViT [1,16,64], grid=(4,4)")
    print("  -> RMSNorm [1,16,64]")
    print("  -> spatial grouping [1,4,256], grid=(2,2)")
    print("  -> projection [1,4,96]")
    trace_tensor("input image [B,C,H,W]", image)

    vision_output = vision_transformer(image)
    trace_tensor("real ViT output [B,N,64]", vision_output.last_hidden_state)
    print("real ViT grid:", vision_output.grid_size)

    try:
        merger = VisionLanguageMerger(merger_config)
        output = merger(
            vision_output.last_hidden_state,
            vision_output.grid_size,
        )
    except ExerciseIncomplete as error:
        print_incomplete(error, "merger_steps/vision_language_merger.py")
        print("YOUR TURN: complete TODO A in __init__, then TODO B in forward.")
        return

    assert vision_output.last_hidden_state.shape == (1, 16, 64)
    assert vision_output.grid_size == (4, 4)
    assert output.normalized_tokens.shape == (1, 16, 64)
    assert output.grouped_tokens.shape == (1, 4, 256)
    assert output.visual_tokens.shape == (1, 4, 96)
    assert output.merged_grid_size == (2, 2)

    loss = output.visual_tokens.square().mean()
    loss.backward()

    gradients = {
        "ViT patch projection": (
            vision_transformer.patch_embedding.projection.weight.grad
        ),
        "merger RMSNorm": merger.norm.weight.grad,
        "merger first linear": merger.projection.input_projection.weight.grad,
        "merger second linear": merger.projection.output_projection.weight.grad,
    }
    for name, gradient in gradients.items():
        assert gradient is not None, f"missing gradient for {name}"
        assert torch.isfinite(gradient).all(), f"non-finite gradient for {name}"

    trace_tensor("final visual LLM tokens [B,N_merged,96]", output.visual_tokens)
    print("scalar loss:", loss.item())
    print("PASS: real ViT output reached the merger and gradients reached the ViT.")


if __name__ == "__main__":
    demo()
