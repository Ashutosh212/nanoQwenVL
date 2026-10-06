"""Full student vision transformer assembled from the individual exercises.

Run a synthetic forward/backward inspection (this does not train):
    python -m qwen25_scratch.vision_steps.vision_transformer
"""

from dataclasses import dataclass

import torch
from torch import nn

from qwen25_scratch.tracing import trace_tensor
from qwen25_scratch.vision_steps.config import StudentVisionConfig
from qwen25_scratch.vision_steps.patch_embedding import ManualPatchEmbedding
from qwen25_scratch.vision_steps.rms_norm import StudentRMSNorm
from qwen25_scratch.vision_steps.spatial_positions import SpatialPositionIds
from qwen25_scratch.vision_steps.vision_block import StudentVisionBlock


@dataclass
class StudentVisionOutput:
    """Inspectable outputs from the assembled encoder."""

    last_hidden_state: torch.Tensor  # [B,N,D]
    flattened_patches: torch.Tensor  # [B,N,C*P*P]
    positions: torch.Tensor  # [N,2]
    grid_size: tuple[int, int]
    attentions: tuple[torch.Tensor, ...] | None = None


@dataclass
class VisionClassifierOutput:
    """Classification result plus all encoder outputs for inspection."""

    logits: torch.Tensor  # [B,num_classes]
    pooled_tokens: torch.Tensor  # [B,D]
    vision_output: StudentVisionOutput


class StudentVisionTransformer(nn.Module):
    """Patch embedding -> positions -> repeated transformer blocks."""

    def __init__(self, config: StudentVisionConfig, *, trace: bool = False) -> None:
        super().__init__()
        if config.depth <= 0:
            raise ValueError("depth must be positive")
        if config.hidden_size % config.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if config.head_dim % 4:
            raise ValueError("head_dim must be divisible by four for 2D RoPE")

        self.config = config
        self.trace = trace
        self.patch_embedding = ManualPatchEmbedding(config)
        self.position_ids = SpatialPositionIds()
        self.blocks = nn.ModuleList(
            StudentVisionBlock(config, trace=trace) for _ in range(config.depth)
        )

    def _validate_images(self, images: torch.Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(
                f"images must have shape [B,C,H,W], got {tuple(images.shape)}"
            )
        _, channels, height, width = images.shape
        if channels != self.config.in_channels:
            raise ValueError(
                f"expected {self.config.in_channels} channels, got {channels}"
            )
        if height % self.config.patch_size or width % self.config.patch_size:
            raise ValueError(
                f"height={height} and width={width} must both be divisible by "
                f"patch_size={self.config.patch_size}"
            )

    def forward(
        self, images: torch.Tensor, *, output_attentions: bool = False
    ) -> StudentVisionOutput:
        self._validate_images(images)
        patch_output = self.patch_embedding(images)
        hidden_states = patch_output.tokens
        grid_height, grid_width = patch_output.grid_size
        positions = self.position_ids(
            grid_height, grid_width, hidden_states.device
        )
        all_attentions: list[torch.Tensor] | None = (
            [] if output_attentions else None
        )

        if self.trace:
            trace_tensor("encoder.patch_tokens [B,N,D]", hidden_states)
            trace_tensor("encoder.positions [N,2]", positions)

        for index, block in enumerate(self.blocks):
            hidden_states, probabilities = block(hidden_states, positions)
            if self.trace:
                trace_tensor(
                    f"encoder.block_{index}.output [B,N,D]", hidden_states
                )
            if all_attentions is not None:
                all_attentions.append(probabilities)

        return StudentVisionOutput(
            last_hidden_state=hidden_states,
            flattened_patches=patch_output.flattened_patches,
            positions=positions,
            grid_size=patch_output.grid_size,
            attentions=(
                tuple(all_attentions) if all_attentions is not None else None
            ),
        )


class StudentVisionClassifier(nn.Module):
    """Mean-pool patch tokens and classify them for datasets such as CIFAR-10."""

    def __init__(
        self,
        config: StudentVisionConfig,
        num_classes: int,
        *,
        trace: bool = False,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.trace = trace
        self.encoder = StudentVisionTransformer(config, trace=trace)
        self.final_norm = StudentRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, images: torch.Tensor) -> VisionClassifierOutput:
        vision_output = self.encoder(images)
        normalized_tokens = self.final_norm(vision_output.last_hidden_state)
        pooled_tokens = normalized_tokens.mean(dim=1)
        logits = self.classifier(pooled_tokens)
        if self.trace:
            trace_tensor("classifier.normalized_tokens [B,N,D]", normalized_tokens)
            trace_tensor("classifier.pooled_tokens [B,D]", pooled_tokens)
            trace_tensor("classifier.logits [B,K]", logits)
        return VisionClassifierOutput(logits, pooled_tokens, vision_output)


def demo() -> None:
    torch.manual_seed(7)
    config = StudentVisionConfig(image_height=28, image_width=42, depth=2)
    images = torch.randn(2, 3, 28, 42)
    model = StudentVisionClassifier(config, num_classes=10, trace=True)

    print("INTEGRATION 2 — FULL STUDENT VISION TRANSFORMER")
    print("[2,3,28,42] -> 2x3 grid -> [2,6,64] -> [2,10]")
    output = model(images)
    assert output.vision_output.last_hidden_state.shape == (2, 6, 64)
    assert output.logits.shape == (2, 10)
    output.logits.square().mean().backward()
    print("PASS: full vision transformer forward/backward is correct.")


if __name__ == "__main__":
    demo()
