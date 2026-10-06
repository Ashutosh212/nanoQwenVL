"""Small dimensions shared by the vision-language merger exercises."""

from dataclasses import dataclass

@dataclass(frozen=True)
class StudentMergerConfig:
    # Output width of the vision transformer.
    vision_hidden_size: int = 64

    # Future small language-model width; intentionally not Qwen's full size.
    language_hidden_size: int = 96

    # Qwen2.5-VL combines each non-overlapping 2x2 spatial token block.
    spatial_merge_size: int = 2

    @property
    def grouped_hidden_size(self) -> int:
        return self.vision_hidden_size * self.spatial_merge_size**2
