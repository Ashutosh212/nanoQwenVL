"""Shared dimensions only; no vision operations are implemented here."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StudentVisionConfig:
    image_height: int = 700
    image_width: int = 1260
    in_channels: int = 3
    patch_size: int = 14

    # Paper vision width 1280 divided by 20.
    hidden_size: int = 64

    # 16/20 is not a valid head count. A=4 gives head_dim=16, suitable for
    # splitting equally across height and width in axial 2D RoPE.
    num_heads: int = 4
    intermediate_size: int = 192
    depth: int = 2
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0

    @property
    def grid_height(self) -> int:
        return self.image_height // self.patch_size

    @property
    def grid_width(self) -> int:
        return self.image_width // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_height * self.grid_width

    @property
    def patch_vector_size(self) -> int:
        return self.in_channels * self.patch_size * self.patch_size

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads
