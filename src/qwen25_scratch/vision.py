"""Readable Qwen2.5-VL-style vision encoder building blocks.

Tensor notation: B=batch, C=channels, T=frames, H/W=image dimensions,
N=patch tokens, D=hidden size, A=attention heads, and d=D/A=head size.

This first implementation deliberately uses full attention and equally sized
images in a batch. Window packing and mixed native resolutions are subsequent
engineering layers; keeping them out initially makes the core math inspectable.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn

from qwen25_scratch.tracing import trace_tensor


@dataclass(frozen=True)
class VisionConfig:
    """Small defaults that preserve the important architectural ratios."""

    in_channels: int = 3
    patch_size: int = 14
    temporal_patch_size: int = 2
    hidden_size: int = 64
    intermediate_size: int = 192
    num_heads: int = 4
    depth: int = 2
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000.0
    trace: bool = False

    def __post_init__(self) -> None:
        positive = {
            "in_channels": self.in_channels,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_heads": self.num_heads,
            "depth": self.depth,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        head_dim = self.hidden_size // self.num_heads
        if head_dim % 4 != 0:
            raise ValueError(
                "vision head_dim must be divisible by 4 so half can encode "
                "height and half can encode width"
            )
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")


@dataclass
class PatchEmbeddingOutput:
    tokens: torch.Tensor
    positions: torch.Tensor
    grid_thw: tuple[int, int, int]


@dataclass
class VisionEncoderOutput:
    last_hidden_state: torch.Tensor
    grid_thw: tuple[int, int, int]
    attentions: tuple[torch.Tensor, ...] | None = None


class RMSNorm(nn.Module):
    """RMSNorm(x) = scale * x / sqrt(mean(x**2) + epsilon).

    Unlike LayerNorm, RMSNorm does not subtract the feature mean. Reduction is
    performed in float32 for numerical stability, matching production practice.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(dtype=input_dtype)


class SwiGLU(nn.Module):
    """SwiGLU(x) = down(SiLU(gate(x)) * up(x))."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(hidden_states))
        values = self.up_proj(hidden_states)
        return self.down_proj(gate * values)


class VisionPatchEmbedding(nn.Module):
    """Turn images or videos into tokens with a non-overlapping Conv3d.

    Images arrive as [B,C,H,W]. A still image is repeated
    ``temporal_patch_size`` times, yielding [B,C,T,H,W]. The Conv3d kernel and
    stride are both [temporal_patch_size, patch_size, patch_size]. This is a
    learned linear map over every flattened patch, efficiently vectorized.
    """

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        kernel = (
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        )
        self.projection = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=kernel,
            stride=kernel,
            bias=False,
        )

    def _as_video(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(2).repeat(
                1, 1, self.config.temporal_patch_size, 1, 1
            )
        elif pixels.ndim != 5:
            raise ValueError(
                "pixels must be [B,C,H,W] or [B,C,T,H,W], "
                f"got shape {tuple(pixels.shape)}"
            )
        return pixels

    def _validate(self, pixels: torch.Tensor) -> None:
        _, channels, frames, height, width = pixels.shape
        if channels != self.config.in_channels:
            raise ValueError(
                f"expected {self.config.in_channels} channels, got {channels}"
            )
        divisors = (
            ("frames", frames, self.config.temporal_patch_size),
            ("height", height, self.config.patch_size),
            ("width", width, self.config.patch_size),
        )
        for name, size, divisor in divisors:
            if size % divisor:
                raise ValueError(
                    f"{name}={size} must be divisible by patch extent {divisor}"
                )

    @staticmethod
    def _spatial_positions(
        temporal: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        # Vision RoPE is spatial: every frame reuses its H/W coordinate grid.
        rows, columns = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        spatial = torch.stack((rows, columns), dim=-1).reshape(-1, 2)
        return spatial.repeat(temporal, 1)

    def forward(self, pixels: torch.Tensor) -> PatchEmbeddingOutput:
        pixels = self._as_video(pixels)
        self._validate(pixels)
        if self.config.trace:
            trace_tensor("vision.pixels_5d [B,C,T,H,W]", pixels)

        grid = self.projection(pixels.to(dtype=self.projection.weight.dtype))
        if self.config.trace:
            trace_tensor("vision.patch_grid [B,D,T',H',W']", grid)

        batch, hidden, temporal, height, width = grid.shape
        tokens = grid.permute(0, 2, 3, 4, 1).reshape(batch, -1, hidden)
        positions = self._spatial_positions(temporal, height, width, grid.device)
        if self.config.trace:
            trace_tensor("vision.patch_tokens [B,N,D]", tokens)
            trace_tensor("vision.positions_hw [N,2]", positions)
        return PatchEmbeddingOutput(tokens, positions, (temporal, height, width))


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """Map [x1,x2] to [-x2,x1], a 90-degree plane rotation."""
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Axial2DRotaryEmbedding(nn.Module):
    """Encode height in half of each head and width in the other half."""

    def __init__(self, head_dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 4:
            raise ValueError("head_dim must be divisible by 4 for axial 2D RoPE")
        spatial_dim = head_dim // 2
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, spatial_dim, 2, dtype=torch.float32) / spatial_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, positions: torch.Tensor, *, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim != 2 or positions.shape[-1] != 2:
            raise ValueError("positions must have shape [N,2] for height and width")
        angles = positions[..., None].float() * self.inv_freq.float()
        height_angles, width_angles = angles.unbind(dim=1)
        half = torch.cat((height_angles, width_angles), dim=-1)
        full = torch.cat((half, half), dim=-1)
        # [1,1,N,d] broadcasts across batch and heads.
        return (
            full.cos().to(dtype=dtype)[None, None, :, :],
            full.sin().to(dtype=dtype)[None, None, :, :],
        )


def apply_2d_rope(
    queries: torch.Tensor,
    keys: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the position-dependent orthogonal rotation to Q and K."""
    query_dtype, key_dtype = queries.dtype, keys.dtype
    queries_float, keys_float = queries.float(), keys.float()
    cos_float, sin_float = cos.float(), sin.float()
    rotated_queries = queries_float * cos_float + rotate_half(queries_float) * sin_float
    rotated_keys = keys_float * cos_float + rotate_half(keys_float) * sin_float
    return rotated_queries.to(query_dtype), rotated_keys.to(key_dtype)


class VisionAttention(nn.Module):
    """Full non-causal attention: softmax(Q K^T / sqrt(d)) V."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.rope = Axial2DRotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, hidden = hidden_states.shape
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(batch, tokens, 3, self.num_heads, self.head_dim)
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        if self.config.trace:
            trace_tensor("vision.queries [B,A,N,d]", queries)
            trace_tensor("vision.keys [B,A,N,d]", keys)
            trace_tensor("vision.values [B,A,N,d]", values)

        cos, sin = self.rope(positions, dtype=queries.dtype)
        queries, keys = apply_2d_rope(queries, keys, cos, sin)
        scores = queries @ keys.transpose(-2, -1) / sqrt(self.head_dim)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
            queries.dtype
        )
        attended = probabilities @ values
        attended = attended.transpose(1, 2).contiguous().view(batch, tokens, hidden)
        output = self.output_projection(attended)
        if self.config.trace:
            trace_tensor("vision.attention_scores [B,A,N,N]", scores)
            trace_tensor("vision.attention_probabilities [B,A,N,N]", probabilities)
            trace_tensor("vision.attention_output [B,N,D]", output)
        return output, probabilities


class VisionBlock(nn.Module):
    """Pre-norm attention and SwiGLU sublayers with residual connections."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = VisionAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(
        self, hidden_states: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attention_output, probabilities = self.attention(
            self.norm1(hidden_states), positions
        )
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states, probabilities


class VisionEncoder(nn.Module):
    """Patch embedding followed by a stack of Qwen2.5-VL-style ViT blocks."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedding = VisionPatchEmbedding(config)
        self.blocks = nn.ModuleList(VisionBlock(config) for _ in range(config.depth))

    def forward(
        self, pixels: torch.Tensor, *, output_attentions: bool = False
    ) -> VisionEncoderOutput:
        patch_output = self.patch_embedding(pixels)
        hidden_states = patch_output.tokens
        all_attentions: list[torch.Tensor] | None = [] if output_attentions else None

        for index, block in enumerate(self.blocks):
            hidden_states, attention = block(hidden_states, patch_output.positions)
            if self.config.trace:
                trace_tensor(f"vision.block_{index}.output [B,N,D]", hidden_states)
            if all_attentions is not None:
                all_attentions.append(attention)

        return VisionEncoderOutput(
            last_hidden_state=hidden_states,
            grid_thw=patch_output.grid_thw,
            attentions=tuple(all_attentions) if all_attentions is not None else None,
        )
