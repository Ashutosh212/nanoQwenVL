"""Exercise 4: implement Qwen2.5-VL axial 2D rotary position encoding.

This file supplies every input directly; it does not require patch embedding or
spatial-position code to be finished first.

Run independently:
    python -m qwen25_scratch.vision_steps.rope_2d

Theory
------
For head width d=16, use d/2=8 channels for height and 8 for width. Each axis
needs d/4=4 inverse frequencies. RoPE rotates pairs rather than adding a learned
position vector. Therefore it preserves each Q/K vector's L2 norm while changing
relative dot products between positions.
"""

import torch
from torch import nn

from qwen25_scratch.tracing import trace_tensor
from qwen25_scratch.vision_steps.common import ExerciseIncomplete, print_incomplete


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """Return [-x_second_half, x_first_half] along the last axis.

    Pseudocode:
      1. Split the final dimension into two equal tensors.
      2. Negate the second half.
      3. Concatenate negative-second followed by first.
    """

    final_dim = hidden_states.shape[-1]
    if final_dim % 2 != 0:
        raise ValueError("rotate_half requires an even final dimension")
    x_first_half, x_second_half = torch.split(hidden_states, final_dim // 2, dim=-1)
    x_second_half = -x_second_half
    return torch.cat([x_second_half, x_first_half], dim=-1)


class Axial2DRoPE(nn.Module):
    """Generate height/width cosine and sine rotations for one head."""

    def __init__(self, head_dim: int = 16, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError("head_dim must be divisible by four")
        self.head_dim = head_dim
        self.theta = theta

        # TODO A — constructor pseudocode:
        # spatial_dim = head_dim // 2
        # exponents = arange(0, spatial_dim, 2) / spatial_dim
        # inv_freq = 1 / theta**exponents       shape [head_dim/4]
        # register inv_freq as a non-persistent buffer.

        # This constructor creates the fixed frequencies used to convert integer positions into rotation angles.

        spatial_dim = head_dim // 2
        exponents = torch.arange(0, spatial_dim, 2) / spatial_dim
        inv_freq = 1.0 / (theta ** exponents)
        self.register_buffer("inv_freq", inv_freq, persistent=False)



    def forward(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin with shape [1,1,N,d].

        Input positions are [N,2], where column 0 is height and column 1 width.

        TODO B — pseudocode:
          1. positions[...,None] -> [N,2,1].
          2. Multiply by inv_freq [d/4] -> angles [N,2,d/4].
          3. Separate height_angles and width_angles -> each [N,d/4]. 
          WHY we separate them: ome Q/K dimensions will encode height position, while other dimensions will encode
          4. concatenate(height,width) -> half_angles [N,d/2].
          5. concatenate(half_angles,half_angles) -> angles [N,d].
             The duplication aligns frequencies with rotate_half pairs.
          6. cos(), sin(), cast to requested dtype.
          7. Add singleton batch and head dimensions -> [1,1,N,d].
        """

        positions = positions[..., None]  # [N,2,1]
        angles = positions * self.inv_freq  # [N,2,d/4]
        height_angles, width_angles = angles[:, 0, :], angles[:, 1, :]  # each [N,d/4]
        half_angles = torch.cat([height_angles, width_angles], dim=-1)  # [N,d/2]
        angles = torch.cat([half_angles, half_angles], dim=-1)  # [N,d]
        cos = torch.cos(angles).to(dtype)  # [N,d]
        sin = torch.sin(angles).to(dtype)  # [N,d]
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,N,d]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1,1,N,d]
        return cos, sin
        raise ExerciseIncomplete("Implement Axial2DRoPE.forward")


def apply_2d_rope(
    queries: torch.Tensor,
    keys: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q and K; each input Q/K has shape [B,A,N,d].

    TODO C — for both Q and K:
        rotated = input * cos + rotate_half(input) * sin

    Do the multiplication in float32, then restore the original dtype.
    """

    B, A, N, d = queries.shape
    queries_rotated = queries.float() * cos + rotate_half(queries.float()) * sin
    keys_rotated = keys.float() * cos + rotate_half(keys.float()) * sin
    return queries_rotated.to(queries.dtype), keys_rotated.to(keys.dtype)
    raise ExerciseIncomplete("Implement apply_2d_rope")


def demo() -> None:
    torch.manual_seed(7)
    batch, heads, tokens, head_dim = 1, 4, 6, 16

    # Explicit 2x3 spatial grid; no dependency on SpatialPositionIds.
    positions = torch.tensor(
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]],
        dtype=torch.long,
    )
    queries = torch.randn(batch, heads, tokens, head_dim)
    keys = torch.randn(batch, heads, tokens, head_dim)

    print("EXERCISE 4 — AXIAL 2D RoPE")
    print("positions [N,2] (height, width):")
    print(positions)
    print("expected inv_freq shape: [4]")
    print("expected cos/sin shape: [1,1,6,16]")
    trace_tensor("queries [B,A,N,d]", queries)
    trace_tensor("keys [B,A,N,d]", keys)

    try:
        rope = Axial2DRoPE(head_dim=head_dim)
        cos, sin = rope(positions, queries.dtype)
        rotated_queries, rotated_keys = apply_2d_rope(queries, keys, cos, sin)
    except ExerciseIncomplete as error:
        print_incomplete(error, "vision_steps/rope_2d.py")
        return

    assert cos.shape == (1, 1, tokens, head_dim)
    assert sin.shape == (1, 1, tokens, head_dim)
    assert rotated_queries.shape == queries.shape
    assert rotated_keys.shape == keys.shape

    # Position [0,0] has angle zero, hence cos=1, sin=0 and no rotation.
    torch.testing.assert_close(rotated_queries[:, :, 0], queries[:, :, 0])
    torch.testing.assert_close(rotated_keys[:, :, 0], keys[:, :, 0])

    # Every RoPE operation is orthogonal, so it preserves vector magnitude.
    torch.testing.assert_close(
        rotated_queries.norm(dim=-1), queries.norm(dim=-1), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        rotated_keys.norm(dim=-1), keys.norm(dim=-1), rtol=1e-5, atol=1e-6
    )
    trace_tensor("cos [1,1,N,d]", cos)
    trace_tensor("sin [1,1,N,d]", sin)
    trace_tensor("rotated queries [B,A,N,d]", rotated_queries)
    print("PASS: origin identity, shapes, and norm preservation are correct.")


if __name__ == "__main__":
    demo()
