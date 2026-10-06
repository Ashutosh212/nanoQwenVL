"""Language exercise 3: one-dimensional rotary position encoding for text.

Unlike vision 2D RoPE, text has one ordered position axis: 0, 1, ..., L-1.

Run independently:
    python -m qwen25_scratch.language_steps.text_rope
"""

import torch
from torch import nn

from qwen25_scratch.language_steps.config import StudentLanguageConfig
from qwen25_scratch.tracing import trace_tensor
from qwen25_scratch.vision_steps.common import ExerciseIncomplete, print_incomplete
from qwen25_scratch.vision_steps.rope_2d import rotate_half


DEMO_TEXT_LENGTH = 5


class TextRotaryEmbedding(nn.Module):
    """Generate text-position cosine and sine values for one attention head."""

    def __init__(self, config: StudentLanguageConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.theta = config.rope_theta

        # TODO A — construct fixed inverse frequencies:
        #   1. exponents = arange(0, head_dim, 2, float32) / head_dim
        #   2. inv_freq = 1 / theta**exponents -> [head_dim/2]
        #   3. register inv_freq as a non-persistent buffer.
        #
        # Default head_dim=24, so inv_freq has shape [12].

        self.exponents = torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim
        inv_freq = 1.0 / (self.theta ** self.exponents)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        sequence_length: int,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin with shape [1,1,L,head_dim].

        TODO B — implement:
          1. Validate sequence_length is positive.
          2. positions = arange(L, device=inv_freq.device, float32) -> [L].
          3. angles_half = positions[:,None] * inv_freq[None,:] -> [L,d/2].
          4. angles = cat(angles_half, angles_half, dim=-1) -> [L,d].
             Duplication aligns frequencies with rotate_half pairs.
          5. cos = cos(angles), sin = sin(angles), cast to requested dtype.
          6. Add singleton batch and head axes -> each [1,1,L,d].
          7. Trace positions, angles, cos, and sin.
          8. Return cos, sin.
        """

        if sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length}")
        
        positions = torch.arange(sequence_length, device=self.inv_freq.device, dtype=torch.float32)  # [L]
        angles_half = positions[:, None] * self.inv_freq[None, :]  # [L, d/2]
        angles = torch.cat([angles_half, angles_half], dim=-1)  # [L, d]
        cos = torch.cos(angles).to(dtype)
        sin = torch.sin(angles).to(dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,L,d]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1,1,L,d]
        return cos, sin


def apply_text_rope(
    queries: torch.Tensor,
    keys: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q and K tensors shaped [B,A,L,d].

    TODO C — for both Q and K:
      1. Save the original dtype.
      2. Convert the working tensors, cos, and sin to float32.
      3. rotated = input * cos + rotate_half(input) * sin.
      4. Restore each original input dtype.
      5. Return rotated queries and keys.

    Values are not rotated; RoPE is applied only to queries and keys.
    """

    original_dtype_queries = queries.dtype
    original_dtype_keys = keys.dtype
    queries = queries.to(torch.float32)
    keys = keys.to(torch.float32)
    cos = cos.to(torch.float32)
    sin = sin.to(torch.float32)

    rotated_queries = queries * cos + rotate_half(queries) * sin
    rotated_keys = keys * cos + rotate_half(keys) * sin

    rotated_queries = rotated_queries.to(original_dtype_queries)
    rotated_keys = rotated_keys.to(original_dtype_keys)

    return rotated_queries, rotated_keys

    raise ExerciseIncomplete("Implement apply_text_rope")


def demo() -> None:
    torch.manual_seed(7)
    config = StudentLanguageConfig()
    batch = 2
    heads = config.num_attention_heads
    length = DEMO_TEXT_LENGTH
    head_dim = config.head_dim
    queries = torch.randn(batch, heads, length, head_dim)
    keys = torch.randn(batch, heads, length, head_dim)

    print("LANGUAGE EXERCISE 3 — TEXT 1D RoPE")
    print(f"Q/K input: [{batch},{heads},{length},{head_dim}]")
    print(f"inv_freq expected: [{head_dim // 2}]")
    print(f"cos/sin expected: [1,1,{length},{head_dim}]")
    print("positions: [0,1,2,3,4]")
    trace_tensor("queries [B,A,L,d]", queries)
    trace_tensor("keys [B,A,L,d]", keys)

    try:
        rope = TextRotaryEmbedding(config)
        cos, sin = rope(length, dtype=queries.dtype)
        rotated_queries, rotated_keys = apply_text_rope(
            queries, keys, cos, sin
        )
    except ExerciseIncomplete as error:
        print_incomplete(error, "language_steps/text_rope.py")
        print("YOUR TURN: complete TODO A, then TODO B, then TODO C.")
        return

    assert rope.inv_freq.shape == (head_dim // 2,)
    assert cos.shape == (1, 1, length, head_dim)
    assert sin.shape == (1, 1, length, head_dim)
    assert rotated_queries.shape == queries.shape
    assert rotated_keys.shape == keys.shape

    # Position zero has angle zero, so RoPE is the identity there.
    torch.testing.assert_close(rotated_queries[:, :, 0], queries[:, :, 0])
    torch.testing.assert_close(rotated_keys[:, :, 0], keys[:, :, 0])

    # Every rotary operation is orthogonal and preserves vector length.
    torch.testing.assert_close(
        rotated_queries.norm(dim=-1),
        queries.norm(dim=-1),
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        rotated_keys.norm(dim=-1),
        keys.norm(dim=-1),
        rtol=1e-5,
        atol=1e-6,
    )

    # A nonzero position should normally be changed by its rotation.
    assert not torch.allclose(rotated_queries[:, :, 1], queries[:, :, 1])

    trace_tensor("cos [1,1,L,d]", cos)
    trace_tensor("sin [1,1,L,d]", sin)
    trace_tensor("rotated queries [B,A,L,d]", rotated_queries)
    print("PASS: shapes, origin identity, position change, and norms are correct.")


if __name__ == "__main__":
    demo()
