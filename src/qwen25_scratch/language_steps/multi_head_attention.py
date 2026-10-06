"""Language exercise 4: standard multi-head causal self-attention (MHA).

Run independently:
    python -m qwen25_scratch.language_steps.multi_head_attention

This file provides the structure and validator while leaving the attention
tensor operations for you to implement.
"""

from dataclasses import dataclass
from math import sqrt

import torch
from torch import nn

from qwen25_scratch.language_steps.causal_mask import CausalAttentionMask
from qwen25_scratch.language_steps.config import StudentLanguageConfig
from qwen25_scratch.language_steps.text_rope import (
    TextRotaryEmbedding,
    apply_text_rope,
)
from qwen25_scratch.tracing import trace_tensor
from qwen25_scratch.vision_steps.common import ExerciseIncomplete, print_incomplete


DEMO_BATCH_SIZE = 2
DEMO_SEQUENCE_LENGTH = 5


@dataclass
class CausalAttentionOutput:
    """The transformed tokens and inspectable attention probabilities."""

    hidden_states: torch.Tensor  # [B,L,D]
    probabilities: torch.Tensor  # [B,A,L,L]


class MultiHeadCausalSelfAttention(nn.Module):
    """Causal attention where Q, K, and V all have the same head count."""

    def __init__(
        self,
        config: StudentLanguageConfig,
        *,
        trace: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.trace = trace

        # TODO A — create the learned layers and helpers:
        #   D = config.hidden_size
        #   A = config.num_attention_heads
        #   d = config.head_dim
        #
        #   query_projection: Linear(D, A*d, bias=False)  -> 96 to 96
        #   key_projection:   Linear(D, A*d, bias=False)  -> 96 to 96
        #   value_projection: Linear(D, A*d, bias=False)  -> 96 to 96
        #   output_projection: Linear(A*d, D, bias=False) -> 96 to 96
        #   rope: TextRotaryEmbedding(config)
        #   causal_mask: CausalAttentionMask()
        #   scale: 1 / sqrt(d)
        #
        # Use these exact attribute names so the validator can inspect them.

        self.D = config.hidden_size
        self.A = config.num_attention_heads
        self.d = config.head_dim

        self.query_projection = nn.Linear(self.D, self.A * self.d, bias=False)
        self.key_projection = nn.Linear(self.D, self.A * self.d, bias=False)
        self.value_projection = nn.Linear(self.D, self.A * self.d, bias=False)
        self.output_projection = nn.Linear(self.A * self.d, self.D, bias=False)
        self.rope = TextRotaryEmbedding(config)
        self.causal_mask = CausalAttentionMask()
        self.scale = 1 / sqrt(self.d)


    def forward(self, hidden_states: torch.Tensor) -> CausalAttentionOutput:
        """Transform hidden states shaped [B,L,D] without seeing the future.

        TODO B — implement one tensor movement at a time:
          1. Validate hidden_states is [B,L,D] and D matches the config.
          2. Project Q, K, and V. Each has shape [B,L,A*d].
          3. Reshape and transpose:
               Q, K, V: [B,L,A*d] -> [B,A,L,d].
          4. Generate cos/sin and apply text RoPE to Q and K only.
          5. scores = (Q @ K.transpose(-2,-1)) * scale -> [B,A,L,L].
          6. Add causal_mask(L,...).additive -> future scores become -inf.
          7. probabilities = softmax(scores, dim=-1). For numerical safety,
             compute softmax in float32, then cast to V's dtype.
          8. context = probabilities @ V -> [B,A,L,d].
          9. Transpose to [B,L,A,d], call contiguous(), then reshape to
             [B,L,A*d].
         10. Apply output_projection -> [B,L,D].
         11. When self.trace is true, trace projected Q/K/V, split heads,
             scores, probabilities, context, and final output.
         12. Return CausalAttentionOutput(output, probabilities).
        """

        B, L, D = hidden_states.shape
        if D != self.D:
            raise ValueError(f"hidden_states last dim {D} != config.hidden_size {self.D}")
        
        q = self.query_projection(hidden_states)  # [B,L,A*d]
        k = self.key_projection(hidden_states)    # [B,L,A*d]
        v = self.value_projection(hidden_states)  # [B,L,A*d]

        q = q.view(B, L, self.A, self.d).transpose(1, 2)  # [B,A,L,d]
        k = k.view(B, L, self.A, self.d).transpose(1, 2)  # [B,A,L,d]
        v = v.view(B, L, self.A, self.d).transpose(1, 2)  # [B,A,L,d]

        cos, sin = self.rope(L, dtype=hidden_states.dtype)

        q, k = apply_text_rope(q, k, cos, sin)

        causal_mask_output = self.causal_mask(L, device=hidden_states.device, dtype=hidden_states.dtype)

        p = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,A,L,L]
        p = p + causal_mask_output.additive  # [B,A,L,L]    

        context = torch.matmul(torch.softmax(p, dim=-1), v)  # [B,A,L,d]

        context = context.transpose(1, 2).contiguous().view(B, L, self.A * self.d)  # [B,L,A*d]

        output = self.output_projection(context)  # [B,L,D]

        if self.trace:
            trace_tensor("projected Q [B,L,A*d]", q)
            trace_tensor("projected K [B,L,A*d]", k)
            trace_tensor("projected V [B,L,A*d]", v)
            trace_tensor("attention scores [B,A,L,L]", p)
            trace_tensor("attention probabilities [B,A,L,L]", torch.softmax(p, dim=-1))
            trace_tensor("context [B,A,L,d]", context)
            trace_tensor("output [B,L,D]", output)
        
        return CausalAttentionOutput(hidden_states=output, probabilities=torch.softmax(p, dim=-1))



        raise ExerciseIncomplete("Implement multi-head causal self-attention")


def make_demo_hidden_states() -> torch.Tensor:
    """Return deterministic [B,L,D] inputs for this exercise."""

    generator = torch.Generator().manual_seed(7)
    config = StudentLanguageConfig()
    return torch.randn(
        DEMO_BATCH_SIZE,
        DEMO_SEQUENCE_LENGTH,
        config.hidden_size,
        generator=generator,
    )

def demo() -> None:
    config = StudentLanguageConfig()
    hidden_states = make_demo_hidden_states().requires_grad_(True)

    print("LANGUAGE EXERCISE 4 — MULTI-HEAD CAUSAL SELF-ATTENTION")
    print("input:                 [B,L,D]       = [2,5,96]")
    print("Q/K/V projections:     [B,L,A*d]     = [2,5,96]")
    print("Q/K/V after split:     [B,A,L,d]     = [2,4,5,24]")
    print("attention scores:      [B,A,L,L]     = [2,4,5,5]")
    print("output:                 [B,L,D]       = [2,5,96]")
    trace_tensor("hidden states [B,L,D]", hidden_states)

    try:
        attention = MultiHeadCausalSelfAttention(config, trace=True)
        result = attention(hidden_states)
    except ExerciseIncomplete as error:
        print_incomplete(error, "language_steps/multi_head_attention.py")
        print("YOUR TURN: complete TODO A first, then implement TODO B in order.")
        return

    assert result.hidden_states.shape == (2, 5, 96)
    assert result.probabilities.shape == (2, 4, 5, 5)

    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert torch.count_nonzero(result.probabilities[..., future]) == 0
    torch.testing.assert_close(
        result.probabilities.sum(dim=-1),
        torch.ones_like(result.probabilities.sum(dim=-1)),
    )

    result.hidden_states.square().mean().backward()
    for name in (
        "query_projection",
        "key_projection",
        "value_projection",
        "output_projection",
    ):
        layer = getattr(attention, name)
        assert layer.weight.grad is not None, f"no gradient reached {name}"

    # Changing the final token must not change outputs at earlier positions.
    attention.trace = False
    changed = hidden_states.detach().clone()
    changed[:, -1] += 10.0
    original_output = attention(hidden_states.detach()).hidden_states
    changed_output = attention(changed).hidden_states
    torch.testing.assert_close(
        original_output[:, :-1], changed_output[:, :-1], rtol=1e-5, atol=1e-6
    )

    print("PASS: MHA shapes, causal probabilities, gradients, and causality are correct.")


if __name__ == "__main__":
    demo()
