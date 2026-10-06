"""Language exercise 1: convert integer token IDs into learned vectors.

A tokenizer is responsible for text -> integers. This module begins after that
step and learns one hidden vector for every vocabulary ID.

Run independently:
    python -m qwen25_scratch.language_steps.token_embedding
"""

import torch
from torch import nn

from qwen25_scratch.language_steps.config import StudentLanguageConfig
from qwen25_scratch.tracing import trace_tensor
from qwen25_scratch.vision_steps.common import ExerciseIncomplete, print_incomplete


class StudentTokenEmbedding(nn.Module):
    """Lookup table with shape [vocabulary_size, hidden_size]."""

    def __init__(self, config: StudentLanguageConfig) -> None:
        super().__init__()
        self.config = config

        # TODO A — create one learned embedding table:
        #   self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        #
        # Default weight shape: [128,96].
        # Row i is the learned 96-dimensional vector for token ID i.

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)  


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map integer IDs [B,L] to embeddings [B,L,D].

        TODO B — implement these operations:
          1. Validate token_ids has rank 2: [batch, sequence_length].
          2. Validate token_ids.dtype is torch.long.
          3. Validate every ID satisfies 0 <= ID < config.vocab_size.
          4. embeddings = self.embedding(token_ids) -> [B,L,96].
          5. Trace token_ids and embeddings.
          6. Return embeddings.

        Mathematical view:
          output[b, position, :] = embedding.weight[token_ids[b, position], :]

        This is a row lookup, not a projection of token IDs as scalar numbers.
        """

        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must have rank 2, got {token_ids.ndim}")
        if token_ids.dtype != torch.long:
            raise ValueError(f"token_ids must have dtype torch.long, got {token_ids.dtype}")
        
        if not torch.all((token_ids >= 0) & (token_ids < self.config.vocab_size)):
            raise ValueError(f"token_ids must be in range [0, {self.config.vocab_size}), got {token_ids}")

        embedding = self.embedding(token_ids)  # [B,L,D]
        trace_tensor("token_ids [B,L]", token_ids)
        trace_tensor("embeddings [B,L,D]", embedding)
        return embedding


def make_demo_token_ids() -> torch.Tensor:
    """Two explicit sequences; repeated IDs make lookup behavior visible."""
    return torch.tensor(
        [
            [1, 7, 11, 11, 14, 2],
            [1, 19, 14, 17, 11, 2],
        ],
        dtype=torch.long,
    )


def demo() -> None:
    torch.manual_seed(7)
    config = StudentLanguageConfig()
    token_ids = make_demo_token_ids()

    print("LANGUAGE EXERCISE 1 — TOKEN EMBEDDING")
    print("embedding table: [vocab_size,D] = [128,96]")
    print("expected: token IDs [2,6] -> embeddings [2,6,96]")
    print("token IDs:")
    print(token_ids)
    trace_tensor("token IDs [B,L]", token_ids)

    try:
        model = StudentTokenEmbedding(config)
        embeddings = model(token_ids)
    except ExerciseIncomplete as error:
        print_incomplete(error, "language_steps/token_embedding.py")
        print("YOUR TURN: complete TODO A in __init__, then TODO B in forward.")
        return

    assert embeddings.shape == (2, 6, config.hidden_size)

    # The same vocabulary ID must select the same embedding row everywhere.
    torch.testing.assert_close(embeddings[0, 2], embeddings[0, 3])  # token 11
    torch.testing.assert_close(embeddings[0, 2], embeddings[1, 4])  # token 11
    torch.testing.assert_close(embeddings[0, 4], embeddings[1, 2])  # token 14

    loss = embeddings.square().mean()
    loss.backward()
    gradient = model.embedding.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()

    used_ids = torch.unique(token_ids)
    assert torch.count_nonzero(gradient[used_ids]) > 0
    assert torch.count_nonzero(gradient[127]) == 0  # unused row is untouched

    trace_tensor("text embeddings [B,L,D]", embeddings)
    print("scalar loss:", loss.item())
    print("PASS: lookup equality, shape, and embedding gradients are correct.")


if __name__ == "__main__":
    demo()
