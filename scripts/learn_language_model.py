#!/usr/bin/env python
"""Print the small Qwen-style language-model exercises in learning order."""

EXERCISES = (
    (
        "1. Token embedding lookup",
        "qwen25_scratch.language_steps.token_embedding",
    ),
    (
        "2. Causal attention mask",
        "qwen25_scratch.language_steps.causal_mask",
    ),
    (
        "3. One-dimensional text RoPE",
        "qwen25_scratch.language_steps.text_rope",
    ),
    (
        "4. Multi-head causal self-attention",
        "qwen25_scratch.language_steps.multi_head_attention",
    ),
    (
        "5. Qwen-style SwiGLU MLP",
        "qwen25_scratch.language_steps.language_mlp",
    ),
)


def main() -> None:
    print("Small Qwen-style language-model exercises\n")
    for title, module in EXERCISES:
        print(title)
        print(f"  python -m {module}")
    print("\nComplete the exercises in order. The decoder block comes next.")


if __name__ == "__main__":
    main()
