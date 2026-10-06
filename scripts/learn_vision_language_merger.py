#!/usr/bin/env python
"""Print the vision-language merger exercises in learning order."""

EXERCISES = (
    (
        "1. Spatial 2x2 token grouping",
        "qwen25_scratch.merger_steps.spatial_merge",
    ),
    (
        "2. Learned projection into language width",
        "qwen25_scratch.merger_steps.merger_projection",
    ),
    (
        "3. Complete merger with real ViT output",
        "qwen25_scratch.merger_steps.vision_language_merger",
    ),
)


def main() -> None:
    print("Qwen2.5-VL vision-language merger exercises\n")
    for title, module in EXERCISES:
        print(title)
        print(f"  python -m {module}")
    print("\nComplete the exercises in order. The language model comes next.")


if __name__ == "__main__":
    main()
