# Vision-language merger — independent exercises

The vision transformer produces one vector for every image patch:

```text
vision tokens: [B, Gh*Gw, Dv]
```

Before those vectors can enter the language model, the Qwen2.5-VL-style merger
does two conceptual jobs:

1. combine each neighboring 2x2 block of vision tokens;
2. project each combined vector from vision width into language-model width.

We separate these jobs so every tensor movement remains visible.

## Exercise 1: spatial 2x2 grouping

Run:

```bash
cd /sfs/qwen2.5
source .venv/bin/activate
python -m qwen25_scratch.merger_steps.spatial_merge
```

Open:

```text
src/qwen25_scratch/merger_steps/spatial_merge.py
```

The supplied example uses:

```text
input grid:       4 x 6
input tokens:     [1, 24, 2]
merge size:       2
merged grid:      2 x 3
grouped tokens:   [1, 6, 8]
```

The token count decreases by four while the feature width increases by four:

```text
N_merged = (Gh / 2) * (Gw / 2) = N / 4
D_grouped = 2 * 2 * Dv = 4Dv
```

The first group contains spatial neighbors with IDs `[0, 1, 6, 7]`. It must
not contain the first four flat tokens `[0, 1, 2, 3]`. That is why this
exercise requires both reshape and permutation.

The next exercise implements the learned projection used after grouping.
The complete merger will import RMSNorm separately and then perform:

```text
[B, N/4, 4Dv] -> Linear -> GELU -> Linear -> [B, N/4, D_language]
```

## Exercise 2: learned projection

Open `src/qwen25_scratch/merger_steps/merger_projection.py` and run:

```bash
python -m qwen25_scratch.merger_steps.merger_projection
```

Complete it in two stages:

1. In `__init__`, construct `Linear(256,256)` and `Linear(256,96)`.
2. In `forward`, apply the first linear layer, GELU, and the second linear
   layer, tracing every intermediate tensor.

The exercise validates the equation, output shape, and parameter gradients.
RMSNorm is deliberately not copied into this exercise. During final merger
integration, the existing `StudentRMSNorm(64)` will normalize each ViT token
before `SpatialPatchGrouper` concatenates neighboring tokens.

## Exercise 3: complete merger integration

Open `src/qwen25_scratch/merger_steps/vision_language_merger.py` and run:

```bash
python -m qwen25_scratch.merger_steps.vision_language_merger
```

This exercise supplies a real `56x56` image and runs the completed vision
transformer first. Implement only the integration TODOs:

1. Construct the existing RMSNorm, spatial grouper, and projection modules.
2. Normalize the ViT tokens.
3. Group each 2x2 spatial neighborhood.
4. Project the grouped vectors into language width.
5. Return all intermediate tensors in `VisionLanguageMergerOutput`.

The validator expects:

```text
image                 [1,3,56,56]
ViT output            [1,16,64], grid=(4,4)
normalized tokens     [1,16,64]
grouped tokens        [1,4,256], grid=(2,2)
visual LLM tokens     [1,4,96]
```

It also checks that a scalar loss on the final visual tokens sends gradients
back through both merger linear layers, RMSNorm, and the ViT patch projection.
