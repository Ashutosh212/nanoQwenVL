# Vision encoder — independent student exercises

Each component is a separate importable Python module. Every module supplies
its own small input tensors and validator, so you can learn it without having
completed earlier components.

List the exercises:

```bash
cd /sfs/qwen2.5
source .venv/bin/activate
python scripts/learn_vision_encoder.py
```

Run one component at a time:

```bash
python -m qwen25_scratch.vision_steps.patch_embedding
python -m qwen25_scratch.vision_steps.spatial_positions
python -m qwen25_scratch.vision_steps.rms_norm
python -m qwen25_scratch.vision_steps.rope_2d
python -m qwen25_scratch.vision_steps.self_attention
python -m qwen25_scratch.vision_steps.swiglu
python -m qwen25_scratch.vision_steps.vision_block
python -m qwen25_scratch.vision_steps.vision_transformer
```

## Recommended order

1. `patch_embedding.py`: `[1,3,700,1260]` to 4,500 flattened patches and
   64-dimensional projected tokens.
2. `spatial_positions.py`: row/column IDs on a tiny visible grid.
3. `rms_norm.py`: one reduction along the hidden dimension.
4. `rope_2d.py`: fully self-contained position IDs, Q, K, frequencies,
   cosine/sine construction, rotate-half, and norm-preservation checks.
5. `self_attention.py`: QKV reshaping and explicit attention on only six tokens.
6. `swiglu.py`: separate gate/value paths and elementwise gating.
7. `vision_block.py`: pre-norm attention and MLP residual paths.
8. `vision_transformer.py`: patching, positions, stacked blocks, pooling, and
   classification logits.

The files keep their pseudocode beside the current implementations. To practise
again, replace one implementation at a time and rerun only that module.

## Current implementation status

Exercises 1–6 remain independently runnable and can be reimplemented one at a time.

## Patching dimensions

```text
images: [B,C,H,W] = [1,3,700,1260]
Gh = 700/14 = 50
Gw = 1260/14 = 90
N  = 50*90 = 4,500
patch width = 3*14*14 = 588
D = 1280/20 = 64
```

## Attention memory warning

Do not begin attention with all 4,500 tokens. Four attention heads create
`4*4500*4500 = 81,000,000` score values—about 324 MB in float32 before
gradients or other activations. The attention exercise uses six tokens. Later,
window attention will make native-resolution computation practical.

## Final assembly

`vision_block.py` combines RMSNorm, attention, SwiGLU, and residual connections.
`vision_transformer.py` combines patching, spatial IDs, and a configurable stack
of blocks. Its optional mean-pooling classifier makes the encoder trainable on
CIFAR-10 while keeping the merger and language model as later stages.
