# Mini Qwen2.5-VL from scratch

This project is an educational, small-scale reimplementation of the important
ideas in Qwen2.5-VL. It is **not** intended to reproduce the parameter count,
training corpus, or benchmark scores of the released models. Every important
operation will have an optional trace mode so tensor shapes, layouts, devices,
values, and gradients can be inspected.

## Scope

The target is an image-and-text model that can be trained end to end on a small
dataset. The implementation will follow the architectural ideas in stages:

1. image loading, batching, normalization, and dynamic resolution;
2. patch embedding, 2D RoPE, RMSNorm, SwiGLU, and vision attention;
3. spatial patch merging and projection into language-token dimensions;
4. a small Qwen2.5-style causal decoder with grouped-query attention and RoPE;
5. replacement of image placeholder tokens with visual embeddings;
6. masked next-token loss, generation, KV caching, training, and fine-tuning;
7. performance engineering: mixed precision, gradient accumulation,
   checkpointing, efficient attention, and distributed training.

## Implemented: vision encoder foundation

The first model component is in `src/qwen25_scratch/vision.py`. It includes:

- Conv3d spatiotemporal patch embedding with still-image frame repetition;
- RMSNorm implemented directly from its equation;
- SwiGLU with separate gate and value projections;
- axial 2D RoPE, splitting each attention head across height and width;
- explicit, non-causal multi-head self-attention;
- pre-normalized residual vision blocks and an encoder stack.

The current path uses equally sized images and full attention. Native
mixed-resolution packing and window attention are the next vision stage. The
vision-language merger is intentionally not part of this module.

## Student implementation path

To implement every tensor operation yourself, follow `LEARNING_VISION.md`.
Each component is a separate runnable module under
`src/qwen25_scratch/vision_steps/`. The components are assembled in
`vision_block.py` and `vision_transformer.py`; the latter also contains the
small classification head used by the CIFAR-10 training script.

## CIFAR-10 training code

The training entry point is `scripts/train_cifar10.py`. It resizes images to
56x56, giving a 4x4 grid of 14x14 patches, and trains the two-block, width-64
student transformer. It saves `latest.pt`, `best.pt`, and `history.json` under
`checkpoints/cifar10/` by default.

Training and downloading happen only when this explicit command is run:

```bash
cd /sfs/qwen2.5
source .venv/bin/activate
python scripts/train_cifar10.py --download
```

Do not use trace mode for full training batches; it is intended for the small
standalone demos.

## Student vision-language merger

The merger exercises under `src/qwen25_scratch/merger_steps/` assemble RMSNorm,
2x2 spatial grouping, and projection from vision width 64 to language width 96.
See `LEARNING_MERGER.md` and run:

```bash
python scripts/learn_vision_language_merger.py
python -m qwen25_scratch.merger_steps.vision_language_merger
```

## Small language model

The independent language-model exercises are under
`src/qwen25_scratch/language_steps/`. Start with learned token embeddings before
adding causal masking, text RoPE, and grouped-query attention. See
`LEARNING_LANGUAGE.md` and run:

```bash
python scripts/learn_language_model.py
python -m qwen25_scratch.language_steps.token_embedding
```

## Environment setup

The environment is project-local and does not modify the existing `qwen_vl`,
`sft`, or other environments under `/sfs/envs`.

```bash
cd /sfs/qwen2.5
./setup_env.sh
source .venv/bin/activate
```

`setup_env.sh` asks `uv` to select the PyTorch backend automatically. On a CPU
worker this produces a CPU build. On an NVIDIA worker, you may select a backend
explicitly when creating/rebuilding the environment, for example:

```bash
QWEN25_TORCH_BACKEND=cu124 ./setup_env.sh
```

The chosen CUDA wheel must be supported by the worker's NVIDIA driver. Check
`nvidia-smi` before choosing it. Do not assume an old worker's CUDA details
still apply.

## Verify the installation

```bash
source /sfs/qwen2.5/.venv/bin/activate
python scripts/check_environment.py
python scripts/inspect_vision_encoder.py
pytest
```

The environment check prints these movements:

```text
images [B,C,H,W]
  -> convolutional patch grid [B,D,H/P,W/P]
  -> flattened patch tokens [B,N,D]
  -> scalar loss
  -> patch-projection gradients
```

## Why the dependency list is small

The runtime deliberately does not depend on `transformers`, `accelerate`, or
`qwen-vl-utils`. Those libraries are useful for production and comparison, but
using them as the implementation would hide the mechanisms this project is
meant to teach. PyTorch provides tensor operations and autograd; model layers,
masking, position encodings, multimodal fusion, training loops, and generation
will live in this repository.

