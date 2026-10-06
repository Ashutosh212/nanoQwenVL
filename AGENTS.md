# Project instructions

## Purpose

Build a small, readable Qwen2.5-VL-style model from scratch for mathematical
and engineering study. Do not optimize for matching the released model's size.

## Environment

Run `./setup_env.sh`, then use `/sfs/qwen2.5/.venv/bin/python`. Keep this
environment separate from every existing `/sfs/envs/*` project environment.

## Implementation principles

- Implement model mechanics locally instead of wrapping a Transformers model.
- Keep optimized paths and explicit educational paths side by side when useful.
- Every major module should document its tensor shapes and underlying equation.
- Trace mode should expose shape, dtype, device, layout, and gradient flow.
- Add small deterministic tests before adding training-scale optimizations.
- Keep project documentation and the staged roadmap current after changes.

