"""Navigation index for the split vision exercises.

The exercises no longer live in this file. Each component is independently
runnable under ``qwen25_scratch.vision_steps`` so its tensors can be studied in
isolation. No vision block or full encoder is assembled yet.
"""

COMPONENT_MODULES = (
    "qwen25_scratch.vision_steps.patch_embedding",
    "qwen25_scratch.vision_steps.spatial_positions",
    "qwen25_scratch.vision_steps.rms_norm",
    "qwen25_scratch.vision_steps.rope_2d",
    "qwen25_scratch.vision_steps.self_attention",
    "qwen25_scratch.vision_steps.swiglu",
)
