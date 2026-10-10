"""Shared, dependency-free configuration for the isolated Qwen Image 2.1 runtime."""
from __future__ import annotations

import os
from pathlib import Path

QWEN_IMAGE_21_PROFILE_ID = "qwen-image-2.1"
QWEN_IMAGE_21_TURBO_PROFILE_ID = "qwen-image-2.1-turbo"
QWEN_IMAGE_21_TURBO8_PROFILE_ID = "qwen-image-2.1-turbo8"
QWEN_IMAGE_21_PROVIDER = "comfy.qwen_image21"
# The existing main ComfyUI checkout predates 2.1. Use its separate runtime.
_ROOT = Path("~/soft/ComfyUI").expanduser()
_EXPERIMENT = _ROOT / "experiments/qwen-image-2.1"


def qwen_image21_turbo8_adapters():
    from .image_edit_profiles import ImageEditLoraAdapter

    return (ImageEditLoraAdapter(
        "turbo8", "Turbo8 8-step", os.environ.get(
            "DIFFUSION_EDITOR_QWEN21_TURBO8_LORA",
            str(_ROOT / "models/loras/turbo8_lora_step2500.safetensors")),
    ),)


def qwen_image21_parameters(*, turbo8=False, turbo=False):
    from .image_edit_profiles import (
        ImageEditParameter as P, ParameterChoice as C, ParameterKind as K,
    )

    def path(key, label, default, env):
        return P(key, label, K.STRING, os.environ.get(env, str(default)),
                 "Model", load_time=True)

    accelerated = turbo8 or turbo
    return (
        P("prompt", "Prompt", K.TEXT, "", "Conditioning"),
        P("negative_prompt", "Negative prompt", K.TEXT, "", "Conditioning"),
        P("steps", "Steps", K.INTEGER, 8 if accelerated else 40, "Sampling",
          8 if accelerated else 1, 8 if accelerated else 100, 1),
        P("true_cfg_scale", "CFG scale", K.FLOAT, 1.0, "Sampling",
          1, 1 if accelerated else 20, 0.1, 2),
        P("seed", "Seed", K.INTEGER, -1, "Sampling", -1, 2**32 - 1, 1),
        P("output_mode", "Output", K.CHOICE, "RGBA", "Output",
          choices=(C("RGBA", "Preserve transparency"), C("RGB", "Opaque RGB"))),
        path("model", "Transformer checkpoint",
             _ROOT / "models/diffusion_models" / (
                 "qwen_image_2.1_turbo_bf16.safetensors" if turbo else "qwen_image_2.1_bf16.safetensors"),
             "DIFFUSION_EDITOR_QWEN21_TURBO_MODEL" if turbo else "DIFFUSION_EDITOR_QWEN21_MODEL"),
        path("text_encoder_checkpoint", "Text encoder checkpoint",
             _ROOT / "models/text_encoders/qwen3vl_8b_int8_convrot.safetensors",
             "DIFFUSION_EDITOR_QWEN21_TEXT_ENCODER"),
        path("vae_checkpoint", "VAE checkpoint",
             _ROOT / "models/vae/qwen_image_2.1_vae_bf16.safetensors",
             "DIFFUSION_EDITOR_QWEN21_VAE"),
        path("runtime_root", "Runtime directory", _EXPERIMENT / "ComfyUI",
             "DIFFUSION_EDITOR_QWEN21_ROOT"),
        path("runtime_python", "Runtime Python",
             _EXPERIMENT / ("venv/Scripts/python.exe" if os.name == "nt"
                            else "venv/bin/python"),
             "DIFFUSION_EDITOR_QWEN21_PYTHON"),
        P("device", "Device", K.CHOICE, "cuda", "Runtime",
          choices=(C("cuda", "CUDA"), C("cpu", "CPU")), load_time=True),
        P("dtype", "Dtype", K.CHOICE, "bfloat16", "Runtime",
          choices=(C("bfloat16", "bfloat16"), C("float32", "float32")),
          load_time=True),
        P("memory_mode", "VRAM strategy", K.CHOICE, "auto", "Runtime",
          choices=(C("auto", "Automatic offload"), C("lowvram", "Low VRAM")),
          load_time=True),
        P("vae_tiling", "VAE tiling", K.BOOLEAN, False, "Runtime"),
    )
