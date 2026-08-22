"""Standalone SenseNova-U1 image-edit adapter for the isolated ML worker."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image


_PATCH_SIZE = 32


def _ensure_provider_overlay() -> None:
    """Expose the provider isolated from incompatible upstream metadata pins."""

    configured = os.environ.get(
        "DIFFUSION_EDITOR_SENSENOVA_PYTHONPATH", "").strip()
    if configured:
        candidate = Path(configured).expanduser().absolute()
    else:
        candidate = (
            Path(sys.prefix).absolute()
            / "share" / "diffusion-editor" / "sensenova-u1"
        )
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _import_runtime():
    _ensure_provider_overlay()
    try:
        import sensenova_u1
        from sensenova_u1.models.neo_unify.utils import smart_resize
        from sensenova_u1.utils import (
            load_model_and_tokenizer,
            make_offload_ctx,
            vram_mode_keeps_generation_resident,
            vram_mode_to_prefetch_count,
        )
    except ImportError as exc:
        raise RuntimeError(
            "SenseNova provider is not installed for the ML worker. "
            "Run ./setup-workers.sh."
        ) from exc
    return (
        sensenova_u1,
        smart_resize,
        load_model_and_tokenizer,
        make_offload_ctx,
        vram_mode_keeps_generation_resident,
        vram_mode_to_prefetch_count,
    )


def _tensor_to_image(batch) -> Image.Image:
    if batch.ndim == 3:
        batch = batch.unsqueeze(0)
    pixels = (
        (batch[0].float() * 0.5 + 0.5)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(
        np.rint(pixels * 255.0).astype(np.uint8), "RGB")


class SenseNovaImageEditPipeline:
    """Own a loaded U1 model/tokenizer and expose image editing."""

    scheduler = None

    def __init__(
            self,
            *,
            model_path: str,
            gguf_checkpoint: str,
            device: str,
            dtype: str,
            attention_backend: str,
            vram_mode: str,
            fast_vram_fraction: float,
            fast_vram_headroom_gib: float,
            fast_activation_reserve_gib: float,
            fast_vram_budget_gib: float,
    ) -> None:
        import torch

        (
            sensenova_u1,
            self._smart_resize,
            load_model_and_tokenizer,
            self._make_offload_ctx,
            self._keeps_generation_resident,
            vram_mode_to_prefetch_count,
        ) = _import_runtime()
        if not gguf_checkpoint:
            raise ValueError("SenseNova GGUF checkpoint path cannot be empty")
        checkpoint = Path(gguf_checkpoint).expanduser().absolute()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"SenseNova GGUF checkpoint not found: {checkpoint}")
        model_candidate = Path(model_path).expanduser()
        if model_candidate.exists():
            model_path = str(model_candidate.absolute())
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        sensenova_u1.set_attn_backend(attention_backend)
        self.model_path = model_path
        self.gguf_checkpoint = str(checkpoint)
        self.device = device
        self.dtype = dtype
        self.attention_backend = attention_backend
        self.effective_attention_backend = (
            sensenova_u1.effective_attn_backend())
        self.vram_mode = vram_mode
        self.prefetch_count = int(vram_mode_to_prefetch_count(vram_mode))
        self.fast_vram_fraction = float(fast_vram_fraction)
        self.fast_vram_headroom_gib = float(fast_vram_headroom_gib)
        self.fast_activation_reserve_gib = float(
            fast_activation_reserve_gib)
        self.fast_vram_budget_gib = (
            float(fast_vram_budget_gib) or None)
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_path,
            dtype=torch_dtype,
            device=device,
            gguf_checkpoint=self.gguf_checkpoint,
            for_offload=self.prefetch_count > 0,
        )

    def _offload_context(self):
        return self._make_offload_ctx(
            self.model,
            self.prefetch_count,
            self.device,
            keep_generation_resident=self._keeps_generation_resident(
                self.vram_mode),
            fast_vram_fraction=self.fast_vram_fraction,
            fast_vram_headroom_gib=self.fast_vram_headroom_gib,
            fast_activation_reserve_gib=self.fast_activation_reserve_gib,
            fast_vram_budget_gib=self.fast_vram_budget_gib,
        )

    def edit(
            self,
            image: Image.Image,
            parameters: dict[str, Any],
            seed: int,
            *,
            reference_image: Image.Image | None = None,
    ) -> Image.Image:
        import torch

        prompt = str(parameters["prompt"]).strip()
        if not prompt:
            raise ValueError("SenseNova image edit prompt cannot be empty")
        start = float(parameters["cfg_interval_start"])
        end = float(parameters["cfg_interval_end"])
        if not 0.0 <= start <= end <= 1.0:
            raise ValueError(
                "SenseNova CFG interval must satisfy 0 <= start <= end <= 1")
        target_pixels = max(
            _PATCH_SIZE * _PATCH_SIZE,
            math.floor(float(parameters["target_megapixels"]) * 1_000_000),
        )
        inputs = [image]
        if reference_image is not None:
            inputs.append(reference_image)
        sources = []
        for input_image in inputs:
            source = input_image.convert("RGB")
            resized_height, resized_width = self._smart_resize(
                height=source.height,
                width=source.width,
                factor=_PATCH_SIZE,
                min_pixels=target_pixels,
                max_pixels=target_pixels,
            )
            sources.append(source.resize(
                (resized_width, resized_height),
                Image.Resampling.LANCZOS,
            ))
        source = sources[0]
        width = int(parameters["width"])
        height = int(parameters["height"])
        if bool(width) != bool(height):
            raise ValueError(
                "SenseNova width and height must both be zero or both set")
        if width == 0:
            # Match the official terminal/Comfy path: the source is first
            # snapped to the pixel budget, then the requested output size is
            # resolved from that snapped image.  The second grid rounding is
            # observable for non-square inputs (e.g. 1248x832 at 1 MP).
            output_height, output_width = self._smart_resize(
                height=source.height,
                width=source.width,
                factor=_PATCH_SIZE,
                min_pixels=target_pixels,
                max_pixels=target_pixels,
            )
            width, height = output_width, output_height
        if width % _PATCH_SIZE or height % _PATCH_SIZE:
            raise ValueError(
                "SenseNova output width and height must be multiples of 32")
        with torch.inference_mode(), self._offload_context() as model:
            output = model.it2i_generate(
                self.tokenizer,
                prompt,
                sources,
                image_size=(width, height),
                cfg_scale=float(parameters["cfg_scale"]),
                img_cfg_scale=float(parameters["img_cfg_scale"]),
                cfg_norm=str(parameters["cfg_norm"]),
                timestep_shift=float(parameters["timestep_shift"]),
                cfg_interval=(start, end),
                num_steps=int(parameters["steps"]),
                batch_size=1,
                think_mode=False,
                seed=seed,
            )
        return _tensor_to_image(output)

    @property
    def runtime_info(self) -> dict[str, Any]:
        return {
            "attention_backend": self.attention_backend,
            "effective_attention_backend": self.effective_attention_backend,
            "vram_mode": self.vram_mode,
            "prefetch_count": self.prefetch_count,
            "gguf_checkpoint": self.gguf_checkpoint,
        }
