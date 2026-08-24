#!/usr/bin/env python3
"""Test DiTF-style Qwen features against an exact image translation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion_editor.generation.image_edit_profiles import (
    QWEN_IMAGE_EDIT_PROFILE_ID,
    image_edit_profile,
    qwen_multiple_angles_lora_adapter,
)
from diffusion_editor.workers.ml_backend import RealMlBackend


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dx", type=int, default=32)
    parser.add_argument("--dy", type=int, default=16)
    parser.add_argument("--block", type=int, default=45)
    parser.add_argument("--timestep", type=float, default=0.0)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args()


def _translated(image: Image.Image, dx: int, dy: int) -> Image.Image:
    # Shift the complete raster. This gives exact ground-truth correspondence
    # for every source pixel that remains inside the destination canvas.
    border = image.getpixel((0, 0))
    result = Image.new("RGB", image.size, border)
    result.paste(image, (dx, dy))
    return result


def main() -> int:
    args = _arguments()
    if not args.image.is_file():
        raise SystemExit(f"missing image: {args.image}")
    if not 0.0 <= args.timestep <= 1.0:
        raise SystemExit("timestep must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch

    profile = image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID)
    parameters = profile.defaults()
    # A full-resolution Qwen VAE encode can peak above the VRAM left beside
    # the FP8 transformer.  Tiling changes only how the latent is computed;
    # the feature grid and the exact raster translation stay unchanged.
    parameters["vae_tiling"] = True
    adapters = [
        adapter.to_dict()
        for adapter in (
            *profile.default_lora_adapters,
            qwen_multiple_angles_lora_adapter(),
        )
    ]
    backend = RealMlBackend()
    print("Loading Qwen Multiple Angles...", flush=True)
    loaded = backend.load_image_edit({
        "profile_id": profile.stable_id,
        "parameters": parameters,
        "lora_adapters": adapters,
    })
    pipe = backend._instruct_pipe
    if pipe is None:
        raise RuntimeError("Qwen pipeline did not load")
    # Encode on CPU first.  The FP8 transformer plus both BF16 LoRAs leaves too
    # little VRAM for Qwen's VAE, even when the latter is tiled.
    pipe.remove_all_hooks()
    pipe.to("cpu", silence_dtype_warnings=True)
    pipe.vae.enable_tiling()

    with Image.open(args.image) as source:
        original = source.convert("RGB")
    width, height = original.size
    multiple = pipe.vae_scale_factor * 2
    if width % multiple or height % multiple:
        raise SystemExit(
            f"image size must be divisible by {multiple}: {(width, height)}"
        )
    shifted = _translated(original, args.dx, args.dy)
    original_path = args.output_dir / "original.png"
    shifted_path = args.output_dir / "shifted.png"
    original.save(original_path)
    shifted.save(shifted_path)

    def encode_latent(image: Image.Image, image_index: int) -> torch.Tensor:
        print(f"CPU VAE encode {image_index + 1}/2...", flush=True)
        pixels = pipe.image_processor.preprocess(image, height, width).unsqueeze(2)
        pixels = pixels.to(device="cpu", dtype=pipe.vae.dtype)
        return pipe._encode_vae_image(
            pixels,
            generator=torch.Generator(device="cpu").manual_seed(args.seed),
        ).cpu()

    latents = [
        encode_latent(image, index)
        for index, image in enumerate((original, shifted))
    ]

    # Keep only one transformer submodule resident at a time.  This is slower
    # than the editor's normal path but leaves room for captured activations.
    pipe.enable_sequential_cpu_offload()
    transformer = pipe.transformer
    block = transformer.transformer_blocks[args.block]

    # Text is not a variable in this equivariance control.  A single neutral
    # conditioning token also avoids loading Qwen-VL beside the VAE: Diffusers'
    # component offload chain assumes the regular end-to-end pipeline order
    # and otherwise leaves too little VRAM for a direct VAE call.
    prompt_embeds = torch.zeros(
        (args.ensemble, 1, transformer.config.joint_attention_dim),
        device=pipe._execution_device,
        dtype=transformer.dtype,
    )
    prompt_mask = None

    state: dict[str, torch.Tensor | None] = {
        "mod": None,
        "raw": None,
    }

    def mod_hook(_module, _inputs, output):
        state["mod"] = output.detach()

    def norm2_hook(_module, inputs, _output):
        state["raw"] = inputs[0].detach()

    handles = [
        block.img_mod.register_forward_hook(mod_hook),
        block.img_norm2.register_forward_hook(norm2_hook),
    ]

    def extract(latent: torch.Tensor, image_index: int) -> dict[str, np.ndarray]:
        state["mod"] = None
        state["raw"] = None
        latent = latent.to(
            device=pipe._execution_device,
            dtype=prompt_embeds.dtype,
        )
        latent = latent.repeat(args.ensemble, 1, 1, 1, 1)
        if args.timestep == 0.0:
            noisy = latent
        else:
            noises = []
            for ensemble_index in range(args.ensemble):
                generator = torch.Generator(device=latent.device).manual_seed(
                    args.seed + image_index * 10_000 + ensemble_index
                )
                noises.append(
                    torch.randn(
                        latent[ensemble_index : ensemble_index + 1].shape,
                        generator=generator,
                        device=latent.device,
                        dtype=latent.dtype,
                    )
                )
            noise = torch.cat(noises, dim=0)
            noisy = args.timestep * noise + (1.0 - args.timestep) * latent
        latent_height, latent_width = noisy.shape[3:]
        packed = pipe._pack_latents(
            noisy,
            args.ensemble,
            noisy.shape[1],
            latent_height,
            latent_width,
        )
        grid_height, grid_width = latent_height // 2, latent_width // 2
        img_shapes = [[(1, grid_height, grid_width)]] * args.ensemble
        timestep = torch.full(
            (args.ensemble,),
            args.timestep,
            device=packed.device,
            dtype=packed.dtype,
        )
        del latent, noisy
        print(
            f"Extracting image {image_index + 1}/2 at t={args.timestep:g}, "
            f"ensemble={args.ensemble}...",
            flush=True,
        )
        with torch.inference_mode(), transformer.cache_context("cond"):
            transformer(
                hidden_states=packed,
                timestep=timestep,
                guidance=None,
                encoder_hidden_states_mask=prompt_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                return_dict=False,
            )
        if state["raw"] is None or state["mod"] is None:
            raise RuntimeError("feature hooks did not run")
        raw = state["raw"].float().mean(dim=0)
        _mod1, mod2 = state["mod"].float().chunk(2, dim=-1)
        shift, scale, _gate = mod2.chunk(3, dim=-1)
        result = {
            "raw": raw.reshape(grid_height, grid_width, -1).cpu().numpy(),
            "shift": shift.mean(dim=0).cpu().numpy(),
            "scale": scale.mean(dim=0).cpu().numpy(),
        }
        return result

    try:
        extracted = [extract(latent, index) for index, latent in enumerate(latents)]
    finally:
        for handle in handles:
            handle.remove()
        backend.unload_image_edit()

    feature_paths = []
    for name, arrays in zip(("original", "shifted"), extracted):
        path = args.output_dir / f"features-{name}.npz"
        np.savez_compressed(path, **arrays)
        feature_paths.append(path)
    grid_height, grid_width = extracted[0]["raw"].shape[:2]
    manifest = {
        "profile": profile.stable_id,
        "loaded": loaded,
        "source": str(args.image.resolve()),
        "images": [str(original_path.resolve()), str(shifted_path.resolve())],
        "features": [str(path.resolve()) for path in feature_paths],
        "translation_pixels": [args.dx, args.dy],
        "translation_tokens": [args.dx / 16.0, args.dy / 16.0],
        "block": args.block,
        "timestep": args.timestep,
        "ensemble": args.ensemble,
        "conditioning": "single-zero-token",
        "output_size": [width, height],
        "feature_grid": [grid_width, grid_height],
        "feature_channels": extracted[0]["raw"].shape[-1],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
