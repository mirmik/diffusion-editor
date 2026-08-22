#!/usr/bin/env python3
"""Capture full-dimensional Qwen features from several blocks across a turn."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion_editor.generation.image_edit_profiles import (
    QWEN_MULTIPLE_ANGLES_PROFILE_ID,
    image_edit_profile,
)
from diffusion_editor.workers.ml_backend import RealMlBackend


AZIMUTHS = {
    0: "front view",
    45: "front-right quarter view",
    90: "right side view",
    135: "back-right quarter view",
    180: "back view",
    225: "back-left quarter view",
    270: "left side view",
    315: "front-left quarter view",
}

ELEVATIONS = {
    "low": (-30, "low-angle shot"),
    "eye": (0, "eye-level shot"),
    "elevated": (30, "elevated shot"),
    "high": (60, "high-angle shot"),
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--back", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--elevation", choices=tuple(ELEVATIONS), default="elevated")
    parser.add_argument("--azimuths", default="45,90")
    parser.add_argument("--blocks", default="15,30,45,59")
    parser.add_argument("--feature-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--steps", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    for path in (args.front, args.back):
        if not path.is_file():
            raise SystemExit(f"missing input image: {path}")
    azimuths = [int(value.strip()) for value in args.azimuths.split(",")]
    invalid_azimuths = [value for value in azimuths if value not in AZIMUTHS]
    if invalid_azimuths:
        raise SystemExit(f"invalid azimuths: {invalid_azimuths}")
    blocks = None
    if args.blocks.strip().lower() != "all":
        blocks = [int(value.strip()) for value in args.blocks.split(",")]
        if len(set(blocks)) != len(blocks):
            raise SystemExit("block list contains duplicates")
    if not 0 <= args.feature_step < args.steps:
        raise SystemExit("feature-step must be inside the denoising schedule")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        calculate_dimensions,
    )

    profile = image_edit_profile(QWEN_MULTIPLE_ANGLES_PROFILE_ID)
    parameters = profile.defaults()
    parameters.update(seed=args.seed, steps=args.steps)
    adapters = [adapter.to_dict() for adapter in profile.default_lora_adapters]
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
    transformer = pipe.transformer
    if blocks is None:
        blocks = list(range(len(transformer.transformer_blocks)))
    invalid_blocks = [
        value for value in blocks
        if not 0 <= value < len(transformer.transformer_blocks)
    ]
    if invalid_blocks:
        raise SystemExit(f"invalid blocks: {invalid_blocks}")

    with Image.open(args.front) as source:
        front = source.convert("RGB")
    with Image.open(args.back) as source:
        back = source.convert("RGB")
    width, height = calculate_dimensions(1024 * 1024, front.width / front.height)
    multiple = pipe.vae_scale_factor * 2
    width = width // multiple * multiple
    height = height // multiple * multiple
    grid_width = width // pipe.vae_scale_factor // 2
    grid_height = height // pipe.vae_scale_factor // 2
    target_tokens = grid_width * grid_height

    capture: dict[str, object] = {
        "enabled": False,
        "step_by_timestep": {},
        "mods": {},
        "mod_steps": {},
        "captured": set(),
        "raw_store": None,
        "shift_store": None,
        "scale_store": None,
    }
    block_slots = {block_index: slot for slot, block_index in enumerate(blocks)}

    def step_index() -> int:
        timestep = float(pipe._current_timestep.detach().cpu())
        key = f"{timestep:.8f}"
        mapping = capture["step_by_timestep"]
        return mapping.setdefault(key, len(mapping))

    def make_mod_hook(block_index: int):
        def hook(_module, _inputs, output):
            if not capture["enabled"]:
                return
            capture["mods"][block_index] = output.detach()
            capture["mod_steps"][block_index] = step_index()

        return hook

    def make_norm2_hook(block_index: int):
        def hook(_module, inputs, _output):
            if not capture["enabled"]:
                return
            index = step_index()
            if index != args.feature_step:
                return
            if capture["mod_steps"].get(block_index) != index:
                raise RuntimeError(
                    f"block {block_index} AdaLN parameters are out of sync"
                )
            raw = inputs[0][0, :target_tokens]
            _mod1, mod2 = capture["mods"][block_index].chunk(2, dim=-1)
            shift, scale, _gate = mod2.chunk(3, dim=-1)
            slot = block_slots[block_index]
            capture["raw_store"][slot] = (
                raw.reshape(grid_height, grid_width, -1)
                .detach().float().cpu().numpy()
            )
            capture["shift_store"][slot] = (
                shift[0].detach().float().cpu().numpy()
            )
            capture["scale_store"][slot] = (
                scale[0].detach().float().cpu().numpy()
            )
            capture["captured"].add(block_index)

        return hook

    handles = []
    for block_index in blocks:
        block = transformer.transformer_blocks[block_index]
        handles.extend((
            block.img_mod.register_forward_hook(make_mod_hook(block_index)),
            block.img_norm2.register_forward_hook(make_norm2_hook(block_index)),
        ))

    manifest_views = []
    elevation_degrees, elevation_descriptor = ELEVATIONS[args.elevation]
    try:
        for azimuth in azimuths:
            prompt = (
                f"<sks> {AZIMUTHS[azimuth]} {elevation_descriptor} medium shot"
            )
            capture["enabled"] = True
            capture["step_by_timestep"] = {}
            capture["mods"] = {}
            capture["mod_steps"] = {}
            capture["captured"] = set()
            stem = f"{args.elevation}-{azimuth:03d}"
            raw_path = args.output_dir / f"raw-{stem}.npy"
            shift_path = args.output_dir / f"shift-{stem}.npy"
            scale_path = args.output_dir / f"scale-{stem}.npy"
            capture["raw_store"] = np.lib.format.open_memmap(
                raw_path,
                mode="w+",
                dtype=np.float32,
                shape=(
                    len(blocks), grid_height, grid_width,
                    int(transformer.inner_dim),
                ),
            )
            capture["shift_store"] = np.lib.format.open_memmap(
                shift_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(blocks), int(transformer.inner_dim)),
            )
            capture["scale_store"] = np.lib.format.open_memmap(
                scale_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(blocks), int(transformer.inner_dim)),
            )
            view_parameters = dict(parameters)
            view_parameters["prompt"] = prompt
            print(
                f"Generating and capturing {args.elevation}/{azimuth:03d}°...",
                flush=True,
            )
            result, result_seed, _provenance = backend.image_edit(
                {
                    "profile_id": profile.stable_id,
                    "parameters": view_parameters,
                    "lora_adapters": adapters,
                },
                front,
                back,
            )
            missing = sorted(set(blocks) - capture["captured"])
            if missing:
                raise RuntimeError(f"feature hooks did not capture blocks: {missing}")
            image_path = args.output_dir / f"view-{stem}.png"
            result.save(image_path)
            for key in ("raw_store", "shift_store", "scale_store"):
                capture[key].flush()
                capture[key] = None
            manifest_views.append({
                "elevation": args.elevation,
                "elevation_degrees": elevation_degrees,
                "azimuth_degrees": azimuth,
                "prompt": prompt,
                "seed": int(result_seed),
                "image": str(image_path.resolve()),
                "features": {
                    "format": "npy-memmap-float32",
                    "raw": str(raw_path.resolve()),
                    "shift": str(shift_path.resolve()),
                    "scale": str(scale_path.resolve()),
                },
                "timesteps": {
                    value: int(index)
                    for value, index in capture["step_by_timestep"].items()
                },
            })
            print(
                f"Saved {image_path.name}: {len(blocks)} blocks as raw NPY",
                flush=True,
            )
    finally:
        capture["enabled"] = False
        capture["raw_store"] = None
        capture["shift_store"] = None
        capture["scale_store"] = None
        for handle in handles:
            handle.remove()
        backend.unload_image_edit()

    manifest = {
        "profile": profile.stable_id,
        "loaded": loaded,
        "front": str(args.front.resolve()),
        "back": str(args.back.resolve()),
        "blocks": blocks,
        "steps": args.steps,
        "feature_step": args.feature_step,
        "output_size": [width, height],
        "feature_grid": [grid_width, grid_height],
        "feature_channels": int(transformer.inner_dim),
        "target_tokens": target_tokens,
        "views": manifest_views,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
