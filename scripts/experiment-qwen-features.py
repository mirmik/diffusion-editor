#!/usr/bin/env python3
"""Capture projected spatial features from Qwen during angle generation."""

from __future__ import annotations

import argparse
import json
import math
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


AZIMUTHS = (
    (0, "front view"),
    (45, "front-right quarter view"),
    (90, "right side view"),
    (135, "back-right quarter view"),
    (180, "back view"),
    (225, "back-left quarter view"),
    (270, "left side view"),
    (315, "front-left quarter view"),
)

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
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--blocks", default="15,30,45,59")
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument(
        "--elevation",
        action="append",
        choices=tuple(ELEVATIONS),
        dest="elevations",
        help="Repeat to capture multiple elevation rings; defaults to eye.",
    )
    parser.add_argument(
        "--azimuths",
        default="45,90,135,180",
        help="Comma-separated azimuths selected from 0,45,...,315.",
    )
    return parser.parse_args()


def _parse_blocks(text: str) -> tuple[int, ...]:
    blocks = tuple(sorted({int(value.strip()) for value in text.split(",")}))
    if not blocks or blocks[0] < 0:
        raise ValueError("blocks must contain non-negative indices")
    return blocks


def main() -> int:
    args = _arguments()
    blocks = _parse_blocks(args.blocks)
    elevations = args.elevations or ["eye"]
    requested_azimuths = {
        int(value.strip()) for value in args.azimuths.split(",") if value.strip()
    }
    azimuths = [item for item in AZIMUTHS if item[0] in requested_azimuths]
    if len(azimuths) != len(requested_azimuths):
        valid = ",".join(str(value) for value, _descriptor in AZIMUTHS)
        raise SystemExit(f"azimuths must be selected from: {valid}")
    for path in (args.front, args.back):
        if not path.is_file():
            raise SystemExit(f"missing input image: {path}")
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
    if blocks[-1] >= len(transformer.transformer_blocks):
        raise ValueError(
            f"block {blocks[-1]} is outside the transformer with "
            f"{len(transformer.transformer_blocks)} blocks"
        )

    with Image.open(args.front) as source:
        front = source.convert("RGB")
    with Image.open(args.back) as source:
        back = source.convert("RGB")
    width, height = calculate_dimensions(
        1024 * 1024, front.width / front.height
    )
    multiple = pipe.vae_scale_factor * 2
    width = width // multiple * multiple
    height = height // multiple * multiple
    grid_width = width // pipe.vae_scale_factor // 2
    grid_height = height // pipe.vae_scale_factor // 2
    target_tokens = grid_width * grid_height
    channels = int(transformer.inner_dim)

    cpu_generator = torch.Generator(device="cpu").manual_seed(20260822)
    projection_cpu = torch.randn(
        channels,
        args.projection_dim,
        generator=cpu_generator,
        dtype=torch.float32,
    ) / math.sqrt(args.projection_dim)
    projection_by_device: dict[str, torch.Tensor] = {}
    capture: dict[str, object] = {
        "view": None,
        "features": {},
        "step_by_timestep": {},
    }

    def hook_for(block_index: int):
        def hook(_module, _inputs, output):
            view = capture["view"]
            if view is None:
                return
            timestep = float(pipe._current_timestep.detach().cpu())
            timestep_key = f"{timestep:.8f}"
            step_by_timestep = capture["step_by_timestep"]
            step_index = step_by_timestep.setdefault(
                timestep_key, len(step_by_timestep)
            )
            hidden = output[1][:, :target_tokens]
            device_key = str(hidden.device)
            projected_matrix = projection_by_device.get(device_key)
            if projected_matrix is None:
                projected_matrix = projection_cpu.to(
                    device=hidden.device, dtype=hidden.dtype
                )
                projection_by_device[device_key] = projected_matrix
            projected = hidden @ projected_matrix
            projected = torch.nn.functional.normalize(
                projected.float(), dim=-1
            )
            array = (
                projected[0]
                .reshape(grid_height, grid_width, args.projection_dim)
                .to(dtype=torch.float16, device="cpu")
                .numpy()
            )
            capture["features"][(block_index, step_index)] = array

        return hook

    handles = [
        transformer.transformer_blocks[index].register_forward_hook(
            hook_for(index)
        )
        for index in blocks
    ]
    manifest_views = []
    try:
        for elevation_name in elevations:
            elevation_degrees, elevation_descriptor = ELEVATIONS[elevation_name]
            for azimuth, descriptor in azimuths:
                prompt = (
                    f"<sks> {descriptor} {elevation_descriptor} medium shot"
                )
                capture["view"] = (elevation_name, azimuth)
                capture["features"] = {}
                capture["step_by_timestep"] = {}
                view_parameters = dict(parameters)
                view_parameters["prompt"] = prompt
                print(
                    f"Generating and capturing {elevation_name}/{azimuth:03d}°...",
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
                stem = f"{elevation_name}-{azimuth:03d}"
                image_path = args.output_dir / f"view-{stem}.png"
                result.save(image_path)
                arrays = {
                    f"block_{block:02d}_step_{step}": value
                    for (block, step), value in capture["features"].items()
                }
                feature_path = args.output_dir / f"features-{stem}.npz"
                np.savez_compressed(feature_path, **arrays)
                manifest_views.append({
                    "elevation": elevation_name,
                    "elevation_degrees": elevation_degrees,
                    "azimuth_degrees": azimuth,
                    "prompt": prompt,
                    "seed": int(result_seed),
                    "image": str(image_path.resolve()),
                    "features": str(feature_path.resolve()),
                    "timesteps": {
                        value: int(index)
                        for value, index in capture["step_by_timestep"].items()
                    },
                    "feature_keys": sorted(arrays),
                })
                print(
                    f"Saved {image_path.name} and {len(arrays)} feature maps",
                    flush=True,
                )
    finally:
        capture["view"] = None
        for handle in handles:
            handle.remove()
        backend.unload_image_edit()

    manifest = {
        "profile": profile.stable_id,
        "loaded": loaded,
        "front": str(args.front.resolve()),
        "back": str(args.back.resolve()),
        "blocks": list(blocks),
        "steps": args.steps,
        "projection_dim": args.projection_dim,
        "projection_seed": 20260822,
        "elevations": elevations,
        "azimuths": [value for value, _descriptor in azimuths],
        "output_size": [width, height],
        "feature_grid": [grid_width, grid_height],
        "feature_channels_before_projection": channels,
        "views": manifest_views,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
