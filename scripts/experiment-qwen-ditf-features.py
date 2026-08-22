#!/usr/bin/env python3
"""Capture full-dimensional DiTF-style features and anchor Q/K from Qwen."""

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
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--block", type=int, default=45)
    parser.add_argument(
        "--attention-step",
        type=int,
        default=1,
        help="Denoising step at which normalized pre-RoPE image Q/K are saved.",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    for path in (args.front, args.back):
        if not path.is_file():
            raise SystemExit(f"missing input image: {path}")
    azimuths = [int(value.strip()) for value in args.azimuths.split(",")]
    invalid = [value for value in azimuths if value not in AZIMUTHS]
    if invalid:
        raise SystemExit(f"invalid azimuths: {invalid}")
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
    if not 0 <= args.block < len(transformer.transformer_blocks):
        raise ValueError(f"block {args.block} is outside the transformer")
    block = transformer.transformer_blocks[args.block]

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
        "view": None,
        "arrays": {},
        "step_by_timestep": {},
        "mod_params": None,
        "mod_step": None,
    }

    def step_index() -> int:
        timestep = float(pipe._current_timestep.detach().cpu())
        key = f"{timestep:.8f}"
        mapping = capture["step_by_timestep"]
        return mapping.setdefault(key, len(mapping))

    def save_array(
        name: str,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        capture["arrays"][name] = (
            tensor.detach().to(dtype=dtype, device="cpu").numpy()
        )

    def mod_hook(_module, _inputs, output):
        if capture["view"] is None:
            return
        capture["mod_params"] = output.detach()
        capture["mod_step"] = step_index()

    def norm2_hook(_module, inputs, _output):
        if capture["view"] is None:
            return
        index = step_index()
        if capture["mod_step"] != index or capture["mod_params"] is None:
            raise RuntimeError("AdaLN modulation parameters are out of sync")
        raw = inputs[0][:, :target_tokens]
        _mod1, mod2 = capture["mod_params"].chunk(2, dim=-1)
        shift, scale, _gate = mod2.chunk(3, dim=-1)
        # Qwen's bfloat16 residual stream contains massive activations that can
        # exceed float16's finite range. Preserve these tensors as float32 so
        # channel statistics and discard decisions remain meaningful.
        save_array(
            f"raw_step_{index}",
            raw[0].reshape(grid_height, grid_width, -1),
            dtype=torch.float32,
        )
        save_array(f"shift_step_{index}", shift[0], dtype=torch.float32)
        save_array(f"scale_step_{index}", scale[0], dtype=torch.float32)

    def q_hook(_module, _inputs, output):
        if capture["view"] is None or step_index() != args.attention_step:
            return
        # QK normalization happens before Qwen applies image RoPE. Keeping the
        # head dimension lets the analysis average head-wise anchor affinities.
        save_array("target_q", output[0, :target_tokens])

    def k_hook(_module, _inputs, output):
        if capture["view"] is None or step_index() != args.attention_step:
            return
        save_array("anchor_k", output[0, target_tokens:])

    handles = [
        block.img_mod.register_forward_hook(mod_hook),
        block.img_norm2.register_forward_hook(norm2_hook),
        block.attn.norm_q.register_forward_hook(q_hook),
        block.attn.norm_k.register_forward_hook(k_hook),
    ]
    manifest_views = []
    elevation_degrees, elevation_descriptor = ELEVATIONS[args.elevation]
    try:
        for azimuth in azimuths:
            prompt = (
                f"<sks> {AZIMUTHS[azimuth]} {elevation_descriptor} medium shot"
            )
            capture["view"] = (args.elevation, azimuth)
            capture["arrays"] = {}
            capture["step_by_timestep"] = {}
            capture["mod_params"] = None
            capture["mod_step"] = None
            view_parameters = dict(parameters)
            view_parameters["prompt"] = prompt
            print(f"Generating and capturing {args.elevation}/{azimuth:03d}°...", flush=True)
            result, result_seed, _provenance = backend.image_edit(
                {
                    "profile_id": profile.stable_id,
                    "parameters": view_parameters,
                    "lora_adapters": adapters,
                },
                front,
                back,
            )
            stem = f"{args.elevation}-{azimuth:03d}"
            image_path = args.output_dir / f"view-{stem}.png"
            feature_path = args.output_dir / f"ditf-{stem}.npz"
            result.save(image_path)
            np.savez_compressed(feature_path, **capture["arrays"])
            manifest_views.append({
                "elevation": args.elevation,
                "elevation_degrees": elevation_degrees,
                "azimuth_degrees": azimuth,
                "prompt": prompt,
                "seed": int(result_seed),
                "image": str(image_path.resolve()),
                "features": str(feature_path.resolve()),
                "feature_keys": sorted(capture["arrays"]),
                "timesteps": {
                    value: int(index)
                    for value, index in capture["step_by_timestep"].items()
                },
            })
            print(
                f"Saved {image_path.name}: {len(capture['arrays'])} arrays, "
                f"Q={capture['arrays']['target_q'].shape}, "
                f"K={capture['arrays']['anchor_k'].shape}",
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
        "block": args.block,
        "steps": args.steps,
        "attention_step": args.attention_step,
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
