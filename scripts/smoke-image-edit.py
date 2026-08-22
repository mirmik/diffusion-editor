#!/usr/bin/env python3
"""Opt-in GPU smoke for one registered standalone image-edit provider."""

from __future__ import annotations

import argparse
from pathlib import Path
import threading

from PIL import Image

from diffusion_editor.generation.image_edit_profiles import (
    image_edit_profile,
    image_edit_profiles,
)
from diffusion_editor.workers.ml_process import MlProcessClient


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=[profile.stable_id for profile in image_edit_profiles()],
        required=True,
    )
    parser.add_argument(
        "--preload-profile",
        choices=[profile.stable_id for profile in image_edit_profiles()],
        default=None,
        help="Load this profile first to exercise backend switching/unload.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--worker-python", type=Path, default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--lora-path", default=None)
    parser.add_argument(
        "--lora",
        action="append",
        nargs=2,
        metavar=("SOURCE", "WEIGHT"),
        help="Append one LoRA adapter; may be repeated.",
    )
    parser.add_argument("--gguf-checkpoint", default=None)
    parser.add_argument("--target-megapixels", type=float, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--img-cfg-scale", type=float, default=None)
    parser.add_argument("--timestep-shift", type=float, default=None)
    parser.add_argument("--vram-mode", default=None)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    profile = image_edit_profile(args.profile)
    parameters = profile.defaults()
    overrides = {
        "prompt": args.prompt,
        "seed": args.seed,
        "local_files_only": args.local_files_only,
    }
    if args.cpu_offload is not None:
        overrides["cpu_offload"] = args.cpu_offload
    for key, value in {
        "model": args.model,
        "negative_prompt": args.negative_prompt,
        "steps": args.steps,
        "dtype": args.dtype,
        "device": args.device,
        "gguf_checkpoint": args.gguf_checkpoint,
        "target_megapixels": args.target_megapixels,
        "cfg_scale": args.cfg_scale,
        "img_cfg_scale": args.img_cfg_scale,
        "timestep_shift": args.timestep_shift,
        "vram_mode": args.vram_mode,
        "attention_backend": args.attention_backend,
        "width": args.width,
        "height": args.height,
    }.items():
        if value is not None and key in parameters:
            overrides[key] = value
    parameters.update(overrides)
    lora_adapters = [
        adapter.to_dict() for adapter in profile.default_lora_adapters]
    if args.lora_path is not None:
        if lora_adapters:
            lora_adapters[0].update(
                source=args.lora_path, enabled=bool(args.lora_path))
        else:
            lora_adapters.append({
                "stable_id": "smoke-lora-1",
                "label": Path(args.lora_path).stem or "LoRA 1",
                "source": args.lora_path,
                "weight": 1.0,
                "enabled": bool(args.lora_path),
            })
    for index, (source, weight) in enumerate(args.lora or (), start=1):
        lora_adapters.append({
            "stable_id": f"smoke-custom-{index}",
            "label": Path(source).stem or f"Custom LoRA {index}",
            "source": source,
            "weight": float(weight),
            "enabled": True,
        })
    cancel = threading.Event()
    client = MlProcessClient(python=args.worker_python)
    try:
        if args.preload_profile is not None:
            preload = image_edit_profile(args.preload_profile)
            print(f"Preloading {preload.title}...")
            client.request(
                "load_image_edit",
                {
                    "profile_id": preload.stable_id,
                    "parameters": preload.defaults(),
                    "lora_adapters": [
                        adapter.to_dict()
                        for adapter in preload.default_lora_adapters
                    ],
                },
                cancel,
                on_progress=print,
            )
        loaded = client.request(
            "load_image_edit",
            {
                "profile_id": profile.stable_id,
                "parameters": parameters,
                "lora_adapters": lora_adapters,
            },
            cancel,
            on_progress=print,
        )
        print(
            f"Loaded {loaded.get('profile_title', profile.title)} "
            f"on {loaded.get('device')} as {loaded.get('dtype')}"
        )
        with Image.open(args.input) as source:
            result = client.request(
                "image_edit",
                {
                    "profile_id": profile.stable_id,
                    "parameters": parameters,
                    "lora_adapters": lora_adapters,
                },
                cancel,
                images={"image": source.convert("RGB")},
                on_progress=print,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result["image"].save(args.output)
        print(f"Saved {args.output} (seed={result['seed']})")
        return 0
    finally:
        client.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
