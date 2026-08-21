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
        "lora_path": args.lora_path,
    }.items():
        if value is not None and key in parameters:
            overrides[key] = value
    parameters.update(overrides)
    cancel = threading.Event()
    client = MlProcessClient(python=args.worker_python)
    try:
        loaded = client.request(
            "load_image_edit",
            {"profile_id": profile.stable_id, "parameters": parameters},
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
                {"profile_id": profile.stable_id, "parameters": parameters},
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
