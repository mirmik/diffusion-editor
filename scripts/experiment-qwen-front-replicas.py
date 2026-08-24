#!/usr/bin/env python3
"""Generate independent Qwen samples for one identical front-view request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time

from PIL import Image, ImageDraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from diffusion_editor.generation.image_edit_profiles import (
    QWEN_IMAGE_EDIT_PROFILE_ID,
    image_edit_profile,
    qwen_multiple_angles_lora_adapter,
)
from diffusion_editor.workers.ml_process import MlProcessClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--back", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=9)
    parser.add_argument("--seed", type=int, default=2026082300)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument(
        "--scene-prompt",
        default=(
            "plain neutral gray background, no floor, no props, no text, "
            "full character visible, centered"
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.count < 2:
        parser.error("--count must be at least two")
    return args


def replica_seeds(seed: int, count: int) -> list[int]:
    return [int(seed) + index for index in range(count)]


def _contact_sheet(paths: list[Path], target: Path) -> None:
    cell = 320
    label = 26
    columns = 3
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * (cell + label)), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((cell, cell))
        x = (index % columns) * cell
        y = (index // columns) * (cell + label)
        sheet.paste(image, (x + (cell - image.width) // 2, y + label))
        draw.text((x + 6, y + 6), f"#{index:02d}  seed {path.stem.split('-')[-1]}", fill=(235, 235, 235))
    sheet.save(target)


def main() -> int:
    args = parse_args()
    for path in (args.front, args.back):
        if not path.is_file():
            raise SystemExit(f"missing conditioning image: {path}")
    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    prompt = (
        "<sks> front view eye-level shot medium shot " + args.scene_prompt
    ).strip()
    seeds = replica_seeds(args.seed, args.count)
    outputs = [image_dir / f"front-{index:02d}-{seed}.png" for index, seed in enumerate(seeds)]
    pending = [
        (index, seed, output)
        for index, (seed, output) in enumerate(zip(seeds, outputs))
        if args.force or not output.is_file()
    ]

    profile = image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID)
    base_parameters = profile.defaults()
    base_parameters.update(steps=args.steps, prompt=prompt)
    adapters = [
        adapter.to_dict()
        for adapter in (
            *profile.default_lora_adapters,
            qwen_multiple_angles_lora_adapter(),
        )
    ]
    client = MlProcessClient(python=args.worker_python)
    cancel = threading.Event()
    started = time.monotonic()
    try:
        if pending:
            loaded = client.request(
                "load_image_edit",
                {
                    "profile_id": profile.stable_id,
                    "parameters": base_parameters,
                    "lora_adapters": adapters,
                },
                cancel,
                on_progress=lambda message: print(message, flush=True),
            )
            print(f"Loaded on {loaded.get('device')} as {loaded.get('dtype')}", flush=True)
        with Image.open(args.front) as front_source, Image.open(args.back) as back_source:
            front = front_source.convert("RGB")
            back = back_source.convert("RGB")
            for completed, (index, seed, output) in enumerate(pending, start=1):
                parameters = dict(base_parameters)
                parameters["seed"] = seed
                print(f"[{completed}/{len(pending)}] sample {index}, seed {seed}", flush=True)
                result = client.request(
                    "image_edit",
                    {
                        "profile_id": profile.stable_id,
                        "parameters": parameters,
                        "lora_adapters": adapters,
                    },
                    cancel,
                    images={"image": front, "reference_image": back},
                    on_progress=lambda message: print(message, flush=True),
                )
                saved_image = result["image"]
                if saved_image.size != front.size:
                    saved_image = saved_image.resize(
                        front.size, Image.Resampling.LANCZOS,
                    )
                saved_image.save(output)
    finally:
        client.shutdown()

    manifest = {
        "schema": "diffusion-editor.qwen-front-replicas",
        "schema_version": 1,
        "profile": profile.stable_id,
        "front": str(args.front.resolve()),
        "back": str(args.back.resolve()),
        "prompt": prompt,
        "steps": args.steps,
        "base_seed": args.seed,
        "elapsed_seconds": time.monotonic() - started,
        "images": [
            {"index": index, "seed": seed, "path": str(output.resolve())}
            for index, (seed, output) in enumerate(zip(seeds, outputs))
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    _contact_sheet(outputs, args.output_dir / "contact-front9.png")
    print(f"Complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
