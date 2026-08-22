#!/usr/bin/env python3
"""Generate a reproducible Qwen Multiple Angles elevation grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading
import time

from PIL import Image, ImageDraw

from diffusion_editor.generation.image_edit_profiles import (
    QWEN_MULTIPLE_ANGLES_PROFILE_ID,
    image_edit_profile,
)
from diffusion_editor.workers.ml_process import MlProcessClient


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


def _write_contact_sheets(output_dir: Path, jobs: list[dict]) -> None:
    cell_width, cell_height, label_height = 240, 272, 24
    row_height = cell_height + label_height
    elevations = list(dict.fromkeys(str(job["elevation"]) for job in jobs))
    combined = Image.new(
        "RGB", (cell_width * len(AZIMUTHS), row_height * len(elevations)),
        (12, 12, 12),
    )
    for row, elevation in enumerate(elevations):
        row_jobs = [job for job in jobs if job["elevation"] == elevation]
        sheet = Image.new(
            "RGB", (cell_width * 4, row_height * 2), (12, 12, 12)
        )
        for column, job in enumerate(row_jobs):
            with Image.open(job["output"]) as source:
                thumbnail = source.convert("RGB")
                thumbnail.thumbnail((cell_width, cell_height))
            label = (
                f"{job['azimuth_degrees']:03d}° / "
                f"{job['elevation_degrees']:+d}°"
            )
            for target, x, y in (
                (
                    sheet,
                    (column % 4) * cell_width,
                    (column // 4) * row_height,
                ),
                (combined, column * cell_width, row * row_height),
            ):
                image_x = x + (cell_width - thumbnail.width) // 2
                target.paste(thumbnail, (image_x, y + label_height))
                ImageDraw.Draw(target).text(
                    (x + 6, y + 5), label, fill=(235, 235, 235)
                )
        sheet.save(output_dir / f"contact-{elevation}.png")
    combined.save(output_dir / "contact-all.png")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--back", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--elevation",
        action="append",
        choices=tuple(ELEVATIONS),
        dest="elevations",
        help="Repeat to select rows; defaults to low, elevated, and high.",
    )
    parser.add_argument("--distance", default="medium shot")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    for path in (args.front, args.back):
        if not path.is_file():
            raise SystemExit(f"missing input image: {path}")
    elevations = args.elevations or ["low", "elevated", "high"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for elevation_name in elevations:
        degrees, descriptor = ELEVATIONS[elevation_name]
        for azimuth_degrees, azimuth_descriptor in AZIMUTHS:
            output = args.output_dir / (
                f"mv-{elevation_name}-{azimuth_degrees:03d}.png"
            )
            jobs.append({
                "elevation": elevation_name,
                "elevation_degrees": degrees,
                "azimuth_degrees": azimuth_degrees,
                "prompt": (
                    f"<sks> {azimuth_descriptor} {descriptor} "
                    f"{args.distance}"
                ),
                "output": output,
            })

    pending = [job for job in jobs if args.force or not job["output"].is_file()]
    print(
        f"Grid contains {len(jobs)} views; {len(pending)} need generation.",
        flush=True,
    )
    profile = image_edit_profile(QWEN_MULTIPLE_ANGLES_PROFILE_ID)
    parameters = profile.defaults()
    parameters.update(seed=args.seed, steps=args.steps)
    adapters = [adapter.to_dict() for adapter in profile.default_lora_adapters]
    cancel = threading.Event()
    client = MlProcessClient(python=args.worker_python)
    started = time.monotonic()
    try:
        if pending:
            loaded = client.request(
                "load_image_edit",
                {
                    "profile_id": profile.stable_id,
                    "parameters": parameters,
                    "lora_adapters": adapters,
                },
                cancel,
                on_progress=lambda message: print(message, flush=True),
            )
            print(
                f"Loaded {profile.title} on {loaded.get('device')} "
                f"as {loaded.get('dtype')}",
                flush=True,
            )
        with (
            Image.open(args.front) as front_source,
            Image.open(args.back) as back_source,
        ):
            front = front_source.convert("RGB")
            back = back_source.convert("RGB")
            for index, job in enumerate(pending, start=1):
                job_parameters = dict(parameters)
                job_parameters["prompt"] = job["prompt"]
                print(
                    f"[{index}/{len(pending)}] {job['prompt']}",
                    flush=True,
                )
                result = client.request(
                    "image_edit",
                    {
                        "profile_id": profile.stable_id,
                        "parameters": job_parameters,
                        "lora_adapters": adapters,
                    },
                    cancel,
                    images={"image": front, "reference_image": back},
                    on_progress=lambda message: print(message, flush=True),
                )
                result["image"].save(job["output"])
                job["result_seed"] = int(result["seed"])
                print(f"Saved {job['output']}", flush=True)
    finally:
        client.shutdown()

    manifest = {
        "profile": profile.stable_id,
        "front": str(args.front.resolve()),
        "back": str(args.back.resolve()),
        "seed": args.seed,
        "steps": args.steps,
        "distance": args.distance,
        "elapsed_seconds": time.monotonic() - started,
        "jobs": [
            {
                **{key: value for key, value in job.items() if key != "output"},
                "output": str(job["output"].resolve()),
            }
            for job in jobs
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    _write_contact_sheets(args.output_dir, jobs)
    print(f"Complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
