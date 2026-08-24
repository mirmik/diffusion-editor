#!/usr/bin/env python3
"""Render deterministic prompts with the editor's Qwen LoRA stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading

from PIL import Image, ImageDraw

from diffusion_editor.generation.image_edit_profiles import (
    QWEN_IMAGE_EDIT_PROFILE_ID,
    image_edit_profile,
    qwen_multiple_angles_lora_adapter,
)
from diffusion_editor.workers.ml_process import MlProcessClient


AZIMUTHS = (
    (0, "front view"),
    (315, "front-left quarter view"),
    (45, "front-right quarter view"),
)

ELEVATIONS = (
    ("low", -30, "low-angle shot"),
    ("eye", 0, "eye-level shot"),
    ("elevated", 30, "elevated shot"),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--prompt",
        default="<sks> front-left quarter view eye-level shot medium shot",
    )
    parser.add_argument("--seed", type=int, default=2399806712)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument(
        "--variant",
        choices=("on", "off", "both"),
        default="both",
        help="Render only the selected Multiple Angles activation state.",
    )
    parser.add_argument(
        "--all-nine",
        action="store_true",
        help="Probe the 3 azimuth by 3 elevation front-sector grid.",
    )
    return parser.parse_args()


def _contact_sheet(paths: list[tuple[str, Path]], target: Path) -> None:
    cell_width, cell_height, label_height = 480, 544, 28
    sheet = Image.new(
        "RGB", (cell_width * len(paths), cell_height + label_height),
        (16, 16, 16),
    )
    draw = ImageDraw.Draw(sheet)
    for column, (label, path) in enumerate(paths):
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((cell_width, cell_height))
        x = column * cell_width
        sheet.paste(image, (x + (cell_width - image.width) // 2, label_height))
        draw.text((x + 8, 6), label, fill=(240, 240, 240))
    sheet.save(target)


def _grid_contact_sheet(records: list[dict], target: Path) -> None:
    cell_width, cell_height, label_height = 240, 272, 28
    labels = list(dict.fromkeys(record["label"] for record in records))
    columns = len(AZIMUTHS) * len(labels)
    rows = len(ELEVATIONS)
    sheet = Image.new(
        "RGB",
        (cell_width * columns, (cell_height + label_height) * rows),
        (16, 16, 16),
    )
    draw = ImageDraw.Draw(sheet)
    lookup = {
        (record["elevation"], record["azimuth_degrees"], record["label"]):
        Path(record["output"])
        for record in records
    }
    for row, (elevation, elevation_degrees, _descriptor) in enumerate(ELEVATIONS):
        for azimuth_index, (azimuth, _azimuth_descriptor) in enumerate(AZIMUTHS):
            for variant_index, label in enumerate(labels):
                column = azimuth_index * len(labels) + variant_index
                path = lookup[(elevation, azimuth, label)]
                with Image.open(path) as source:
                    image = source.convert("RGB")
                    image.thumbnail((cell_width, cell_height))
                x = column * cell_width
                y = row * (cell_height + label_height)
                sheet.paste(
                    image,
                    (x + (cell_width - image.width) // 2, y + label_height),
                )
                state = "ON" if label.endswith("-on") else "OFF"
                draw.text(
                    (x + 6, y + 6),
                    f"{azimuth:03d}°/{elevation_degrees:+d}°  LoRA {state}",
                    fill=(240, 240, 240),
                )
    sheet.save(target)


def main() -> int:
    args = _arguments()
    if not args.image.is_file():
        raise SystemExit(f"missing input image: {args.image}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Match the editor setup exactly: the primary Qwen profile with Lightning
    # at 1.0 and the optional angle LoRA explicitly added at 1.0.
    profile = image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID)
    parameters = profile.defaults()
    parameters.update(seed=args.seed, steps=args.steps)
    jobs = [{
        "prompt": args.prompt,
        "elevation": None,
        "elevation_degrees": None,
        "azimuth_degrees": None,
        "stem": "single",
    }]
    if args.all_nine:
        jobs = []
        for elevation, elevation_degrees, elevation_descriptor in ELEVATIONS:
            for azimuth, azimuth_descriptor in AZIMUTHS:
                jobs.append({
                    "prompt": (
                        f"<sks> {azimuth_descriptor} "
                        f"{elevation_descriptor} medium shot"
                    ),
                    "elevation": elevation,
                    "elevation_degrees": elevation_degrees,
                    "azimuth_degrees": azimuth,
                    "stem": f"{elevation}-{azimuth:03d}",
                })
    default_adapters = [
        adapter.to_dict() for adapter in profile.default_lora_adapters
    ]
    angle_adapter = qwen_multiple_angles_lora_adapter().to_dict()
    default_adapters.append(angle_adapter)
    variants = []
    enabled_states = {
        "on": (True,),
        "off": (False,),
        "both": (True, False),
    }[args.variant]
    for enabled in enabled_states:
        adapters = [dict(adapter) for adapter in default_adapters]
        for adapter in adapters:
            if adapter["stable_id"] == "multiple-angles":
                adapter["enabled"] = enabled
        variants.append((
            "multiple-angles-on" if enabled else "multiple-angles-off",
            adapters,
        ))

    cancel = threading.Event()
    client = MlProcessClient(python=args.worker_python)
    records = []
    outputs: list[tuple[str, Path]] = []
    try:
        with Image.open(args.image) as source:
            image = source.convert("RGB")
            for label, adapters in variants:
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
                    f"{label}: active={loaded.get('active_lora_adapters')} "
                    f"weights={loaded.get('active_lora_weights')}",
                    flush=True,
                )
                for index, job in enumerate(jobs, start=1):
                    job_parameters = dict(parameters)
                    job_parameters["prompt"] = job["prompt"]
                    print(
                        f"{label} [{index}/{len(jobs)}]: {job['prompt']}",
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
                        images={"image": image},
                        on_progress=lambda message: print(message, flush=True),
                    )
                    output = args.output_dir / f"{job['stem']}-{label}.png"
                    raw_size = result["image"].size
                    saved_image = result["image"]
                    if saved_image.size != image.size:
                        saved_image = saved_image.resize(
                            image.size, Image.Resampling.LANCZOS,
                        )
                    saved_image.save(output)
                    outputs.append((label, output))
                    records.append({
                        **job,
                        "label": label,
                        "adapters": adapters,
                        "loaded": loaded,
                        "raw_output_size": list(raw_size),
                        "saved_output_size": list(saved_image.size),
                        "output": str(output.resolve()),
                    })
    finally:
        client.shutdown()

    manifest = {
        "schema": "diffusion-editor.qwen-multiple-angles-activation-probe",
        "schema_version": 1,
        "input": str(args.image.resolve()),
        "profile_id": profile.stable_id,
        "prompt": args.prompt if not args.all_nine else None,
        "all_nine": args.all_nine,
        "seed": args.seed,
        "steps": args.steps,
        "variant": args.variant,
        "variants": records,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.all_nine:
        _grid_contact_sheet(records, args.output_dir / "contact-grid.png")
    else:
        _contact_sheet(outputs, args.output_dir / "contact.png")
    print(f"Complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
