#!/usr/bin/env python3
"""Generate a reproducible Qwen Multiple Angles elevation grid."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from PIL import Image, ImageDraw

from diffusion_editor.generation.image_edit_profiles import (
    QWEN_IMAGE_EDIT_PROFILE_ID,
    image_edit_profile,
    qwen_multiple_angles_lora_adapter,
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
    columns = max(
        sum(job["elevation"] == elevation for job in jobs)
        for elevation in elevations
    )
    combined = Image.new(
        "RGB", (cell_width * columns, row_height * len(elevations)),
        (12, 12, 12),
    )
    for row, elevation in enumerate(elevations):
        row_jobs = [job for job in jobs if job["elevation"] == elevation]
        sheet_columns = min(4, len(row_jobs))
        sheet_rows = (len(row_jobs) + sheet_columns - 1) // sheet_columns
        sheet = Image.new(
            "RGB",
            (cell_width * sheet_columns, row_height * sheet_rows),
            (12, 12, 12),
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
                    (column % sheet_columns) * cell_width,
                    (column // sheet_columns) * row_height,
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
    references = parser.add_mutually_exclusive_group()
    references.add_argument(
        "--back",
        type=Path,
        help="Optional second reference image; omit for a front-only source.",
    )
    references.add_argument(
        "--reference-dir",
        type=Path,
        help=(
            "Directory containing one second reference per requested view, "
            "named like mv-eye-045.png."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--elevation",
        action="append",
        choices=tuple(ELEVATIONS),
        dest="elevations",
        help="Repeat to select rows; defaults to low, eye, and elevated.",
    )
    parser.add_argument(
        "--azimuth",
        action="append",
        type=int,
        choices=tuple(value for value, _ in AZIMUTHS),
        dest="azimuths",
        help="Repeat to select azimuths; defaults to the complete eight-view ring.",
    )
    parser.add_argument("--distance", default="medium shot")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--max-seed-attempts",
        type=int,
        default=10,
        help="Reject a bad complete batch and try this many consecutive seeds.",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument(
        "--without-multiple-angles",
        action="store_true",
        help="Keep the default Lightning adapter but omit Multiple Angles.",
    )
    parser.add_argument(
        "--fixed-prompt",
        help="Use this exact prompt for every view instead of angle tokens.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-orientation-verification",
        action="store_true",
        help="Keep unverified views (intended only for verifier diagnostics).",
    )
    parser.add_argument(
        "--verifier-api-base",
        default=os.environ.get(
            "DIFFUSION_EDITOR_ORIENTATION_VERIFIER_API",
            "http://192.168.0.61:8096/v1",
        ),
    )
    parser.add_argument(
        "--verifier-model", default="qwen3.8-27b-uncensored-q4-mtp"
    )
    return parser.parse_args()


def candidate_seeds(initial_seed: int, attempts: int) -> list[int]:
    if attempts < 1:
        raise ValueError("seed attempts must be positive")
    return [initial_seed + offset for offset in range(attempts)]


def per_view_reference_path(reference_dir: Path, output: Path) -> Path:
    return reference_dir / output.name


def _manifest_jobs(jobs: list[dict]) -> list[dict]:
    return [
        {
            **{
                key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in job.items()
                if key != "output"
            },
            "output": str(job["output"].resolve()),
        }
        for job in jobs
    ]


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _verification_command(args: argparse.Namespace, manifest_path: Path) -> list[str]:
    verifier = Path(__file__).with_name("verify-qwen-view-orientation.py")
    return [
        sys.executable,
        str(verifier),
        str(manifest_path),
        "--api-base", args.verifier_api_base,
        "--model", args.verifier_model,
    ]


def _verify_batch(args: argparse.Namespace, manifest_path: Path) -> bool:
    verification = subprocess.run(
        _verification_command(args, manifest_path), check=False
    )
    if verification.returncode == 2:
        return False
    if verification.returncode != 0:
        raise RuntimeError(
            "orientation verifier failed with exit code "
            f"{verification.returncode}"
        )
    return True


def _archive_batch(
    output_dir: Path,
    jobs: list[dict],
    *,
    seed: int | None,
    reason: str,
) -> Path:
    archive_root = output_dir / "rejected-batches"
    seed_label = "unknown" if seed is None else str(seed)
    archive_dir = archive_root / f"seed-{seed_label}"
    suffix = 2
    while archive_dir.exists():
        archive_dir = archive_root / f"seed-{seed_label}-{suffix}"
        suffix += 1
    archive_dir.mkdir(parents=True)

    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "seed": seed,
            "jobs": _manifest_jobs(jobs),
        }
    manifest["batch_status"] = "rejected"
    manifest["rejection_reason"] = reason
    manifest["archive_dir"] = str(archive_dir.resolve())
    records = manifest.get("jobs")
    if isinstance(records, list):
        for record in records:
            raw_output = record.get("output")
            if raw_output is not None:
                record["output"] = str(
                    (archive_dir / Path(raw_output).name).resolve()
                )
    _write_manifest(manifest_path, manifest)

    paths = [job["output"] for job in jobs]
    paths.extend(sorted(output_dir.glob("contact-*.png")))
    paths.append(manifest_path)
    for path in paths:
        if path.is_file():
            path.replace(archive_dir / path.name)
    return archive_dir


def _accepted_manifest(path: Path) -> bool:
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    verification = manifest.get("orientation_verification")
    return (
        isinstance(verification, dict)
        and verification.get("accepted") is True
    )


def main() -> int:
    args = _arguments()
    if args.max_seed_attempts < 1:
        raise SystemExit("--max-seed-attempts must be positive")
    for path in (args.front, args.back):
        if path is None:
            continue
        if not path.is_file():
            raise SystemExit(f"missing input image: {path}")
    if args.reference_dir is not None and not args.reference_dir.is_dir():
        raise SystemExit(f"missing reference directory: {args.reference_dir}")
    elevations = args.elevations or ["low", "eye", "elevated"]
    azimuths_by_value = dict(AZIMUTHS)
    selected_azimuths = args.azimuths or [value for value, _ in AZIMUTHS]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for elevation_name in elevations:
        degrees, descriptor = ELEVATIONS[elevation_name]
        for azimuth_degrees in selected_azimuths:
            azimuth_descriptor = azimuths_by_value[azimuth_degrees]
            output = args.output_dir / (
                f"mv-{elevation_name}-{azimuth_degrees:03d}.png"
            )
            jobs.append({
                "elevation": elevation_name,
                "elevation_degrees": degrees,
                "azimuth_degrees": azimuth_degrees,
                "prompt": args.fixed_prompt or (
                    f"<sks> {azimuth_descriptor} {descriptor} "
                    f"{args.distance}"
                ),
                "output": output,
            })
    if args.reference_dir is not None:
        for job in jobs:
            reference = per_view_reference_path(
                args.reference_dir, job["output"]
            )
            if not reference.is_file():
                raise SystemExit(f"missing per-view reference image: {reference}")
            job["reference_image"] = reference.resolve()

    manifest_path = args.output_dir / "manifest.json"
    existing_outputs = [job for job in jobs if job["output"].is_file()]
    rejected_batches: list[dict] = []
    initial_seed = args.seed
    if existing_outputs:
        existing_seed = None
        if manifest_path.is_file():
            existing_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            if existing_manifest.get("seed") is not None:
                existing_seed = int(existing_manifest["seed"])
        complete = len(existing_outputs) == len(jobs)
        if complete and not args.force and not args.skip_orientation_verification:
            if not _accepted_manifest(manifest_path):
                if not manifest_path.is_file():
                    raise SystemExit(
                        "complete existing batch has no manifest; use --force"
                    )
                if "orientation_verification" not in existing_manifest:
                    _verify_batch(args, manifest_path)
            if _accepted_manifest(manifest_path):
                print(f"Existing verified batch is accepted: {args.output_dir}")
                return 0
        if complete and not args.force and args.skip_orientation_verification:
            print(f"Existing batch kept without verification: {args.output_dir}")
            return 0
        reason = (
            "explicitly replaced with --force"
            if args.force
            else "pre-existing batch was incomplete or failed verification"
        )
        archived = _archive_batch(
            args.output_dir,
            jobs,
            seed=existing_seed,
            reason=reason,
        )
        rejected_batches.append({
            "seed": existing_seed,
            "archive_dir": str(archived.resolve()),
            "reason": reason,
        })
        if existing_seed is not None and not args.force:
            initial_seed = existing_seed + 1

    print(
        f"Grid contains {len(jobs)} views; complete batches only. "
        f"Seeds: {candidate_seeds(initial_seed, args.max_seed_attempts)}",
        flush=True,
    )
    profile = image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID)
    parameters = profile.defaults()
    parameters.update(seed=initial_seed, steps=args.steps)
    selected_adapters = list(profile.default_lora_adapters)
    if not args.without_multiple_angles:
        selected_adapters.append(qwen_multiple_angles_lora_adapter())
    adapters = [adapter.to_dict() for adapter in selected_adapters]
    cancel = threading.Event()
    client = MlProcessClient(python=args.worker_python)
    started = time.monotonic()
    try:
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
        with Image.open(args.front) as front_source:
            front = front_source.convert("RGB")
            back = None
            if args.back is not None:
                with Image.open(args.back) as back_source:
                    back = back_source.convert("RGB")
            for attempt_index, seed in enumerate(
                candidate_seeds(initial_seed, args.max_seed_attempts), start=1
            ):
                attempt_started = time.monotonic()
                print(
                    f"Batch attempt {attempt_index}/{args.max_seed_attempts}, "
                    f"seed={seed}",
                    flush=True,
                )
                for index, job in enumerate(jobs, start=1):
                    job_parameters = dict(parameters)
                    job_parameters.update(seed=seed, prompt=job["prompt"])
                    print(
                        f"[{index}/{len(jobs)}] {job['prompt']}",
                        flush=True,
                    )
                    images = {"image": front}
                    if back is not None:
                        images["reference_image"] = back
                    elif job.get("reference_image") is not None:
                        with Image.open(job["reference_image"]) as reference_source:
                            reference = reference_source.convert("RGB")
                        if reference.size != front.size:
                            raise RuntimeError(
                                "per-view reference size does not match the front: "
                                f"{job['reference_image']} is {reference.size}, "
                                f"front is {front.size}"
                            )
                        images["reference_image"] = reference
                    result = client.request(
                        "image_edit",
                        {
                            "profile_id": profile.stable_id,
                            "parameters": job_parameters,
                            "lora_adapters": adapters,
                        },
                        cancel,
                        images=images,
                        on_progress=lambda message: print(message, flush=True),
                    )
                    job["raw_output_size"] = list(result["image"].size)
                    saved_image = result["image"]
                    if saved_image.size != front.size:
                        saved_image = saved_image.resize(
                            front.size, Image.Resampling.LANCZOS,
                        )
                    saved_image.save(job["output"])
                    job["saved_output_size"] = list(saved_image.size)
                    job["result_seed"] = int(result["seed"])
                    print(f"Saved {job['output']}", flush=True)

                manifest = {
                    "profile": profile.stable_id,
                    "front": str(args.front.resolve()),
                    "back": (
                        str(args.back.resolve())
                        if args.back is not None else None
                    ),
                    "reference_dir": (
                        str(args.reference_dir.resolve())
                        if args.reference_dir is not None else None
                    ),
                    "seed": seed,
                    "initial_seed": initial_seed,
                    "seed_attempt": attempt_index,
                    "max_seed_attempts": args.max_seed_attempts,
                    "steps": args.steps,
                    "distance": args.distance,
                    "multiple_angles_enabled": (
                        not args.without_multiple_angles
                    ),
                    "fixed_prompt": args.fixed_prompt,
                    "lora_adapters": adapters,
                    "attempt_elapsed_seconds": (
                        time.monotonic() - attempt_started
                    ),
                    "elapsed_seconds": time.monotonic() - started,
                    "rejected_batches": list(rejected_batches),
                    "jobs": _manifest_jobs(jobs),
                }
                _write_manifest(manifest_path, manifest)
                _write_contact_sheets(args.output_dir, jobs)

                if args.skip_orientation_verification:
                    manifest["batch_status"] = "unverified"
                    _write_manifest(manifest_path, manifest)
                    print(f"Complete without verification: {args.output_dir}")
                    return 0
                if _verify_batch(args, manifest_path):
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    manifest["batch_status"] = "accepted"
                    manifest["accepted_seed"] = seed
                    manifest["rejected_batches"] = list(rejected_batches)
                    manifest["elapsed_seconds"] = time.monotonic() - started
                    _write_manifest(manifest_path, manifest)
                    print(
                        f"Complete: {args.output_dir}; accepted seed={seed}",
                        flush=True,
                    )
                    return 0

                archived = _archive_batch(
                    args.output_dir,
                    jobs,
                    seed=seed,
                    reason="one or more views failed orientation verification",
                )
                rejected_batches.append({
                    "seed": seed,
                    "archive_dir": str(archived.resolve()),
                    "reason": "orientation verification failed",
                })
                print(
                    f"Rejected entire seed={seed} batch -> {archived}",
                    flush=True,
                )
    finally:
        client.shutdown()
    raise SystemExit(
        "No orientation-valid batch found after "
        f"{args.max_seed_attempts} seeds; rejected batches are preserved in "
        f"{args.output_dir / 'rejected-batches'}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
