#!/usr/bin/env python3
"""Run one persistent, reproducible high-quality Pixal3D reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import threading
import time

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_editor.generation.types import (
    ReconstructionParameters,
    ReconstructionStage,
)
from diffusion_editor.workers.pixal3d_process import Pixal3DProcessClient


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=1961831400)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--resolution", type=int, choices=(1024, 1280, 1536), default=1536
    )
    parser.add_argument(
        "--lr-conditioning-resolution",
        type=int,
        choices=(512, 1024),
        default=1024,
    )
    parser.add_argument("--decimation-target", type=int, default=500_000)
    parser.add_argument(
        "--texture-size", type=int, choices=(1024, 2048, 4096), default=4096
    )
    parser.add_argument("--manual-fov-degrees", type=float, default=0.0)
    parser.add_argument(
        "--foreground-mask",
        type=Path,
        help=(
            "Optional L-mode mask to install as input alpha. Supplying it "
            "also skips loading Pixal3D's gated RMBG model."
        ),
    )
    parser.add_argument("--no-low-vram", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _copy_optional(path: Path | None, output: Path) -> str | None:
    if path is None or not path.is_file():
        return None
    destination = output / path.name
    shutil.copy2(path, destination)
    return str(destination.resolve())


def main() -> int:
    args = _arguments()
    if not args.image.is_file():
        raise SystemExit(f"missing input image: {args.image}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.force:
            raise SystemExit(
                f"output directory is not empty: {args.output_dir}; use --force"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parameters = ReconstructionParameters(
        seed=args.seed,
        steps=args.steps,
        resolution=args.resolution,
        lr_conditioning_resolution=args.lr_conditioning_resolution,
        manual_fov_degrees=args.manual_fov_degrees,
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        low_vram=not args.no_low_vram,
    )
    events: list[dict] = []
    started = time.monotonic()

    def on_event(event) -> None:
        record = {
            "stage": event.stage.value,
            "status": event.status.value,
            "progress": event.progress,
            "total": event.total,
            "elapsed_seconds": time.monotonic() - started,
            "artifact": None,
        }
        if event.artifact is not None:
            record["artifact"] = {
                "path": event.artifact.path,
                "preview_kind": event.artifact.preview_kind,
            }
        events.append(record)
        suffix = ""
        if event.total:
            suffix = f" {event.progress}/{event.total}"
        print(
            f"Pixal3D {event.stage.value}: {event.status.value}{suffix}",
            flush=True,
        )

    client = None
    try:
        with Image.open(args.image) as source:
            image = source.convert("RGBA")
            source_size = list(source.size)
        if args.foreground_mask is not None:
            if not args.foreground_mask.is_file():
                raise SystemExit(
                    f"missing foreground mask: {args.foreground_mask}"
                )
            with Image.open(args.foreground_mask) as mask_source:
                mask = mask_source.convert("L")
            if mask.size != image.size:
                raise SystemExit(
                    f"foreground mask is {mask.size}, input is {image.size}"
                )
            image.putalpha(mask)
        alpha_extrema = image.getchannel("A").getextrema()
        has_foreground_alpha = alpha_extrema[0] < 255
        client = Pixal3DProcessClient(
            skip_rembg_model=has_foreground_alpha
        )
        model_path, source_path = client.generate(
            image,
            args.seed,
            threading.Event(),
            parameters=parameters,
            target_stage=ReconstructionStage.FINAL_MESH,
            on_event=on_event,
        )

        # Preserve the complete temporary run before client.shutdown() removes it.
        raw_dir = args.output_dir / "raw"
        shutil.copytree(model_path.parent, raw_dir, dirs_exist_ok=True)
        final_model = args.output_dir / "model.glb"
        shutil.copy2(model_path, final_model)
        copied_source = args.output_dir / "source.png"
        shutil.copy2(source_path, copied_source)
        checkpoints_dir = args.output_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        checkpoints = {
            "shape": _copy_optional(client.checkpoint_path, checkpoints_dir),
            "texture": _copy_optional(
                client.texture_checkpoint_path, checkpoints_dir
            ),
            "resume": _copy_optional(
                client.resume_checkpoint_path, checkpoints_dir
            ),
        }
        artifacts = []
        for artifact in client.artifacts:
            path = Path(artifact.path)
            artifacts.append({
                "stage": artifact.stage.value,
                "preview_kind": artifact.preview_kind,
                "temporary_path": str(path),
                "preserved_path": str((raw_dir / path.name).resolve()),
            })

        report = {
            "schema": "diffusion-editor.pixal3d-high-quality-reconstruction",
            "schema_version": 1,
            "input": str(args.image.resolve()),
            "foreground_mask": (
                str(args.foreground_mask.resolve())
                if args.foreground_mask is not None else None
            ),
            "input_size": source_size,
            "parameters": parameters.to_dict(),
            "elapsed_seconds": time.monotonic() - started,
            "model": str(final_model.resolve()),
            "source": str(copied_source.resolve()),
            "raw_run": str(raw_dir.resolve()),
            "checkpoints": checkpoints,
            "artifacts": artifacts,
            "events": events,
        }
        report_path = args.output_dir / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({
            "model": str(final_model.resolve()),
            "report": str(report_path.resolve()),
            "elapsed_seconds": report["elapsed_seconds"],
        }, indent=2), flush=True)
    finally:
        if client is not None:
            client.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
