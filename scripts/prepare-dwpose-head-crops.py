#!/usr/bin/env python3
"""Create consistently framed head crops from a Qwen multiview manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_editor.workers.pose_backend import PoseBackend


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--face-width-scale", type=float, default=2.15)
    parser.add_argument("--face-height-scale", type=float, default=2.35)
    parser.add_argument("--shoulder-width-scale", type=float, default=0.72)
    parser.add_argument("--upward-shift", type=float, default=0.22)
    return parser.parse_args()


def main_person(result):
    if not result.poses:
        raise RuntimeError("DWPose found no person")
    return max(result.poses, key=lambda pose: sum(point.score for point in pose.keypoints))


def robust_bounds(points) -> tuple[float, float, float, float]:
    xy = np.asarray([(point.x, point.y) for point in points], dtype=np.float32)
    low = np.percentile(xy, 2.0, axis=0)
    high = np.percentile(xy, 98.0, axis=0)
    return float(low[0]), float(low[1]), float(high[0]), float(high[1])


def head_crop_box(points: dict, confidence: float, args: argparse.Namespace):
    face = [
        point for name, point in points.items()
        if name.startswith("face_") and point.score >= confidence
    ]
    if len(face) < 12:
        face = [
            points[name] for name in (
                "nose", "left_eye", "right_eye", "left_ear", "right_ear"
            )
            if name in points and points[name].score >= confidence
        ]
    if len(face) < 3:
        raise RuntimeError(f"Only {len(face)} confident head keypoints")

    x0, y0, x1, y1 = robust_bounds(face)
    face_width = max(x1 - x0, 1.0)
    face_height = max(y1 - y0, 1.0)
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5 - args.upward_shift * face_height
    side = max(
        face_width * args.face_width_scale,
        face_height * args.face_height_scale,
    )

    left_shoulder = points.get("left_shoulder")
    right_shoulder = points.get("right_shoulder")
    if (
        left_shoulder is not None and right_shoulder is not None
        and left_shoulder.score >= confidence and right_shoulder.score >= confidence
    ):
        shoulder_width = float(np.hypot(
            left_shoulder.x - right_shoulder.x,
            left_shoulder.y - right_shoulder.y,
        ))
        side = max(side, shoulder_width * args.shoulder_width_scale)

    return (
        center_x - side * 0.5,
        center_y - side * 0.5,
        center_x + side * 0.5,
        center_y + side * 0.5,
    ), {
        "face_keypoints_used": len(face),
        "face_bounds": [x0, y0, x1, y1],
        "face_mean_confidence": float(np.mean([point.score for point in face])),
        "crop_center": [center_x, center_y],
        "crop_side": side,
    }


def padded_square_crop(image: Image.Image, box, size: int) -> Image.Image:
    left, top, right, bottom = box
    corners = np.asarray(image)[
        np.ix_([0, image.height - 1], [0, image.width - 1])
    ].reshape(-1, 3)
    fill = tuple(np.median(corners, axis=0).astype(np.uint8).tolist())
    width = max(1, int(round(right - left)))
    height = max(1, int(round(bottom - top)))
    canvas = Image.new("RGB", (width, height), fill)
    source_box = (
        max(0, int(np.floor(left))),
        max(0, int(np.floor(top))),
        min(image.width, int(np.ceil(right))),
        min(image.height, int(np.ceil(bottom))),
    )
    if source_box[2] > source_box[0] and source_box[3] > source_box[1]:
        region = image.crop(source_box)
        destination = (
            int(round(source_box[0] - left)),
            int(round(source_box[1] - top)),
        )
        canvas.paste(region, destination)
    return canvas.resize((size, size), Image.Resampling.LANCZOS)


def labeled_sheet(images: list[Image.Image], labels: list[str], columns: int = 8):
    thumb = 256
    label_height = 28
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb, rows * (thumb + label_height)), "#202020")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index % columns * thumb
        y = index // columns * (thumb + label_height)
        sheet.paste(image.resize((thumb, thumb), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 6, y + thumb + 5), label, fill="white", font=font)
    return sheet


def load_source_jobs(manifest_path: Path, payload: dict) -> list[dict]:
    jobs = payload.get("jobs")
    if isinstance(jobs, list):
        return [{
            **job,
            "pose_source": job["output"],
            "name": Path(job["output"]).stem,
            "full_camera": job.get("full_camera"),
        } for job in jobs]

    input_views = payload.get("input_views")
    cameras = payload.get("cameras_final")
    fixed = payload.get("parameters", {}).get("fixed_cameras")
    if (
        not isinstance(input_views, list)
        or not isinstance(cameras, list)
        or len(input_views) != 24
        or len(cameras) != len(input_views)
        or not isinstance(fixed, dict)
    ):
        raise SystemExit(
            "Expected either a 24-job Qwen manifest or a 24-view TripoSPLAT report"
        )
    azimuths = fixed.get("azimuth_degrees")
    elevations = fixed.get("elevation_degrees")
    if not isinstance(azimuths, list) or not isinstance(elevations, list):
        raise SystemExit("TripoSPLAT report has no fixed camera angles")
    run_dir = manifest_path.parent
    result = []
    for index, (view, camera, azimuth, elevation) in enumerate(zip(
        input_views, cameras, azimuths, elevations
    )):
        prepared = (
            run_dir / "01-prepared-1024.png"
            if index == 0
            else run_dir / "00-conditioning" / f"view-{index:02d}-prepared-1024.png"
        )
        if not prepared.is_file():
            raise FileNotFoundError(prepared)
        result.append({
            "name": f"view-{index:02d}-{Path(view['image']).stem}",
            "pose_source": str(prepared),
            "original_source": str(view["image"]),
            "elevation": "fixed",
            "elevation_degrees": float(elevation),
            "azimuth_degrees": float(azimuth),
            "full_camera": camera,
        })
    return result


def main() -> int:
    args = arguments()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    jobs = load_source_jobs(args.manifest, manifest)
    if len(jobs) != 24:
        raise SystemExit("Expected exactly 24 source views")

    backend = PoseBackend()
    output_jobs = []
    crops = []
    overlays = []
    labels = []
    for index, job in enumerate(jobs, start=1):
        source = Path(job["pose_source"]).expanduser()
        if not source.is_file():
            source = args.manifest.parent / source.name
        print(f"DWPose crop {index:02d}/24: {source.name}", flush=True)
        result = backend.estimate("dwpose", source)
        pose = main_person(result)
        points = {point.name: point for point in pose.keypoints}
        box, crop_info = head_crop_box(points, args.confidence, args)
        image = Image.open(source).convert("RGB")
        crop = padded_square_crop(image, box, args.crop_size)
        output_path = args.output / f"{job['name']}-head.png"
        crop.save(output_path)

        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(box, outline="#00ff60", width=6)
        for name, point in points.items():
            if name.startswith("face_") and point.score >= args.confidence:
                radius = 3
                draw.ellipse(
                    (point.x - radius, point.y - radius, point.x + radius, point.y + radius),
                    fill="#ff3050",
                )
        overlays.append(padded_square_crop(overlay, box, args.crop_size))
        crops.append(crop)
        labels.append(
            f"{int(job['elevation_degrees']):+03d}/"
            f"{int(job['azimuth_degrees']) % 360:03d}"
        )
        output_jobs.append({
            "name": job["name"],
            "source": str(source.resolve()),
            "original_source": job.get("original_source", str(source.resolve())),
            "output": str(output_path.resolve()),
            "elevation": job["elevation"],
            "elevation_degrees": float(job["elevation_degrees"]),
            "azimuth_degrees": float(job["azimuth_degrees"]),
            "source_size": [result.width, result.height],
            "crop_box": list(map(float, box)),
            "full_camera": job.get("full_camera"),
            **crop_info,
        })

    labeled_sheet(crops, labels).save(args.output / "contact-head-crops.jpg", quality=94)
    labeled_sheet(overlays, labels).save(args.output / "contact-dwpose-overlays.jpg", quality=92)
    report = {
        "schema": "diffusion-editor.dwpose-head-crops.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest.resolve()),
        "crop_size": args.crop_size,
        "confidence": args.confidence,
        "crop_parameters": {
            "face_width_scale": args.face_width_scale,
            "face_height_scale": args.face_height_scale,
            "shoulder_width_scale": args.shoulder_width_scale,
            "upward_shift": args.upward_shift,
        },
        "jobs": output_jobs,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "views": len(output_jobs),
        "crop_side_min": min(job["crop_side"] for job in output_jobs),
        "crop_side_max": max(job["crop_side"] for job in output_jobs),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
