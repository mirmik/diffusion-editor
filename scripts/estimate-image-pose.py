#!/usr/bin/env python3
"""Run editor pose backends on one image and export inspectable artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image

from diffusion_editor.generation.pose_estimation import (
    POSE_ESTIMATOR_PROFILES,
    render_pose_overlay,
)
from diffusion_editor.workers.pose_process import PoseProcessClient


def _composite(source: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    background = Image.fromarray(source, "RGBA")
    foreground = Image.fromarray(overlay, "RGBA")
    return np.asarray(Image.alpha_composite(background, foreground))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--profile",
        action="append",
        choices=[profile.stable_id for profile in POSE_ESTIMATOR_PROFILES],
        help="May be repeated; defaults to every profile",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    profiles = args.profile or [
        profile.stable_id for profile in POSE_ESTIMATOR_PROFILES]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = np.asarray(Image.open(args.image).convert("RGBA"), dtype=np.uint8)
    client = PoseProcessClient()
    try:
        for profile_id in profiles:
            print(f"Running {profile_id}...", flush=True)
            result = client.estimate(
                source,
                profile_id,
                threading.Event(),
                on_progress=lambda message: print(message, flush=True),
            )
            overlay = render_pose_overlay(
                result, confidence_threshold=args.confidence)
            stem = args.output_dir / profile_id
            Path(f"{stem}.json").write_text(
                json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            Image.fromarray(overlay, "RGBA").save(f"{stem}-overlay.png")
            Image.fromarray(_composite(source, overlay), "RGBA").save(
                f"{stem}-preview.png")
            visible = sum(
                point.score >= args.confidence
                for pose in result.poses
                for point in pose.keypoints
            )
            print(
                f"  {len(result.poses)} pose(s), {visible} visible points",
                flush=True,
            )
    finally:
        client.shutdown()
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
