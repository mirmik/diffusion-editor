#!/usr/bin/env python3
"""Run selected depth profiles through the real isolated ML worker."""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
from PIL import Image

from diffusion_editor.generation.depth_point_cloud import (
    project_depth_point_cloud,
)
from diffusion_editor.generation.types import (
    DEPTH_MODEL_PROFILES,
    DepthValueKind,
    depth_model_profile,
)
from diffusion_editor.workers.ml_process import MlProcessClient


def _input_image(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(255, 0, height, dtype=np.uint8)[:, None]
    image = np.empty((height, width, 4), dtype=np.uint8)
    image[:, :, 0] = x
    image[:, :, 1] = y
    image[:, :, 2] = ((x.astype(np.uint16) + y) // 2).astype(np.uint8)
    image[:, :, 3] = 255
    return image


def main() -> int:
    available = tuple(profile.stable_id for profile in DEPTH_MODEL_PROFILES)
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="*", choices=available)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    profile_ids = args.profiles or available
    image = _input_image((args.width, args.height))
    client = MlProcessClient(backend="real", request_timeout=args.timeout)
    cancel = threading.Event()
    try:
        for profile_id in profile_ids:
            profile = depth_model_profile(profile_id)
            started = time.monotonic()
            result = client.request(
                "depth",
                {
                    "profile_id": profile.stable_id,
                    "model_id": profile.model_id,
                    "backend": profile.backend.value,
                    "title": profile.title,
                    "process_resolution": profile.process_resolution,
                    "direct_depth": profile.direct_depth,
                    "value_kind": profile.value_kind.value,
                    "use_ray_pose": profile.use_ray_pose,
                    "device": args.device,
                },
                cancel,
                images={"image": image},
                on_progress=lambda message: print(f"  {message}"),
            )
            output = np.asarray(result["depth"])
            if output.ndim != 2 or output.size == 0:
                raise RuntimeError(
                    f"{profile.title}: invalid shape {output.shape}"
                )
            if output.dtype != np.float32 or not np.isfinite(output).all():
                raise RuntimeError(
                    f"{profile.title}: invalid canonical depth payload")
            if profile.predicts_intrinsics and result.get("intrinsics") is None:
                raise RuntimeError(
                    f"{profile.title}: missing camera intrinsics")
            value_kind = DepthValueKind(result["value_kind"])
            intrinsics = result.get("intrinsics")
            cloud_status = "unavailable without camera calibration"
            if (
                    intrinsics is not None
                    or value_kind is DepthValueKind.INVERSE_RELATIVE):
                cloud_image = np.asarray(
                    Image.fromarray(image).resize(
                        (output.shape[1], output.shape[0]),
                        Image.Resampling.LANCZOS,
                    ),
                    dtype=np.uint8,
                )
                cloud = project_depth_point_cloud(
                    cloud_image,
                    output,
                    intrinsics=intrinsics,
                    value_kind=value_kind,
                    fallback_fov_y_degrees=(
                        55.0
                        if value_kind is DepthValueKind.INVERSE_RELATIVE
                        else None
                    ),
                )
                cloud_status = f"{cloud.point_count} raw-grid points"
            print(
                f"{profile.title}: OK, shape={output.shape}, "
                f"dtype={output.dtype}, range={output.min()}..{output.max()}, "
                f"value_kind={result['value_kind']}, "
                "intrinsics="
                f"{'yes' if result.get('intrinsics') is not None else 'no'}, "
                f"cloud={cloud_status}, "
                f"elapsed={time.monotonic() - started:.2f}s"
            )
    finally:
        client.shutdown(timeout=10.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
