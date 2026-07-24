"""Run an end-to-end mask round trip through the segmentation worker."""

from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

from diffusion_editor.workers.segmentation_process import (
    SegmentationProcessClient,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("rembg", "threshold"),
        default="rembg",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    image = np.full((128, 128, 4), 255, dtype=np.uint8)
    image[32:96, 40:88, :3] = (20, 40, 180)
    progress: list[str] = []
    gil_before = sys._is_gil_enabled()
    client = SegmentationProcessClient(
        backend=args.backend,
        request_timeout=args.timeout,
    )
    started = time.monotonic()
    try:
        mask = client.segment(
            image,
            threading.Event(),
            on_progress=progress.append,
        )
    finally:
        client.shutdown()
    if mask.shape != image.shape[:2] or mask.dtype != np.uint8:
        raise RuntimeError(
            f"Unexpected segmentation mask: {mask.shape} {mask.dtype}"
        )
    if not progress:
        raise RuntimeError("Segmentation worker emitted no progress")
    if sys._is_gil_enabled() is not gil_before or gil_before:
        raise RuntimeError(
            "Main-process GIL state changed during segmentation smoke"
        )
    print(
        f"Segmentation {args.backend} round trip OK: "
        f"{mask.shape[1]}x{mask.shape[0]}, "
        f"range={int(mask.min())}..{int(mask.max())}, "
        f"{time.monotonic() - started:.2f}s; main GIL disabled"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
