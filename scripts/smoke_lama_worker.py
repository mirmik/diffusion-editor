"""Run an end-to-end image round trip through the configured LaMa worker."""

from __future__ import annotations

import argparse
import sys
import threading
import time

from PIL import Image, ImageDraw

from diffusion_editor.workers.lama_process import LamaProcessClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("lama", "identity"), default="lama")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    image = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 43, 43), fill="red")
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((18, 18, 45, 45), fill=255)

    gil_before = sys._is_gil_enabled()
    client = LamaProcessClient(
        backend=args.backend,
        request_timeout=args.timeout,
    )
    started = time.monotonic()
    try:
        result = client.inpaint(image, mask, threading.Event())
    finally:
        client.shutdown()
    if result.size != image.size:
        raise RuntimeError(
            f"LaMa result size mismatch: {result.size} != {image.size}"
        )
    if sys._is_gil_enabled() is not gil_before or gil_before:
        raise RuntimeError("Main-process GIL state changed during LaMa smoke")
    print(
        f"LaMa {args.backend} round trip OK: "
        f"{result.size[0]}x{result.size[1]} in "
        f"{time.monotonic() - started:.2f}s; main GIL disabled"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
