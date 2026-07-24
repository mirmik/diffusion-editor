"""Standalone rembg worker without editor or graphics imports."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

from PIL import Image

from .segmentation_protocol import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    encode_message,
)


class _Backend:
    def __init__(self, name: str) -> None:
        self.name = name
        self._session = None

    def segment(self, image: Image.Image) -> Image.Image:
        if self.name == "threshold":
            source = image.convert("RGB")
            return source.convert("L").point(
                lambda value: 255 if value < 128 else 0
            )
        if self.name == "hang":
            while True:
                time.sleep(1.0)
        if self.name == "crash":
            os._exit(38)
        if self.name == "malformed":
            sys.__stdout__.buffer.write(b"not-json\n")
            sys.__stdout__.buffer.flush()
            while True:
                time.sleep(1.0)
        if self.name != "rembg":
            raise RuntimeError(
                f"Unknown segmentation worker backend: {self.name}"
            )
        from rembg import new_session, remove

        if self._session is None:
            self._session = new_session("u2net")
        return remove(image, session=self._session, only_mask=True).convert("L")


def _send(wire, payload: dict[str, Any]) -> None:
    wire.write(encode_message(payload))
    wire.flush()


def _runtime(backend: str) -> dict[str, Any]:
    gil_probe = getattr(sys, "_is_gil_enabled", None)
    return {
        "python": platform.python_implementation(),
        "version": platform.python_version(),
        "abiflags": getattr(sys, "abiflags", ""),
        "gil_enabled": True if gil_probe is None else bool(gil_probe()),
        "backend": backend,
    }


def run(backend_name: str) -> int:
    wire = sys.stdout.buffer
    backend = _Backend(backend_name)
    _send(
        wire,
        {
            "protocol": PROTOCOL_VERSION,
            "type": "ready",
            "runtime": _runtime(backend_name),
        },
    )
    while message := sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 1):
        request_id = "unknown"
        try:
            if len(message) > MAX_MESSAGE_BYTES:
                raise RuntimeError("Segmentation request is too large")
            request = json.loads(message)
            if (
                not isinstance(request, dict)
                or request.get("protocol") != PROTOCOL_VERSION
                or request.get("type") != "segment"
                or not isinstance(request.get("request_id"), str)
            ):
                raise RuntimeError("Invalid segmentation request")
            request_id = request["request_id"]
            paths = [
                request.get("image_path"),
                request.get("output_path"),
            ]
            if not all(isinstance(path, str) and path for path in paths):
                raise RuntimeError(
                    "Segmentation request has invalid image paths"
                )
            image_path, output_path = map(Path, paths)
            _send(
                wire,
                {
                    "protocol": PROTOCOL_VERSION,
                    "type": "progress",
                    "request_id": request_id,
                    "message": "Running segmentation",
                },
            )
            with (
                Image.open(image_path) as source,
                redirect_stdout(sys.stderr),
            ):
                mask = backend.segment(source.convert("RGB"))
                mask.save(output_path, format="PNG")
            _send(
                wire,
                {
                    "protocol": PROTOCOL_VERSION,
                    "type": "result",
                    "request_id": request_id,
                    "output_path": str(output_path),
                },
            )
        except Exception as exc:
            _send(
                wire,
                {
                    "protocol": PROTOCOL_VERSION,
                    "type": "error",
                    "request_id": request_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("rembg", "threshold", "hang", "crash", "malformed"),
        default="rembg",
    )
    args = parser.parse_args()
    return run(args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
