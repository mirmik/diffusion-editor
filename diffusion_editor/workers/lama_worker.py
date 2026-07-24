"""Standalone LaMa worker entry point.

This module intentionally imports no editor UI, Termin, SDL, OpenGL, or Vulkan
packages. The legacy ML stack is imported only after an inpaint request.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

from PIL import Image

from .lama_protocol import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    encode_message,
)


class _Backend:
    def __init__(self, name: str) -> None:
        self.name = name
        self._model = None

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> Image.Image:
        if self.name == "identity":
            return image.copy()
        if self.name == "hang":
            while True:
                time.sleep(1.0)
        if self.name == "crash":
            os._exit(37)
        if self.name == "malformed":
            sys.__stdout__.buffer.write(b"not-json\n")
            sys.__stdout__.buffer.flush()
            while True:
                time.sleep(1.0)
        if self.name != "lama":
            raise RuntimeError(f"Unknown LaMa worker backend: {self.name}")
        if self._model is None:
            from .lama_model import LamaModel

            self._model = LamaModel()
        return self._model(image, mask)


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
                raise RuntimeError("LaMa request is too large")
            import json

            request = json.loads(message)
            if (
                not isinstance(request, dict)
                or request.get("protocol") != PROTOCOL_VERSION
                or request.get("type") != "inpaint"
                or not isinstance(request.get("request_id"), str)
            ):
                raise RuntimeError("Invalid LaMa inpaint request")
            request_id = request["request_id"]
            paths = [
                request.get("image_path"),
                request.get("mask_path"),
                request.get("output_path"),
            ]
            if not all(isinstance(path, str) and path for path in paths):
                raise RuntimeError("LaMa request has invalid image paths")
            image_path, mask_path, output_path = map(Path, paths)
            with (
                Image.open(image_path) as source,
                Image.open(mask_path) as source_mask,
                redirect_stdout(sys.stderr),
            ):
                image = source.convert("RGB")
                mask = source_mask.convert("L")
                result = backend.inpaint(image, mask)
                result.save(output_path, format="PNG")
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
        choices=("lama", "identity", "hang", "crash", "malformed"),
        default="lama",
    )
    args = parser.parse_args()
    return run(args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
