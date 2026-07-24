"""Standalone Diffusers/Transformers worker without editor or graphics imports."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import numpy as np
from PIL import Image

from .ml_protocol import MAX_MESSAGE_BYTES, PROTOCOL_VERSION, encode_message


def _send(wire, payload: dict[str, Any]) -> None:
    wire.write(encode_message(payload))
    wire.flush()


class _Backend:
    def __init__(self, name: str) -> None:
        self.name = name
        self._real = None
        self.diffusion_loaded = False
        self.instruct_loaded = False

    def execute(
        self,
        operation: str,
        data: dict[str, Any],
        progress: Callable[[str], None],
    ) -> dict[str, Any]:
        if self.name == "hang":
            while True:
                time.sleep(1)
        if self.name == "crash":
            os._exit(39)
        if self.name == "malformed":
            sys.__stdout__.buffer.write(b"not-json\n")
            sys.__stdout__.buffer.flush()
            while True:
                time.sleep(1)
        if self.name == "fake":
            return self._fake(operation, data, progress)
        if self.name != "real":
            raise RuntimeError(f"Unknown ML worker backend: {self.name}")
        if self._real is None:
            from .ml_backend import RealMlBackend
            self._real = RealMlBackend()
        return self._execute_real(operation, data, progress)

    def _execute_real(
        self,
        operation: str,
        data: dict[str, Any],
        progress: Callable[[str], None],
    ) -> dict[str, Any]:
        output_dir = Path(data["output_dir"])
        if operation == "gpu_available":
            return {"available": self._real.gpu_available()}
        if operation == "load_diffusion":
            progress("Loading diffusion model...")
            return self._real.load_diffusion(data)
        if operation == "load_ip_adapter":
            progress("Loading IP-Adapter...")
            return self._real.load_ip_adapter()
        if operation == "diffusion":
            progress("Running diffusion...")
            images = self._open_images(data, ("image", "mask", "ip_adapter"))
            result, seed = self._real.diffusion(data, images)
            output = output_dir / "result.png"
            result.save(output, format="PNG")
            return {"output_path": str(output), "seed": seed}
        if operation == "load_instruct":
            progress("Loading InstructPix2Pix model...")
            return self._real.load_instruct(data)
        if operation == "instruct":
            progress("Running InstructPix2Pix...")
            image = self._open_required(data, "image")
            result, seed = self._real.instruct(data, image)
            output = output_dir / "result.png"
            result.save(output, format="PNG")
            return {"output_path": str(output), "seed": seed}
        if operation == "grounding":
            image = self._open_required(data, "image")
            detections = self._real.grounding(data, image, progress)
            return self._write_detections(output_dir, detections)
        raise RuntimeError(f"Unknown ML operation: {operation}")

    def _fake(
        self,
        operation: str,
        data: dict[str, Any],
        progress: Callable[[str], None],
    ) -> dict[str, Any]:
        output_dir = Path(data["output_dir"])
        if operation == "gpu_available":
            return {"available": False}
        if operation == "load_diffusion":
            self.diffusion_loaded = True
            progress("Loading diffusion model...")
            return {
                "model_path": str(data["model_path"]),
                "model_info": {"path": Path(data["model_path"]).name, "device": "cpu"},
            }
        if operation == "load_ip_adapter":
            if not self.diffusion_loaded:
                raise RuntimeError("No diffusion model loaded")
            return {"loaded": True}
        if operation == "diffusion":
            if not self.diffusion_loaded:
                raise RuntimeError("No diffusion model loaded")
            progress("Running diffusion...")
            if data["mode"] == "txt2img":
                result = Image.new(
                    "RGB", (int(data["width"]), int(data["height"])), "purple"
                )
            else:
                result = self._open_required(data, "image").convert("RGB")
            output = output_dir / "result.png"
            result.save(output, format="PNG")
            seed = 4242 if int(data["seed"]) == -1 else int(data["seed"])
            return {"output_path": str(output), "seed": seed}
        if operation == "load_instruct":
            self.instruct_loaded = True
            return {"loaded": True, "device": "cpu"}
        if operation == "instruct":
            if not self.instruct_loaded:
                raise RuntimeError("InstructPix2Pix model not loaded")
            progress("Running InstructPix2Pix...")
            output = output_dir / "result.png"
            self._open_required(data, "image").convert("RGB").save(
                output, format="PNG"
            )
            seed = 4343 if int(data["seed"]) == -1 else int(data["seed"])
            return {"output_path": str(output), "seed": seed}
        if operation == "grounding":
            progress("Grounding: detecting...")
            image = self._open_required(data, "image")
            mask = np.zeros((image.height, image.width), dtype=bool)
            mask[:, : max(1, image.width // 2)] = True
            return self._write_detections(
                output_dir,
                [{
                    "label": str(data["prompt"]),
                    "x0": 0,
                    "y0": 0,
                    "x1": max(1, image.width // 2),
                    "y1": image.height,
                    "score": 0.9,
                    "mask": mask if data.get("sam2_model_id") else None,
                }],
            )
        raise RuntimeError(f"Unknown ML operation: {operation}")

    @staticmethod
    def _open_required(data: dict[str, Any], name: str) -> Image.Image:
        path = data.get(f"{name}_path")
        if not isinstance(path, str):
            raise RuntimeError(f"ML operation requires {name}")
        with Image.open(path) as image:
            return image.copy()

    @classmethod
    def _open_images(
        cls,
        data: dict[str, Any],
        names: tuple[str, ...],
    ) -> dict[str, Image.Image | None]:
        result = {}
        for name in names:
            path = data.get(f"{name}_path")
            result[name] = cls._open_required(data, name) if path else None
        return result

    @staticmethod
    def _write_detections(
        output_dir: Path,
        detections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        serialized = []
        for index, raw in enumerate(detections):
            item = dict(raw)
            mask = item.pop("mask", None)
            if mask is not None:
                path = output_dir / f"mask-{index}.png"
                Image.fromarray(
                    np.asarray(mask, dtype=np.uint8) * 255, "L"
                ).save(path, format="PNG")
                item["mask_path"] = str(path)
            else:
                item["mask_path"] = None
            serialized.append(item)
        path = output_dir / "detections.json"
        path.write_text(json.dumps(serialized), encoding="utf-8")
        return {"detections_path": str(path)}


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
                raise RuntimeError("ML request is too large")
            request = json.loads(message)
            if (
                not isinstance(request, dict)
                or request.get("protocol") != PROTOCOL_VERSION
                or request.get("type") != "request"
                or not isinstance(request.get("request_id"), str)
                or not isinstance(request.get("operation"), str)
                or not isinstance(request.get("data"), dict)
            ):
                raise RuntimeError("Invalid ML request")
            request_id = request["request_id"]
            progress = lambda text: _send(
                wire,
                {
                    "protocol": PROTOCOL_VERSION,
                    "type": "progress",
                    "request_id": request_id,
                    "message": text,
                },
            )
            with redirect_stdout(sys.stderr):
                result = backend.execute(
                    request["operation"], request["data"], progress
                )
            _send(
                wire,
                {
                    "protocol": PROTOCOL_VERSION,
                    "type": "result",
                    "request_id": request_id,
                    "data": result,
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
        choices=("real", "fake", "hang", "crash", "malformed"),
        default="real",
    )
    return run(parser.parse_args().backend)


if __name__ == "__main__":
    raise SystemExit(main())
