"""Standalone pose-estimation worker without editor or graphics imports."""

from __future__ import annotations

from contextlib import redirect_stdout
import json
from pathlib import Path
import platform
import sys
from typing import Any

from .pose_backend import PoseBackend
from .pose_protocol import MAX_MESSAGE_BYTES, PROTOCOL_VERSION, encode_message


def _send(wire, payload: dict[str, Any]) -> None:
    wire.write(encode_message(payload))
    wire.flush()


def run() -> int:
    wire = sys.stdout.buffer
    gil_probe = getattr(sys, "_is_gil_enabled", None)
    _send(wire, {
        "protocol": PROTOCOL_VERSION,
        "type": "ready",
        "runtime": {
            "python": platform.python_implementation(),
            "version": platform.python_version(),
            "abiflags": getattr(sys, "abiflags", ""),
            "gil_enabled": True if gil_probe is None else bool(gil_probe()),
        },
    })
    backend = PoseBackend()
    while message := sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 1):
        request_id = "unknown"
        try:
            if len(message) > MAX_MESSAGE_BYTES:
                raise RuntimeError("Pose request is too large")
            request = json.loads(message)
            if (
                    not isinstance(request, dict)
                    or request.get("protocol") != PROTOCOL_VERSION
                    or request.get("type") != "estimate"
                    or not isinstance(request.get("request_id"), str)):
                raise RuntimeError("Invalid pose request")
            request_id = request["request_id"]
            profile_id = request.get("profile_id")
            image_path = request.get("image_path")
            output_path = request.get("output_path")
            if not all(
                    isinstance(value, str) and value
                    for value in (profile_id, image_path, output_path)):
                raise RuntimeError("Pose request has invalid fields")
            _send(wire, {
                "protocol": PROTOCOL_VERSION,
                "type": "progress",
                "request_id": request_id,
                "message": f"Running {profile_id}",
            })
            with redirect_stdout(sys.stderr):
                result = backend.estimate(profile_id, image_path)
            _send(wire, {
                "protocol": PROTOCOL_VERSION,
                "type": "progress",
                "request_id": request_id,
                "message": f"Serializing {profile_id}",
            })
            Path(output_path).write_text(
                json.dumps(result.to_dict(), ensure_ascii=False),
                encoding="utf-8",
            )
            _send(wire, {
                "protocol": PROTOCOL_VERSION,
                "type": "result",
                "request_id": request_id,
                "output_path": output_path,
            })
        except Exception as exc:
            _send(wire, {
                "protocol": PROTOCOL_VERSION,
                "type": "error",
                "request_id": request_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
