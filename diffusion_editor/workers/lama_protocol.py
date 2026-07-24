"""Versioned wire protocol shared by the LaMa client and worker."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 64 * 1024


class LamaProtocolError(RuntimeError):
    """Raised when the worker sends an invalid or incompatible message."""


@dataclass(frozen=True)
class LamaRuntime:
    python: str
    version: str
    abiflags: str
    gil_enabled: bool
    backend: str


@dataclass(frozen=True)
class LamaResponse:
    kind: Literal["ready", "result", "error"]
    request_id: str | None = None
    output_path: str | None = None
    error: str | None = None
    runtime: LamaRuntime | None = None


def encode_message(payload: dict[str, Any]) -> bytes:
    message = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if len(message) > MAX_MESSAGE_BYTES:
        raise LamaProtocolError("LaMa protocol message is too large")
    return message


def decode_response(message: bytes) -> LamaResponse:
    if not message or len(message) > MAX_MESSAGE_BYTES:
        raise LamaProtocolError("LaMa worker returned an invalid message size")
    try:
        payload = json.loads(message)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LamaProtocolError("LaMa worker returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise LamaProtocolError("LaMa worker response must be an object")
    if payload.get("protocol") != PROTOCOL_VERSION:
        raise LamaProtocolError("LaMa worker protocol version mismatch")

    kind = payload.get("type")
    if kind == "ready":
        runtime = payload.get("runtime")
        if not isinstance(runtime, dict):
            raise LamaProtocolError("LaMa ready response has no runtime")
        values = (
            runtime.get("python"),
            runtime.get("version"),
            runtime.get("abiflags"),
            runtime.get("gil_enabled"),
            runtime.get("backend"),
        )
        if (
            not all(isinstance(value, str) for value in values[:3])
            or not isinstance(values[3], bool)
            or not isinstance(values[4], str)
        ):
            raise LamaProtocolError("LaMa ready response has invalid runtime")
        return LamaResponse(
            kind="ready",
            runtime=LamaRuntime(
                python=values[0],
                version=values[1],
                abiflags=values[2],
                gil_enabled=values[3],
                backend=values[4],
            ),
        )

    request_id = payload.get("request_id")
    if kind not in {"result", "error"} or not isinstance(request_id, str):
        raise LamaProtocolError("LaMa worker returned an unknown response")
    if kind == "result":
        output_path = payload.get("output_path")
        if not isinstance(output_path, str):
            raise LamaProtocolError("LaMa result has no output path")
        return LamaResponse(
            kind="result",
            request_id=request_id,
            output_path=output_path,
        )
    error = payload.get("error")
    if not isinstance(error, str):
        raise LamaProtocolError("LaMa error response has no message")
    return LamaResponse(kind="error", request_id=request_id, error=error)
