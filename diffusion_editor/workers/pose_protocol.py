"""Versioned JSON-lines protocol shared by pose client and worker."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 64 * 1024


class PoseProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class PoseRuntime:
    python: str
    version: str
    abiflags: str
    gil_enabled: bool


@dataclass(frozen=True)
class PoseResponse:
    kind: Literal["ready", "progress", "result", "error"]
    request_id: str | None = None
    output_path: str | None = None
    message: str | None = None
    error: str | None = None
    runtime: PoseRuntime | None = None


def encode_message(payload: dict[str, Any]) -> bytes:
    message = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if len(message) > MAX_MESSAGE_BYTES:
        raise PoseProtocolError("Pose protocol message is too large")
    return message


def decode_response(message: bytes) -> PoseResponse:
    if not message or len(message) > MAX_MESSAGE_BYTES:
        raise PoseProtocolError("Pose worker returned an invalid message size")
    try:
        payload = json.loads(message)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PoseProtocolError("Pose worker returned malformed JSON") from exc
    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL_VERSION:
        raise PoseProtocolError("Pose worker protocol version mismatch")
    kind = payload.get("type")
    if kind == "ready":
        runtime = payload.get("runtime")
        if not isinstance(runtime, dict):
            raise PoseProtocolError("Pose ready response has no runtime")
        try:
            gil_enabled = runtime["gil_enabled"]
            if not isinstance(gil_enabled, bool):
                raise TypeError
            parsed = PoseRuntime(
                python=str(runtime["python"]),
                version=str(runtime["version"]),
                abiflags=str(runtime["abiflags"]),
                gil_enabled=gil_enabled,
            )
        except (KeyError, TypeError) as exc:
            raise PoseProtocolError("Pose ready response has invalid runtime") from exc
        return PoseResponse(kind="ready", runtime=parsed)
    request_id = payload.get("request_id")
    if kind not in {"progress", "result", "error"} or not isinstance(request_id, str):
        raise PoseProtocolError("Pose worker returned an unknown response")
    if kind == "progress":
        message_value = payload.get("message")
        if not isinstance(message_value, str):
            raise PoseProtocolError("Pose progress response has no message")
        return PoseResponse(
            kind="progress", request_id=request_id, message=message_value)
    if kind == "result":
        output_path = payload.get("output_path")
        if not isinstance(output_path, str):
            raise PoseProtocolError("Pose result has no output path")
        return PoseResponse(
            kind="result", request_id=request_id, output_path=output_path)
    error = payload.get("error")
    if not isinstance(error, str):
        raise PoseProtocolError("Pose error response has no message")
    return PoseResponse(kind="error", request_id=request_id, error=error)
