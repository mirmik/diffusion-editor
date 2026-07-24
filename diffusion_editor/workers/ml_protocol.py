"""Versioned JSON-lines protocol shared by the ML client and worker."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 256 * 1024


class MlProtocolError(RuntimeError):
    """Raised when the ML worker sends an invalid or incompatible message."""


@dataclass(frozen=True)
class MlRuntime:
    python: str
    version: str
    abiflags: str
    gil_enabled: bool
    backend: str


@dataclass(frozen=True)
class MlResponse:
    kind: Literal["ready", "progress", "result", "error"]
    request_id: str | None = None
    message: str | None = None
    data: dict[str, Any] | None = None
    error: str | None = None
    runtime: MlRuntime | None = None


def encode_message(payload: dict[str, Any]) -> bytes:
    message = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if len(message) > MAX_MESSAGE_BYTES:
        raise MlProtocolError("ML protocol message is too large")
    return message


def decode_response(message: bytes) -> MlResponse:
    if not message or len(message) > MAX_MESSAGE_BYTES:
        raise MlProtocolError("ML worker returned an invalid message size")
    try:
        payload = json.loads(message)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MlProtocolError("ML worker returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise MlProtocolError("ML worker response must be an object")
    if payload.get("protocol") != PROTOCOL_VERSION:
        raise MlProtocolError("ML worker protocol version mismatch")

    kind = payload.get("type")
    if kind == "ready":
        runtime = payload.get("runtime")
        if not isinstance(runtime, dict):
            raise MlProtocolError("ML ready response has no runtime")
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
            raise MlProtocolError("ML ready response has invalid runtime")
        return MlResponse(
            kind="ready",
            runtime=MlRuntime(
                python=values[0],
                version=values[1],
                abiflags=values[2],
                gil_enabled=values[3],
                backend=values[4],
            ),
        )

    request_id = payload.get("request_id")
    if (
        kind not in {"progress", "result", "error"}
        or not isinstance(request_id, str)
    ):
        raise MlProtocolError("ML worker returned an unknown response")
    if kind == "progress":
        message_text = payload.get("message")
        if not isinstance(message_text, str):
            raise MlProtocolError("ML progress response has no message")
        return MlResponse(
            kind="progress",
            request_id=request_id,
            message=message_text,
        )
    if kind == "result":
        data = payload.get("data")
        if not isinstance(data, dict):
            raise MlProtocolError("ML result response has no data")
        return MlResponse(kind="result", request_id=request_id, data=data)
    error = payload.get("error")
    if not isinstance(error, str):
        raise MlProtocolError("ML error response has no message")
    return MlResponse(kind="error", request_id=request_id, error=error)
