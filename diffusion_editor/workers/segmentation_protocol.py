"""Versioned wire protocol shared by the segmentation client and worker."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 64 * 1024


class SegmentationProtocolError(RuntimeError):
    """Raised when the worker sends an invalid or incompatible message."""


@dataclass(frozen=True)
class SegmentationRuntime:
    python: str
    version: str
    abiflags: str
    gil_enabled: bool
    backend: str


@dataclass(frozen=True)
class SegmentationResponse:
    kind: Literal["ready", "progress", "result", "error"]
    request_id: str | None = None
    output_path: str | None = None
    message: str | None = None
    error: str | None = None
    runtime: SegmentationRuntime | None = None


def encode_message(payload: dict[str, Any]) -> bytes:
    message = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if len(message) > MAX_MESSAGE_BYTES:
        raise SegmentationProtocolError(
            "Segmentation protocol message is too large"
        )
    return message


def decode_response(message: bytes) -> SegmentationResponse:
    if not message or len(message) > MAX_MESSAGE_BYTES:
        raise SegmentationProtocolError(
            "Segmentation worker returned an invalid message size"
        )
    try:
        payload = json.loads(message)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SegmentationProtocolError(
            "Segmentation worker returned malformed JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise SegmentationProtocolError(
            "Segmentation worker response must be an object"
        )
    if payload.get("protocol") != PROTOCOL_VERSION:
        raise SegmentationProtocolError(
            "Segmentation worker protocol version mismatch"
        )

    kind = payload.get("type")
    if kind == "ready":
        runtime = payload.get("runtime")
        if not isinstance(runtime, dict):
            raise SegmentationProtocolError(
                "Segmentation ready response has no runtime"
            )
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
            raise SegmentationProtocolError(
                "Segmentation ready response has invalid runtime"
            )
        return SegmentationResponse(
            kind="ready",
            runtime=SegmentationRuntime(
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
        raise SegmentationProtocolError(
            "Segmentation worker returned an unknown response"
        )
    if kind == "progress":
        progress = payload.get("message")
        if not isinstance(progress, str):
            raise SegmentationProtocolError(
                "Segmentation progress response has no message"
            )
        return SegmentationResponse(
            kind="progress",
            request_id=request_id,
            message=progress,
        )
    if kind == "result":
        output_path = payload.get("output_path")
        if not isinstance(output_path, str):
            raise SegmentationProtocolError(
                "Segmentation result has no output path"
            )
        return SegmentationResponse(
            kind="result",
            request_id=request_id,
            output_path=output_path,
        )
    error = payload.get("error")
    if not isinstance(error, str):
        raise SegmentationProtocolError(
            "Segmentation error response has no message"
        )
    return SegmentationResponse(
        kind="error",
        request_id=request_id,
        error=error,
    )
