"""Small JSON-lines protocol for the persistent Pixal3D stage worker."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 256 * 1024


class Pixal3DProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Pixal3DWorkerResponse:
    kind: Literal["ready", "result", "error"]
    request_id: str | None = None
    error: str | None = None


def encode_message(payload: dict[str, Any]) -> bytes:
    message = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if len(message) > MAX_MESSAGE_BYTES:
        raise Pixal3DProtocolError("Pixal3D protocol message is too large")
    return message


def decode_response(message: bytes) -> Pixal3DWorkerResponse:
    if not message or len(message) > MAX_MESSAGE_BYTES:
        raise Pixal3DProtocolError("Pixal3D worker returned an invalid message")
    try:
        payload = json.loads(message)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Pixal3DProtocolError(
            "Pixal3D worker returned malformed JSON"
        ) from exc
    if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL_VERSION:
        raise Pixal3DProtocolError("Pixal3D worker protocol version mismatch")
    kind = payload.get("type")
    if kind == "ready":
        return Pixal3DWorkerResponse("ready")
    request_id = payload.get("request_id")
    if kind not in {"result", "error"} or not isinstance(request_id, str):
        raise Pixal3DProtocolError("Pixal3D worker returned an unknown response")
    if kind == "result":
        return Pixal3DWorkerResponse("result", request_id=request_id)
    error = payload.get("error")
    if not isinstance(error, str):
        raise Pixal3DProtocolError("Pixal3D worker error has no message")
    return Pixal3DWorkerResponse("error", request_id=request_id, error=error)
