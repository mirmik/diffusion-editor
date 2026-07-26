"""Versioned, forward-compatible generation provenance value objects."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


PROVENANCE_SCHEMA_VERSION = 1
REQUEST_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ModelIdentityStatus(str, Enum):
    CONFIRMED_IMMUTABLE = "confirmed_immutable"
    FLOATING = "floating"
    UNKNOWN = "unknown"
    HASH_MISMATCH = "hash_mismatch"


class ModelIdentityPolicy(str, Enum):
    WARN = "warn"
    REQUIRE_IMMUTABLE = "require_immutable"


class ModelIdentityPolicyError(RuntimeError):
    pass


def _validate_json(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            _validate_json(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} contains unsupported value {type(value).__name__}")


@dataclass(frozen=True)
class FrozenJsonObject:
    """An immutable canonical JSON object with copy-on-read semantics."""

    canonical_json: str = "{}"

    @classmethod
    def capture(
            cls,
            value: Mapping[str, Any] | None,
    ) -> "FrozenJsonObject":
        data = dict(value or {})
        _validate_json(data)
        return cls(json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ))

    def to_dict(self) -> dict[str, Any]:
        value = json.loads(self.canonical_json)
        if not isinstance(value, dict):
            raise ValueError("frozen JSON value is not an object")
        return value


def _split_fields(
        raw: Mapping[str, Any],
        known: set[str],
) -> tuple[dict[str, Any], FrozenJsonObject]:
    data = dict(raw)
    selected = {key: data.get(key) for key in known}
    extra = {
        key: value for key, value in data.items()
        if key not in known
    }
    return selected, FrozenJsonObject.capture(extra)


def _merge_extensions(
        known: dict[str, Any],
        extensions: FrozenJsonObject,
) -> dict[str, Any]:
    result = extensions.to_dict()
    result.update(known)
    return result


@dataclass(frozen=True)
class ModelIdentity:
    provider: str
    repository: str | None
    revision: str | None
    content_hash: str | None
    local_override: str | None
    status: ModelIdentityStatus | str
    warning: str | None = None
    extensions: FrozenJsonObject = FrozenJsonObject()

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("model identity provider must not be empty")
        if (
                self.content_hash is not None
                and not _SHA256_RE.fullmatch(self.content_hash)):
            raise ValueError(
                "model content_hash must be 'sha256:' plus 64 lowercase hex digits")
        if (
                self.is_confirmed_immutable
                and self.content_hash is None
                and self.revision is None):
            raise ValueError(
                "confirmed immutable model identity requires a revision or hash")

    @property
    def is_confirmed_immutable(self) -> bool:
        return self.status in (
            ModelIdentityStatus.CONFIRMED_IMMUTABLE,
            ModelIdentityStatus.CONFIRMED_IMMUTABLE.value,
        )

    def to_dict(self) -> dict[str, Any]:
        return _merge_extensions({
            "provider": self.provider,
            "repository": self.repository,
            "revision": self.revision,
            "content_hash": self.content_hash,
            "local_override": self.local_override,
            "status": (
                self.status.value
                if isinstance(self.status, ModelIdentityStatus)
                else str(self.status)
            ),
            "warning": self.warning,
        }, self.extensions)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ModelIdentity":
        known = {
            "provider",
            "repository",
            "revision",
            "content_hash",
            "local_override",
            "status",
            "warning",
        }
        fields, extensions = _split_fields(raw, known)
        status_raw = str(fields["status"] or ModelIdentityStatus.UNKNOWN.value)
        try:
            status = ModelIdentityStatus(status_raw)
        except ValueError:
            # Keep a future wire value intact. Current code treats it as
            # unconfirmed, but saving the project must not erase it.
            status = status_raw
        content_hash = fields["content_hash"]
        if content_hash is not None:
            content_hash = str(content_hash).lower()
        return cls(
            provider=str(fields["provider"] or "unknown"),
            repository=(
                str(fields["repository"])
                if fields["repository"] is not None else None
            ),
            revision=(
                str(fields["revision"])
                if fields["revision"] is not None else None
            ),
            content_hash=content_hash,
            local_override=(
                str(fields["local_override"])
                if fields["local_override"] is not None else None
            ),
            status=status,
            warning=(
                str(fields["warning"])
                if fields["warning"] is not None else None
            ),
            extensions=extensions,
        )


@dataclass(frozen=True)
class RequestProvenance:
    kind: str
    parameters: FrozenJsonObject
    schema_version: int = REQUEST_SCHEMA_VERSION
    extensions: FrozenJsonObject = FrozenJsonObject()

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("request provenance schema_version must be >= 1")
        if not self.kind:
            raise ValueError("request provenance kind must not be empty")

    @classmethod
    def capture(
            cls,
            kind: str,
            parameters: Mapping[str, Any],
    ) -> "RequestProvenance":
        return cls(
            kind=kind,
            parameters=FrozenJsonObject.capture(parameters),
        )

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        return f"request-sha256:{digest}"

    def to_dict(self) -> dict[str, Any]:
        return _merge_extensions({
            "schema_version": self.schema_version,
            "kind": self.kind,
            "parameters": self.parameters.to_dict(),
        }, self.extensions)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RequestProvenance":
        known = {"schema_version", "kind", "parameters"}
        fields, extensions = _split_fields(raw, known)
        parameters = fields["parameters"]
        if not isinstance(parameters, dict):
            parameters = {}
        return cls(
            schema_version=int(fields["schema_version"] or 1),
            kind=str(fields["kind"] or "unknown"),
            parameters=FrozenJsonObject.capture(parameters),
            extensions=extensions,
        )


@dataclass(frozen=True)
class GenerationProvenance:
    operation: str
    model: ModelIdentity
    request: RequestProvenance
    seed: int | None
    width: int | None
    height: int | None
    runtime: FrozenJsonObject
    warnings: tuple[str, ...] = ()
    schema_version: int = PROVENANCE_SCHEMA_VERSION
    extensions: FrozenJsonObject = FrozenJsonObject()

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("generation provenance schema_version must be >= 1")
        if not self.operation:
            raise ValueError("generation provenance operation must not be empty")
        if self.width is not None and self.width < 1:
            raise ValueError("provenance width must be positive")
        if self.height is not None and self.height < 1:
            raise ValueError("provenance height must be positive")

    def with_request(
            self,
            request: RequestProvenance,
    ) -> "GenerationProvenance":
        return replace(self, request=request)

    def to_dict(self) -> dict[str, Any]:
        return _merge_extensions({
            "schema_version": self.schema_version,
            "operation": self.operation,
            "model": self.model.to_dict(),
            "request": self.request.to_dict(),
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "runtime": self.runtime.to_dict(),
            "warnings": list(self.warnings),
        }, self.extensions)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "GenerationProvenance":
        known = {
            "schema_version",
            "operation",
            "model",
            "request",
            "seed",
            "width",
            "height",
            "runtime",
            "warnings",
        }
        fields, extensions = _split_fields(raw, known)
        model = fields["model"] if isinstance(fields["model"], dict) else {}
        request = (
            fields["request"] if isinstance(fields["request"], dict) else {}
        )
        runtime = (
            fields["runtime"] if isinstance(fields["runtime"], dict) else {}
        )
        warnings = fields["warnings"]
        if not isinstance(warnings, list):
            warnings = []
        return cls(
            schema_version=int(fields["schema_version"] or 1),
            operation=str(fields["operation"] or "unknown"),
            model=ModelIdentity.from_dict(model),
            request=RequestProvenance.from_dict(request),
            seed=(
                int(fields["seed"])
                if fields["seed"] is not None else None
            ),
            width=(
                int(fields["width"])
                if fields["width"] is not None else None
            ),
            height=(
                int(fields["height"])
                if fields["height"] is not None else None
            ),
            runtime=FrozenJsonObject.capture(runtime),
            warnings=tuple(str(item) for item in warnings),
            extensions=extensions,
        )


def enforce_model_identity_policy(
        identity: ModelIdentity,
        policy: ModelIdentityPolicy | str,
) -> tuple[str, ...]:
    policy = ModelIdentityPolicy(policy)
    if identity.is_confirmed_immutable:
        return ()
    status_label = (
        identity.status.value
        if isinstance(identity.status, ModelIdentityStatus)
        else identity.status
    )
    warning = identity.warning or (
        f"Model identity is {status_label}; "
        "exact artifact is not confirmed"
    )
    if policy == ModelIdentityPolicy.REQUIRE_IMMUTABLE:
        raise ModelIdentityPolicyError(warning)
    return (warning,)


def sha256_file(path: str | Path, *, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def resolve_local_model_identity(
        path: str | Path,
        *,
        expected_content_hash: str | None = None,
        policy: ModelIdentityPolicy | str = ModelIdentityPolicy.WARN,
) -> tuple[ModelIdentity, tuple[str, ...]]:
    candidate = Path(path).expanduser().absolute()
    actual_hash = sha256_file(candidate) if candidate.is_file() else None
    expected = (
        expected_content_hash.lower()
        if expected_content_hash is not None else None
    )
    if expected is not None and not _SHA256_RE.fullmatch(expected):
        raise ValueError("expected model hash must be a sha256 digest")

    if actual_hash is None:
        warning = f"Model artifact does not exist for hashing: {candidate}"
        status = ModelIdentityStatus.UNKNOWN
    elif expected is not None and expected != actual_hash:
        warning = (
            f"Model hash mismatch: expected {expected}, resolved {actual_hash}"
        )
        status = ModelIdentityStatus.HASH_MISMATCH
    else:
        warning = None
        status = ModelIdentityStatus.CONFIRMED_IMMUTABLE

    identity = ModelIdentity(
        provider="local",
        repository=candidate.name,
        revision=None,
        content_hash=actual_hash,
        local_override=str(candidate),
        status=status,
        warning=warning,
    )
    warnings = enforce_model_identity_policy(identity, policy)
    return identity, warnings


def floating_model_identity(
        provider: str,
        repository: str,
        *,
        revision: str | None = None,
        local_override: str | None = None,
) -> ModelIdentity:
    label = (
        f"{provider}/{repository}@{revision}"
        if revision else f"{provider}/{repository}"
    )
    return ModelIdentity(
        provider=provider,
        repository=repository,
        revision=revision,
        content_hash=None,
        local_override=local_override,
        status=ModelIdentityStatus.FLOATING,
        warning=(
            f"Model identity {label} is floating because no confirmed "
            "content hash is available"
        ),
    )


def capture_tool_state(tool: object | None) -> RequestProvenance:
    """Capture every mutable generation setting relevant to a pending job."""
    if tool is None:
        return RequestProvenance.capture("no_tool", {})
    kind = str(getattr(tool, "tool_type", "") or "unknown")
    parameters: dict[str, Any] = {}
    if kind == "diffusion":
        model_identity = getattr(tool, "model_identity", None)
        identity_policy = getattr(
            tool,
            "model_identity_policy",
            ModelIdentityPolicy.WARN,
        )
        parameters.update({
            "prompt": str(getattr(tool, "prompt", "")),
            "negative_prompt": str(getattr(tool, "negative_prompt", "")),
            "strength": float(getattr(tool, "strength", 0.0)),
            "guidance_scale": float(getattr(tool, "guidance_scale", 0.0)),
            "steps": int(getattr(tool, "steps", 0)),
            "seed": int(getattr(tool, "seed", -1)),
            "mode": str(getattr(tool, "mode", "")),
            "masked_content": str(getattr(tool, "masked_content", "")),
            "ip_adapter_layer_id": getattr(
                tool, "ip_adapter_layer_id", None),
            "ip_adapter_scale": float(getattr(
                tool, "ip_adapter_scale", 0.0)),
            "resize_to_model_resolution": bool(getattr(
                tool, "resize_to_model_resolution", False)),
            "model_path": str(getattr(tool, "model_path", "")),
            "prediction_type": str(getattr(tool, "prediction_type", "")),
            "model_identity": (
                model_identity.to_dict()
                if isinstance(model_identity, ModelIdentity) else None
            ),
            "model_identity_policy": (
                identity_policy.value
                if isinstance(identity_policy, ModelIdentityPolicy)
                else str(identity_policy)
            ),
        })
    elif kind == "instruct":
        parameters.update({
            "instruction": str(getattr(tool, "instruction", "")),
            "image_guidance_scale": float(getattr(
                tool, "image_guidance_scale", 0.0)),
            "guidance_scale": float(getattr(tool, "guidance_scale", 0.0)),
            "steps": int(getattr(tool, "steps", 0)),
            "seed": int(getattr(tool, "seed", -1)),
        })
    return RequestProvenance.capture(f"{kind}_tool_state", parameters)
