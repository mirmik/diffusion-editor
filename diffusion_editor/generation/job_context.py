"""Immutable identity and input snapshots for asynchronous generation jobs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Callable, Literal
import uuid

import numpy as np
from PIL import Image, ImageFilter

from ..document.layer import Layer
from ..document.result_paste import paste_result
from ..document.tool import Tool
from .provenance import (
    GenerationProvenance,
    RequestProvenance,
    capture_tool_state,
)
from .types import Rect


GenerationKind = Literal["diffusion", "instruct", "lama", "segmentation"]


class ResultApplicationPolicy(str, Enum):
    """Explicit stale-result behavior; only strict rejection is implemented."""

    REJECT_STALE = "reject_stale"
    REBASE = "rebase"
    NEW_LAYER = "new_layer"


@dataclass(frozen=True)
class JobDocumentState:
    """Document identity captured atomically when a request is submitted."""

    session_id: str
    revision: int


@dataclass(frozen=True)
class FrozenImage:
    """A PIL image represented by immutable value data."""

    mode: str
    size: tuple[int, int]
    data: bytes

    @classmethod
    def capture(cls, image: Image.Image | None) -> "FrozenImage | None":
        if image is None:
            return None
        snapshot = image.copy()
        return cls(
            mode=snapshot.mode,
            size=(int(snapshot.width), int(snapshot.height)),
            data=snapshot.tobytes(),
        )

    def to_image(self) -> Image.Image:
        return Image.frombytes(self.mode, self.size, self.data)

    @property
    def content_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.mode.encode("utf-8"))
        digest.update(f"{self.size[0]}x{self.size[1]}".encode("ascii"))
        digest.update(self.data)
        return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class FrozenArray:
    """A NumPy array represented by immutable value data."""

    shape: tuple[int, ...]
    dtype: str
    data: bytes

    @classmethod
    def capture(cls, array: np.ndarray | None) -> "FrozenArray | None":
        if array is None:
            return None
        snapshot = np.ascontiguousarray(array)
        return cls(
            shape=tuple(int(value) for value in snapshot.shape),
            dtype=snapshot.dtype.str,
            data=snapshot.tobytes(),
        )

    def to_array(self) -> np.ndarray:
        array = np.frombuffer(self.data, dtype=np.dtype(self.dtype))
        return np.ascontiguousarray(array.reshape(self.shape)).copy()

    @property
    def content_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.dtype.encode("ascii"))
        digest.update(repr(self.shape).encode("ascii"))
        digest.update(self.data)
        return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class FrozenPasteContext:
    """All target geometry and masking needed to apply a generated image."""

    canvas_rect: Rect
    layer_local_rect: Rect
    layer_mask: FrozenArray | None

    @classmethod
    def capture(cls, layer: Layer, tool: Tool) -> "FrozenPasteContext":
        patch_x = int(getattr(tool, "patch_x"))
        patch_y = int(getattr(tool, "patch_y"))
        patch_w = int(getattr(tool, "patch_w"))
        patch_h = int(getattr(tool, "patch_h"))
        canvas_rect = (
            patch_x,
            patch_y,
            patch_x + patch_w,
            patch_y + patch_h,
        )
        local_x = patch_x - int(layer.x)
        local_y = patch_y - int(layer.y)
        layer_mask = (
            FrozenArray.capture(layer.mask.to_uint8())
            if layer.has_mask()
            else None
        )
        return cls(
            canvas_rect=canvas_rect,
            layer_local_rect=(
                local_x,
                local_y,
                local_x + patch_w,
                local_y + patch_h,
            ),
            layer_mask=layer_mask,
        )

    @property
    def width(self) -> int:
        return self.canvas_rect[2] - self.canvas_rect[0]

    @property
    def height(self) -> int:
        return self.canvas_rect[3] - self.canvas_rect[1]

    def matches_layer(self, layer: Layer) -> bool:
        """Check mutable target state without using it for result placement."""
        tool = layer.tool
        if tool is None:
            return False
        current_rect = (
            int(getattr(tool, "patch_x", 0)),
            int(getattr(tool, "patch_y", 0)),
            int(getattr(tool, "patch_x", 0))
            + int(getattr(tool, "patch_w", 0)),
            int(getattr(tool, "patch_y", 0))
            + int(getattr(tool, "patch_h", 0)),
        )
        if current_rect != self.canvas_rect:
            return False
        if self.layer_mask is None:
            return not layer.has_mask()
        if not layer.has_mask():
            return False
        current_mask = FrozenArray.capture(layer.mask.to_uint8())
        return current_mask == self.layer_mask


@dataclass(frozen=True)
class InferenceJobContext:
    """Immutable contract joining an engine terminal event to a document."""

    job_id: str
    kind: GenerationKind
    document_session_id: str
    base_revision: int
    layer_id: str
    layer_bounds: Rect
    target_pixel_revision: int
    tool_type: str | None
    input_image: FrozenImage | None = None
    input_array: FrozenArray | None = None
    request_mask: FrozenImage | None = None
    reference_image: FrozenImage | None = None
    target_mask: FrozenArray | None = None
    paste: FrozenPasteContext | None = None
    model_provenance: tuple[tuple[str, str], ...] = ()
    request_provenance: RequestProvenance | None = None
    tool_state_fingerprint: str = ""
    result_provenance: GenerationProvenance | None = None
    application_policy: ResultApplicationPolicy = (
        ResultApplicationPolicy.REJECT_STALE
    )

    def provenance(self, key: str, default: str = "") -> str:
        return dict(self.model_provenance).get(key, default)


def new_job_id() -> str:
    return f"job_{uuid.uuid4().hex}"


def standalone_document_state() -> JobDocumentState:
    """Compatibility state for controllers used without an application."""
    return JobDocumentState("standalone", 0)


def capture_job_context(
        *,
        kind: GenerationKind,
        document_state: JobDocumentState,
        layer: Layer,
        job_id: str,
        input_image: Image.Image | None = None,
        input_array: np.ndarray | None = None,
        request_mask: Image.Image | None = None,
        reference_image: Image.Image | None = None,
        paste: FrozenPasteContext | None = None,
        model_provenance: tuple[tuple[str, str], ...] = (),
        request_provenance: RequestProvenance | None = None,
) -> InferenceJobContext:
    tool = layer.tool
    return InferenceJobContext(
        job_id=job_id,
        kind=kind,
        document_session_id=document_state.session_id,
        base_revision=int(document_state.revision),
        layer_id=layer.id,
        layer_bounds=layer.bounds,
        target_pixel_revision=layer.pixel_revision,
        tool_type=(
            str(tool.tool_type)
            if tool is not None
            else None
        ),
        input_image=FrozenImage.capture(input_image),
        input_array=FrozenArray.capture(input_array),
        request_mask=FrozenImage.capture(request_mask),
        reference_image=FrozenImage.capture(reference_image),
        target_mask=FrozenArray.capture(layer.mask.to_uint8()),
        paste=paste,
        model_provenance=tuple(
            (str(key), str(value))
            for key, value in model_provenance
        ),
        request_provenance=request_provenance,
        tool_state_fingerprint=capture_tool_state(tool).fingerprint,
    )


def engine_supports_job_ids(engine: object) -> bool:
    return bool(getattr(engine, "supports_job_ids", False))


def submit_with_job_id(
        engine: object,
        method_name: str,
        *args,
        job_id: str,
        **kwargs,
) -> bool:
    """Submit to a real job-aware engine or a legacy test double."""
    method: Callable = getattr(engine, method_name)
    if engine_supports_job_ids(engine):
        return bool(method(*args, job_id=job_id, **kwargs))
    return bool(method(*args, **kwargs))


def terminal_job_matches(
        engine: object,
        event_job_id: str | None,
        expected_job_id: str,
) -> bool:
    if event_job_id is not None:
        return event_job_id == expected_job_id
    # Old third-party/test engines predate the identity protocol. Production
    # engines explicitly advertise support and must always return the token.
    return not engine_supports_job_ids(engine)


def cancel_engine(engine: object) -> bool:
    cancel = getattr(engine, "cancel", None)
    return bool(cancel()) if callable(cancel) else False


@dataclass(frozen=True)
class ApplyFrozenGeneratedResultCommand:
    """Apply an output using only the immutable job paste snapshot."""

    layer: Layer
    result_image: Image.Image
    paste: FrozenPasteContext
    label: str
    provenance: GenerationProvenance | None = None

    def apply(self, layer_stack) -> None:
        replacement = np.zeros_like(self.layer.image)
        mask_arg: np.ndarray | None = None
        if self.paste.layer_mask is not None:
            mask_pil = Image.fromarray(
                self.paste.layer_mask.to_array(), "L")
            mask_pil = mask_pil.filter(ImageFilter.MaxFilter(7))
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=4))
            mask_arg = np.array(mask_pil, dtype=np.uint8)

        x0, y0, x1, y1 = self.paste.layer_local_rect
        intersects = paste_result(
            replacement,
            self.result_image,
            x0,
            y0,
            x1 - x0,
            y1 - y0,
            mask=mask_arg,
        )
        if not intersects:
            return
        self.layer.image[:] = replacement
        if self.provenance is not None and self.layer.tool is not None:
            self.layer.tool.generation_provenance = self.provenance
            if hasattr(self.layer.tool, "model_identity"):
                self.layer.tool.model_identity = self.provenance.model
        layer_stack.mark_layer_dirty(self.layer)
        if layer_stack.on_changed:
            layer_stack.on_changed()
