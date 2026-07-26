"""Application workflow for LaMa inpainting tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from PIL import Image
from tcbase import log

from ..document.layer import Layer
from ..document.tool import LamaTool
from ..engines.lama_engine import LamaEngine
from .job_context import (
    FrozenImage,
    FrozenPasteContext,
    InferenceJobContext,
    JobDocumentState,
    cancel_engine,
    capture_job_context,
    new_job_id,
    standalone_document_state,
    submit_with_job_id,
    terminal_job_matches,
)
from .patch_resolver import (
    apply_patch_source_to_tool,
    extract_layer_mask_patch,
    source_patch_from_mask,
)
from .provenance import RequestProvenance
from .types import LamaRequest, LamaResult


@dataclass(frozen=True)
class LamaControllerEvent:
    status: str | None = None
    inference_result: (
        tuple[InferenceJobContext, Image.Image] | None
    ) = None
    inference_error: str | None = None


@dataclass(frozen=True)
class _PendingLamaJob:
    context: InferenceJobContext
    request: LamaRequest


class LamaGenerationController:
    def __init__(
            self,
            *,
            engine: LamaEngine,
            composite_below: Callable[[Layer], np.ndarray | None],
            document_state: Callable[[], JobDocumentState] | None = None,
            job_id_factory: Callable[[], str] = new_job_id):
        self._engine = engine
        self._composite_below = composite_below
        self._document_state = document_state or standalone_document_state
        self._job_id_factory = job_id_factory
        self._pending_job: _PendingLamaJob | None = None
        self._active_operation_id: str | None = None
        self._invalidated_operations: dict[str, str] = {}

    @property
    def pending_context(self) -> InferenceJobContext | None:
        job = self._pending_job
        return job.context if job is not None else None

    @property
    def pending_contexts(self) -> tuple[InferenceJobContext, ...]:
        context = self.pending_context
        return (context,) if context is not None else ()

    @property
    def pending_layer(self) -> None:
        return None

    def clear_pending_layer(self, layer: Layer) -> None:
        self.invalidate_layer(layer.id, "target layer or tool was detached")

    def invalidate_layer(self, layer_id: str, reason: str) -> bool:
        job = self._pending_job
        if job is None or job.context.layer_id != layer_id:
            return False
        self._pending_job = None
        self._invalidate_active(reason)
        return True

    def invalidate_job(self, job_id: str, reason: str) -> bool:
        job = self._pending_job
        if job is None or job.context.job_id != job_id:
            return False
        self._pending_job = None
        self._invalidate_active(reason)
        return True

    def invalidate_all(self, reason: str) -> bool:
        if self._pending_job is None:
            return False
        self._pending_job = None
        self._invalidate_active(reason)
        return True

    def _invalidate_active(self, reason: str) -> None:
        operation_id = self._active_operation_id
        if operation_id is not None:
            self._invalidated_operations[operation_id] = reason
            cancel_engine(self._engine)

    def start_remove(self, layer: Layer) -> LamaControllerEvent:
        if self._engine.is_busy or self._active_operation_id is not None:
            return LamaControllerEvent()
        tool = layer.tool
        if not isinstance(tool, LamaTool):
            return LamaControllerEvent()
        if not layer.has_mask():
            return LamaControllerEvent()

        composite = self._composite_below(layer)
        if composite is None:
            return LamaControllerEvent()
        patch = source_patch_from_mask(layer, composite)
        if patch is None:
            return LamaControllerEvent()
        apply_patch_source_to_tool(tool, patch)

        mask = extract_layer_mask_patch(layer, patch.canvas_rect)
        input_image = FrozenImage.capture(patch.image)
        request_mask = FrozenImage.capture(mask)
        assert input_image is not None and request_mask is not None
        request = LamaRequest(
            image=input_image.to_image(),
            mask_image=request_mask.to_image(),
        )
        context = capture_job_context(
            kind="lama",
            document_state=self._document_state(),
            layer=layer,
            job_id=self._job_id_factory(),
            input_image=input_image.to_image(),
            request_mask=request_mask.to_image(),
            paste=FrozenPasteContext.capture(layer, tool),
            model_provenance=(("model_id", "big-lama"),),
            request_provenance=RequestProvenance.capture(
                "lama",
                {
                    "input_image_hash": input_image.content_hash,
                    "mask_image_hash": request_mask.content_hash,
                },
            ),
        )
        submitted = submit_with_job_id(
            self._engine,
            "submit_request",
            request,
            job_id=context.job_id,
        )
        if not submitted:
            return LamaControllerEvent()

        self._pending_job = _PendingLamaJob(context=context, request=request)
        self._active_operation_id = context.job_id
        return LamaControllerEvent(status="Removing objects (LaMa)...")

    def poll(self) -> LamaControllerEvent | None:
        engine_event = self._engine.poll_event()
        if engine_event is None:
            return None

        expected_id = self._active_operation_id
        if expected_id is None:
            return LamaControllerEvent(
                status="LaMa event ignored: no active job")
        if not terminal_job_matches(
                self._engine, engine_event.job_id, expected_id):
            return LamaControllerEvent(
                status="LaMa event ignored: stale job identity")

        invalidation_reason = self._invalidated_operations.pop(
            expected_id, None)
        self._active_operation_id = None
        if invalidation_reason is not None:
            return LamaControllerEvent(
                status=f"LaMa result ignored: {invalidation_reason}")

        job = self._pending_job
        self._pending_job = None
        if engine_event.error is not None:
            log.error(f"LaMa error: {engine_event.error}")
            return LamaControllerEvent(
                inference_error=engine_event.error,
                status=f"LaMa error: {engine_event.error[:80]}",
            )
        if job is None or job.context.job_id != expected_id:
            return LamaControllerEvent(
                status="LaMa result ignored: no matching pending job"
            )
        if engine_event.task_type != "inference":
            return LamaControllerEvent(
                status="LaMa result ignored: invalid task type")
        result = engine_event.result
        if not isinstance(result, LamaResult):
            log.error(
                "LaMa inference returned unexpected result "
                f"type: {type(result)}"
            )
            return LamaControllerEvent(
                status="LaMa result ignored: invalid result")
        provenance = result.provenance
        if (
                provenance is not None
                and job.context.request_provenance is not None):
            provenance = provenance.with_request(
                job.context.request_provenance)
        completed_context = replace(
            job.context,
            result_provenance=provenance,
        )
        return LamaControllerEvent(
            inference_result=(completed_context, result.image))
