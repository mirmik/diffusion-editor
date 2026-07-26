"""Application workflow for background segmentation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from tcbase import log

from ..document.layer import Layer
from ..engines.segmentation_engine import SegmentationEngine
from .job_context import (
    FrozenArray,
    InferenceJobContext,
    JobDocumentState,
    cancel_engine,
    capture_job_context,
    new_job_id,
    standalone_document_state,
    submit_with_job_id,
    terminal_job_matches,
)
from .types import SegmentationRequest, SegmentationResult
from .provenance import RequestProvenance


@dataclass(frozen=True)
class SegmentationControllerEvent:
    status: str | None = None
    segmentation_result: (
        tuple[InferenceJobContext, np.ndarray] | None
    ) = None
    segmentation_error: str | None = None


@dataclass(frozen=True)
class _PendingSegmentationJob:
    context: InferenceJobContext
    request: SegmentationRequest


class SegmentationGenerationController:
    def __init__(
            self,
            *,
            engine: SegmentationEngine,
            composite_below: Callable[[Layer], np.ndarray | None],
            document_state: Callable[[], JobDocumentState] | None = None,
            job_id_factory: Callable[[], str] = new_job_id):
        self._engine = engine
        self._composite_below = composite_below
        self._document_state = document_state or standalone_document_state
        self._job_id_factory = job_id_factory
        self._pending_job: _PendingSegmentationJob | None = None
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

    def start_select_background(self, layer: Layer) -> SegmentationControllerEvent:
        if self._engine.is_busy or self._active_operation_id is not None:
            return SegmentationControllerEvent()
        composite = self._composite_below(layer)
        if composite is None:
            return SegmentationControllerEvent()

        input_array = FrozenArray.capture(composite)
        assert input_array is not None
        request = SegmentationRequest(
            image=input_array.to_array(),
            invert=True,
        )
        context = capture_job_context(
            kind="segmentation",
            document_state=self._document_state(),
            layer=layer,
            job_id=self._job_id_factory(),
            input_array=input_array.to_array(),
            model_provenance=(("model_id", "sam2.1-segmentation"),),
            request_provenance=RequestProvenance.capture(
                "segmentation",
                {
                    "invert": request.invert,
                    "input_array_hash": input_array.content_hash,
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
            return SegmentationControllerEvent()

        self._pending_job = _PendingSegmentationJob(
            context=context,
            request=request,
        )
        self._active_operation_id = context.job_id
        return SegmentationControllerEvent(status="Segmenting background...")

    def poll(self) -> SegmentationControllerEvent | None:
        engine_event = self._engine.poll_event()
        if engine_event is None:
            return None

        expected_id = self._active_operation_id
        if expected_id is None:
            return SegmentationControllerEvent(
                status="Segmentation event ignored: no active job")
        if not terminal_job_matches(
                self._engine, engine_event.job_id, expected_id):
            return SegmentationControllerEvent(
                status="Segmentation event ignored: stale job identity")

        invalidation_reason = self._invalidated_operations.pop(
            expected_id, None)
        self._active_operation_id = None
        if invalidation_reason is not None:
            return SegmentationControllerEvent(
                status=f"Segmentation result ignored: {invalidation_reason}")

        job = self._pending_job
        self._pending_job = None
        if engine_event.error is not None:
            log.error(f"Segmentation error: {engine_event.error}")
            return SegmentationControllerEvent(
                segmentation_error=engine_event.error,
                status=f"Segmentation error: {engine_event.error[:80]}",
            )
        if job is None or job.context.job_id != expected_id:
            return SegmentationControllerEvent(
                status="Segmentation result ignored: no matching pending job"
            )
        if engine_event.task_type != "segmentation":
            return SegmentationControllerEvent(
                status="Segmentation result ignored: invalid task type")
        result = engine_event.result
        if not isinstance(result, SegmentationResult):
            log.error(
                "Segmentation returned unexpected result "
                f"type: {type(result)}"
            )
            return SegmentationControllerEvent(
                status="Segmentation result ignored: invalid result"
            )
        return SegmentationControllerEvent(
            segmentation_result=(
                job.context,
                np.ascontiguousarray(result.mask),
            ),
        )
