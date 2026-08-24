"""Application workflow for pose-estimation jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from tcbase import log

from ..document.layer import Layer
from ..engines.pose_engine import PoseEstimationEngine
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
from .pose_estimation import (
    PoseEstimationRequest,
    PoseEstimationResult,
    pose_estimator_profile,
)
from .provenance import RequestProvenance


@dataclass(frozen=True)
class PoseControllerEvent:
    status: str | None = None
    pose_result: tuple[InferenceJobContext, PoseEstimationResult] | None = None
    pose_error: str | None = None


class PoseEstimationController:
    def __init__(
            self,
            engine: PoseEstimationEngine,
            *,
            composite: Callable[[], np.ndarray] | None = None,
            document_state: Callable[[], JobDocumentState] | None = None,
            job_id_factory: Callable[[], str] = new_job_id) -> None:
        self._engine = engine
        self._composite = composite
        self._document_state = document_state or standalone_document_state
        self._job_id_factory = job_id_factory
        self._pending_context: InferenceJobContext | None = None
        self._active_operation_id: str | None = None
        self._invalidated_operations: dict[str, str] = {}

    @property
    def is_busy(self) -> bool:
        return (
            self._active_operation_id is not None
            or bool(getattr(self._engine, "is_busy", False))
        )

    @property
    def pending_context(self) -> InferenceJobContext | None:
        return self._pending_context

    @property
    def pending_contexts(self) -> tuple[InferenceJobContext, ...]:
        return (() if self._pending_context is None else (self._pending_context,))

    def start(self, layer: Layer, profile_id: str) -> PoseControllerEvent:
        if self.is_busy:
            return PoseControllerEvent()
        profile = pose_estimator_profile(profile_id)
        image = np.ascontiguousarray(
            self._composite() if self._composite is not None else layer.image)
        request = PoseEstimationRequest(image=image, profile_id=profile_id)
        input_array = FrozenArray.capture(image)
        assert input_array is not None
        job_id = self._job_id_factory()
        context = capture_job_context(
            kind="pose",
            document_state=self._document_state(),
            layer=layer,
            job_id=job_id,
            input_array=image,
            model_provenance=(("model_id", profile.stable_id),),
            request_provenance=RequestProvenance.capture(
                "pose-estimation",
                {
                    "profile_id": profile_id,
                    "input_array_hash": input_array.content_hash,
                },
            ),
        )
        if not submit_with_job_id(
                self._engine,
                "submit_request",
                request,
                job_id=job_id):
            return PoseControllerEvent()
        self._pending_context = context
        self._active_operation_id = job_id
        return PoseControllerEvent(status=f"Estimating pose — {profile.title}...")

    def invalidate_layer(self, layer_id: str, reason: str) -> bool:
        context = self._pending_context
        if context is None or context.layer_id != layer_id:
            return False
        self._pending_context = None
        self._invalidate_active(reason)
        return True

    def invalidate_job(self, job_id: str, reason: str) -> bool:
        context = self._pending_context
        if context is None or context.job_id != job_id:
            return False
        self._pending_context = None
        self._invalidate_active(reason)
        return True

    def invalidate_all(self, reason: str) -> bool:
        if self._pending_context is None:
            return False
        self._pending_context = None
        self._invalidate_active(reason)
        return True

    def _invalidate_active(self, reason: str) -> None:
        operation_id = self._active_operation_id
        if operation_id is not None:
            self._invalidated_operations[operation_id] = reason
            cancel_engine(self._engine)

    def poll(self) -> PoseControllerEvent | None:
        event = self._engine.poll_event()
        if event is None:
            return None
        expected_id = self._active_operation_id
        if expected_id is None:
            return PoseControllerEvent(
                status="Pose event ignored: no active job")
        if not terminal_job_matches(self._engine, event.job_id, expected_id):
            return PoseControllerEvent(
                status="Pose event ignored: stale job identity")
        invalidation_reason = self._invalidated_operations.pop(expected_id, None)
        self._active_operation_id = None
        if invalidation_reason is not None:
            return PoseControllerEvent(
                status=f"Pose result ignored: {invalidation_reason}")
        context = self._pending_context
        self._pending_context = None
        if event.error is not None:
            log.error(f"Pose estimation error: {event.error}")
            return PoseControllerEvent(
                pose_error=event.error,
                status=f"Pose estimation error: {event.error[:100]}",
            )
        if context is None or context.job_id != expected_id:
            return PoseControllerEvent(
                status="Pose result ignored: no matching pending job")
        if event.task_type != "pose-estimation":
            return PoseControllerEvent(
                status="Pose result ignored: invalid task type")
        result = event.result
        if not isinstance(result, PoseEstimationResult):
            return PoseControllerEvent(
                status="Pose result ignored: invalid result")
        return PoseControllerEvent(pose_result=(context, result))
