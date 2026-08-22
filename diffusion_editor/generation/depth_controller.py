"""Application workflow for monocular depth-map generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from tcbase import log

from ..document.layer import Layer
from ..engines.depth_engine import DepthEstimationEngine
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
from .provenance import RequestProvenance
from .types import (
    DEFAULT_DEPTH_MODEL_PROFILE_ID,
    DepthEstimationRequest,
    DepthEstimationResult,
    depth_model_profile,
)


@dataclass(frozen=True)
class DepthControllerEvent:
    status: str | None = None
    depth_result: (
        tuple[
            InferenceJobContext,
            DepthEstimationResult,
            np.ndarray | None,
        ] | None
    ) = None
    depth_error: str | None = None


@dataclass(frozen=True)
class _PendingDepthJob:
    context: InferenceJobContext
    request: DepthEstimationRequest
    output_mask: FrozenArray | None = None


class DepthGenerationController:
    def __init__(
            self,
            *,
            engine: DepthEstimationEngine,
            composite: Callable[[], np.ndarray],
            selection: Callable[[], np.ndarray | None] | None = None,
            document_state: Callable[[], JobDocumentState] | None = None,
            job_id_factory: Callable[[], str] = new_job_id) -> None:
        self._engine = engine
        self._composite = composite
        self._selection = selection
        self._document_state = document_state or standalone_document_state
        self._job_id_factory = job_id_factory
        self._pending_job: _PendingDepthJob | None = None
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
        job = self._pending_job
        return job.context if job is not None else None

    @property
    def pending_contexts(self) -> tuple[InferenceJobContext, ...]:
        context = self.pending_context
        return (context,) if context is not None else ()

    def clear_pending_layer(self, layer: Layer) -> None:
        self.invalidate_layer(layer.id, "target layer was detached")

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

    def start(
            self,
            layer: Layer,
            profile_id: str = DEFAULT_DEPTH_MODEL_PROFILE_ID,
            *,
            output_mask: np.ndarray | None = None,
    ) -> DepthControllerEvent:
        if self.is_busy:
            return DepthControllerEvent()
        try:
            profile = depth_model_profile(profile_id)
        except ValueError as exc:
            return DepthControllerEvent(status=str(exc))
        try:
            composite = np.ascontiguousarray(self._composite())
        except Exception as exc:
            log.error(f"Depth estimation: failed to get composite - {exc}")
            return DepthControllerEvent(
                status="Depth estimation: failed to get current image"
            )
        if composite.ndim != 3 or composite.shape[2] not in (3, 4):
            return DepthControllerEvent(
                status="Depth estimation: current image has an invalid format"
            )

        frozen = FrozenArray.capture(composite)
        assert frozen is not None
        if output_mask is None and self._selection is not None:
            output_mask = self._selection()
        if output_mask is None and composite.shape[2] == 4:
            alpha = composite[:, :, 3]
            if np.any(alpha < 255):
                output_mask = alpha
        try:
            frozen_mask = self._freeze_output_mask(
                output_mask, composite.shape[:2])
        except ValueError as exc:
            return DepthControllerEvent(
                status=f"Depth estimation: {exc}"
            )
        request = DepthEstimationRequest(
            image=frozen.to_array(),
            profile_id=profile.stable_id,
        )
        context = capture_job_context(
            kind="depth",
            document_state=self._document_state(),
            layer=layer,
            job_id=self._job_id_factory(),
            input_array=frozen.to_array(),
            model_provenance=(
                ("profile_id", profile.stable_id),
                ("model_id", request.model_id),
                ("backend", profile.backend.value),
                ("license", profile.license_name),
                ("metric", str(profile.metric).lower()),
            ),
            request_provenance=RequestProvenance.capture(
                "depth",
                {
                    "model_id": request.model_id,
                    "profile_id": profile.stable_id,
                    "backend": profile.backend.value,
                    "process_resolution": profile.process_resolution,
                    "inference_scope": "full_frame",
                    "output_mask_stage": (
                        "post_inference"
                        if frozen_mask is not None else "none"
                    ),
                    "input_array_hash": frozen.content_hash,
                    "output_mask_hash": (
                        frozen_mask.content_hash
                        if frozen_mask is not None else None
                    ),
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
            return DepthControllerEvent()
        self._pending_job = _PendingDepthJob(context, request, frozen_mask)
        self._active_operation_id = context.job_id
        return DepthControllerEvent(
            status=(
                f"{profile.title}: starting full-frame inference; "
                "output mask will be applied afterward..."
                if frozen_mask is not None
                else f"{profile.title}: starting..."
            )
        )

    @staticmethod
    def _freeze_output_mask(
            mask: np.ndarray | None,
            expected_shape: tuple[int, int],
    ) -> FrozenArray | None:
        if mask is None:
            return None
        values = np.asarray(mask)
        if values.shape != expected_shape:
            raise ValueError(
                f"output mask shape {values.shape} does not match image "
                f"shape {expected_shape}"
            )
        try:
            values = values.astype(np.float32, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("output mask is not numeric") from exc
        if not np.isfinite(values).all():
            raise ValueError("output mask contains non-finite values")
        if values.size and float(values.max()) > 1.0:
            values = values / 255.0
        values = np.clip(values, 0.0, 1.0)
        if not np.any(values > 0.0):
            raise ValueError("output mask is empty")
        return FrozenArray.capture(
            np.ascontiguousarray(np.rint(values * 255.0).astype(np.uint8))
        )

    def poll(self) -> DepthControllerEvent | None:
        engine_event = self._engine.poll_event()
        if engine_event is None:
            return None

        expected_id = self._active_operation_id
        if expected_id is None:
            return DepthControllerEvent(
                status="Depth result ignored: no active job"
            )
        if not terminal_job_matches(
                self._engine, engine_event.job_id, expected_id):
            return DepthControllerEvent(
                status="Depth result ignored: stale job identity"
            )

        progress = (
            engine_event.meta.get("progress")
            if isinstance(engine_event.meta, dict)
            else None
        )
        if (
                engine_event.result is None
                and engine_event.error is None
                and isinstance(progress, str)):
            return DepthControllerEvent(status=progress)

        invalidation_reason = self._invalidated_operations.pop(
            expected_id, None)
        self._active_operation_id = None
        if invalidation_reason is not None:
            return DepthControllerEvent(
                status=f"Depth result ignored: {invalidation_reason}"
            )

        job = self._pending_job
        self._pending_job = None
        if engine_event.error is not None:
            log.error(f"Depth estimation error: {engine_event.error}")
            return DepthControllerEvent(
                depth_error=engine_event.error,
                status=f"Depth estimation error: {engine_event.error[:80]}",
            )
        if job is None or job.context.job_id != expected_id:
            return DepthControllerEvent(
                status="Depth result ignored: no matching pending job"
            )
        if engine_event.task_type != "depth":
            return DepthControllerEvent(
                status="Depth result ignored: invalid task type"
            )
        result = engine_event.result
        if not isinstance(result, DepthEstimationResult):
            return DepthControllerEvent(
                status="Depth result ignored: invalid result"
            )
        if result.profile_id != job.request.profile_id:
            return DepthControllerEvent(
                status="Depth result ignored: model profile mismatch"
            )
        return DepthControllerEvent(
            depth_result=(
                job.context,
                result,
                (
                    job.output_mask.to_array()
                    if job.output_mask is not None else None
                ),
            )
        )
