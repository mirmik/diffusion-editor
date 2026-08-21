"""Application workflow for provider-neutral AI image editing tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
import copy
from typing import Callable

import numpy as np
from PIL import Image
from tcbase import log

from ..document.layer import Layer
from ..document.tool import InstructTool
from ..engines.instruct_engine import InstructEngine
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
from .patch_resolver import apply_patch_source_to_tool, resolve_source_patch
from .provenance import RequestProvenance
from .image_edit_profiles import (
    LEGACY_INSTRUCT_PROFILE_ID,
    image_edit_profile,
)
from .types import (
    GenerationError,
    ImageEditRequest,
    InstructInferenceResult,
)


@dataclass(frozen=True)
class InstructControllerEvent:
    status: str | None = None
    model_loading: bool = False
    model_loaded: bool = False
    model_error: str | None = None
    inference_result: (
        tuple[InferenceJobContext, Image.Image, int] | None
    ) = None
    inference_error: str | None = None


@dataclass(frozen=True)
class _PendingInstructJob:
    context: InferenceJobContext
    request: ImageEditRequest


class InstructGenerationController:
    def __init__(
            self,
            *,
            engine: InstructEngine,
            composite_below: Callable[[Layer], np.ndarray | None],
            document_state: Callable[[], JobDocumentState] | None = None,
            job_id_factory: Callable[[], str] = new_job_id):
        self._engine = engine
        self._composite_below = composite_below
        self._document_state = document_state or standalone_document_state
        self._job_id_factory = job_id_factory
        self._pending_job: _PendingInstructJob | None = None
        self._active_operation_id: str | None = None
        self._active_target_layer_id: str | None = None
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
        if (
                job is None
                or job.context.layer_id != layer_id):
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

    def submit_load_model(
            self,
            profile_id: str = LEGACY_INSTRUCT_PROFILE_ID,
            parameters: dict[str, object] | None = None,
    ) -> InstructControllerEvent:
        if self._engine.is_busy or self._active_operation_id is not None:
            return InstructControllerEvent()
        operation_id = self._job_id_factory()
        submitted = self._submit_load(
            profile_id, parameters or image_edit_profile(profile_id).defaults(),
            operation_id)
        if not submitted:
            return InstructControllerEvent()
        self._active_operation_id = operation_id
        self._active_target_layer_id = None
        return InstructControllerEvent(
            model_loading=True,
            status=self._loading_status(profile_id),
        )

    def start_apply(self, layer: Layer) -> InstructControllerEvent:
        if self._engine.is_busy or self._active_operation_id is not None:
            return InstructControllerEvent()
        prepared = self._prepare_job(layer)
        if isinstance(prepared, InstructControllerEvent):
            return prepared
        job = prepared

        matches = getattr(
            self._engine, "loaded_configuration_matches", None)
        needs_load = not self._engine.is_loaded
        if callable(matches):
            needs_load = not matches(
                job.request.model_profile_id, job.request.parameters)
        elif hasattr(self._engine, "loaded_profile_id"):
            needs_load = (
                getattr(self._engine, "loaded_profile_id")
                != job.request.model_profile_id)
        if needs_load:
            self._pending_job = job
            submitted = self._submit_load(
                job.request.model_profile_id,
                job.request.parameters,
                job.context.job_id,
            )
            if not submitted:
                self._pending_job = None
                return InstructControllerEvent()
            self._active_operation_id = job.context.job_id
            self._active_target_layer_id = job.context.layer_id
            return InstructControllerEvent(
                model_loading=True,
                status=self._loading_status(job.request.model_profile_id),
            )

        return self._submit_job(job)

    def _prepare_job(
            self, layer: Layer,
    ) -> _PendingInstructJob | InstructControllerEvent:
        tool = layer.tool
        if not isinstance(tool, InstructTool):
            return InstructControllerEvent()

        composite = self._composite_below(layer)
        if composite is None:
            return InstructControllerEvent()
        patch = resolve_source_patch(
            layer,
            composite,
            default_to_full=True,
        )
        if isinstance(patch, GenerationError):
            log.error(patch.log_message or patch.message)
            return InstructControllerEvent(status=patch.message)
        if patch is None:
            return InstructControllerEvent(
                status="No source patch for instruction")

        apply_patch_source_to_tool(tool, patch)
        input_image = FrozenImage.capture(tool.source_patch)
        if input_image is None:
            return InstructControllerEvent(
                status="No source patch for instruction")
        parameters = copy.deepcopy(tool.parameters)
        request = ImageEditRequest(
            image=input_image.to_image(),
            model_profile_id=tool.model_profile_id,
            parameters=parameters,
        )
        profile = image_edit_profile(tool.model_profile_id)
        context = capture_job_context(
            kind="instruct",
            document_state=self._document_state(),
            layer=layer,
            job_id=self._job_id_factory(),
            input_image=input_image.to_image(),
            paste=FrozenPasteContext.capture(layer, tool),
            model_provenance=(
                ("model_profile_id", profile.stable_id),
                ("provider", profile.provider),
                ("model_id", str(parameters.get("model", profile.model_id))),
            ),
            request_provenance=RequestProvenance.capture(
                "image_edit",
                {
                    "model_profile_id": request.model_profile_id,
                    "parameters": request.parameters,
                    "input_image_hash": input_image.content_hash,
                },
            ),
        )
        return _PendingInstructJob(context=context, request=request)

    def _submit_load(
            self,
            profile_id: str,
            parameters: dict[str, object],
            job_id: str,
    ) -> bool:
        if hasattr(self._engine, "loaded_profile_id"):
            return submit_with_job_id(
                self._engine,
                "submit_load",
                profile_id,
                parameters,
                job_id=job_id,
            )
        return submit_with_job_id(
            self._engine, "submit_load", job_id=job_id)

    @staticmethod
    def _loading_status(profile_id: str) -> str:
        if profile_id == LEGACY_INSTRUCT_PROFILE_ID:
            return "Loading InstructPix2Pix model..."
        return f"Loading {image_edit_profile(profile_id).title}..."

    def _submit_job(
            self, job: _PendingInstructJob) -> InstructControllerEvent:
        submitted = submit_with_job_id(
            self._engine,
            "submit_request",
            job.request,
            job_id=job.context.job_id,
        )
        if not submitted:
            return InstructControllerEvent()
        self._pending_job = job
        self._active_operation_id = job.context.job_id
        self._active_target_layer_id = job.context.layer_id
        return InstructControllerEvent(status="Applying instruction...")

    def poll(self) -> InstructControllerEvent | None:
        engine_event = self._engine.poll_event()
        if engine_event is None:
            return None

        expected_id = self._active_operation_id
        if expected_id is None:
            return InstructControllerEvent(
                status="InstructPix2Pix event ignored: no active job")
        if not terminal_job_matches(
                self._engine, engine_event.job_id, expected_id):
            return InstructControllerEvent(
                status="InstructPix2Pix event ignored: stale job identity")

        invalidation_reason = self._invalidated_operations.pop(
            expected_id, None)
        self._active_operation_id = None
        self._active_target_layer_id = None
        if invalidation_reason is not None:
            return InstructControllerEvent(
                status=(
                    "InstructPix2Pix result ignored: "
                    f"{invalidation_reason}"
                )
            )

        if engine_event.task_type == "load":
            if engine_event.error:
                log.error(f"InstructPix2Pix load error: {engine_event.error}")
                self._pending_job = None
                return InstructControllerEvent(
                    model_error=engine_event.error,
                    status=(
                        "InstructPix2Pix load error: "
                        f"{engine_event.error[:80]}"
                    ),
                )
            pending = self._pending_job
            event = InstructControllerEvent(
                model_loaded=True,
                status="InstructPix2Pix model loaded",
            )
            if pending is not None:
                return replace(
                    self._submit_job(pending),
                    model_loaded=True,
                )
            return event

        if engine_event.task_type == "inference":
            job = self._pending_job
            self._pending_job = None
            if engine_event.error:
                log.error(
                    f"InstructPix2Pix inference error: {engine_event.error}")
                return InstructControllerEvent(
                    inference_error=engine_event.error,
                    status=(
                        "InstructPix2Pix error: "
                        f"{engine_event.error[:80]}"
                    ),
                )
            if job is None or job.context.job_id != expected_id:
                return InstructControllerEvent(
                    status=(
                        "InstructPix2Pix result ignored: "
                        "no matching pending job"
                    )
                )
            result = engine_event.result
            if not isinstance(result, InstructInferenceResult):
                log.error(
                    "InstructPix2Pix inference returned unexpected result "
                    f"type: {type(result)}"
                )
                return InstructControllerEvent(
                    status="InstructPix2Pix result ignored: invalid result"
                )
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
            return InstructControllerEvent(
                inference_result=(
                    completed_context,
                    result.image,
                    result.seed,
                )
            )

        return InstructControllerEvent(
            status="InstructPix2Pix event ignored: invalid task type")
