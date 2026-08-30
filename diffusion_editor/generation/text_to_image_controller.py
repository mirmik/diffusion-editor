"""Application workflow for generation without source pixels."""

from __future__ import annotations

from dataclasses import dataclass, replace
import copy
from typing import Callable

from PIL import Image
from tcbase import log

from ..document.layer import Layer
from ..document.tool import TextToImageTool
from ..engines.text_to_image_engine import TextToImageEngine
from .job_context import (
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
from .provenance import RequestProvenance
from .text_to_image_profiles import text_to_image_profile
from .types import TextToImageInferenceResult, TextToImageRequest


@dataclass(frozen=True)
class TextToImageControllerEvent:
    status: str | None = None
    model_loading: bool = False
    model_loaded: bool = False
    model_error: str | None = None
    inference_result: (
        tuple[InferenceJobContext, Image.Image, int] | None
    ) = None
    inference_error: str | None = None


@dataclass(frozen=True)
class _PendingTextToImageJob:
    context: InferenceJobContext
    request: TextToImageRequest


class TextToImageGenerationController:
    def __init__(
            self, *, engine: TextToImageEngine,
            document_state: Callable[[], JobDocumentState] | None = None,
            job_id_factory: Callable[[], str] = new_job_id):
        self._engine = engine
        self._document_state = document_state or standalone_document_state
        self._job_id_factory = job_id_factory
        self._pending_job: _PendingTextToImageJob | None = None
        self._active_operation_id: str | None = None
        self._invalidated_operations: dict[str, str] = {}

    @property
    def pending_context(self) -> InferenceJobContext | None:
        return self._pending_job.context if self._pending_job else None

    @property
    def pending_contexts(self) -> tuple[InferenceJobContext, ...]:
        return (self.pending_context,) if self.pending_context else ()

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
        if self._active_operation_id is not None:
            self._invalidated_operations[self._active_operation_id] = reason
            cancel_engine(self._engine)

    def submit_load_model(
            self, profile_id: str, parameters: dict[str, object] | None = None,
            lora_adapters=None) -> TextToImageControllerEvent:
        if self._engine.is_busy or self._active_operation_id is not None:
            return TextToImageControllerEvent()
        profile = text_to_image_profile(profile_id)
        operation_id = self._job_id_factory()
        submitted = submit_with_job_id(
            self._engine,
            "submit_load",
            profile_id,
            parameters or profile.defaults(),
            lora_adapters,
            job_id=operation_id,
        )
        if not submitted:
            return TextToImageControllerEvent()
        self._active_operation_id = operation_id
        return TextToImageControllerEvent(
            model_loading=True,
            status=f"Loading {profile.title}...",
        )

    def start(self, layer: Layer) -> TextToImageControllerEvent:
        if self._engine.is_busy or self._active_operation_id is not None:
            return TextToImageControllerEvent()
        tool = layer.tool
        if not isinstance(tool, TextToImageTool):
            return TextToImageControllerEvent()
        profile = text_to_image_profile(tool.model_profile_id)
        parameters = copy.deepcopy(tool.parameters)
        adapters = tuple(
            adapter.to_dict() for adapter in tool.lora_adapters)
        request = TextToImageRequest(
            model_profile_id=profile.stable_id,
            parameters=parameters,
            width=layer.width,
            height=layer.height,
            lora_adapters=adapters,
        )
        context = capture_job_context(
            kind="text_to_image",
            document_state=self._document_state(),
            layer=layer,
            job_id=self._job_id_factory(),
            paste=FrozenPasteContext.capture_layer(layer),
            model_provenance=(
                ("model_profile_id", profile.stable_id),
                ("provider", profile.provider),
                ("model_id", str(parameters.get("model", profile.model_id))),
            ),
            request_provenance=RequestProvenance.capture(
                "text_to_image",
                {
                    "model_profile_id": profile.stable_id,
                    "parameters": parameters,
                    "lora_adapters": list(adapters),
                    "width": layer.width,
                    "height": layer.height,
                },
            ),
        )
        job = _PendingTextToImageJob(context, request)
        matches = getattr(
            self._engine, "loaded_configuration_matches", None)
        needs_load = not self._engine.is_loaded
        if callable(matches):
            needs_load = not matches(
                profile.stable_id, parameters, adapters)
        elif hasattr(self._engine, "loaded_profile_id"):
            needs_load = self._engine.loaded_profile_id != profile.stable_id
        if needs_load:
            self._pending_job = job
            submitted = submit_with_job_id(
                self._engine,
                "submit_load",
                profile.stable_id,
                parameters,
                adapters,
                job_id=context.job_id,
            )
            if not submitted:
                self._pending_job = None
                return TextToImageControllerEvent()
            self._active_operation_id = context.job_id
            return TextToImageControllerEvent(
                model_loading=True,
                status=f"Loading {profile.title}...",
            )
        return self._submit_job(job)

    def _submit_job(
            self, job: _PendingTextToImageJob) -> TextToImageControllerEvent:
        submitted = submit_with_job_id(
            self._engine, "submit_request", job.request,
            job_id=job.context.job_id)
        if not submitted:
            return TextToImageControllerEvent()
        self._pending_job = job
        self._active_operation_id = job.context.job_id
        return TextToImageControllerEvent(
            status=f"Generating {job.request.width}x{job.request.height}...")

    def poll(self) -> TextToImageControllerEvent | None:
        event = self._engine.poll_event()
        if event is None:
            return None
        expected_id = self._active_operation_id
        if expected_id is None:
            return TextToImageControllerEvent(
                status="Text to Image event ignored: no active job")
        if not terminal_job_matches(self._engine, event.job_id, expected_id):
            return TextToImageControllerEvent(
                status="Text to Image event ignored: stale job identity")
        invalidation = self._invalidated_operations.pop(expected_id, None)
        self._active_operation_id = None
        if invalidation is not None:
            return TextToImageControllerEvent(
                status=f"Text to Image result ignored: {invalidation}")

        if event.task_type == "load":
            if event.error:
                self._pending_job = None
                log.error(f"Text to Image load error: {event.error}")
                return TextToImageControllerEvent(
                    model_error=event.error,
                    status=f"Text to Image load error: {event.error[:80]}",
                )
            pending = self._pending_job
            if pending is not None:
                return replace(
                    self._submit_job(pending), model_loaded=True)
            return TextToImageControllerEvent(
                model_loaded=True, status="Text to Image model loaded")

        if event.task_type == "inference":
            job = self._pending_job
            self._pending_job = None
            if event.error:
                log.error(f"Text to Image inference error: {event.error}")
                return TextToImageControllerEvent(
                    inference_error=event.error,
                    status=f"Text to Image error: {event.error[:80]}",
                )
            if job is None or job.context.job_id != expected_id:
                return TextToImageControllerEvent(
                    status="Text to Image result ignored: no matching job")
            result = event.result
            if not isinstance(result, TextToImageInferenceResult):
                return TextToImageControllerEvent(
                    status="Text to Image result ignored: invalid result")
            if result.image.size != (job.request.width, job.request.height):
                return TextToImageControllerEvent(
                    status="Text to Image result ignored: size mismatch")
            provenance = result.provenance
            if provenance is not None:
                provenance = provenance.with_request(
                    job.context.request_provenance)
            completed = replace(
                job.context, result_provenance=provenance)
            return TextToImageControllerEvent(
                inference_result=(completed, result.image, result.seed))

        return TextToImageControllerEvent(
            status="Text to Image event ignored: invalid task type")
