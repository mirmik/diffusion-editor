"""Application workflow for generation without source pixels."""

from __future__ import annotations

from .progress import GenerationProgress

from dataclasses import dataclass, replace
import copy
from typing import Callable

from PIL import Image
from termin.base import log

from ..document.layer import Layer
from ..document.layer_stack import LayerStack
from ..document.tool import TextToImageTool
from ..engines.text_to_image_engine import TextToImageEngine
from .job_context import (
    FrozenPasteContext,
    FrozenImage,
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
from .types import GenerationError, TextToImageInferenceResult, TextToImageRequest
from .reference_resolver import resolve_text_to_image_references, layer_reference_image


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
    progress: GenerationProgress | None = None


@dataclass(frozen=True)
class _PendingTextToImageJob:
    context: InferenceJobContext
    request: TextToImageRequest
    reference_layer_ids: tuple[str | None, ...] = ()


class TextToImageGenerationController:
    def __init__(
            self, *, engine: TextToImageEngine,
            layer_stack: LayerStack | None = None,
            document_state: Callable[[], JobDocumentState] | None = None,
            job_id_factory: Callable[[], str] = new_job_id):
        self._engine = engine
        self._layer_stack = layer_stack
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
        if job is None or layer_id not in (job.context.layer_id, *job.reference_layer_ids):
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
        references = resolve_text_to_image_references(tool, self._layer_stack)
        if isinstance(references, GenerationError):
            return TextToImageControllerEvent(status=references.message, inference_error=references.message)
        frozen_references = tuple(FrozenImage.capture(image) for image in references)
        reference = references[0] if references else None
        frozen_reference = frozen_references[0] if frozen_references else None
        request = TextToImageRequest(
            model_profile_id=profile.stable_id,
            parameters=parameters,
            width=layer.width,
            height=layer.height,
            lora_adapters=adapters,
            reference_images=references,
        )
        context = capture_job_context(
            kind="text_to_image",
            reference_image=reference,
            document_state=self._document_state(),
            layer=layer,
            job_id=self._job_id_factory(),
            paste=FrozenPasteContext.capture_layer(
                layer, preserve_alpha=profile.supports_alpha),
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
                    "reference_image_hash": (frozen_reference.content_hash if frozen_reference else None),
                    "reference_layer_id": tool.reference_layer_id if reference is not None else None,
                    "references": [{"stable_id": ref.stable_id, "layer_id": ref.layer_id,
                                    "image_hash": frozen.content_hash}
                                   for ref, frozen in zip(tool.references, frozen_references)],
                },
            ),
        )
        context = replace(context, reference_images=frozen_references)
        job = _PendingTextToImageJob(
            context, request, tuple(ref.layer_id for ref in tool.references) if references else ())
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

    def _reference_is_current(self, job: _PendingTextToImageJob) -> bool:
        for layer_id, frozen in zip(job.reference_layer_ids, job.context.reference_images):
            if layer_id is None:
                continue
            layer = self._layer_stack.find_layer_by_id(layer_id)
            if layer is None:
                return False
            reference = layer_reference_image(
                layer, purpose="Text to Image", preserve_alpha=text_to_image_profile(
                    job.request.model_profile_id).supports_alpha)
            if isinstance(reference, GenerationError) or FrozenImage.capture(reference.image) != frozen:
                return False
        return True

    def _submit_job(
            self, job: _PendingTextToImageJob) -> TextToImageControllerEvent:
        if not self._reference_is_current(job):
            self._pending_job = None
            return TextToImageControllerEvent(
                status="Text to Image cancelled: reference changed",
                inference_error="Text to Image reference changed")
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
        if event.progress is not None and (
                expected_id is None or not terminal_job_matches(
                    self._engine, event.job_id, expected_id)):
            return None
        if expected_id is None:
            return TextToImageControllerEvent(
                status="Text to Image event ignored: no active job")
        if not terminal_job_matches(self._engine, event.job_id, expected_id):
            return TextToImageControllerEvent(
                status="Text to Image event ignored: stale job identity")
        if event.progress is not None:
            if expected_id in self._invalidated_operations:
                return None
            return TextToImageControllerEvent(progress=event.progress, status=event.progress.text)

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
            if not self._reference_is_current(job):
                return TextToImageControllerEvent(
                    status="Text to Image result ignored: reference changed",
                    inference_error="Text to Image reference changed")
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
