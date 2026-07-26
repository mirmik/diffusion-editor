"""Application workflow for Stable Diffusion generation tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from PIL import Image
from tcbase import log

from ..document.layer import Layer
from ..document.layer_stack import LayerStack
from ..document.tool import DiffusionTool
from ..engines.diffusion_engine import DiffusionEngine
from .diffusion_request_builder import DiffusionRequestBuilder
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
from .provenance import (
    ModelIdentity,
    ModelIdentityPolicy,
    RequestProvenance,
)
from .types import DiffusionInferenceResult, DiffusionRequest


@dataclass(frozen=True)
class DiffusionControllerEvent:
    status: str | None = None
    model_loaded_path: str | None = None
    model_error: str | None = None
    ip_adapter_loaded: bool = False
    ip_adapter_error: str | None = None
    inference_result: (
        tuple[InferenceJobContext, Image.Image, int] | None
    ) = None
    inference_error: str | None = None


@dataclass(frozen=True)
class _PendingDiffusionJob:
    context: InferenceJobContext
    request: DiffusionRequest


class DiffusionGenerationController:
    def __init__(
            self,
            *,
            engine: DiffusionEngine,
            layer_stack: LayerStack,
            composite_below: Callable[[Layer], np.ndarray | None],
            document_state: Callable[[], JobDocumentState] | None = None,
            job_id_factory: Callable[[], str] = new_job_id):
        self._engine = engine
        self._layer_stack = layer_stack
        self._composite_below = composite_below
        self._document_state = document_state or standalone_document_state
        self._job_id_factory = job_id_factory
        self._pending_job: _PendingDiffusionJob | None = None
        self._queued_job: _PendingDiffusionJob | None = None
        self._active_task: str | None = None
        self._active_operation_id: str | None = None
        self._active_target_layer_id: str | None = None
        self._invalidated_operations: dict[str, str] = {}

    @property
    def pending_context(self) -> InferenceJobContext | None:
        job = self._pending_job
        return job.context if job is not None else None

    @property
    def pending_contexts(self) -> tuple[InferenceJobContext, ...]:
        contexts: list[InferenceJobContext] = []
        for job in (self._pending_job, self._queued_job):
            if job is not None and job.context not in contexts:
                contexts.append(job.context)
        return tuple(contexts)

    @property
    def pending_layer(self) -> None:
        """Legacy introspection seam; controllers no longer retain Layer objects."""
        return None

    def clear_pending_layer(self, layer: Layer) -> None:
        self.invalidate_layer(layer.id, "target layer or tool was detached")

    def invalidate_layer(self, layer_id: str, reason: str) -> bool:
        changed = False
        if (
                self._queued_job is not None
                and self._queued_job.context.layer_id == layer_id):
            self._queued_job = None
            changed = True
        if (
                self._pending_job is not None
                and self._pending_job.context.layer_id == layer_id):
            self._pending_job = None
            changed = True
        if self._active_target_layer_id == layer_id:
            self._invalidate_active(reason)
            changed = True
        return changed

    def invalidate_job(self, job_id: str, reason: str) -> bool:
        changed = False
        if (
                self._queued_job is not None
                and self._queued_job.context.job_id == job_id):
            self._queued_job = None
            changed = True
        if (
                self._pending_job is not None
                and self._pending_job.context.job_id == job_id):
            layer_id = self._pending_job.context.layer_id
            self._pending_job = None
            changed = True
            if self._active_target_layer_id == layer_id:
                self._invalidate_active(reason)
        return changed

    def invalidate_all(self, reason: str) -> bool:
        had_jobs = bool(self._pending_job or self._queued_job)
        self._pending_job = None
        self._queued_job = None
        if self._active_target_layer_id is not None:
            self._invalidate_active(reason)
            had_jobs = True
        return had_jobs

    def _invalidate_active(self, reason: str) -> None:
        operation_id = self._active_operation_id
        if operation_id is not None:
            self._invalidated_operations[operation_id] = reason
            cancel_engine(self._engine)

    def submit_load_model(
            self,
            path: str,
            prediction_type: str | None = None) -> DiffusionControllerEvent:
        if self._engine.is_busy or self._active_operation_id is not None:
            return DiffusionControllerEvent()
        pred = prediction_type if prediction_type else None
        operation_id = self._job_id_factory()
        submitted = submit_with_job_id(
            self._engine,
            "submit_load",
            path,
            pred,
            job_id=operation_id,
        )
        if not submitted:
            return DiffusionControllerEvent()
        self._begin_operation("load", operation_id, target_layer_id=None)
        return DiffusionControllerEvent(status="Loading model...")

    def submit_load_ip_adapter(self) -> DiffusionControllerEvent:
        if (
                self._engine.is_busy
                or self._active_operation_id is not None
                or not self._engine.is_loaded):
            return DiffusionControllerEvent()
        operation_id = self._job_id_factory()
        submitted = submit_with_job_id(
            self._engine,
            "submit_load_ip_adapter",
            job_id=operation_id,
        )
        if not submitted:
            return DiffusionControllerEvent()
        self._begin_operation(
            "load_ip_adapter", operation_id, target_layer_id=None)
        return DiffusionControllerEvent(status="Loading IP-Adapter...")

    def start_regeneration(self, layer: Layer) -> DiffusionControllerEvent:
        prepared = self._prepare_job(layer)
        if isinstance(prepared, DiffusionControllerEvent):
            return prepared
        job = prepared

        if self._engine.is_busy or self._active_operation_id is not None:
            # During an automatic model/adapter load, replace the deferred
            # request with the newest immutable snapshot. Otherwise keep one
            # latest queued regeneration.
            if (
                    self._active_task in ("load", "load_ip_adapter")
                    and self._active_target_layer_id == layer.id):
                self._pending_job = job
            else:
                self._queued_job = job
            return DiffusionControllerEvent(status="Regeneration queued...")

        tool = layer.tool
        assert isinstance(tool, DiffusionTool)
        if (
                tool.model_path
                and (
                    tool.model_path != self._engine.model_path
                    or not self._loaded_model_satisfies(tool)
                )):
            self._pending_job = job
            submitted = self._submit_model_load(
                tool,
                job_id=job.context.job_id,
            )
            if not submitted:
                self._pending_job = None
                return DiffusionControllerEvent()
            self._begin_operation(
                "load", job.context.job_id, target_layer_id=layer.id)
            return DiffusionControllerEvent(
                status="Loading model for regeneration...")

        if not self._engine.is_loaded:
            return DiffusionControllerEvent()
        return self._submit_job(job)

    def _loaded_model_satisfies(self, tool: DiffusionTool) -> bool:
        if (
                tool.model_identity_policy
                != ModelIdentityPolicy.REQUIRE_IMMUTABLE):
            return True
        model_info = getattr(self._engine, "model_info", {})
        if not isinstance(model_info, dict):
            return False
        raw = model_info.get("model_identity")
        if not isinstance(raw, dict):
            return False
        try:
            loaded = ModelIdentity.from_dict(raw)
        except (TypeError, ValueError):
            return False
        if not loaded.is_confirmed_immutable:
            return False
        expected = tool.model_identity
        if expected is None or not expected.is_confirmed_immutable:
            return True
        if (
                expected.content_hash is not None
                and expected.content_hash != loaded.content_hash):
            return False
        if (
                expected.revision is not None
                and expected.revision != loaded.revision):
            return False
        return True

    def _submit_model_load(
            self,
            tool: DiffusionTool,
            *,
            job_id: str,
    ) -> bool:
        prediction_type = (
            tool.prediction_type if tool.prediction_type else None
        )
        identity = tool.model_identity
        optional: dict[str, str | None] = {}
        if getattr(
                self._engine,
                "supports_model_identity_policy",
                False):
            optional["expected_content_hash"] = (
                identity.content_hash
                if (
                    identity is not None
                    and identity.is_confirmed_immutable
                )
                else None
            )
            optional["model_identity_policy"] = (
                tool.model_identity_policy.value
            )
        return submit_with_job_id(
            self._engine,
            "submit_load",
            tool.model_path,
            prediction_type,
            job_id=job_id,
            **optional,
        )

    def _prepare_job(
            self, layer: Layer,
    ) -> _PendingDiffusionJob | DiffusionControllerEvent:
        tool = layer.tool
        if not isinstance(tool, DiffusionTool):
            return DiffusionControllerEvent()

        composite_below = None
        if tool.mode != "txt2img":
            composite_below = self._composite_below(layer)
        build = DiffusionRequestBuilder(self._layer_stack).build(
            layer,
            composite_below,
        )
        if build.error is not None:
            log.error(build.error.log_message or build.error.message)
            return DiffusionControllerEvent(status=build.error.message)
        request = build.request
        if request is None:
            log.error(
                f"Diffusion request builder returned no request for {layer.name}")
            return DiffusionControllerEvent(
                status="Could not build diffusion request")

        input_image = FrozenImage.capture(request.image)
        request_mask = FrozenImage.capture(request.mask_image)
        ip_adapter = FrozenImage.capture(request.ip_adapter_image)
        frozen_request = replace(
            request,
            image=(
                input_image.to_image()
                if input_image is not None else None
            ),
            mask_image=(
                request_mask.to_image()
                if request_mask is not None else None
            ),
            ip_adapter_image=(
                ip_adapter.to_image()
                if ip_adapter is not None else None
            ),
        )
        context = capture_job_context(
            kind="diffusion",
            document_state=self._document_state(),
            layer=layer,
            job_id=self._job_id_factory(),
            input_image=(
                input_image.to_image()
                if input_image is not None else None
            ),
            request_mask=(
                request_mask.to_image()
                if request_mask is not None else None
            ),
            reference_image=(
                ip_adapter.to_image()
                if ip_adapter is not None else None
            ),
            paste=FrozenPasteContext.capture(layer, tool),
            model_provenance=(
                ("model_path", tool.model_path or str(
                    self._engine.model_path or "")),
                ("prediction_type", tool.prediction_type),
                ("mode", tool.mode),
                ("ip_adapter_layer_id", tool.ip_adapter_layer_id or ""),
            ),
            request_provenance=RequestProvenance.capture(
                "diffusion",
                {
                    "prompt": frozen_request.prompt,
                    "negative_prompt": frozen_request.negative_prompt,
                    "strength": frozen_request.strength,
                    "steps": frozen_request.steps,
                    "guidance_scale": frozen_request.guidance_scale,
                    "seed": frozen_request.seed,
                    "mode": frozen_request.mode,
                    "masked_content": frozen_request.masked_content,
                    "ip_adapter_scale": frozen_request.ip_adapter_scale,
                    "width": frozen_request.width,
                    "height": frozen_request.height,
                    "model_path": tool.model_path or str(
                        self._engine.model_path or ""),
                    "prediction_type": tool.prediction_type,
                    "model_identity": (
                        tool.model_identity.to_dict()
                        if tool.model_identity is not None else None
                    ),
                    "model_identity_policy": (
                        tool.model_identity_policy.value
                    ),
                    "input_image_hash": (
                        input_image.content_hash
                        if input_image is not None else None
                    ),
                    "mask_image_hash": (
                        request_mask.content_hash
                        if request_mask is not None else None
                    ),
                    "reference_image_hash": (
                        ip_adapter.content_hash
                        if ip_adapter is not None else None
                    ),
                },
            ),
        )
        return _PendingDiffusionJob(context=context, request=frozen_request)

    def _submit_job(
            self, job: _PendingDiffusionJob) -> DiffusionControllerEvent:
        request_parameters = (
            job.context.request_provenance.parameters.to_dict()
            if job.context.request_provenance is not None
            else {}
        )
        if (
                job.request.ip_adapter_image is not None
                and request_parameters.get("model_identity_policy")
                == ModelIdentityPolicy.REQUIRE_IMMUTABLE.value):
            message = (
                "IP-Adapter identity is floating; strict reproducibility "
                "requires a pinned immutable adapter artifact"
            )
            if self._pending_job is job:
                self._pending_job = None
            return DiffusionControllerEvent(
                ip_adapter_error=message,
                status=message,
            )
        if (
                job.request.ip_adapter_image is not None
                and not self._engine.ip_adapter_loaded):
            self._pending_job = job
            submitted = submit_with_job_id(
                self._engine,
                "submit_load_ip_adapter",
                job_id=job.context.job_id,
            )
            if not submitted:
                self._pending_job = None
                return DiffusionControllerEvent(
                    status="Could not load IP-Adapter")
            self._begin_operation(
                "load_ip_adapter",
                job.context.job_id,
                target_layer_id=job.context.layer_id,
            )
            return DiffusionControllerEvent(status="Loading IP-Adapter...")

        submitted = submit_with_job_id(
            self._engine,
            "submit_request",
            job.request,
            job_id=job.context.job_id,
        )
        if not submitted:
            return DiffusionControllerEvent()
        self._pending_job = job
        self._begin_operation(
            "inference",
            job.context.job_id,
            target_layer_id=job.context.layer_id,
        )
        return DiffusionControllerEvent(
            status=(
                f"Regenerating "
                f"({job.request.width}x{job.request.height})..."
            )
        )

    def _begin_operation(
            self,
            task: str,
            operation_id: str,
            *,
            target_layer_id: str | None) -> None:
        self._active_task = task
        self._active_operation_id = operation_id
        self._active_target_layer_id = target_layer_id

    def poll(self) -> DiffusionControllerEvent | None:
        engine_event = self._engine.poll_event()
        if engine_event is None:
            return None

        expected_id = self._active_operation_id
        if expected_id is None:
            return DiffusionControllerEvent(
                status="Diffusion event ignored: no active job")
        if not terminal_job_matches(
                self._engine, engine_event.job_id, expected_id):
            return DiffusionControllerEvent(
                status="Diffusion event ignored: stale job identity")

        invalidation_reason = self._invalidated_operations.pop(
            expected_id, None)
        active_task = self._active_task
        self._active_task = None
        self._active_operation_id = None
        self._active_target_layer_id = None
        if invalidation_reason is not None:
            self._start_queued_if_possible()
            return DiffusionControllerEvent(
                status=f"Diffusion result ignored: {invalidation_reason}")

        log.debug(
            f"[DiffusionGenerationController] task_type={engine_event.task_type}, "
            f"job_id={engine_event.job_id}, error={engine_event.error}, "
            f"result_type={type(engine_event.result)}"
        )

        if engine_event.task_type == "load":
            if engine_event.error:
                log.error(f"Diffusion model load error: {engine_event.error}")
                self._pending_job = None
                return DiffusionControllerEvent(
                    model_error=engine_event.error,
                    status=f"Model load error: {engine_event.error[:80]}",
                )
            model_path = None
            if isinstance(engine_event.result, str):
                model_path = engine_event.result
            else:
                log.error(
                    "Diffusion model load returned unexpected result "
                    f"type: {type(engine_event.result)}"
                )
            event = DiffusionControllerEvent(
                model_loaded_path=model_path,
                status="Model loaded",
            )
            pending = self._pending_job
            if pending is not None:
                return replace(
                    self._submit_job(pending),
                    model_loaded_path=model_path,
                )
            queued_event = self._start_queued_if_possible()
            return (
                replace(queued_event, model_loaded_path=model_path)
                if queued_event is not None
                else event
            )

        if engine_event.task_type == "load_ip_adapter":
            if engine_event.error:
                log.error(f"IP-Adapter load error: {engine_event.error}")
                self._pending_job = None
                return DiffusionControllerEvent(
                    ip_adapter_error=engine_event.error,
                    status=f"IP-Adapter error: {engine_event.error[:80]}",
                )
            event = DiffusionControllerEvent(
                ip_adapter_loaded=True,
                status="IP-Adapter loaded",
            )
            pending = self._pending_job
            if pending is not None:
                return replace(
                    self._submit_job(pending),
                    ip_adapter_loaded=True,
                )
            return event

        if engine_event.task_type == "inference":
            job = self._pending_job
            self._pending_job = None
            if engine_event.error:
                log.error(f"Diffusion inference error: {engine_event.error}")
                self._start_queued_if_possible()
                return DiffusionControllerEvent(
                    inference_error=engine_event.error,
                    status=f"Diffusion error: {engine_event.error[:80]}",
                )
            if job is None or job.context.job_id != expected_id:
                self._start_queued_if_possible()
                return DiffusionControllerEvent(
                    status="Diffusion result ignored: no matching pending job"
                )
            result = engine_event.result
            if not isinstance(result, DiffusionInferenceResult):
                log.error(
                    "Diffusion inference returned unexpected result "
                    f"type: {type(result)}"
                )
                self._start_queued_if_possible()
                return DiffusionControllerEvent(
                    status="Diffusion result ignored: invalid result"
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
            self._start_queued_if_possible()
            return DiffusionControllerEvent(
                inference_result=(
                    completed_context,
                    result.image,
                    result.seed,
                )
            )

        log.error(
            "Diffusion engine returned unexpected task type "
            f"{engine_event.task_type!r} for active task {active_task!r}"
        )
        return DiffusionControllerEvent(
            status="Diffusion event ignored: invalid task type")

    def _start_queued_if_possible(self) -> DiffusionControllerEvent | None:
        job = self._queued_job
        self._queued_job = None
        if job is None:
            return None
        return self._submit_job(job)
