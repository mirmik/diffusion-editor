"""Application workflow for Stable Diffusion generation tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from PIL import Image
from tcbase import log

from ..engines.diffusion_engine import DiffusionEngine
from .diffusion_request_builder import DiffusionRequestBuilder
from .types import DiffusionInferenceResult
from ..document.layer import Layer
from ..document.layer_stack import LayerStack
from ..document.tool import DiffusionTool


@dataclass(frozen=True)
class DiffusionControllerEvent:
    status: str | None = None
    model_loaded_path: str | None = None
    model_error: str | None = None
    ip_adapter_loaded: bool = False
    ip_adapter_error: str | None = None
    inference_result: tuple[Layer, Image.Image, int] | None = None
    inference_error: str | None = None


class DiffusionGenerationController:
    def __init__(
            self,
            *,
            engine: DiffusionEngine,
            layer_stack: LayerStack,
            composite_below: Callable[[Layer], np.ndarray | None]):
        self._engine = engine
        self._layer_stack = layer_stack
        self._composite_below = composite_below
        self._pending_layer: Layer | None = None

    @property
    def pending_layer(self) -> Layer | None:
        return self._pending_layer

    def clear_pending_layer(self, layer: Layer) -> None:
        if self._pending_layer is layer:
            self._pending_layer = None

    def submit_load_model(
            self,
            path: str,
            prediction_type: str | None = None) -> DiffusionControllerEvent:
        if self._engine.is_busy:
            return DiffusionControllerEvent()
        pred = prediction_type if prediction_type else None
        submitted = self._engine.submit_load(path, pred)
        if not submitted:
            return DiffusionControllerEvent()
        return DiffusionControllerEvent(status="Loading model...")

    def submit_load_ip_adapter(self) -> DiffusionControllerEvent:
        if self._engine.is_busy or not self._engine.is_loaded:
            return DiffusionControllerEvent()
        submitted = self._engine.submit_load_ip_adapter()
        if not submitted:
            return DiffusionControllerEvent()
        return DiffusionControllerEvent(status="Loading IP-Adapter...")

    def start_regeneration(self, layer: Layer) -> DiffusionControllerEvent:
        if self._engine.is_busy:
            return DiffusionControllerEvent()
        tool = layer.tool
        if not isinstance(tool, DiffusionTool):
            return DiffusionControllerEvent()

        if tool.model_path and tool.model_path != self._engine.model_path:
            self._pending_layer = layer
            pred = tool.prediction_type if tool.prediction_type else None
            submitted = self._engine.submit_load(tool.model_path, pred)
            if not submitted:
                self._pending_layer = None
                return DiffusionControllerEvent()
            return DiffusionControllerEvent(status="Loading model for regeneration...")

        if not self._engine.is_loaded:
            return DiffusionControllerEvent()
        return self._submit_regeneration(layer)

    def _submit_regeneration(self, layer: Layer) -> DiffusionControllerEvent:
        tool = layer.tool
        if not isinstance(tool, DiffusionTool):
            return DiffusionControllerEvent()

        self._pending_layer = layer

        composite_below = None
        if tool.mode != "txt2img":
            composite_below = self._composite_below(layer)

        build = DiffusionRequestBuilder(self._layer_stack).build(
            layer,
            composite_below,
        )
        if build.error is not None:
            log.error(build.error.log_message or build.error.message)
            self._pending_layer = None
            return DiffusionControllerEvent(status=build.error.message)

        request = build.request
        if request is None:
            log.error(f"Diffusion request builder returned no request for {layer.name}")
            self._pending_layer = None
            return DiffusionControllerEvent(status="Could not build diffusion request")

        if request.ip_adapter_image is not None and not self._engine.ip_adapter_loaded:
            submitted = self._engine.submit_load_ip_adapter()
            if not submitted:
                self._pending_layer = None
                return DiffusionControllerEvent(status="Could not load IP-Adapter")
            return DiffusionControllerEvent(status="Loading IP-Adapter...")

        submitted = self._engine.submit_request(request)
        if not submitted:
            self._pending_layer = None
            return DiffusionControllerEvent()

        return DiffusionControllerEvent(
            status=f"Regenerating ({request.width}x{request.height})..."
        )

    def poll(self) -> DiffusionControllerEvent | None:
        engine_event = self._engine.poll_event()
        if engine_event is None:
            return None

        log.debug(
            f"[DiffusionGenerationController] task_type={engine_event.task_type}, "
            f"error={engine_event.error}, result_type={type(engine_event.result)}"
        )

        if engine_event.task_type == "load":
            if engine_event.error:
                log.error(f"Diffusion model load error: {engine_event.error}")
                self._pending_layer = None
                return DiffusionControllerEvent(
                    model_error=engine_event.error,
                    status=f"Model load error: {engine_event.error[:80]}",
                )
            pending = self._pending_layer
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
            if pending is not None and isinstance(pending.tool, DiffusionTool):
                return replace(
                    self._submit_regeneration(pending),
                    model_loaded_path=model_path,
                )
            return event

        if engine_event.task_type == "load_ip_adapter":
            if engine_event.error:
                log.error(f"IP-Adapter load error: {engine_event.error}")
                self._pending_layer = None
                return DiffusionControllerEvent(
                    ip_adapter_error=engine_event.error,
                    status=f"IP-Adapter error: {engine_event.error[:80]}",
                )
            pending = self._pending_layer
            event = DiffusionControllerEvent(
                ip_adapter_loaded=True,
                status="IP-Adapter loaded",
            )
            if pending is not None and isinstance(pending.tool, DiffusionTool):
                return replace(
                    self._submit_regeneration(pending),
                    ip_adapter_loaded=True,
                )
            return event

        if engine_event.task_type == "inference":
            if engine_event.error:
                log.error(f"Diffusion inference error: {engine_event.error}")
                self._pending_layer = None
                return DiffusionControllerEvent(
                    inference_error=engine_event.error,
                    status=f"Diffusion error: {engine_event.error[:80]}",
                )

            pending = self._pending_layer
            self._pending_layer = None
            if pending is None:
                return DiffusionControllerEvent(
                    status="Diffusion result ignored: no pending layer"
                )
            result = engine_event.result
            if not isinstance(result, DiffusionInferenceResult):
                log.error(
                    "Diffusion inference returned unexpected result "
                    f"type: {type(result)}"
                )
                return DiffusionControllerEvent(
                    status="Diffusion result ignored: invalid result"
                )
            log.debug(
                "[DiffusionGenerationController] inference OK, "
                f"seed={result.seed}, pending={type(pending).__name__}"
            )
            return DiffusionControllerEvent(
                inference_result=(pending, result.image, result.seed)
            )

        return DiffusionControllerEvent()
