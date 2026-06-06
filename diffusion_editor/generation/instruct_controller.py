"""Application workflow for InstructPix2Pix generation tasks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from PIL import Image
from tcbase import log

from .types import (
    GenerationError,
    InstructInferenceResult,
    InstructRequest,
)
from ..engines.instruct_engine import InstructEngine
from ..document.layer import Layer
from .patch_resolver import apply_patch_source_to_tool, resolve_source_patch
from ..document.tool import InstructTool


@dataclass(frozen=True)
class InstructControllerEvent:
    status: str | None = None
    model_loading: bool = False
    model_loaded: bool = False
    model_error: str | None = None
    inference_result: tuple[Layer, Image.Image, int] | None = None
    inference_error: str | None = None


class InstructGenerationController:
    def __init__(
            self,
            *,
            engine: InstructEngine,
            composite_below: Callable[[Layer], np.ndarray | None]):
        self._engine = engine
        self._composite_below = composite_below
        self._pending_layer: Layer | None = None

    @property
    def pending_layer(self) -> Layer | None:
        return self._pending_layer

    def clear_pending_layer(self, layer: Layer) -> None:
        if self._pending_layer is layer:
            self._pending_layer = None

    def submit_load_model(self) -> InstructControllerEvent:
        if self._engine.is_busy:
            return InstructControllerEvent()
        submitted = self._engine.submit_load()
        if not submitted:
            return InstructControllerEvent()
        return InstructControllerEvent(
            model_loading=True,
            status="Loading InstructPix2Pix model...",
        )

    def start_apply(self, layer: Layer) -> InstructControllerEvent:
        if self._engine.is_busy:
            return InstructControllerEvent()
        if not isinstance(layer.tool, InstructTool):
            return InstructControllerEvent()

        if not self._engine.is_loaded:
            self._pending_layer = layer
            event = self.submit_load_model()
            if not event.model_loading:
                self._pending_layer = None
            return event

        return self._submit_inference(layer)

    def _submit_inference(self, layer: Layer) -> InstructControllerEvent:
        tool = layer.tool
        if not isinstance(tool, InstructTool):
            return InstructControllerEvent()

        self._pending_layer = layer

        composite = self._composite_below(layer)
        if composite is None:
            self._pending_layer = None
            return InstructControllerEvent()

        patch = resolve_source_patch(
            layer,
            composite,
            fallback_canvas_rect=(
                tool.patch_x,
                tool.patch_y,
                tool.patch_x + tool.patch_w,
                tool.patch_y + tool.patch_h,
            ),
        )
        if isinstance(patch, GenerationError):
            log.error(patch.log_message or patch.message)
            self._pending_layer = None
            return InstructControllerEvent(status=patch.message)
        if patch is None:
            self._pending_layer = None
            return InstructControllerEvent(status="No source patch for instruction")

        apply_patch_source_to_tool(tool, patch)
        request = InstructRequest(
            image=tool.source_patch,
            instruction=tool.instruction,
            guidance_scale=tool.guidance_scale,
            image_guidance_scale=tool.image_guidance_scale,
            steps=tool.steps,
            seed=tool.seed,
        )
        submitted = self._engine.submit_request(request)
        if not submitted:
            self._pending_layer = None
            return InstructControllerEvent()

        return InstructControllerEvent(status="Applying instruction...")

    def poll(self) -> InstructControllerEvent | None:
        engine_event = self._engine.poll_event()
        if engine_event is None:
            return None

        if engine_event.task_type == "load":
            if engine_event.error:
                log.error(f"InstructPix2Pix load error: {engine_event.error}")
                self._pending_layer = None
                return InstructControllerEvent(
                    model_error=engine_event.error,
                    status=f"InstructPix2Pix load error: {engine_event.error[:80]}",
                )
            pending = self._pending_layer
            event = InstructControllerEvent(
                model_loaded=True,
                status="InstructPix2Pix model loaded",
            )
            if pending is not None and isinstance(pending.tool, InstructTool):
                return replace(
                    self._submit_inference(pending),
                    model_loaded=True,
                )
            return event

        if engine_event.task_type == "inference":
            if engine_event.error:
                log.error(f"InstructPix2Pix inference error: {engine_event.error}")
                self._pending_layer = None
                return InstructControllerEvent(
                    inference_error=engine_event.error,
                    status=f"InstructPix2Pix error: {engine_event.error[:80]}",
                )
            pending = self._pending_layer
            self._pending_layer = None
            if pending is None:
                return InstructControllerEvent(
                    status="InstructPix2Pix result ignored: no pending layer"
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
            return InstructControllerEvent(
                inference_result=(pending, result.image, result.seed)
            )

        return InstructControllerEvent()
