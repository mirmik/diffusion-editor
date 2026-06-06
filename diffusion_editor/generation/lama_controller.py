"""Application workflow for LaMa inpainting tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image
from tcbase import log

from .types import LamaRequest, LamaResult
from ..engines.lama_engine import LamaEngine
from ..document.layer import Layer
from .patch_resolver import (
    apply_patch_source_to_tool,
    extract_layer_mask_patch,
    source_patch_from_mask,
)
from ..document.tool import LamaTool


@dataclass(frozen=True)
class LamaControllerEvent:
    status: str | None = None
    inference_result: tuple[Layer, Image.Image] | None = None
    inference_error: str | None = None


class LamaGenerationController:
    def __init__(
            self,
            *,
            engine: LamaEngine,
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

    def start_remove(self, layer: Layer) -> LamaControllerEvent:
        if self._engine.is_busy:
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
        request = LamaRequest(image=patch.image, mask_image=mask)
        submitted = self._engine.submit_request(request)
        if not submitted:
            return LamaControllerEvent()

        self._pending_layer = layer
        return LamaControllerEvent(status="Removing objects (LaMa)...")

    def poll(self) -> LamaControllerEvent | None:
        engine_event = self._engine.poll_event()
        if engine_event is None:
            return None

        if engine_event.error is not None:
            log.error(f"LaMa error: {engine_event.error}")
            self._pending_layer = None
            return LamaControllerEvent(
                inference_error=engine_event.error,
                status=f"LaMa error: {engine_event.error[:80]}",
            )

        pending = self._pending_layer
        self._pending_layer = None
        if pending is None:
            return LamaControllerEvent(
                status="LaMa result ignored: no pending layer"
            )
        result = engine_event.result
        if not isinstance(result, LamaResult):
            log.error(
                "LaMa inference returned unexpected result "
                f"type: {type(result)}"
            )
            return LamaControllerEvent(status="LaMa result ignored: invalid result")
        return LamaControllerEvent(inference_result=(pending, result.image))
