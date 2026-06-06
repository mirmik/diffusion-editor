"""Application workflow for background segmentation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from tcbase import log

from .types import SegmentationRequest, SegmentationResult
from ..document.layer import Layer
from ..engines.segmentation_engine import SegmentationEngine


@dataclass(frozen=True)
class SegmentationControllerEvent:
    status: str | None = None
    segmentation_result: tuple[Layer, np.ndarray] | None = None
    segmentation_error: str | None = None


class SegmentationGenerationController:
    def __init__(
            self,
            *,
            engine: SegmentationEngine,
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

    def start_select_background(self, layer: Layer) -> SegmentationControllerEvent:
        if self._engine.is_busy:
            return SegmentationControllerEvent()
        composite = self._composite_below(layer)
        if composite is None:
            return SegmentationControllerEvent()

        request = SegmentationRequest(image=composite, invert=True)
        submitted = self._engine.submit_request(request)
        if not submitted:
            return SegmentationControllerEvent()

        self._pending_layer = layer
        return SegmentationControllerEvent(status="Segmenting background...")

    def poll(self) -> SegmentationControllerEvent | None:
        engine_event = self._engine.poll_event()
        if engine_event is None:
            return None

        if engine_event.error is not None:
            log.error(f"Segmentation error: {engine_event.error}")
            self._pending_layer = None
            return SegmentationControllerEvent(
                segmentation_error=engine_event.error,
                status=f"Segmentation error: {engine_event.error[:80]}",
            )

        pending = self._pending_layer
        self._pending_layer = None
        if pending is None:
            return SegmentationControllerEvent(
                status="Segmentation result ignored: no pending layer"
            )

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
            segmentation_result=(pending, result.mask),
        )
