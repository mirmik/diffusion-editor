"""Application workflow for Grounding DINO + SAM detection tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from tcbase import log

from ..engines.grounding_engine import GroundingEngine
from .types import GroundingParams, GroundingRequest, GroundingResult
from ..document.layer import Layer


@dataclass(frozen=True)
class GroundingControllerEvent:
    status: str | None = None
    grounding_result: tuple[Layer, GroundingResult] | None = None
    grounding_error: str | None = None


class GroundingController:
    def __init__(
            self,
            *,
            engine: GroundingEngine,
            composite: Callable[[], np.ndarray]):
        self._engine = engine
        self._composite = composite
        self._pending_layer: Layer | None = None

    @property
    def pending_layer(self) -> Layer | None:
        return self._pending_layer

    def gpu_available(self) -> bool:
        return self._engine.gpu_available()

    def clear_pending_layer(self, layer: Layer) -> None:
        if self._pending_layer is layer:
            self._pending_layer = None

    def start_detection(
            self,
            layer: Layer,
            params: GroundingParams) -> GroundingControllerEvent:
        if self._engine.is_busy:
            return GroundingControllerEvent()

        try:
            image = self._composite()
        except Exception as exc:
            log.error(f"Grounding: failed to get composite - {exc}")
            return GroundingControllerEvent(status="Grounding: failed to get composite")

        request = GroundingRequest(image=image, params=params)
        submitted = self._engine.submit_request(request)
        if not submitted:
            return GroundingControllerEvent()

        self._pending_layer = layer
        return GroundingControllerEvent(status="Grounding: detecting...")

    def poll(self) -> GroundingControllerEvent | None:
        engine_event = self._engine.poll_event()
        if engine_event is None:
            return None

        if engine_event.error is not None:
            log.error(f"Grounding: {engine_event.error}")
            self._pending_layer = None
            return GroundingControllerEvent(
                grounding_error=engine_event.error,
                status=f"Grounding error: {engine_event.error[:80]}",
            )

        if engine_event.result is not None:
            pending = self._pending_layer
            self._pending_layer = None
            if pending is None:
                return GroundingControllerEvent(
                    status="Grounding result ignored: no pending layer"
                )
            return GroundingControllerEvent(
                status=engine_event.status,
                grounding_result=(pending, engine_event.result),
            )

        if engine_event.status is not None:
            return GroundingControllerEvent(status=engine_event.status)

        return GroundingControllerEvent()
