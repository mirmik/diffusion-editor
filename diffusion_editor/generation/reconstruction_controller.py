"""Toolkit-neutral coordinator for one image-to-3D job."""

from __future__ import annotations

from dataclasses import dataclass
import uuid

from PIL import Image

from ..engines.reconstruction_engine import ReconstructionEngine
from .types import ReconstructionRequest, ReconstructionResult


@dataclass(frozen=True)
class ReconstructionControllerEvent:
    status: str | None = None
    result: ReconstructionResult | None = None
    error: str | None = None


class ReconstructionController:
    def __init__(self, engine: ReconstructionEngine) -> None:
        self._engine = engine
        self._active_job_id: str | None = None

    @property
    def is_busy(self) -> bool:
        return self._active_job_id is not None or self._engine.is_busy

    def start(self, image: Image.Image, *, seed: int = 42) -> ReconstructionControllerEvent:
        if self.is_busy:
            return ReconstructionControllerEvent()
        job_id = f"reconstruction_{uuid.uuid4().hex}"
        request = ReconstructionRequest(image=image.copy(), seed=int(seed))
        if not self._engine.submit_request(request, job_id=job_id):
            return ReconstructionControllerEvent()
        self._active_job_id = job_id
        return ReconstructionControllerEvent(status="Generating 3D model with Pixal3D...")

    def cancel(self) -> bool:
        return self._engine.cancel()

    def poll(self) -> ReconstructionControllerEvent | None:
        event = self._engine.poll_event()
        if event is None:
            return None
        expected = self._active_job_id
        if expected is None or event.job_id != expected:
            return ReconstructionControllerEvent(status="Ignored stale 3D result")
        self._active_job_id = None
        if event.error is not None:
            return ReconstructionControllerEvent(
                error=event.error,
                status=f"3D generation failed: {event.error[:120]}",
            )
        if event.task_type != "reconstruction" or not isinstance(
            event.result, ReconstructionResult
        ):
            return ReconstructionControllerEvent(
                error="invalid Pixal3D result",
                status="3D generation failed: invalid worker result",
            )
        return ReconstructionControllerEvent(
            result=event.result,
            status="3D model generated",
        )
