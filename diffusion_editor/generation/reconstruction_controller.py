"""Toolkit-neutral coordinator for one image-to-3D job."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import uuid

from PIL import Image

from ..engines.reconstruction_engine import ReconstructionEngine
from .types import (
    ReconstructionRefineParameters,
    ReconstructionRefineRequest,
    ReconstructionRequest,
    ReconstructionParameters,
    ReconstructionResult,
    ReconstructionStage,
    ReconstructionStageEvent,
)


@dataclass(frozen=True)
class ReconstructionControllerEvent:
    status: str | None = None
    result: ReconstructionResult | None = None
    stage_event: ReconstructionStageEvent | None = None
    error: str | None = None


class ReconstructionController:
    def __init__(self, engine: ReconstructionEngine) -> None:
        self._engine = engine
        self._active_job_id: str | None = None

    @property
    def is_busy(self) -> bool:
        return self._active_job_id is not None or self._engine.is_busy

    def start(
        self,
        image: Image.Image,
        *,
        seed: int | None = None,
        parameters: ReconstructionParameters | None = None,
        target_stage: ReconstructionStage = ReconstructionStage.FINAL_MESH,
    ) -> ReconstructionControllerEvent:
        if self.is_busy:
            return ReconstructionControllerEvent()
        job_id = f"reconstruction_{uuid.uuid4().hex}"
        snapshot = parameters or ReconstructionParameters()
        if seed is not None:
            snapshot = replace(snapshot, seed=int(seed))
        request = ReconstructionRequest(
            image=image.copy(),
            parameters=snapshot,
            target_stage=target_stage,
        )
        if not self._engine.submit_request(request, job_id=job_id):
            return ReconstructionControllerEvent()
        self._active_job_id = job_id
        return ReconstructionControllerEvent(status="Generating 3D model with Pixal3D...")

    def start_refine(
        self,
        conditioning_image: Image.Image,
        mask_image: Image.Image,
        base_checkpoint_path: str,
        *,
        parameters: ReconstructionRefineParameters | None = None,
    ) -> ReconstructionControllerEvent:
        """Start geometry-only refinement; UI-to-mask mapping stays upstream."""
        if self.is_busy:
            return ReconstructionControllerEvent()
        if conditioning_image.size != mask_image.size:
            return ReconstructionControllerEvent(
                error="refine mask dimensions do not match conditioning image",
                status="Cannot refine: mask dimensions do not match conditioning",
            )
        job_id = f"reconstruction_refine_{uuid.uuid4().hex}"
        request = ReconstructionRefineRequest(
            conditioning_image=conditioning_image.copy(),
            mask_image=mask_image.copy(),
            base_checkpoint_path=str(base_checkpoint_path),
            parameters=parameters or ReconstructionRefineParameters(),
        )
        if not self._engine.submit_refine_request(request, job_id=job_id):
            return ReconstructionControllerEvent()
        self._active_job_id = job_id
        return ReconstructionControllerEvent(status="Refining 3D geometry with Pixal3D...")

    def cancel(self) -> bool:
        return self._engine.cancel()

    def poll(self) -> ReconstructionControllerEvent | None:
        event = self._engine.poll_event()
        if event is None:
            return None
        expected = self._active_job_id
        if expected is None or event.job_id != expected:
            return ReconstructionControllerEvent(status="Ignored stale 3D result")
        if isinstance(event.meta, ReconstructionStageEvent):
            return ReconstructionControllerEvent(stage_event=event.meta)
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
