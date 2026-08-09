"""Background image-to-3D engine."""

from __future__ import annotations

from tcbase import log

from ..generation.types import (
    ReconstructionRequest,
    ReconstructionResult,
    ReconstructionStageEvent,
)
from ..workers.pixal3d_process import Pixal3DProcessClient
from .threaded_lifecycle import EngineTaskQueue


class ReconstructionEngine:
    supports_job_ids = True

    def __init__(self, client: Pixal3DProcessClient | None = None) -> None:
        self._client = client or Pixal3DProcessClient()
        self._tasks = EngineTaskQueue()

    @property
    def is_busy(self) -> bool:
        return self._tasks.is_busy

    def submit_request(
        self,
        request: ReconstructionRequest,
        *,
        job_id: str | None = None,
    ) -> bool:
        def emit(event: ReconstructionStageEvent) -> None:
            from ..generation.types import EnginePollEvent

            self._tasks.emit(EnginePollEvent(
                task_type="reconstruction",
                meta=event,
                job_id=job_id,
            ))

        return self._tasks.submit(
            "reconstruction",
            lambda cancel: self._run(request, cancel, emit),
            job_id=job_id,
            name="pixal3d-reconstruction",
            on_error=lambda _exc: log.exception("Pixal3D reconstruction failed"),
        )

    def _run(
        self,
        request: ReconstructionRequest,
        cancel,
        emit,
    ) -> ReconstructionResult:
        glb_path, source_path = self._client.generate(
            request.image,
            request.parameters.seed,
            cancel,
            parameters=request.parameters,
            target_stage=request.target_stage,
            on_event=emit,
        )
        return ReconstructionResult(
            glb_path=str(glb_path),
            source_path=str(source_path),
            completed_stage=request.target_stage,
            artifacts=tuple(self._client.artifacts),
        )

    def poll_event(self):
        return self._tasks.poll_event()

    def cancel(self) -> bool:
        return self._tasks.cancel()

    def shutdown(self, timeout: float = 2.0) -> None:
        self._tasks.cancel()
        if self._tasks.shutdown(timeout):
            self._client.shutdown(timeout)
