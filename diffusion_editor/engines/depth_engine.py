"""Main-process facade for monocular depth estimation."""

from __future__ import annotations

import numpy as np
from tcbase import log

from ..generation.types import (
    DepthEstimationRequest,
    DepthEstimationResult,
    EnginePollEvent,
)
from ..workers.ml_process import MlProcessClient
from .threaded_lifecycle import EngineTaskQueue


class DepthEstimationEngine:
    supports_job_ids = True

    def __init__(self, client: MlProcessClient | None = None) -> None:
        self._client = client or MlProcessClient()
        self._tasks = EngineTaskQueue()

    @property
    def is_busy(self) -> bool:
        return self._tasks.is_busy

    def submit_request(
            self,
            request: DepthEstimationRequest,
            *,
            job_id: str | None = None) -> bool:
        return self._tasks.submit(
            "depth",
            lambda cancel: self._run(request, cancel, job_id),
            job_id=job_id,
            name="depth-estimation",
            on_error=lambda _exc: log.exception("Depth estimation failed"),
        )

    def _run(self, request, cancel, job_id) -> DepthEstimationResult:
        result = self._client.request(
            "depth",
            {"model_id": request.model_id},
            cancel,
            images={"image": request.image},
            on_progress=lambda message: self._tasks.emit(EnginePollEvent(
                task_type="depth",
                meta={"progress": message},
                job_id=job_id,
            )),
        )
        depth_map = np.asarray(result["image"].convert("L"), dtype=np.uint8)
        expected_shape = tuple(int(value) for value in request.image.shape[:2])
        if depth_map.shape != expected_shape:
            raise RuntimeError(
                "Depth worker returned an unexpected image size: "
                f"{depth_map.shape} != {expected_shape}"
            )
        return DepthEstimationResult(
            depth_map=np.ascontiguousarray(depth_map)
        )

    def poll_event(self):
        return self._tasks.poll_event()

    def cancel(self) -> bool:
        return self._tasks.cancel()

    def shutdown(self, timeout: float = 1.0) -> None:
        if self._tasks.shutdown(timeout):
            self._client.shutdown(timeout)
