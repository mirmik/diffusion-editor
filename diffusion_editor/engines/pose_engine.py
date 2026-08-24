"""Threaded facade for the isolated pose-estimation process."""

from __future__ import annotations

from tcbase import log

from ..generation.pose_estimation import PoseEstimationRequest
from ..workers.pose_process import PoseProcessClient
from .threaded_lifecycle import EngineTaskQueue


class PoseEstimationEngine:
    supports_job_ids = True

    def __init__(self, client: PoseProcessClient | None = None) -> None:
        self._client = client or PoseProcessClient()
        self._tasks = EngineTaskQueue()

    @property
    def is_loaded(self) -> bool:
        return self._client.is_running

    @property
    def is_busy(self) -> bool:
        return self._tasks.is_busy

    def submit_request(
            self,
            request: PoseEstimationRequest,
            *,
            job_id: str | None = None) -> bool:
        return self._tasks.submit(
            "pose-estimation",
            lambda cancel: self._client.estimate(
                request.image,
                request.profile_id,
                cancel,
                on_progress=lambda message: log.debug(
                    f"[Pose estimation] {message}"),
            ),
            job_id=job_id,
            name="pose-estimation-inference",
            on_error=lambda _exc: log.exception("Pose estimation failed"),
        )

    def poll_event(self):
        return self._tasks.poll_event()

    def cancel(self) -> bool:
        return self._tasks.cancel()

    def shutdown(self, timeout: float = 1.0) -> None:
        if self._tasks.shutdown(timeout):
            self._client.shutdown(timeout)
