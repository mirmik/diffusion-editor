"""Main-process facade for monocular depth estimation."""

from __future__ import annotations

import numpy as np
from tcbase import log

from ..generation.types import (
    DepthEstimationRequest,
    DepthEstimationResult,
    DepthValueKind,
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
        profile = request.profile
        result = self._client.request(
            "depth",
            {
                "profile_id": profile.stable_id,
                "model_id": profile.model_id,
                "backend": profile.backend.value,
                "title": profile.title,
                "process_resolution": profile.process_resolution,
                "direct_depth": profile.direct_depth,
                "value_kind": profile.value_kind.value,
                "use_ray_pose": profile.use_ray_pose,
            },
            cancel,
            images={"image": request.image},
            on_progress=lambda message: self._tasks.emit(EnginePollEvent(
                task_type="depth",
                meta={"progress": message},
                job_id=job_id,
            )),
        )
        depth_map = np.asarray(result["depth"])
        if depth_map.ndim != 2 or depth_map.size == 0:
            raise RuntimeError(
                "Depth worker returned an invalid depth shape"
            )
        if depth_map.dtype != np.float32:
            raise RuntimeError(
                f"Depth worker returned {depth_map.dtype}, expected float32")
        if not np.isfinite(depth_map).all():
            raise RuntimeError("Depth worker returned non-finite depth")
        try:
            value_kind = DepthValueKind(str(result["value_kind"]))
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                "Depth worker returned an unknown value convention") from exc
        if profile.value_kind is DepthValueKind.DIRECT_METRIC:
            if value_kind is not DepthValueKind.DIRECT_METRIC:
                raise RuntimeError("Metric depth profile lost metric semantics")
        elif profile.value_kind is DepthValueKind.INVERSE_RELATIVE:
            if value_kind is not DepthValueKind.INVERSE_RELATIVE:
                raise RuntimeError("Relative depth profile changed convention")
        elif value_kind not in {
                DepthValueKind.DIRECT_SCALE_AMBIGUOUS,
                DepthValueKind.DIRECT_METRIC,
        }:
            raise RuntimeError("Direct depth profile changed convention")

        intrinsics = result.get("intrinsics")
        if (
                profile.predicts_intrinsics
                or profile.value_kind is DepthValueKind.DIRECT_METRIC
        ) and intrinsics is None:
            raise RuntimeError(
                f"{profile.title} did not return camera intrinsics")
        confidence = result.get("confidence")
        if confidence is not None and confidence.shape != depth_map.shape:
            raise RuntimeError(
                "Depth worker returned an unexpected confidence size")
        return DepthEstimationResult(
            depth_map=np.ascontiguousarray(depth_map),
            profile_id=profile.stable_id,
            value_kind=value_kind,
            intrinsics=(
                np.ascontiguousarray(intrinsics, dtype=np.float32)
                if intrinsics is not None else None
            ),
            confidence=(
                np.ascontiguousarray(confidence, dtype=np.float32)
                if confidence is not None else None
            ),
            field_of_view_degrees=result.get("field_of_view_degrees"),
            scale_factor=result.get("scale_factor"),
        )

    def poll_event(self):
        return self._tasks.poll_event()

    def cancel(self) -> bool:
        return self._tasks.cancel()

    def shutdown(self, timeout: float = 1.0) -> None:
        if self._tasks.shutdown(timeout):
            self._client.shutdown(timeout)
