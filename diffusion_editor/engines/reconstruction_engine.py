"""Background image-to-3D engine."""

from __future__ import annotations

from tcbase import log

from ..generation.types import (
    RECONSTRUCTION_BACKEND_LABELS,
    ReconstructionBackend,
    ReconstructionRefineRequest,
    ReconstructionRequest,
    ReconstructionResult,
    ReconstructionRunKind,
    ReconstructionStageEvent,
    ReconstructionTextureRefineRequest,
)
from ..workers.pixal3d_process import Pixal3DProcessClient
from ..workers.hi3dgen_process import Hi3DGenProcessClient
from ..workers.hunyuan3d21_process import Hunyuan3D21ProcessClient
from ..workers.sam3d_objects_process import Sam3DObjectsProcessClient
from ..workers.spar3d_process import Spar3DProcessClient
from ..workers.trellis2_process import Trellis2ProcessClient
from .threaded_lifecycle import EngineTaskQueue


class ReconstructionEngine:
    supports_job_ids = True

    def __init__(
        self,
        client: Pixal3DProcessClient | None = None,
        *,
        trellis_client: Trellis2ProcessClient | None = None,
        spar3d_client: Spar3DProcessClient | None = None,
        hi3dgen_client: Hi3DGenProcessClient | None = None,
        hunyuan3d21_client: Hunyuan3D21ProcessClient | None = None,
        sam3d_objects_client: Sam3DObjectsProcessClient | None = None,
    ) -> None:
        self._client = client or Pixal3DProcessClient()
        self._trellis_client = trellis_client or Trellis2ProcessClient()
        self._spar3d_client = spar3d_client or Spar3DProcessClient()
        self._hi3dgen_client = hi3dgen_client or Hi3DGenProcessClient()
        self._hunyuan3d21_client = (
            hunyuan3d21_client or Hunyuan3D21ProcessClient()
        )
        self._sam3d_objects_client = (
            sam3d_objects_client or Sam3DObjectsProcessClient()
        )
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

        backend_label = RECONSTRUCTION_BACKEND_LABELS[
            request.parameters.backend
        ]
        return self._tasks.submit(
            "reconstruction",
            lambda cancel: self._run(request, cancel, emit),
            job_id=job_id,
            name=f"{request.parameters.backend.value}-reconstruction",
            on_error=lambda _exc: log.exception(
                "%s reconstruction failed", backend_label
            ),
        )

    def _run(
        self,
        request: ReconstructionRequest,
        cancel,
        emit,
    ) -> ReconstructionResult:
        client = {
            ReconstructionBackend.PIXAL3D: self._client,
            ReconstructionBackend.TRELLIS2: self._trellis_client,
            ReconstructionBackend.SPAR3D: self._spar3d_client,
            ReconstructionBackend.HI3DGEN: self._hi3dgen_client,
            ReconstructionBackend.HUNYUAN3D21: self._hunyuan3d21_client,
            ReconstructionBackend.SAM3D_OBJECTS: self._sam3d_objects_client,
        }[request.parameters.backend]
        glb_path, source_path = client.generate(
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
            artifacts=tuple(client.artifacts),
            kind=ReconstructionRunKind.BASE,
            conditioning_path=(
                str(client.conditioning_path)
                if client.conditioning_path else None
            ),
            checkpoint_path=(
                str(client.checkpoint_path)
                if client.checkpoint_path else None
            ),
            texture_checkpoint_path=(
                str(client.texture_checkpoint_path)
                if client.texture_checkpoint_path else None
            ),
            backend=request.parameters.backend,
        )

    def submit_refine_request(
        self,
        request: ReconstructionRefineRequest,
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
            lambda cancel: self._run_refine(request, cancel, emit),
            job_id=job_id,
            name="pixal3d-masked-refine",
            on_error=lambda _exc: log.exception("Pixal3D refinement failed"),
        )

    def _run_refine(self, request, cancel, emit) -> ReconstructionResult:
        glb_path, condition_path = self._client.refine(
            request.conditioning_image,
            request.mask_image,
            request.base_checkpoint_path,
            cancel,
            parameters=request.parameters,
            generation_parameters=request.generation_parameters,
            on_event=emit,
        )
        return ReconstructionResult(
            glb_path=str(glb_path),
            source_path=str(condition_path),
            artifacts=tuple(self._client.artifacts),
            kind=ReconstructionRunKind.MASKED_REFINE,
            conditioning_path=(
                str(self._client.conditioning_path)
                if self._client.conditioning_path else None
            ),
            checkpoint_path=(
                str(self._client.checkpoint_path)
                if self._client.checkpoint_path else None
            ),
            texture_checkpoint_path=(
                str(self._client.texture_checkpoint_path)
                if self._client.texture_checkpoint_path else None
            ),
            backend=ReconstructionBackend.PIXAL3D,
        )

    def submit_texture_refine_request(
        self,
        request: ReconstructionTextureRefineRequest,
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
            lambda cancel: self._run_texture_refine(request, cancel, emit),
            job_id=job_id,
            name="pixal3d-masked-texture-refine",
            on_error=lambda _exc: log.exception(
                "Pixal3D texture refinement failed"
            ),
        )

    def _run_texture_refine(self, request, cancel, emit) -> ReconstructionResult:
        glb_path, source_path = self._client.refine_texture(
            request.conditioning_image,
            request.mask_image,
            request.shape_checkpoint_path,
            request.texture_checkpoint_path,
            cancel,
            parameters=request.parameters,
            generation_parameters=request.generation_parameters,
            on_event=emit,
        )
        return ReconstructionResult(
            glb_path=str(glb_path),
            source_path=str(source_path),
            artifacts=tuple(self._client.artifacts),
            kind=ReconstructionRunKind.MASKED_TEXTURE_REFINE,
            conditioning_path=(
                str(self._client.conditioning_path)
                if self._client.conditioning_path else None
            ),
            checkpoint_path=request.shape_checkpoint_path,
            texture_checkpoint_path=(
                str(self._client.texture_checkpoint_path)
                if self._client.texture_checkpoint_path else None
            ),
            backend=ReconstructionBackend.PIXAL3D,
        )

    def poll_event(self):
        return self._tasks.poll_event()

    def cancel(self) -> bool:
        return self._tasks.cancel()

    def shutdown(self, timeout: float = 2.0) -> None:
        self._tasks.cancel()
        if self._tasks.shutdown(timeout):
            self._client.shutdown(timeout)
            self._trellis_client.shutdown(timeout)
            self._spar3d_client.shutdown(timeout)
            self._hi3dgen_client.shutdown(timeout)
            self._hunyuan3d21_client.shutdown(timeout)
            self._sam3d_objects_client.shutdown(timeout)
