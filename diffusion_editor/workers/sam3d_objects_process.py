"""Subprocess boundary for the gated local SAM 3D Objects runtime."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import tempfile
import threading

from PIL import Image

from ..generation.types import (
    ReconstructionParameters,
    ReconstructionStage,
    ReconstructionStageArtifact,
    ReconstructionStageEvent,
    ReconstructionStageStatus,
)
from .pixal3d_process import Pixal3DProcessClient


DEFAULT_SAM3D_OBJECTS_ROOT = Path("/home/mirmik/soft/sam-3d-objects")
DEFAULT_SAM3D_OBJECTS_PYTHON = DEFAULT_SAM3D_OBJECTS_ROOT / "venv/bin/python"
DEFAULT_SAM3D_OBJECTS_CHECKPOINTS = DEFAULT_SAM3D_OBJECTS_ROOT / "checkpoints"


class Sam3DObjectsProcessClient(Pixal3DProcessClient):
    """Run SAM 3D Objects and publish its point, sparse and mesh artifacts."""

    def __init__(
        self,
        *,
        python: str | Path | None = None,
        root: str | Path | None = None,
        checkpoints: str | Path | None = None,
        runner_path: str | Path | None = None,
    ) -> None:
        resolved_root = Path(root or os.environ.get(
            "DIFFUSION_EDITOR_SAM3D_OBJECTS_ROOT",
            DEFAULT_SAM3D_OBJECTS_ROOT,
        ))
        resolved_checkpoints = Path(checkpoints or os.environ.get(
            "DIFFUSION_EDITOR_SAM3D_OBJECTS_CHECKPOINTS",
            DEFAULT_SAM3D_OBJECTS_CHECKPOINTS,
        ))
        super().__init__(
            python=python or os.environ.get(
                "DIFFUSION_EDITOR_SAM3D_OBJECTS_PYTHON",
                DEFAULT_SAM3D_OBJECTS_PYTHON,
            ),
            root=resolved_root,
            model_path=resolved_checkpoints,
            runner_path=runner_path or Path(__file__).with_name(
                "sam3d_objects_staged_runner.py"
            ),
            staged=True,
        )
        self._checkpoints = resolved_checkpoints.expanduser()
        self._backend_label = "SAM 3D Objects"
        self._environment_overrides.update({
            "LIDRA_SKIP_INIT": "true",
            "ATTN_BACKEND": "sdpa",
            "SPARSE_ATTN_BACKEND": "sdpa",
            "SPCONV_ALGO": "native",
            "MPLCONFIGDIR": "/tmp/sam3d-objects-matplotlib",
            "TORCH_EXTENSIONS_DIR": "/tmp/sam3d-objects-torch-extensions",
        })

    def generate(
        self,
        image: Image.Image,
        seed: int,
        cancel: threading.Event,
        *,
        parameters: ReconstructionParameters | None = None,
        target_stage: ReconstructionStage = ReconstructionStage.FINAL_MESH,
        on_event: Callable[[ReconstructionStageEvent], None] | None = None,
    ) -> tuple[Path, Path]:
        supported = {
            ReconstructionStage.SOURCE_IMAGE,
            ReconstructionStage.POINT_CLOUD,
            ReconstructionStage.SPARSE_OCCUPANCY,
            ReconstructionStage.HR_SHAPE_FLOW,
            ReconstructionStage.HR_SHAPE_LATENT,
            ReconstructionStage.TEXTURE_LATENT,
            ReconstructionStage.FINAL_MESH,
        }
        if target_stage not in supported:
            raise ValueError(
                f"stage {target_stage.value} is unavailable in SAM 3D Objects"
            )
        self._validate_runtime()
        snapshot = parameters or ReconstructionParameters()
        artifact_root = Path(tempfile.mkdtemp(
            prefix="diffusion-editor-sam3d-objects-"
        ))
        self._artifact_roots.append(artifact_root)
        source_path = artifact_root / "source.png"
        output_path = artifact_root / "model.glb"
        events_path = artifact_root / "events.jsonl"
        log_path = artifact_root / "sam3d-objects.log"
        image.convert("RGBA").save(source_path, format="PNG")
        source_artifact = ReconstructionStageArtifact(
            ReconstructionStage.SOURCE_IMAGE, str(source_path), "image"
        )
        self._artifacts = [source_artifact]
        self._checkpoint_path = None
        self._texture_checkpoint_path = None
        self._conditioning_path = None
        if on_event is not None:
            on_event(ReconstructionStageEvent(
                ReconstructionStage.SOURCE_IMAGE,
                ReconstructionStageStatus.READY,
                artifact=source_artifact,
            ))
        if target_stage is ReconstructionStage.SOURCE_IMAGE:
            return output_path, source_path

        command = [
            str(self._python), str(self._runner_path),
            "--sam3d-root", str(self._root),
            "--config", str(self._checkpoints / "pipeline.yaml"),
            "--image", str(source_path),
            "--output", str(output_path),
            "--events", str(events_path),
            "--target-stage", target_stage.value,
            "--seed", str(int(seed)),
            "--sparse-steps", str(snapshot.sam3d_sparse_steps),
            "--slat-steps", str(snapshot.sam3d_slat_steps),
            "--sparse-guidance", str(snapshot.sam3d_sparse_guidance_scale),
            "--slat-guidance", str(snapshot.sam3d_slat_guidance_scale),
            "--simplify", str(snapshot.sam3d_simplify),
            "--texture-size", str(snapshot.texture_size),
        ]
        self._run_command(command, log_path, events_path, cancel, on_event)
        for attribute, name in (
            ("_conditioning_path", "conditioning.png"),
            ("_checkpoint_path", "structured-latent.pt"),
            ("_texture_checkpoint_path", "gaussian-splat.ply"),
        ):
            path = artifact_root / name
            if path.is_file():
                setattr(self, attribute, path)
        if (
            target_stage is ReconstructionStage.FINAL_MESH
            and (not output_path.is_file() or output_path.stat().st_size == 0)
        ):
            raise RuntimeError("SAM 3D Objects completed without producing a GLB")
        return output_path, source_path

    def _validate_runtime(self) -> None:
        if not self._python.is_file():
            raise RuntimeError(
                f"SAM 3D Objects Python not found: {self._python}. Set "
                "DIFFUSION_EDITOR_SAM3D_OBJECTS_PYTHON."
            )
        if not self._runner_path.is_file():
            raise RuntimeError(
                f"SAM 3D Objects staged runner not found: {self._runner_path}"
            )
        if not self._root.is_dir():
            raise RuntimeError(
                f"SAM 3D Objects root not found: {self._root}. Set "
                "DIFFUSION_EDITOR_SAM3D_OBJECTS_ROOT."
            )
        config = self._checkpoints / "pipeline.yaml"
        if not config.is_file():
            raise RuntimeError(
                f"SAM 3D Objects checkpoints not found: {self._checkpoints}. "
                "Download facebook/sam-3d-objects after accepting its license."
            )
