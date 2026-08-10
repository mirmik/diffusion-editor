"""Subprocess boundary for an optional local SPAR3D runtime."""

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


DEFAULT_SPAR3D_ROOT = Path("/home/mirmik/soft/SPAR3D")
DEFAULT_SPAR3D_PYTHON = DEFAULT_SPAR3D_ROOT / "venv/bin/python"
DEFAULT_SPAR3D_MODEL = DEFAULT_SPAR3D_ROOT / "models/stable-point-aware-3d"


class Spar3DProcessClient(Pixal3DProcessClient):
    """Run SPAR3D and expose its editable point-cloud intermediate."""

    def __init__(
        self,
        *,
        python: str | Path | None = None,
        root: str | Path | None = None,
        model_path: str | Path | None = None,
        runner_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            python=python or os.environ.get(
                "DIFFUSION_EDITOR_SPAR3D_PYTHON", DEFAULT_SPAR3D_PYTHON
            ),
            root=root or os.environ.get(
                "DIFFUSION_EDITOR_SPAR3D_ROOT", DEFAULT_SPAR3D_ROOT
            ),
            model_path=model_path or os.environ.get(
                "DIFFUSION_EDITOR_SPAR3D_MODEL", DEFAULT_SPAR3D_MODEL
            ),
            runner_path=runner_path or Path(__file__).with_name(
                "spar3d_staged_runner.py"
            ),
            staged=True,
        )
        self._backend_label = "SPAR3D"

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
        if target_stage not in {
            ReconstructionStage.SOURCE_IMAGE,
            ReconstructionStage.POINT_CLOUD,
            ReconstructionStage.FINAL_MESH,
        }:
            raise ValueError(f"stage {target_stage.value} is unavailable in SPAR3D")
        self._validate_runtime()
        snapshot = parameters or ReconstructionParameters()
        artifact_root = Path(tempfile.mkdtemp(
            prefix="diffusion-editor-spar3d-"
        ))
        self._artifact_roots.append(artifact_root)
        source_path = artifact_root / "source.png"
        output_path = artifact_root / "model.glb"
        events_path = artifact_root / "events.jsonl"
        log_path = artifact_root / "spar3d.log"
        image.convert("RGBA").save(source_path, format="PNG")
        source_artifact = ReconstructionStageArtifact(
            ReconstructionStage.SOURCE_IMAGE,
            str(source_path),
            "image",
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
            str(self._python),
            str(self._runner_path),
            "--spar3d-root", str(self._root),
            "--image", str(source_path),
            "--output", str(output_path),
            "--events", str(events_path),
            "--model-path", str(self._model_path),
            "--target-stage", target_stage.value,
            "--seed", str(int(seed)),
            "--guidance-scale", str(snapshot.spar3d_guidance_scale),
            "--texture-size", str(snapshot.texture_size),
        ]
        if snapshot.low_vram:
            command.append("--low-vram")
        self._run_command(command, log_path, events_path, cancel, on_event)
        preprocessed = artifact_root / "preprocessed.png"
        if preprocessed.is_file():
            self._conditioning_path = preprocessed
        if (
            target_stage is ReconstructionStage.FINAL_MESH
            and (not output_path.is_file() or output_path.stat().st_size == 0)
        ):
            raise RuntimeError("SPAR3D completed without producing a GLB")
        return output_path, source_path

    def _validate_runtime(self) -> None:
        if not self._python.is_file():
            raise RuntimeError(
                f"SPAR3D Python not found: {self._python}. Set "
                "DIFFUSION_EDITOR_SPAR3D_PYTHON."
            )
        if not self._runner_path.is_file():
            raise RuntimeError(
                f"SPAR3D staged runner not found: {self._runner_path}"
            )
        if not self._root.is_dir():
            raise RuntimeError(
                f"SPAR3D root not found: {self._root}. Set "
                "DIFFUSION_EDITOR_SPAR3D_ROOT."
            )
        if not self._model_path.exists():
            raise RuntimeError(
                f"SPAR3D model not found: {self._model_path}. Set "
                "DIFFUSION_EDITOR_SPAR3D_MODEL after accepting the gated "
                "model license."
            )
