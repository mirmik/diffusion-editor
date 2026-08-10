"""Subprocess boundary for the optional local TRELLIS.2 backend."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
from collections.abc import Callable

from PIL import Image

from ..generation.types import (
    ReconstructionParameters,
    ReconstructionStage,
    ReconstructionStageArtifact,
    ReconstructionStageEvent,
    ReconstructionStageStatus,
)
from .pixal3d_process import Pixal3DProcessClient


DEFAULT_TRELLIS2_ROOT = Path("/home/mirmik/soft/TRELLIS.2")
DEFAULT_TRELLIS2_PYTHON = DEFAULT_TRELLIS2_ROOT / "venv/bin/python"
DEFAULT_TRELLIS2_MODEL = DEFAULT_TRELLIS2_ROOT / "models/TRELLIS.2-4B"


class Trellis2ProcessClient(Pixal3DProcessClient):
    """Run TRELLIS.2 while reusing the editor-owned event transport."""

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
                "DIFFUSION_EDITOR_TRELLIS2_PYTHON", DEFAULT_TRELLIS2_PYTHON
            ),
            root=root or os.environ.get(
                "DIFFUSION_EDITOR_TRELLIS2_ROOT", DEFAULT_TRELLIS2_ROOT
            ),
            model_path=model_path or os.environ.get(
                "DIFFUSION_EDITOR_TRELLIS2_MODEL", DEFAULT_TRELLIS2_MODEL
            ),
            runner_path=runner_path or Path(__file__).with_name(
                "trellis2_staged_runner.py"
            ),
            staged=True,
        )
        self._backend_label = "TRELLIS.2"
        # TRELLIS.2 sparse attention does not support PyTorch SDPA.  The
        # installed runtime ships xformers but not flash-attn, while a parent
        # editor environment may still request flash_attn for another backend.
        # Keep the dense path on SDPA and explicitly select the available
        # sparse implementation for this subprocess.
        self._environment_overrides.update({
            "ATTN_BACKEND": "sdpa",
            "SPARSE_ATTN_BACKEND": "xformers",
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
        self._validate_runtime()
        snapshot = parameters or ReconstructionParameters()
        artifact_root = Path(tempfile.mkdtemp(
            prefix="diffusion-editor-trellis2-"
        ))
        self._artifact_roots.append(artifact_root)
        source_path = artifact_root / "source.png"
        output_path = artifact_root / "model.glb"
        events_path = artifact_root / "events.jsonl"
        log_path = artifact_root / "trellis2.log"
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
            "--trellis2-root", str(self._root),
            "--image", str(source_path),
            "--output", str(output_path),
            "--events", str(events_path),
            "--model-path", str(self._model_path),
            "--target-stage", target_stage.value,
            "--resolution", str(snapshot.resolution),
            "--steps", str(snapshot.steps),
            "--seed", str(int(seed)),
            "--decimation-target", str(snapshot.decimation_target),
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
            raise RuntimeError("TRELLIS.2 completed without producing a GLB")
        return output_path, source_path

    def _validate_runtime(self) -> None:
        if not self._python.is_file():
            raise RuntimeError(
                f"TRELLIS.2 Python not found: {self._python}. Set "
                "DIFFUSION_EDITOR_TRELLIS2_PYTHON."
            )
        if not self._runner_path.is_file():
            raise RuntimeError(
                f"TRELLIS.2 staged runner not found: {self._runner_path}"
            )
        if not self._root.is_dir():
            raise RuntimeError(
                f"TRELLIS.2 root not found: {self._root}. Set "
                "DIFFUSION_EDITOR_TRELLIS2_ROOT."
            )
        if not self._model_path.exists():
            raise RuntimeError(
                f"TRELLIS.2 model not found: {self._model_path}. Set "
                "DIFFUSION_EDITOR_TRELLIS2_MODEL."
            )
