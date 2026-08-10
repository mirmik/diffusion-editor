"""Subprocess boundary for the optional local Hi3DGen runtime."""

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


DEFAULT_HI3DGEN_ROOT = Path("/home/mirmik/soft/Stable3DGen")
DEFAULT_HI3DGEN_PYTHON = DEFAULT_HI3DGEN_ROOT / "venv/bin/python"
DEFAULT_HI3DGEN_MODEL = (
    DEFAULT_HI3DGEN_ROOT / "weights/trellis-normal-v0-1"
)
DEFAULT_STABLE_NORMAL_ROOT = Path("/home/mirmik/soft/StableNormal")


class Hi3DGenProcessClient(Pixal3DProcessClient):
    """Run Hi3DGen while reusing the editor-owned event transport."""

    def __init__(
        self,
        *,
        python: str | Path | None = None,
        root: str | Path | None = None,
        model_path: str | Path | None = None,
        stable_normal_root: str | Path | None = None,
        runner_path: str | Path | None = None,
    ) -> None:
        super().__init__(
            python=python or os.environ.get(
                "DIFFUSION_EDITOR_HI3DGEN_PYTHON", DEFAULT_HI3DGEN_PYTHON
            ),
            root=root or os.environ.get(
                "DIFFUSION_EDITOR_HI3DGEN_ROOT", DEFAULT_HI3DGEN_ROOT
            ),
            model_path=model_path or os.environ.get(
                "DIFFUSION_EDITOR_HI3DGEN_MODEL", DEFAULT_HI3DGEN_MODEL
            ),
            runner_path=runner_path or Path(__file__).with_name(
                "hi3dgen_staged_runner.py"
            ),
            staged=True,
        )
        self._stable_normal_root = Path(
            stable_normal_root or os.environ.get(
                "DIFFUSION_EDITOR_STABLE_NORMAL_ROOT",
                DEFAULT_STABLE_NORMAL_ROOT,
            )
        ).expanduser()
        self._backend_label = "Hi3DGen"
        self._environment_overrides.update({
            "SPCONV_ALGO": "native",
            # Official DINOv2 is fp32. xFormers has no fp32 Blackwell kernel;
            # PyTorch SDPA does, while Hi3DGen's own sparse attention still
            # imports and uses xFormers independently.
            "XFORMERS_DISABLED": "1",
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
            ReconstructionStage.NORMAL_MAP,
            ReconstructionStage.SPARSE_OCCUPANCY,
            ReconstructionStage.HR_SHAPE_FLOW,
            ReconstructionStage.HR_SHAPE_LATENT,
            ReconstructionStage.FINAL_MESH,
        }
        if target_stage not in supported:
            raise ValueError(
                f"stage {target_stage.value} is unavailable in Hi3DGen"
            )
        self._validate_runtime()
        snapshot = parameters or ReconstructionParameters()
        artifact_root = Path(tempfile.mkdtemp(
            prefix="diffusion-editor-hi3dgen-"
        ))
        self._artifact_roots.append(artifact_root)
        source_path = artifact_root / "source.png"
        output_path = artifact_root / "model.glb"
        events_path = artifact_root / "events.jsonl"
        log_path = artifact_root / "hi3dgen.log"
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
            "--hi3dgen-root", str(self._root),
            "--stable-normal-root", str(self._stable_normal_root),
            "--image", str(source_path),
            "--output", str(output_path),
            "--events", str(events_path),
            "--model-path", str(self._model_path),
            "--target-stage", target_stage.value,
            "--sparse-steps", str(snapshot.steps),
            "--slat-steps", str(snapshot.hi3dgen_slat_steps),
            "--guidance-scale", str(snapshot.hi3dgen_guidance_scale),
            "--normal-resolution", str(snapshot.hi3dgen_normal_resolution),
            "--seed", str(int(seed)),
            "--decimation-target", str(snapshot.decimation_target),
        ]
        self._run_command(command, log_path, events_path, cancel, on_event)
        preprocessed = artifact_root / "preprocessed.png"
        if preprocessed.is_file():
            self._conditioning_path = preprocessed
        if (
            target_stage is ReconstructionStage.FINAL_MESH
            and (not output_path.is_file() or output_path.stat().st_size == 0)
        ):
            raise RuntimeError("Hi3DGen completed without producing a GLB")
        return output_path, source_path

    def _validate_runtime(self) -> None:
        if not self._python.is_file():
            raise RuntimeError(
                f"Hi3DGen Python not found: {self._python}. Set "
                "DIFFUSION_EDITOR_HI3DGEN_PYTHON."
            )
        if not self._runner_path.is_file():
            raise RuntimeError(
                f"Hi3DGen staged runner not found: {self._runner_path}"
            )
        if not self._root.is_dir():
            raise RuntimeError(
                f"Hi3DGen root not found: {self._root}. Set "
                "DIFFUSION_EDITOR_HI3DGEN_ROOT."
            )
        if not self._model_path.is_dir():
            raise RuntimeError(
                f"Hi3DGen model not found: {self._model_path}. Set "
                "DIFFUSION_EDITOR_HI3DGEN_MODEL."
            )
        if not self._stable_normal_root.is_dir():
            raise RuntimeError(
                f"StableNormal source not found: {self._stable_normal_root}. "
                "Set DIFFUSION_EDITOR_STABLE_NORMAL_ROOT."
            )
        for relative in ("weights/yoso-normal-v1-8-1", "weights/BiRefNet"):
            if not (self._root / relative).is_dir():
                raise RuntimeError(
                    f"Hi3DGen auxiliary weights not found: "
                    f"{self._root / relative}"
                )
