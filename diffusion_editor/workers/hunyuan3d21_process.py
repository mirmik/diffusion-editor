"""Subprocess boundary for the local Hunyuan3D 2.1 runtime."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
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


DEFAULT_HUNYUAN3D21_ROOT = Path("/home/mirmik/soft/ComfyUI")
DEFAULT_HUNYUAN3D21_PYTHON = DEFAULT_HUNYUAN3D21_ROOT / "venv/bin/python"
DEFAULT_HUNYUAN3D21_NODE_ROOT = (
    DEFAULT_HUNYUAN3D21_ROOT / "custom_nodes/ComfyUI-Hunyuan3d-2-1"
)
DEFAULT_HUNYUAN3D21_DIT = (
    DEFAULT_HUNYUAN3D21_ROOT
    / "models/diffusion_models/hunyuan3d-dit-v2-1.ckpt"
)
DEFAULT_HUNYUAN3D21_VAE = (
    DEFAULT_HUNYUAN3D21_ROOT / "models/vae/hunyuan3d-vae-v2-1.ckpt"
)


class Hunyuan3D21ProcessClient(Pixal3DProcessClient):
    """Run geometry and PBR paint as one editor-owned staged job."""

    def __init__(
        self,
        *,
        python: str | Path | None = None,
        root: str | Path | None = None,
        node_root: str | Path | None = None,
        dit_path: str | Path | None = None,
        vae_path: str | Path | None = None,
        runner_path: str | Path | None = None,
        gltf_transform: str | Path | None = None,
    ) -> None:
        resolved_root = Path(root or os.environ.get(
            "DIFFUSION_EDITOR_HUNYUAN3D21_ROOT", DEFAULT_HUNYUAN3D21_ROOT
        ))
        super().__init__(
            python=python or os.environ.get(
                "DIFFUSION_EDITOR_HUNYUAN3D21_PYTHON",
                DEFAULT_HUNYUAN3D21_PYTHON,
            ),
            root=resolved_root,
            model_path=dit_path or os.environ.get(
                "DIFFUSION_EDITOR_HUNYUAN3D21_DIT", DEFAULT_HUNYUAN3D21_DIT
            ),
            runner_path=runner_path or Path(__file__).with_name(
                "hunyuan3d21_staged_runner.py"
            ),
            staged=True,
        )
        self._node_root = Path(node_root or os.environ.get(
            "DIFFUSION_EDITOR_HUNYUAN3D21_NODE_ROOT",
            DEFAULT_HUNYUAN3D21_NODE_ROOT,
        )).expanduser()
        self._vae_path = Path(vae_path or os.environ.get(
            "DIFFUSION_EDITOR_HUNYUAN3D21_VAE", DEFAULT_HUNYUAN3D21_VAE
        )).expanduser()
        self._gltf_transform = str(gltf_transform or os.environ.get(
            "DIFFUSION_EDITOR_GLTF_TRANSFORM",
            shutil.which("gltf-transform") or "gltf-transform",
        ))
        self._backend_label = "Hunyuan3D 2.1"

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
            ReconstructionStage.HR_SHAPE_FLOW,
            ReconstructionStage.HR_SHAPE_LATENT,
            ReconstructionStage.TEXTURE_FLOW,
            ReconstructionStage.TEXTURE_LATENT,
            ReconstructionStage.FINAL_MESH,
        }
        if target_stage not in supported:
            raise ValueError(
                f"stage {target_stage.value} is unavailable in Hunyuan3D 2.1"
            )
        self._validate_runtime()
        snapshot = parameters or ReconstructionParameters()
        artifact_root = Path(tempfile.mkdtemp(
            prefix="diffusion-editor-hunyuan3d21-"
        ))
        self._artifact_roots.append(artifact_root)
        source_path = artifact_root / "source.png"
        output_path = artifact_root / "model.glb"
        events_path = artifact_root / "events.jsonl"
        log_path = artifact_root / "hunyuan3d21.log"
        latent_path = artifact_root / "shape-latent.pt"
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
            "--comfy-root", str(self._root),
            "--node-root", str(self._node_root),
            "--image", str(source_path),
            "--output", str(output_path),
            "--events", str(events_path),
            "--dit", str(self._model_path),
            "--vae", str(self._vae_path),
            "--latent-output", str(latent_path),
            "--target-stage", target_stage.value,
            "--shape-steps", str(snapshot.steps),
            "--shape-guidance", str(snapshot.hunyuan3d21_guidance_scale),
            "--octree-resolution", str(snapshot.hunyuan3d21_octree_resolution),
            "--texture-steps", str(snapshot.hunyuan3d21_texture_steps),
            "--texture-guidance",
            str(snapshot.hunyuan3d21_texture_guidance_scale),
            "--texture-size", str(snapshot.texture_size),
            "--seed", str(int(seed)),
            "--decimation-target", str(snapshot.decimation_target),
            "--gltf-transform", self._gltf_transform,
        ]
        self._run_command(command, log_path, events_path, cancel, on_event)
        if latent_path.is_file():
            self._checkpoint_path = latent_path
        preprocessed = artifact_root / "preprocessed.png"
        if preprocessed.is_file():
            self._conditioning_path = preprocessed
        if (
            target_stage is ReconstructionStage.FINAL_MESH
            and (not output_path.is_file() or output_path.stat().st_size == 0)
        ):
            raise RuntimeError("Hunyuan3D 2.1 completed without producing a GLB")
        return output_path, source_path

    def _validate_runtime(self) -> None:
        if not self._python.is_file():
            raise RuntimeError(
                f"Hunyuan3D 2.1 Python not found: {self._python}. Set "
                "DIFFUSION_EDITOR_HUNYUAN3D21_PYTHON."
            )
        if not self._runner_path.is_file():
            raise RuntimeError(
                f"Hunyuan3D 2.1 staged runner not found: {self._runner_path}"
            )
        if not self._root.is_dir():
            raise RuntimeError(
                f"ComfyUI root not found: {self._root}. Set "
                "DIFFUSION_EDITOR_HUNYUAN3D21_ROOT."
            )
        if not self._node_root.is_dir():
            raise RuntimeError(
                f"Hunyuan3D 2.1 node not found: {self._node_root}. Set "
                "DIFFUSION_EDITOR_HUNYUAN3D21_NODE_ROOT."
            )
        for label, path in (("shape DiT", self._model_path), ("shape VAE", self._vae_path)):
            if not path.is_file():
                raise RuntimeError(f"Hunyuan3D 2.1 {label} not found: {path}")
        executable = Path(self._gltf_transform).expanduser()
        if not executable.is_file() and shutil.which(self._gltf_transform) is None:
            raise RuntimeError(
                f"gltf-transform not found: {self._gltf_transform}; it is needed "
                "to normalize Hunyuan data-URI textures for Termin"
            )
