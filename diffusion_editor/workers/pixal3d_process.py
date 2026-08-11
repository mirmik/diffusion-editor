"""Subprocess boundary for the local Pixal3D runtime."""

from __future__ import annotations

import os
import json
import math
from pathlib import Path
from queue import Empty, Queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable

from PIL import Image

from .pixal3d_protocol import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    Pixal3DProtocolError,
    decode_response,
    encode_message,
)

from ..generation.types import (
    ReconstructionParameters,
    ReconstructionRefineParameters,
    ReconstructionStage,
    ReconstructionStageArtifact,
    ReconstructionStageEvent,
    ReconstructionStageStatus,
)


DEFAULT_PIXAL3D_ROOT = Path("/home/mirmik/soft/Pixal3D")
DEFAULT_PIXAL3D_PYTHON = Path("/home/mirmik/soft/TRELLIS.2/venv/bin/python")
DEFAULT_PIXAL3D_MODEL = Path("/home/mirmik/soft/Pixal3D-hf-check")


class Pixal3DProcessClient:
    """Runs one isolated Pixal3D CLI job and owns its temporary artifacts."""

    def __init__(
        self,
        *,
        python: str | Path | None = None,
        root: str | Path | None = None,
        model_path: str | Path | None = None,
        resolution: int = 1024,
        steps: int = 12,
        decimation_target: int = 200_000,
        texture_size: int = 2048,
        low_vram: bool = True,
        runner_path: str | Path | None = None,
        staged: bool | None = None,
        persistent: bool | None = None,
    ) -> None:
        self._python = Path(python or os.environ.get(
            "DIFFUSION_EDITOR_PIXAL3D_PYTHON", DEFAULT_PIXAL3D_PYTHON))
        self._root = Path(root or os.environ.get(
            "DIFFUSION_EDITOR_PIXAL3D_ROOT", DEFAULT_PIXAL3D_ROOT))
        self._model_path = Path(model_path or os.environ.get(
            "DIFFUSION_EDITOR_PIXAL3D_MODEL", DEFAULT_PIXAL3D_MODEL))
        self._resolution = int(resolution)
        self._steps = int(steps)
        self._decimation_target = int(decimation_target)
        self._texture_size = int(texture_size)
        self._low_vram = bool(low_vram)
        self._runner_path = Path(runner_path) if runner_path else Path(
            __file__
        ).with_name("pixal3d_staged_runner.py")
        self._staged = bool(runner_path is None) if staged is None else bool(staged)
        self._persistent = (
            runner_path is None if persistent is None else bool(persistent)
        ) and self._staged
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._worker_process: subprocess.Popen[bytes] | None = None
        self._worker_responses: Queue[bytes | None] | None = None
        self._worker_reader: threading.Thread | None = None
        self._worker_identity: tuple[str, str, str, bool] | None = None
        self._worker_operation_lock = threading.Lock()
        self._artifact_roots: list[Path] = []
        self._artifacts: list[ReconstructionStageArtifact] = []
        self._checkpoint_path: Path | None = None
        self._texture_checkpoint_path: Path | None = None
        self._resume_checkpoint_path: Path | None = None
        self._conditioning_path: Path | None = None
        self._backend_label = "Pixal3D"
        self._environment_overrides: dict[str, str] = {}

    @property
    def artifacts(self) -> tuple[ReconstructionStageArtifact, ...]:
        return tuple(self._artifacts)

    @property
    def checkpoint_path(self) -> Path | None:
        return self._checkpoint_path

    @property
    def conditioning_path(self) -> Path | None:
        return self._conditioning_path

    @property
    def texture_checkpoint_path(self) -> Path | None:
        return self._texture_checkpoint_path

    @property
    def resume_checkpoint_path(self) -> Path | None:
        return self._resume_checkpoint_path

    def generate(
        self,
        image: Image.Image,
        seed: int,
        cancel: threading.Event,
        *,
        parameters: ReconstructionParameters | None = None,
        target_stage: ReconstructionStage = ReconstructionStage.FINAL_MESH,
        resume_checkpoint_path: str | Path | None = None,
        on_event: Callable[[ReconstructionStageEvent], None] | None = None,
    ) -> tuple[Path, Path]:
        self._validate_runtime()
        artifact_root = Path(tempfile.mkdtemp(prefix="diffusion-editor-pixal3d-"))
        self._artifact_roots.append(artifact_root)
        source_path = artifact_root / "source.png"
        output_path = artifact_root / "model.glb"
        log_path = artifact_root / "pixal3d.log"
        events_path = artifact_root / "events.jsonl"
        checkpoint_path = artifact_root / "shape-checkpoint.npz"
        texture_checkpoint_path = artifact_root / "texture-checkpoint.npz"
        next_resume_checkpoint = artifact_root / "resume-checkpoint.npz"
        self._checkpoint_path = None
        self._texture_checkpoint_path = None
        self._resume_checkpoint_path = None
        self._conditioning_path = None
        image.convert("RGBA").save(source_path, format="PNG")
        self._artifacts = [ReconstructionStageArtifact(
            ReconstructionStage.SOURCE_IMAGE,
            str(source_path),
            "image",
        )]
        if on_event is not None:
            on_event(ReconstructionStageEvent(
                ReconstructionStage.SOURCE_IMAGE,
                ReconstructionStageStatus.READY,
                artifact=self._artifacts[0],
            ))
        if target_stage is ReconstructionStage.SOURCE_IMAGE:
            return output_path, source_path

        resolution = parameters.resolution if parameters else self._resolution
        lr_conditioning_resolution = (
            parameters.lr_conditioning_resolution if parameters else 512
        )
        steps = parameters.steps if parameters else self._steps
        decimation_target = (
            parameters.decimation_target if parameters else self._decimation_target
        )
        texture_size = parameters.texture_size if parameters else self._texture_size
        low_vram = parameters.low_vram if parameters else self._low_vram
        manual_fov = (
            math.radians(parameters.manual_fov_degrees)
            if parameters and parameters.manual_fov_degrees > 0.0
            else -1.0
        )
        command = [
            str(self._python),
            str(self._runner_path if self._staged else self._root / "inference.py"),
            "--image", str(source_path),
            "--output", str(output_path),
            "--model_path", str(self._model_path),
            "--resolution", str(resolution),
            "--steps", str(steps),
            "--seed", str(int(seed)),
            "--decimation-target", str(decimation_target),
            "--texture-size", str(texture_size),
        ]
        if self._staged:
            command.extend((
                "--pixal3d-root", str(self._root),
                "--events", str(events_path),
                "--target-stage", target_stage.value,
                "--manual-fov", str(manual_fov),
                "--lr-conditioning-resolution",
                str(lr_conditioning_resolution),
                "--checkpoint", str(checkpoint_path),
                "--texture-checkpoint", str(texture_checkpoint_path),
                "--session-checkpoint", str(next_resume_checkpoint),
                "--sparse-seed", str(
                    parameters.pixal3d_seed_for("sparse")
                    if parameters else seed
                ),
                "--sparse-steps", str(
                    parameters.pixal3d_steps_for("sparse")
                    if parameters else steps
                ),
                "--lr-seed", str(
                    parameters.pixal3d_seed_for("lr")
                    if parameters else seed
                ),
                "--lr-steps", str(
                    parameters.pixal3d_steps_for("lr")
                    if parameters else steps
                ),
                "--hr-seed", str(
                    parameters.pixal3d_seed_for("hr")
                    if parameters else seed
                ),
                "--hr-steps", str(
                    parameters.pixal3d_steps_for("hr")
                    if parameters else steps
                ),
                "--texture-seed", str(
                    parameters.pixal3d_seed_for("texture")
                    if parameters else seed
                ),
                "--texture-steps", str(
                    parameters.pixal3d_steps_for("texture")
                    if parameters else steps
                ),
            ))
            if resume_checkpoint_path is not None:
                resume_checkpoint = Path(resume_checkpoint_path)
                if not resume_checkpoint.is_file():
                    raise RuntimeError(
                        f"Pixal3D resume checkpoint not found: {resume_checkpoint}"
                    )
                command.extend(("--resume-checkpoint", str(resume_checkpoint)))
        elif manual_fov > 0.0:
            command.extend(("--fov", str(manual_fov)))
        if low_vram:
            command.append("--low_vram")

        if self._persistent:
            self._run_persistent_command(
                command, events_path, cancel, on_event, low_vram=low_vram
            )
        else:
            self._run_command(command, log_path, events_path, cancel, on_event)
        if checkpoint_path.is_file():
            self._checkpoint_path = checkpoint_path
        if texture_checkpoint_path.is_file():
            self._texture_checkpoint_path = texture_checkpoint_path
        if next_resume_checkpoint.is_file():
            self._resume_checkpoint_path = next_resume_checkpoint
        preprocessed = artifact_root / "preprocessed.png"
        if preprocessed.is_file():
            self._conditioning_path = preprocessed

        if (
            target_stage is ReconstructionStage.FINAL_MESH
            and (not output_path.is_file() or output_path.stat().st_size == 0)
        ):
            raise RuntimeError("Pixal3D completed without producing a GLB")
        return output_path, source_path

    def refine_lr(
        self,
        conditioning_image: Image.Image,
        mask_image: Image.Image,
        session_checkpoint_path: str | Path,
        cancel: threading.Event,
        *,
        parameters: ReconstructionRefineParameters | None = None,
        generation_parameters: ReconstructionParameters | None = None,
        on_event: Callable[[ReconstructionStageEvent], None] | None = None,
    ) -> tuple[Path, Path]:
        """Masked-refine an LR latent and return a resumable LR checkpoint."""
        self._validate_runtime()
        if not self._staged:
            raise RuntimeError("LR refinement requires the staged runner")
        base_checkpoint = Path(session_checkpoint_path)
        if not base_checkpoint.is_file():
            raise RuntimeError(
                f"LR session checkpoint not found: {base_checkpoint}"
            )
        if conditioning_image.size != mask_image.size:
            raise ValueError("refine mask must match conditioning image dimensions")

        snapshot = parameters or ReconstructionRefineParameters()
        generation = generation_parameters or ReconstructionParameters()
        artifact_root = Path(tempfile.mkdtemp(
            prefix="diffusion-editor-pixal3d-lr-refine-"
        ))
        self._artifact_roots.append(artifact_root)
        source_path = artifact_root / "conditioning.png"
        mask_path = artifact_root / "mask.png"
        output_path = artifact_root / "lr-shape-refined.glb"
        next_session_checkpoint = artifact_root / "resume-checkpoint.npz"
        events_path = artifact_root / "events.jsonl"
        log_path = artifact_root / "pixal3d.log"
        conditioning_image.convert("RGBA").save(source_path, format="PNG")
        mask_image.convert("L").save(mask_path, format="PNG")
        self._artifacts = [ReconstructionStageArtifact(
            ReconstructionStage.SOURCE_IMAGE, str(source_path), "image"
        )]
        self._checkpoint_path = None
        self._texture_checkpoint_path = None
        self._resume_checkpoint_path = None
        self._conditioning_path = None
        command = [
            str(self._python), str(self._runner_path),
            "--pixal3d-root", str(self._root),
            "--image", str(source_path),
            "--output", str(output_path),
            "--events", str(events_path),
            "--model_path", str(self._model_path),
            "--lr-refine-checkpoint", str(base_checkpoint),
            "--session-checkpoint", str(next_session_checkpoint),
            "--refine-mask", str(mask_path),
            "--refine-strength", str(snapshot.strength),
            "--refine-steps", str(snapshot.steps),
            "--refine-seed", str(snapshot.seed),
            "--refine-rescale-t", str(snapshot.rescale_t),
            "--refine-guidance", str(snapshot.guidance_strength),
            "--lr-conditioning-resolution",
            str(generation.lr_conditioning_resolution),
            "--resolution", str(generation.resolution),
        ]
        if generation.low_vram:
            command.append("--low_vram")
        if snapshot.resize_detail_to_1024:
            command.append("--resize-refine-detail")
        if self._persistent:
            self._run_persistent_command(
                command, events_path, cancel, on_event,
                low_vram=generation.low_vram,
            )
        else:
            self._run_command(command, log_path, events_path, cancel, on_event)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("Pixal3D LR refinement produced no preview")
        if not next_session_checkpoint.is_file():
            raise RuntimeError("Pixal3D LR refinement produced no resume checkpoint")
        self._resume_checkpoint_path = next_session_checkpoint
        preprocessed = artifact_root / "preprocessed.png"
        if preprocessed.is_file():
            self._conditioning_path = preprocessed
        return output_path, source_path

    def refine(
        self,
        conditioning_image: Image.Image,
        mask_image: Image.Image,
        base_checkpoint_path: str | Path,
        cancel: threading.Event,
        *,
        parameters: ReconstructionRefineParameters | None = None,
        generation_parameters: ReconstructionParameters | None = None,
        low_vram: bool | None = None,
        on_event: Callable[[ReconstructionStageEvent], None] | None = None,
    ) -> tuple[Path, Path]:
        """Refine HR shape, then regenerate and bake its texture."""
        self._validate_runtime()
        if not self._staged:
            raise RuntimeError("masked refinement requires the staged runner")
        base_checkpoint = Path(base_checkpoint_path)
        if not base_checkpoint.is_file():
            raise RuntimeError(f"refine checkpoint not found: {base_checkpoint}")
        if conditioning_image.size != mask_image.size:
            raise ValueError("refine mask must match conditioning image dimensions")

        snapshot = parameters or ReconstructionRefineParameters()
        generation = generation_parameters or ReconstructionParameters()
        artifact_root = Path(tempfile.mkdtemp(
            prefix="diffusion-editor-pixal3d-refine-"
        ))
        self._artifact_roots.append(artifact_root)
        condition_path = artifact_root / "conditioning.png"
        mask_path = artifact_root / "mask.png"
        output_path = artifact_root / "refined.glb"
        checkpoint_path = artifact_root / "shape-checkpoint.npz"
        texture_checkpoint_path = artifact_root / "texture-checkpoint.npz"
        events_path = artifact_root / "events.jsonl"
        log_path = artifact_root / "pixal3d.log"
        conditioning_image.convert("RGBA").save(condition_path, format="PNG")
        mask_image.convert("L").save(mask_path, format="PNG")
        self._artifacts = [ReconstructionStageArtifact(
            ReconstructionStage.SOURCE_IMAGE, str(condition_path), "image"
        )]
        self._checkpoint_path = None
        self._texture_checkpoint_path = None
        self._conditioning_path = None
        command = [
            str(self._python), str(self._runner_path),
            "--pixal3d-root", str(self._root),
            "--image", str(condition_path),
            "--output", str(output_path),
            "--events", str(events_path),
            "--model_path", str(self._model_path),
            "--checkpoint", str(checkpoint_path),
            "--texture-checkpoint", str(texture_checkpoint_path),
            "--refine-checkpoint", str(base_checkpoint),
            "--refine-mask", str(mask_path),
            "--refine-strength", str(snapshot.strength),
            "--refine-steps", str(snapshot.steps),
            "--refine-seed", str(snapshot.seed),
            "--refine-rescale-t", str(snapshot.rescale_t),
            "--refine-guidance", str(snapshot.guidance_strength),
            "--steps", str(generation.steps),
            "--seed", str(generation.seed),
            "--decimation-target", str(generation.decimation_target),
            "--texture-size", str(generation.texture_size),
        ]
        effective_low_vram = (
            generation.low_vram if low_vram is None else bool(low_vram)
        )
        if effective_low_vram:
            command.append("--low_vram")
        if snapshot.resize_detail_to_1024:
            command.append("--resize-refine-detail")
        if self._persistent:
            self._run_persistent_command(
                command, events_path, cancel, on_event,
                low_vram=effective_low_vram,
            )
        else:
            self._run_command(command, log_path, events_path, cancel, on_event)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("Pixal3D refinement completed without a GLB")
        if not checkpoint_path.is_file():
            raise RuntimeError("Pixal3D refinement completed without a checkpoint")
        self._checkpoint_path = checkpoint_path
        if texture_checkpoint_path.is_file():
            self._texture_checkpoint_path = texture_checkpoint_path
        preprocessed = artifact_root / "preprocessed.png"
        if preprocessed.is_file():
            self._conditioning_path = preprocessed
        return output_path, condition_path

    def refine_texture(
        self,
        conditioning_image: Image.Image,
        mask_image: Image.Image,
        shape_checkpoint_path: str | Path,
        texture_checkpoint_path: str | Path,
        cancel: threading.Event,
        *,
        parameters: ReconstructionRefineParameters | None = None,
        generation_parameters: ReconstructionParameters | None = None,
        on_event: Callable[[ReconstructionStageEvent], None] | None = None,
    ) -> tuple[Path, Path]:
        """Refine only the texture latent while retaining the selected shape."""
        self._validate_runtime()
        if not self._staged:
            raise RuntimeError("masked texture refinement requires the staged runner")
        shape_checkpoint = Path(shape_checkpoint_path)
        texture_checkpoint = Path(texture_checkpoint_path)
        for label, path in (
            ("shape", shape_checkpoint), ("texture", texture_checkpoint)
        ):
            if not path.is_file():
                raise RuntimeError(f"{label} refine checkpoint not found: {path}")
        if conditioning_image.size != mask_image.size:
            raise ValueError("refine mask must match conditioning image dimensions")

        snapshot = parameters or ReconstructionRefineParameters()
        generation = generation_parameters or ReconstructionParameters()
        artifact_root = Path(tempfile.mkdtemp(
            prefix="diffusion-editor-pixal3d-texture-refine-"
        ))
        self._artifact_roots.append(artifact_root)
        source_path = artifact_root / "conditioning.png"
        mask_path = artifact_root / "mask.png"
        output_path = artifact_root / "texture-refined.glb"
        next_texture_checkpoint = artifact_root / "texture-checkpoint.npz"
        events_path = artifact_root / "events.jsonl"
        log_path = artifact_root / "pixal3d.log"
        conditioning_image.convert("RGBA").save(source_path, format="PNG")
        mask_image.convert("L").save(mask_path, format="PNG")
        self._artifacts = [ReconstructionStageArtifact(
            ReconstructionStage.SOURCE_IMAGE, str(source_path), "image"
        )]
        self._checkpoint_path = shape_checkpoint
        self._texture_checkpoint_path = None
        self._conditioning_path = None
        command = [
            str(self._python), str(self._runner_path),
            "--pixal3d-root", str(self._root),
            "--image", str(source_path),
            "--output", str(output_path),
            "--events", str(events_path),
            "--model_path", str(self._model_path),
            "--refine-checkpoint", str(shape_checkpoint),
            "--texture-refine-checkpoint", str(texture_checkpoint),
            "--texture-checkpoint", str(next_texture_checkpoint),
            "--refine-mask", str(mask_path),
            "--refine-strength", str(snapshot.strength),
            "--refine-steps", str(snapshot.steps),
            "--refine-seed", str(snapshot.seed),
            "--refine-rescale-t", str(snapshot.rescale_t),
            "--steps", str(generation.steps),
            "--seed", str(generation.seed),
            "--decimation-target", str(generation.decimation_target),
            "--texture-size", str(generation.texture_size),
        ]
        if generation.low_vram:
            command.append("--low_vram")
        if snapshot.resize_detail_to_1024:
            command.append("--resize-refine-detail")
        if self._persistent:
            self._run_persistent_command(
                command, events_path, cancel, on_event,
                low_vram=generation.low_vram,
            )
        else:
            self._run_command(command, log_path, events_path, cancel, on_event)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("Pixal3D texture refinement completed without a GLB")
        if not next_texture_checkpoint.is_file():
            raise RuntimeError(
                "Pixal3D texture refinement completed without a checkpoint"
            )
        self._texture_checkpoint_path = next_texture_checkpoint
        preprocessed = artifact_root / "preprocessed.png"
        if preprocessed.is_file():
            self._conditioning_path = preprocessed
        return output_path, source_path

    def _worker_environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(self._root)
            if not existing_pythonpath
            else os.pathsep.join((str(self._root), existing_pythonpath))
        )
        environment.setdefault("ATTN_BACKEND", "sdpa")
        environment.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
        environment.update(self._environment_overrides)
        return environment

    def _run_persistent_command(
        self,
        command: list[str],
        events_path: Path,
        cancel: threading.Event,
        on_event: Callable[[ReconstructionStageEvent], None] | None,
        *,
        low_vram: bool,
    ) -> None:
        with self._worker_operation_lock:
            process, responses = self._ensure_worker(low_vram)
            request_id = uuid.uuid4().hex
            request = encode_message({
                "protocol": PROTOCOL_VERSION,
                "type": "run",
                "request_id": request_id,
                "arguments": command[2:],
            })
            try:
                assert process.stdin is not None
                process.stdin.write(request)
                process.stdin.flush()
                event_offset = 0
                deadline = time.monotonic() + 1800.0
                while True:
                    event_offset = self._read_stage_events(
                        events_path, event_offset, on_event
                    )
                    if cancel.is_set():
                        raise RuntimeError(
                            f"{self._backend_label} generation cancelled"
                        )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Pixal3D worker request timed out")
                    try:
                        message = responses.get(timeout=min(0.05, remaining))
                    except Empty:
                        if process.poll() is not None:
                            raise RuntimeError(
                                "Pixal3D worker exited with code "
                                f"{process.returncode}"
                            )
                        continue
                    if message is None:
                        raise RuntimeError(
                            "Pixal3D worker exited with code "
                            f"{process.poll()}"
                        )
                    response = decode_response(message)
                    if response.request_id != request_id:
                        raise Pixal3DProtocolError(
                            "Pixal3D response request ID mismatch"
                        )
                    if response.kind == "error":
                        raise RuntimeError(
                            response.error or "Pixal3D worker failed"
                        )
                    if response.kind != "result":
                        raise Pixal3DProtocolError(
                            "Pixal3D worker returned a non-result response"
                        )
                    break
                self._read_stage_events(events_path, event_offset, on_event)
            except Exception:
                self._stop_worker(timeout=0.2)
                raise

    def _ensure_worker(
        self, low_vram: bool
    ) -> tuple[subprocess.Popen[bytes], Queue[bytes | None]]:
        identity = (
            str(self._python), str(self._root.resolve()),
            str(self._model_path.resolve()), bool(low_vram),
        )
        with self._lock:
            process = self._worker_process
            if (
                process is not None
                and process.poll() is None
                and self._worker_identity == identity
            ):
                assert self._worker_responses is not None
                return process, self._worker_responses
        self._stop_worker(timeout=0.5)
        worker_command = [
            str(self._python), "-u", str(self._runner_path), "--server",
            "--pixal3d-root", str(self._root),
            "--model_path", str(self._model_path),
        ]
        if low_vram:
            worker_command.append("--low_vram")
        process = subprocess.Popen(
            worker_command,
            cwd=self._root,
            env=self._worker_environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            start_new_session=(os.name != "nt"),
        )
        responses: Queue[bytes | None] = Queue(maxsize=8)
        reader = threading.Thread(
            target=self._read_worker_responses,
            args=(process, responses),
            name="pixal3d-worker-reader",
            daemon=True,
        )
        with self._lock:
            self._worker_process = process
            self._worker_responses = responses
            self._worker_reader = reader
            self._worker_identity = identity
        reader.start()
        try:
            deadline = time.monotonic() + 300.0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Pixal3D worker startup timed out")
                try:
                    message = responses.get(timeout=min(0.05, remaining))
                except Empty:
                    if process.poll() is not None:
                        raise RuntimeError(
                            "Pixal3D worker exited during startup with code "
                            f"{process.returncode}"
                        )
                    continue
                if message is None:
                    raise RuntimeError(
                        "Pixal3D worker exited during startup with code "
                        f"{process.poll()}"
                    )
                response = decode_response(message)
                if response.kind != "ready":
                    raise Pixal3DProtocolError(
                        "Pixal3D worker did not send ready"
                    )
                break
        except Exception:
            self._stop_worker(timeout=0.2)
            raise
        return process, responses

    @staticmethod
    def _read_worker_responses(
        process: subprocess.Popen[bytes], responses: Queue[bytes | None]
    ) -> None:
        assert process.stdout is not None
        try:
            while message := process.stdout.readline(MAX_MESSAGE_BYTES + 1):
                responses.put(message)
        finally:
            responses.put(None)

    def _stop_worker(self, timeout: float = 1.0) -> None:
        with self._lock:
            process = self._worker_process
            self._worker_process = None
            self._worker_responses = None
            self._worker_reader = None
            self._worker_identity = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._terminate(process, timeout=max(timeout, 0.5))

    def _run_command(
        self,
        command: list[str],
        log_path: Path,
        events_path: Path,
        cancel: threading.Event,
        on_event: Callable[[ReconstructionStageEvent], None] | None,
    ) -> None:
        self._stop_worker(timeout=0.5)
        environment = self._worker_environment()
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(
                command,
                cwd=self._root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name != "nt"),
            )
            with self._lock:
                self._process = process
            try:
                event_offset = 0
                while process.poll() is None:
                    event_offset = self._read_stage_events(
                        events_path, event_offset, on_event
                    )
                    if cancel.wait(0.1):
                        self._terminate(process)
                        raise RuntimeError(
                            f"{self._backend_label} generation cancelled"
                        )
                if process.returncode != 0:
                    raise RuntimeError(
                        self._failure_message(log_path, process.returncode)
                    )
            finally:
                with self._lock:
                    if self._process is process:
                        self._process = None
        self._read_stage_events(events_path, event_offset, on_event)

    def shutdown(self, timeout: float = 2.0) -> None:
        self._stop_worker(timeout=timeout)
        with self._lock:
            process = self._process
        if process is not None and process.poll() is None:
            self._terminate(process, timeout=timeout)
        for root in self._artifact_roots:
            shutil.rmtree(root, ignore_errors=True)
        self._artifact_roots.clear()

    def _validate_runtime(self) -> None:
        if not self._python.is_file():
            raise RuntimeError(
                f"Pixal3D Python not found: {self._python}. Set "
                "DIFFUSION_EDITOR_PIXAL3D_PYTHON."
            )
        script = self._runner_path if self._staged else self._root / "inference.py"
        if not script.is_file():
            raise RuntimeError(
                f"Pixal3D inference entry point not found: {script}. Set "
                "DIFFUSION_EDITOR_PIXAL3D_ROOT."
            )
        if not self._model_path.exists():
            raise RuntimeError(
                f"Pixal3D model not found: {self._model_path}. Set "
                "DIFFUSION_EDITOR_PIXAL3D_MODEL."
            )

    def _read_stage_events(
        self,
        path: Path,
        offset: int,
        callback: Callable[[ReconstructionStageEvent], None] | None,
    ) -> int:
        if not path.is_file():
            return offset
        with path.open("r", encoding="utf-8") as stream:
            stream.seek(offset)
            for line in stream:
                try:
                    payload = json.loads(line)
                    artifact = None
                    artifact_path = payload.get("artifact_path")
                    if artifact_path:
                        artifact = ReconstructionStageArtifact(
                            ReconstructionStage(payload["stage"]),
                            str(artifact_path),
                            payload.get("preview_kind", "latent"),
                        )
                        self._artifacts = [
                            item for item in self._artifacts
                            if item.stage is not artifact.stage
                        ]
                        self._artifacts.append(artifact)
                    event = ReconstructionStageEvent(
                        ReconstructionStage(payload["stage"]),
                        ReconstructionStageStatus(payload["status"]),
                        int(payload.get("progress", 0)),
                        int(payload.get("total", 0)),
                        artifact,
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                if callback is not None:
                    callback(event)
            return stream.tell()
    @staticmethod
    def _terminate(process: subprocess.Popen[bytes], timeout: float = 2.0) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=timeout)

    def _failure_message(self, log_path: Path, return_code: int) -> str:
        try:
            lines = log_path.read_text(errors="replace").splitlines()
        except OSError:
            lines = []
        tail = "\n".join(lines[-20:]).strip()
        base = f"{self._backend_label} exited with code {return_code}"
        return f"{base}:\n{tail}" if tail else base
