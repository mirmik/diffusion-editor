"""Subprocess boundary for the local Pixal3D runtime."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading

from PIL import Image


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
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._artifact_roots: list[Path] = []

    def generate(
        self,
        image: Image.Image,
        seed: int,
        cancel: threading.Event,
    ) -> tuple[Path, Path]:
        self._validate_runtime()
        artifact_root = Path(tempfile.mkdtemp(prefix="diffusion-editor-pixal3d-"))
        self._artifact_roots.append(artifact_root)
        source_path = artifact_root / "source.png"
        output_path = artifact_root / "model.glb"
        log_path = artifact_root / "pixal3d.log"
        image.convert("RGBA").save(source_path, format="PNG")

        command = [
            str(self._python),
            str(self._root / "inference.py"),
            "--image", str(source_path),
            "--output", str(output_path),
            "--model_path", str(self._model_path),
            "--resolution", str(self._resolution),
            "--steps", str(self._steps),
            "--seed", str(int(seed)),
            "--decimation-target", str(self._decimation_target),
            "--texture-size", str(self._texture_size),
        ]
        if self._low_vram:
            command.append("--low_vram")

        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(self._root)
            if not existing_pythonpath
            else os.pathsep.join((str(self._root), existing_pythonpath))
        )
        environment.setdefault("ATTN_BACKEND", "sdpa")
        environment.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

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
                while process.poll() is None:
                    if cancel.wait(0.1):
                        self._terminate(process)
                        raise RuntimeError("Pixal3D generation cancelled")
                if process.returncode != 0:
                    raise RuntimeError(self._failure_message(log_path, process.returncode))
            finally:
                with self._lock:
                    if self._process is process:
                        self._process = None

        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("Pixal3D completed without producing a GLB")
        return output_path, source_path

    def shutdown(self, timeout: float = 2.0) -> None:
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
        script = self._root / "inference.py"
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

    @staticmethod
    def _failure_message(log_path: Path, return_code: int) -> str:
        try:
            lines = log_path.read_text(errors="replace").splitlines()
        except OSError:
            lines = []
        tail = "\n".join(lines[-20:]).strip()
        base = f"Pixal3D exited with code {return_code}"
        return f"{base}:\n{tail}" if tail else base
