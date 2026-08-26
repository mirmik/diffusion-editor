"""Subprocess boundary for shape-only TRELLIS.2 multiview generation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable

from .model import MultiviewProject, ViewKey, view_schedule


DEFAULT_TRELLIS_PYTHON = Path("/home/mirmik/soft/TRELLIS.2/venv/bin/python")
DEFAULT_TRELLIS_ROOT = Path("/home/mirmik/soft/TRELLIS.2")
DEFAULT_MODEL_PATH = DEFAULT_TRELLIS_ROOT / "models/TRELLIS.2-4B"


class TrellisShapeGenerator:
    def __init__(
        self,
        *,
        python: Path = DEFAULT_TRELLIS_PYTHON,
        trellis_root: Path = DEFAULT_TRELLIS_ROOT,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self._python = Path(python)
        self._trellis_root = Path(trellis_root)
        self._model_path = Path(model_path)
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def generate(
        self,
        project: MultiviewProject,
        project_path: Path,
        cancel: threading.Event,
        on_progress: Callable[[str], None] | None = None,
    ) -> Path:
        errors = project.validate_shape_request()
        if errors:
            raise ValueError("; ".join(errors))
        for path in (self._python, self._trellis_root, self._model_path):
            if not path.exists():
                raise FileNotFoundError(path)

        output_root = project_path.parent / "shape-runs"
        output_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output = output_root / f"shape-{stamp}-seed{project.trellis.seed}"
        suffix = 2
        while output.exists():
            output = output_root / (
                f"shape-{stamp}-seed{project.trellis.seed}-{suffix}"
            )
            suffix += 1
        output.mkdir(parents=True)

        included = project.included_slots()
        schedule = view_schedule(
            project.slots,
            project.trellis.total_steps,
            project.trellis.warmup_steps,
        )
        request = {
            "protocol": 1,
            "project": str(project_path.resolve()),
            "front": str(Path(project.front_path).resolve()),
            "views": [
                {
                    "id": slot.key.stable_id,
                    "image": str(Path(slot.image_path).resolve()),
                }
                for slot in included
            ],
            "schedule": [key.stable_id for key in schedule],
            "seed": project.trellis.seed,
            "steps": project.trellis.total_steps,
            "warmup_steps": project.trellis.warmup_steps,
            "resolution": project.trellis.resolution,
            "decimation_target": project.trellis.decimation_target,
            "trellis_root": str(self._trellis_root.resolve()),
            "model_path": str(self._model_path.resolve()),
            "output_dir": str(output.resolve()),
        }
        request_path = output / "request.json"
        request_path.write_text(
            json.dumps(request, indent=2) + "\n", encoding="utf-8"
        )
        runner = Path(__file__).with_name("trellis_shape_runner.py")
        command = [str(self._python), str(runner), str(request_path)]
        self._progress(on_progress, "Starting TRELLIS.2 shape worker")
        process = subprocess.Popen(
            command,
            cwd=str(self._trellis_root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        with self._lock:
            self._process = process
        tail: list[str] = []
        try:
            assert process.stdout is not None
            while True:
                if cancel.is_set():
                    self._terminate(process)
                    raise RuntimeError("TRELLIS.2 shape generation cancelled")
                line = process.stdout.readline()
                if line:
                    message = line.strip()
                    if message:
                        tail.append(message)
                        tail = tail[-20:]
                        self._progress(on_progress, message)
                    continue
                if process.poll() is not None:
                    break
                cancel.wait(0.05)
            return_code = process.wait()
        finally:
            with self._lock:
                if self._process is process:
                    self._process = None
        if return_code:
            detail = "\n".join(tail[-8:])
            raise RuntimeError(
                f"TRELLIS.2 worker exited with code {return_code}"
                + (f":\n{detail}" if detail else "")
            )
        result_path = output / "result.json"
        if not result_path.is_file():
            raise RuntimeError("TRELLIS.2 worker did not produce result.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        shape_path = Path(result["shape"])
        if not shape_path.is_file():
            raise RuntimeError(f"TRELLIS.2 shape is missing: {shape_path}")
        return shape_path

    def cancel(self) -> None:
        with self._lock:
            process = self._process
        if process is not None:
            self._terminate(process)

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.0)

    @staticmethod
    def _progress(callback: Callable[[str], None] | None, message: str) -> None:
        if callback is not None and message:
            callback(message)


def schedule_images(project: MultiviewProject) -> tuple[tuple[ViewKey, Path], ...]:
    """Public, lightweight description useful to tests and diagnostics."""
    paths = {slot.key: Path(slot.image_path) for slot in project.included_slots()}
    paths.setdefault(ViewKey("eye", 0), Path(project.front_path))
    return tuple(
        (key, paths[key])
        for key in view_schedule(
            project.slots,
            project.trellis.total_steps,
            project.trellis.warmup_steps,
        )
    )
