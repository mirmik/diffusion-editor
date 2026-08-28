"""Subprocess boundary for shape-only TRELLIS.2 multiview generation."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable

from tcbase import log

from .model import MultiviewProject, ViewKey, view_schedule
from .trellis_mesh_postprocess import postprocess_key


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

        views = project.populated_slots()
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
                for slot in views
            ],
            "schedule": [key.stable_id for key in schedule],
            "seed": project.trellis.seed,
            "steps": project.trellis.total_steps,
            "warmup_steps": project.trellis.warmup_steps,
            "resolution": project.trellis.resolution,
            "decimation_target": project.trellis.decimation_target,
            "postprocess": _postprocess_payload(project),
            "generation_key": self._generation_key(project),
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
        return self._run_worker(
            command,
            result_path=output / "result.json",
            cancel=cancel,
            on_progress=on_progress,
        )

    def reprocess(
        self,
        project: MultiviewProject,
        cancel: threading.Event,
        on_progress: Callable[[str], None] | None = None,
    ) -> Path:
        shape_path = Path(
            project.geometry_path or project.shape_path
        ).expanduser().resolve()
        result_path = shape_path.parent / "result.json"
        if not result_path.is_file():
            raise ValueError("current shape has no reusable generation cache")
        generated = json.loads(result_path.read_text(encoding="utf-8"))
        if generated.get("generation_key") != self._generation_key(project):
            raise ValueError(
                "views or TRELLIS generation settings changed; rebuild the shape"
            )
        cache_path = _cached_mesh_path(generated, shape_path.parent)
        if not cache_path.is_file():
            raise ValueError("decoded mesh cache is missing; rebuild the shape")
        settings = _postprocess_payload(project)
        actual_resolution = int(generated["actual_resolution"])
        key = postprocess_key(settings, actual_resolution)
        request_path = shape_path.parent / f"reprocess-{key}.request.json"
        response_path = shape_path.parent / f"reprocess-{key}.result.json"
        request_path.write_text(
            json.dumps(
                {
                    "protocol": 1,
                    "cache": str(cache_path.resolve()),
                    "output_dir": str(shape_path.parent),
                    "actual_resolution": actual_resolution,
                    "settings": settings,
                    "result": str(response_path),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        runner = Path(__file__).with_name("trellis_postprocess_runner.py")
        self._progress(on_progress, "Starting cached mesh postprocess")
        return self._run_worker(
            [str(self._python), str(runner), str(request_path)],
            result_path=response_path,
            cancel=cancel,
            on_progress=on_progress,
        )

    def has_reusable_cache(self, project: MultiviewProject) -> bool:
        geometry_path = project.geometry_path or project.shape_path
        if not geometry_path:
            return False
        shape_path = Path(geometry_path).expanduser().resolve()
        result_path = shape_path.parent / "result.json"
        if not result_path.is_file():
            return False
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            return (
                result.get("generation_key") == self._generation_key(project)
                and _cached_mesh_path(result, shape_path.parent).is_file()
            )
        except (OSError, TypeError, ValueError):
            return False

    def _generation_key(self, project: MultiviewProject) -> str:
        payload = {
            "seed": project.trellis.seed,
            "steps": project.trellis.total_steps,
            "warmup_steps": project.trellis.warmup_steps,
            "resolution": project.trellis.resolution,
            "model_path": str(self._model_path.resolve()),
            "views": [
                {
                    "id": slot.key.stable_id,
                    "file": _file_signature(Path(slot.image_path)),
                }
                for slot in project.populated_slots()
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _run_worker(
        self,
        command: list[str],
        *,
        result_path: Path,
        cancel: threading.Event,
        on_progress: Callable[[str], None] | None,
        operation: str = "shape generation",
        log_name: str = "worker.log",
    ) -> Path:
        worker_log_path = result_path.parent / log_name
        worker_log_path.parent.mkdir(parents=True, exist_ok=True)
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
            with worker_log_path.open("w", encoding="utf-8") as worker_log:
                assert process.stdout is not None
                while True:
                    if cancel.is_set():
                        self._terminate(process)
                        raise RuntimeError(
                            f"TRELLIS.2 {operation} cancelled; "
                            f"worker log: {worker_log_path}"
                        )
                    line = process.stdout.readline()
                    if line:
                        worker_log.write(line)
                        worker_log.flush()
                        message = line.rstrip("\r\n")
                        if message:
                            tail.append(message)
                            tail = tail[-20:]
                            log.info(f"[TRELLIS.2 {operation}] {message}")
                            self._progress(on_progress, message.strip())
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
            message = (
                f"TRELLIS.2 worker exited with code {return_code}"
                + (f":\n{detail}" if detail else "")
                + f"\nFull worker log: {worker_log_path}"
            )
            log.error(message)
            raise RuntimeError(message)
        if not result_path.is_file():
            raise RuntimeError("TRELLIS.2 worker did not produce its result")
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
    paths = {slot.key: Path(slot.image_path) for slot in project.populated_slots()}
    paths.setdefault(ViewKey("eye", 0), Path(project.front_path))
    return tuple(
        (key, paths[key])
        for key in view_schedule(
            project.slots,
            project.trellis.total_steps,
            project.trellis.warmup_steps,
        )
    )


def _postprocess_payload(project: MultiviewProject) -> dict[str, object]:
    settings = project.trellis.postprocess
    return {
        "fill_holes": settings.fill_holes,
        "fill_hole_perimeter": settings.fill_hole_perimeter,
        "remesh": settings.remesh,
        "remesh_band": 1.0,
        "remesh_project": 0.0,
        "simplify": settings.simplify,
        "decimation_target": project.trellis.decimation_target,
        "cleanup": settings.cleanup,
        "final_repair": settings.final_repair,
        "remove_isolated_double_faces": settings.remove_isolated_double_faces,
        "remove_degenerate_faces": settings.remove_degenerate_faces,
    }


def _file_signature(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "size": stat.st_size,
        "sha256": _file_digest(
            str(resolved), stat.st_size, stat.st_mtime_ns
        ),
    }


@lru_cache(maxsize=256)
def _file_digest(path: str, _size: int, _mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_mesh_path(result: dict, run_dir: Path) -> Path:
    recorded = Path(str(result.get("shape_cache", "")))
    if recorded.is_file():
        return recorded
    return run_dir / recorded.name
