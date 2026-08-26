"""Cached subprocess boundary for TRELLIS.2 texturing of the current mesh."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
from typing import Callable

from .model import MultiviewProject, ViewKey, view_schedule
from .trellis_generation import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TRELLIS_PYTHON,
    DEFAULT_TRELLIS_ROOT,
    TrellisShapeGenerator,
    _file_signature,
)


class TrellisTextureGenerator(TrellisShapeGenerator):
    """Re-encode the final geometry and generate a cached PBR material."""

    def __init__(
        self,
        *,
        python: Path = DEFAULT_TRELLIS_PYTHON,
        trellis_root: Path = DEFAULT_TRELLIS_ROOT,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> None:
        super().__init__(
            python=python,
            trellis_root=trellis_root,
            model_path=model_path,
        )

    def generate(
        self,
        project: MultiviewProject,
        project_path: Path,
        cancel: threading.Event,
        on_progress: Callable[[str], None] | None = None,
    ) -> Path:
        errors = project.validate_texture_request()
        if errors:
            raise ValueError("; ".join(errors))
        geometry = Path(project.geometry_path).expanduser().resolve()
        for path in (
            self._python,
            self._trellis_root,
            self._model_path,
            geometry,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        for slot in project.populated_slots():
            image = Path(slot.image_path).expanduser().resolve()
            if not image.is_file():
                raise FileNotFoundError(image)

        key = self.texture_key(project)
        output = project_path.parent / "texture-runs" / f"texture-{key[:16]}"
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "result.json"
        cached = _cached_texture(result_path, key)
        if cached is not None:
            self._progress(
                on_progress,
                f"[texture-cache] {cached.name}",
            )
            return cached

        settings = project.texture
        schedule = view_schedule(
            project.slots,
            settings.total_steps,
            settings.warmup_steps,
        )
        request = {
            "protocol": 1,
            "project": str(project_path.resolve()),
            "input_mesh": str(geometry),
            "views": [
                {
                    "id": slot.key.stable_id,
                    "image": str(Path(slot.image_path).expanduser().resolve()),
                }
                for slot in project.populated_slots()
            ],
            "schedule": [key.stable_id for key in schedule],
            "seed": settings.seed,
            "steps": settings.total_steps,
            "warmup_steps": settings.warmup_steps,
            "resolution": settings.resolution,
            "texture_size": settings.texture_size,
            "texture_key": key,
            "trellis_root": str(self._trellis_root.resolve()),
            "model_path": str(self._model_path.resolve()),
            "output_dir": str(output.resolve()),
        }
        request_path = output / "request.json"
        request_path.write_text(
            json.dumps(request, indent=2) + "\n",
            encoding="utf-8",
        )
        runner = Path(__file__).with_name("trellis_texture_runner.py")
        self._progress(on_progress, "Starting TRELLIS.2 texture worker")
        return self._run_worker(
            [str(self._python), str(runner), str(request_path)],
            result_path=result_path,
            cancel=cancel,
            on_progress=on_progress,
            operation="texturing",
        )

    def texture_key(self, project: MultiviewProject) -> str:
        settings = project.texture
        payload = {
            "geometry": _file_signature(Path(project.geometry_path)),
            "views": [
                {
                    "id": slot.key.stable_id,
                    "file": _file_signature(Path(slot.image_path)),
                }
                for slot in project.populated_slots()
            ],
            "seed": settings.seed,
            "steps": settings.total_steps,
            "warmup_steps": settings.warmup_steps,
            "resolution": settings.resolution,
            "texture_size": settings.texture_size,
            "model_path": str(self._model_path.resolve()),
            "pipeline": "dedicated-texturing-final-mesh-v1",
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def schedule_texture_images(
    project: MultiviewProject,
) -> tuple[tuple[ViewKey, Path], ...]:
    paths = {slot.key: Path(slot.image_path) for slot in project.populated_slots()}
    paths.setdefault(ViewKey("eye", 0), Path(project.front_path))
    return tuple(
        (key, paths[key])
        for key in view_schedule(
            project.slots,
            project.texture.total_steps,
            project.texture.warmup_steps,
        )
    )


def _cached_texture(result_path: Path, key: str) -> Path | None:
    if not result_path.is_file():
        return None
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        shape = Path(str(result["shape"]))
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not shape.is_file():
        shape = result_path.parent / shape.name
    if result.get("texture_key") != key or not shape.is_file():
        return None
    return shape
