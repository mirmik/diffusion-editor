"""Subprocess boundary for isolated TRELLIS.2 region refinement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
from typing import Callable

import numpy as np

from .model import MultiviewProject, RefineShapeResult
from .trellis_generation import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TRELLIS_PYTHON,
    DEFAULT_TRELLIS_ROOT,
    TrellisShapeGenerator,
    _file_signature,
)


class TrellisRegionRefineGenerator(TrellisShapeGenerator):
    """Encode and RePaint one already extracted region as a local model."""

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
        region_index: int,
        project_path: Path,
        region_geometry: tuple[np.ndarray, np.ndarray, np.ndarray],
        cancel: threading.Event,
        on_progress: Callable[[str], None] | None = None,
    ) -> RefineShapeResult:
        errors = project.validate_refine_request(region_index)
        if errors:
            raise ValueError("; ".join(errors))
        for path in (self._python, self._trellis_root, self._model_path):
            if not path.exists():
                raise FileNotFoundError(path)
        vertices, faces, weights = region_geometry
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        faces = np.ascontiguousarray(faces, dtype=np.uint32)
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        if not len(vertices) or not len(faces):
            raise ValueError("selected refine region contains no triangles")
        if len(weights) != len(vertices) or not np.any(weights > 1e-6):
            raise ValueError("selected refine region has an empty vertex mask")

        key = self.refine_key(project, region_index, region_geometry)
        output = (
            project_path.parent
            / "refine-runs"
            / f"region-{region_index + 1}-{key[:16]}"
        )
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "result.json"
        cached = _cached_result(result_path, key)
        if cached is not None:
            self._progress(on_progress, f"[refine-cache] {cached.refined_region_path}")
            return cached

        source_path = output / "source-region.npz"
        np.savez_compressed(
            source_path,
            vertices=vertices,
            faces=faces,
            weights=weights,
        )
        settings = project.region_refine_settings(region_index)
        region = project.refine_regions[region_index]
        patches = project.view_patches(region_index)
        schedule = tuple(
            patches[index % len(patches)].key.stable_id
            for index in range(settings.steps)
        )
        request = {
            "protocol": 1,
            "project": str(project_path.resolve()),
            "region_index": region_index,
            "geometry_fingerprint": region.geometry_fingerprint,
            "source_region": str(source_path.resolve()),
            "cube": region.to_dict(),
            "patches": [
                {
                    "id": patch.key.stable_id,
                    "image": str(Path(project.slot(patch.key).image_path).resolve()),
                    "bounds": list(patch.bounds),
                }
                for patch in patches
            ],
            "schedule": list(schedule),
            **settings.to_dict(),
            "request_key": key,
            "trellis_root": str(self._trellis_root.resolve()),
            "model_path": str(self._model_path.resolve()),
            "output_dir": str(output.resolve()),
        }
        request_path = output / "request.json"
        request_path.write_text(
            json.dumps(request, indent=2) + "\n", encoding="utf-8"
        )
        runner = Path(__file__).with_name("trellis_refine_runner.py")
        self._progress(on_progress, "Starting isolated TRELLIS.2 region refine")
        self._run_worker(
            [str(self._python), str(runner), str(request_path)],
            result_path=result_path,
            cancel=cancel,
            on_progress=on_progress,
            operation="region refine",
        )
        result = _cached_result(result_path, key)
        if result is None:
            raise RuntimeError("TRELLIS.2 region worker produced an invalid result")
        return result

    def refine_key(
        self,
        project: MultiviewProject,
        region_index: int,
        region_geometry: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> str:
        arrays = []
        for value in region_geometry:
            array = np.ascontiguousarray(value)
            digest = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
            arrays.append(
                {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": digest}
            )
        patches = project.view_patches(region_index)
        payload = {
            "pipeline": "isolated-region-shape-slat-repaint-v1",
            "geometry_fingerprint": project.refine_regions[
                region_index
            ].geometry_fingerprint,
            "cube": project.refine_regions[region_index].to_dict(),
            "region_arrays": arrays,
            "patches": [
                {
                    "id": patch.key.stable_id,
                    "bounds": list(patch.bounds),
                    "file": _file_signature(Path(project.slot(patch.key).image_path)),
                }
                for patch in patches
            ],
            "settings": project.region_refine_settings(region_index).to_dict(),
            "model_path": str(self._model_path.resolve()),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cached_result(result_path: Path, key: str) -> RefineShapeResult | None:
    if not result_path.is_file():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("request_key") != key:
            return None
        result = RefineShapeResult(
            geometry_fingerprint=str(payload["geometry_fingerprint"]),
            request_key=key,
            source_region_path=str(Path(payload["source_region"]).resolve()),
            refined_region_path=str(Path(payload["shape"]).resolve()),
            source_slat_path=str(Path(payload["source_slat"]).resolve()),
            refined_slat_path=str(Path(payload["refined_slat"]).resolve()),
            manifest_path=str(result_path.resolve()),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    required = (
        result.source_region_path,
        result.refined_region_path,
        result.source_slat_path,
        result.refined_slat_path,
    )
    return result if all(Path(path).is_file() for path in required) else None
