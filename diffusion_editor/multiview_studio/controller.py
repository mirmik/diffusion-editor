"""Toolkit-neutral state transitions for the multiview studio."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

from .model import (
    MultiviewProject,
    RefineCube,
    ViewKey,
    geometry_fingerprint,
)


ProjectListener = Callable[[MultiviewProject, Path | None, bool], None]


class MultiviewStudioController:
    def __init__(self, project: MultiviewProject | None = None) -> None:
        self.project = project or MultiviewProject()
        self.project_path: Path | None = None
        self.dirty = False
        self._listeners: list[ProjectListener] = []

    def connect(self, listener: ProjectListener) -> None:
        self._listeners.append(listener)
        listener(self.project, self.project_path, self.dirty)

    def new_project(self) -> None:
        self.project = MultiviewProject()
        self.project_path = None
        self.dirty = False
        self._publish()

    def open_project(self, path: str | Path) -> None:
        resolved = Path(path).expanduser().resolve()
        self.project = MultiviewProject.load(resolved)
        self.project_path = resolved
        self.dirty = False
        self._publish()

    def save(self, path: str | Path | None = None) -> Path:
        target = (
            Path(path).expanduser().resolve()
            if path is not None
            else self.project_path
        )
        if target is None:
            raise ValueError("project has no manifest path")
        self.project_path = self.project.save(target)
        self.dirty = False
        self._publish()
        return self.project_path

    def set_source(self, source: str, path: str | Path) -> None:
        self._replace(self.project.with_source(source, str(Path(path).resolve())))

    def set_slot_image(self, key: ViewKey, path: str | Path) -> None:
        self._replace(
            self.project.with_slot(key, image_path=str(Path(path).resolve()))
        )

    def set_slot_images(self, images: dict[ViewKey, str | Path]) -> None:
        project = self.project
        for key, path in images.items():
            project = project.with_slot(
                key, image_path=str(Path(path).resolve())
            )
        self._replace(project)

    def clear_slot(self, key: ViewKey) -> None:
        project = self.project.with_slot(key, image_path="")
        if key == ViewKey("eye", 0):
            project = replace(project, front_path="")
        elif key == ViewKey("eye", 180):
            project = replace(project, back_path="")
        self._replace(project)

    def set_qwen_seed(self, seed: int) -> None:
        self._replace(replace(self.project, qwen_seed=int(seed)))

    def set_trellis_setting(self, field: str, value: int) -> None:
        if field not in {
            "seed",
            "total_steps",
            "warmup_steps",
            "resolution",
            "decimation_target",
        }:
            raise ValueError(f"unknown TRELLIS.2 setting: {field}")
        settings = replace(self.project.trellis, **{field: int(value)})
        self._replace(replace(self.project, trellis=settings))

    def set_mesh_postprocess(self, field: str, value: bool | float) -> None:
        boolean_fields = {
            "fill_holes",
            "remesh",
            "simplify",
            "cleanup",
            "final_repair",
            "remove_isolated_double_faces",
            "remove_degenerate_faces",
        }
        if field in boolean_fields:
            normalized: bool | float = bool(value)
        elif field == "fill_hole_perimeter":
            normalized = float(value)
        else:
            raise ValueError(f"unknown mesh postprocess setting: {field}")
        postprocess = replace(
            self.project.trellis.postprocess,
            **{field: normalized},
        )
        settings = replace(self.project.trellis, postprocess=postprocess)
        self._replace(replace(self.project, trellis=settings))

    def set_texture_setting(self, field: str, value: int) -> None:
        if field not in {
            "seed",
            "total_steps",
            "warmup_steps",
            "resolution",
            "texture_size",
        }:
            raise ValueError(f"unknown texture setting: {field}")
        settings = replace(self.project.texture, **{field: int(value)})
        self._replace(replace(self.project, texture=settings))

    def set_geometry_path(self, path: str | Path) -> None:
        resolved = str(Path(path).resolve())
        self.set_model_paths(resolved, resolved)

    def set_textured_shape_path(self, path: str | Path) -> None:
        self._replace(
            replace(self.project, shape_path=str(Path(path).resolve()))
        )

    def set_model_paths(
        self,
        geometry_path: str | Path,
        shape_path: str | Path,
    ) -> None:
        resolved_geometry = str(Path(geometry_path).resolve())
        cube = self.project.refine_cube
        if cube is not None:
            source = Path(resolved_geometry)
            if (
                not source.is_file()
                or geometry_fingerprint(source) != cube.geometry_fingerprint
            ):
                cube = None
        self._replace(replace(
            self.project,
            geometry_path=resolved_geometry,
            shape_path=str(Path(shape_path).resolve()),
            refine_cube=cube,
        ))

    def set_shape_path(self, path: str | Path) -> None:
        """Compatibility alias for callers that produce new geometry."""
        self.set_geometry_path(path)

    def set_refine_cube(
        self,
        center: tuple[float, float, float],
        side: float,
        *,
        confirmed: bool = False,
    ) -> None:
        geometry = Path(self.project.geometry_path)
        if not geometry.is_file():
            raise ValueError("a geometry model is required before selecting a cube")
        cube = RefineCube(
            center=center,
            side=side,
            geometry_fingerprint=geometry_fingerprint(geometry),
            confirmed=confirmed,
        )
        self._replace(replace(self.project, refine_cube=cube))

    def update_refine_cube(
        self,
        center: tuple[float, float, float],
        side: float,
    ) -> None:
        """Apply an interactive edit without re-hashing unchanged geometry."""
        cube = self.project.refine_cube
        if cube is None:
            raise ValueError("no refine cube is selected")
        self._replace(replace(
            self.project,
            refine_cube=replace(
                cube,
                center=tuple(map(float, center)),
                side=float(side),
                confirmed=False,
            ),
        ))

    def set_refine_cube_value(self, field: str, value: float) -> None:
        cube = self.project.refine_cube
        if cube is None:
            raise ValueError("no refine cube is selected")
        if field == "side":
            updated = replace(cube, side=float(value), confirmed=False)
        elif field in {"x", "y", "z"}:
            center = list(cube.center)
            center[{"x": 0, "y": 1, "z": 2}[field]] = float(value)
            updated = replace(cube, center=tuple(center), confirmed=False)
        else:
            raise ValueError(f"unknown refine cube field: {field}")
        self._replace(replace(self.project, refine_cube=updated))

    def confirm_refine_cube(self) -> None:
        cube = self.project.refine_cube
        if cube is None:
            raise ValueError("no refine cube is selected")
        self._replace(replace(
            self.project,
            refine_cube=replace(cube, confirmed=True),
        ))

    def replace_refine_cube(self, cube: RefineCube | None) -> None:
        self._replace(replace(self.project, refine_cube=cube))

    def clear_refine_cube(self) -> None:
        self.replace_refine_cube(None)

    def _replace(self, project: MultiviewProject) -> None:
        if project == self.project:
            return
        self.project = project
        self.dirty = True
        self._publish()

    def _publish(self) -> None:
        for listener in tuple(self._listeners):
            listener(self.project, self.project_path, self.dirty)
