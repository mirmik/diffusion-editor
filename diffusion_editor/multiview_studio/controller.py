"""Toolkit-neutral state transitions for the multiview studio."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

from .model import MultiviewProject, ViewKey


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

    def set_slot_included(self, key: ViewKey, included: bool) -> None:
        self._replace(
            self.project.with_slot(key, include_in_trellis=included)
        )

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

    def set_shape_path(self, path: str | Path) -> None:
        self._replace(
            replace(self.project, shape_path=str(Path(path).resolve()))
        )

    def _replace(self, project: MultiviewProject) -> None:
        if project == self.project:
            return
        self.project = project
        self.dirty = True
        self._publish()

    def _publish(self) -> None:
        for listener in tuple(self._listeners):
            listener(self.project, self.project_path, self.dirty)
