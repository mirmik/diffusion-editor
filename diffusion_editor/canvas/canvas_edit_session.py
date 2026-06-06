"""Transient canvas edit session state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canvas_geometry import union_rect

Point = tuple[int, int]
Rect = tuple[int, int, int, int]


@dataclass
class CanvasEditSession:
    active: bool = False
    label: str | None = None
    target: str | None = None
    layer: Any | None = None
    last_pos: Point | None = None
    dirty_rect: Rect | None = None

    def begin(
            self,
            *,
            label: str,
            target: str,
            layer: Any | None,
            pos: Point) -> None:
        self.active = True
        self.label = label
        self.target = target
        self.layer = layer
        self.last_pos = pos
        self.dirty_rect = None

    def add_dirty(self, dirty: Rect | None) -> None:
        self.dirty_rect = union_rect(self.dirty_rect, dirty)

    def move_to(self, pos: Point) -> None:
        self.last_pos = pos

    def clear(self) -> None:
        self.active = False
        self.label = None
        self.target = None
        self.layer = None
        self.last_pos = None
        self.dirty_rect = None
