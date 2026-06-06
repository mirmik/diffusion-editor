"""Selection painting runtime settings and stroke operations."""

from __future__ import annotations

import numpy as np

from .soft_mask_stroke import SoftMaskBrush, apply_dab, apply_line

Rect = tuple[int, int, int, int]


class CanvasSelectionPainter:
    def __init__(self):
        self._size = 50
        self._hardness = 0.4
        self._flow = 1.0
        self._eraser = False

    @property
    def size(self) -> int:
        return self._size

    @property
    def hardness(self) -> float:
        return self._hardness

    @property
    def flow(self) -> float:
        return self._flow

    @property
    def eraser(self) -> bool:
        return self._eraser

    def set_brush(self, size: int, hardness: float, flow: float = 1.0) -> None:
        self._size = size
        self._hardness = hardness
        self._flow = max(0.0, min(flow, 1.0))

    def set_eraser(self, eraser: bool) -> None:
        self._eraser = eraser

    def dab(self, selection: np.ndarray, cx: int, cy: int):
        if not self._can_paint(selection):
            return None, None
        return apply_dab(
            selection,
            cx,
            cy,
            self._brush(),
            erase=self._eraser,
        )

    def line(self, selection: np.ndarray, x0: int, y0: int, x1: int, y1: int):
        if not self._can_paint(selection):
            return None, None
        return apply_line(
            selection,
            x0,
            y0,
            x1,
            y1,
            self._brush(),
            erase=self._eraser,
        )

    def _can_paint(self, selection: np.ndarray) -> bool:
        return self._size >= 1 and selection.size > 0

    def _brush(self) -> SoftMaskBrush:
        return SoftMaskBrush(
            size=self._size,
            hardness=self._hardness,
            flow=self._flow,
        )
