"""Smudge stroke runtime buffer and pixel operations."""

from __future__ import annotations

import numpy as np

from .canvas_geometry import union_rect

Rect = tuple[int, int, int, int]
BrushSlices = tuple[int, int, int, int, int, int, int, int]


def brush_rect_slices(
        image_shape: tuple[int, int],
        stamp_shape: tuple[int, int],
        cx: int,
        cy: int) -> BrushSlices | None:
    sh, sw = stamp_shape
    ih, iw = image_shape

    x0 = cx - sw // 2
    y0 = cy - sh // 2
    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    sx1 = min(sw, iw - x0)
    sy1 = min(sh, ih - y0)
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)
    if dx0 >= dx1 or dy0 >= dy1:
        return None
    return sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1


class SmudgeStrokeBuffer:
    def __init__(self):
        self._buffer: np.ndarray | None = None

    @property
    def buffer(self) -> np.ndarray | None:
        return self._buffer

    def begin(self, layer_image: np.ndarray, brush, x: int, y: int) -> None:
        stamp = brush._alpha_stamp
        sh, sw = stamp.shape[:2]
        self._buffer = np.zeros((sh, sw, 4), dtype=np.float32)
        slices = brush_rect_slices(layer_image.shape[:2], stamp.shape[:2], x, y)
        if slices is None:
            return
        sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1 = slices
        self._buffer[sy0:sy1, sx0:sx1] = (
            layer_image[dy0:dy1, dx0:dx1].astype(np.float32))

    def dab(self, layer_image: np.ndarray, brush, cx: int, cy: int) -> Rect | None:
        if self._buffer is None:
            self.begin(layer_image, brush, cx, cy)
        if self._buffer is None:
            return None

        stamp = brush._alpha_stamp
        slices = brush_rect_slices(layer_image.shape[:2], stamp.shape[:2], cx, cy)
        if slices is None:
            return None
        sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1 = slices

        amount = (
            stamp[sy0:sy1, sx0:sx1, None]
            * brush.flow
            * (brush.color[3] / 255.0)
        ).astype(np.float32)
        if not np.any(amount > 0.0):
            return None

        carry = self._buffer[sy0:sy1, sx0:sx1]
        dst = layer_image[dy0:dy1, dx0:dx1].astype(np.float32).copy()
        layer_image[dy0:dy1, dx0:dx1] = np.clip(
            dst * (1.0 - amount) + carry * amount,
            0,
            255,
        ).astype(np.uint8)
        self._buffer[sy0:sy1, sx0:sx1] = (
            carry * amount + dst * (1.0 - amount))
        return dx0, dy0, dx1, dy1

    def line(
            self,
            layer_image: np.ndarray,
            brush,
            x0: int,
            y0: int,
            x1: int,
            y1: int) -> Rect | None:
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        dist = float(np.hypot(dx, dy))
        if dist < 0.5:
            return self.dab(layer_image, brush, x1, y1)

        spacing = max(1.0, brush.size * 0.25)
        steps = max(1, int(np.ceil(dist / spacing)))
        dirty = None
        prev_x, prev_y = x0, y0
        for i in range(1, steps + 1):
            t = i / steps
            cx = int(round(x0 + dx * t))
            cy = int(round(y0 + dy * t))
            if cx == prev_x and cy == prev_y:
                continue
            dab_dirty = self.dab(layer_image, brush, cx, cy)
            dirty = union_rect(dirty, dab_dirty)
            prev_x, prev_y = cx, cy
        return dirty

    def clear(self) -> None:
        self._buffer = None
