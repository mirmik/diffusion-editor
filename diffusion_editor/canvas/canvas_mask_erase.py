"""Runtime state for layer mask erase previews."""

from __future__ import annotations

import numpy as np

from .canvas_geometry import union_rect

Rect = tuple[int, int, int, int]


class MaskEraseStrokeBuffer:
    def __init__(self):
        self._mask: np.ndarray | None = None
        self._dirty_rect: Rect | None = None

    @property
    def mask(self) -> np.ndarray | None:
        return self._mask

    @property
    def dirty_rect(self) -> Rect | None:
        return self._dirty_rect

    def begin(self, height: int, width: int) -> bool:
        if height == 0 or width == 0:
            self.clear()
            return False
        self._mask = np.zeros((height, width), dtype=np.float32)
        self._dirty_rect = None
        return True

    def add_dirty(self, dirty: Rect | None) -> None:
        self._dirty_rect = union_rect(self._dirty_rect, dirty)

    def erase_region(self, rect: Rect) -> np.ndarray | None:
        if self._mask is None:
            return None
        x0, y0, x1, y1 = rect
        return self._mask[y0:y1, x0:x1]

    def preview_region(
            self,
            layer_mask: np.ndarray,
            rect: Rect | None) -> np.ndarray | None:
        erase = self.erase_region(rect) if rect is not None else None
        if erase is None:
            return None
        x0, y0, x1, y1 = rect
        return np.minimum(layer_mask[y0:y1, x0:x1], 1.0 - erase)

    def apply_to_layer_mask(self, layer_mask: np.ndarray) -> Rect | None:
        if self._mask is None or self._dirty_rect is None:
            return None
        x0, y0, x1, y1 = self._dirty_rect
        layer_mask[y0:y1, x0:x1] = np.minimum(
            layer_mask[y0:y1, x0:x1],
            1.0 - self._mask[y0:y1, x0:x1],
        )
        return self._dirty_rect

    def clear(self) -> None:
        self._mask = None
        self._dirty_rect = None
