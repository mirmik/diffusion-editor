"""Live paint stroke buffer used by EditorCanvas."""

from __future__ import annotations

import numpy as np

from .canvas_geometry import union_rect

Rect = tuple[int, int, int, int]


def blend_paint_region(
        base: np.ndarray,
        stroke_mask: np.ndarray,
        color: tuple[int, int, int, int],
        dirty: Rect) -> np.ndarray | None:
    x0, y0, x1, y1 = dirty
    if x1 <= x0 or y1 <= y0:
        return None

    base_region = base[y0:y1, x0:x1]
    mask_region = stroke_mask[y0:y1, x0:x1]
    out = base_region.copy()
    where = mask_region > 0
    if not np.any(where):
        return out

    r, g, b, _a = color
    sa = mask_region[where].astype(np.float32) / 255.0
    da = base_region[where, 3].astype(np.float32) / 255.0
    out_a = sa + da * (1.0 - sa)
    safe_a = np.maximum(out_a, 1.0 / 255.0)
    inv_sa = 1.0 - sa

    for c, src_val in enumerate((r, g, b)):
        dst_c = base_region[where, c].astype(np.float32)
        out[where, c] = np.clip(
            (src_val * sa + dst_c * da * inv_sa) / safe_a,
            0,
            255,
        ).astype(np.uint8)
    out[where, 3] = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)
    return out


class PaintStrokeBuffer:
    def __init__(self):
        self._mask: np.ndarray | None = None
        self._color: tuple[int, int, int, int] | None = None
        self._base_image: np.ndarray | None = None
        self._live_dirty_rect: Rect | None = None

    @property
    def mask(self) -> np.ndarray | None:
        return self._mask

    @property
    def live_dirty_rect(self) -> Rect | None:
        return self._live_dirty_rect

    def begin(self, layer_image: np.ndarray, color: tuple[int, int, int, int]) -> bool:
        h, w = layer_image.shape[:2]
        if h == 0 or w == 0:
            self.clear()
            return False
        self._mask = np.zeros((h, w), dtype=np.uint8)
        self._color = color
        self._base_image = layer_image.copy()
        self._live_dirty_rect = None
        return True

    def apply_region(self, layer_image: np.ndarray, dirty: Rect | None) -> Rect | None:
        if (
                dirty is None
                or self._base_image is None
                or self._mask is None
                or self._color is None):
            return None
        out = blend_paint_region(
            self._base_image,
            self._mask,
            self._color,
            dirty,
        )
        if out is None:
            return None

        x0, y0, x1, y1 = dirty
        layer_image[y0:y1, x0:x1] = out
        self._live_dirty_rect = union_rect(self._live_dirty_rect, dirty)
        return dirty

    def clear(self) -> None:
        self._mask = None
        self._color = None
        self._base_image = None
        self._live_dirty_rect = None
