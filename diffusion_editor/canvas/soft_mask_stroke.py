"""Pure soft-brush stroke operations for float masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .canvas_geometry import Rect


@dataclass(frozen=True)
class SoftMaskBrush:
    size: int
    hardness: float
    flow: float


def apply_dab(
        mask: np.ndarray,
        cx: int,
        cy: int,
        brush: SoftMaskBrush,
        *,
        erase: bool = False) -> tuple[Rect | None, np.ndarray | None]:
    """Apply one soft brush dab to a float mask."""
    d = brush.size
    if d < 1:
        return None, None

    y, x = np.ogrid[-d / 2:d / 2, -d / 2:d / 2]
    dist = np.sqrt(x * x + y * y)
    radius = d / 2
    alpha_mask = _alpha_from_distance(dist, radius, brush.hardness)

    stamp_height, stamp_width = alpha_mask.shape
    mask_height, mask_width = mask.shape
    x0 = cx - stamp_width // 2
    y0 = cy - stamp_height // 2
    sx0 = max(0, -x0)
    sy0 = max(0, -y0)
    sx1 = min(stamp_width, mask_width - x0)
    sy1 = min(stamp_height, mask_height - y0)
    dx0 = max(0, x0)
    dy0 = max(0, y0)
    dx1 = dx0 + (sx1 - sx0)
    dy1 = dy0 + (sy1 - sy0)
    if dx0 >= dx1 or dy0 >= dy1:
        return None, None

    stamp_slice = alpha_mask[sy0:sy1, sx0:sx1] * brush.flow
    _apply_stamp(mask, (dx0, dy0, dx1, dy1), stamp_slice, erase=erase)
    return (dx0, dy0, dx1, dy1), stamp_slice


def apply_line(
        mask: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        brush: SoftMaskBrush,
        *,
        erase: bool = False) -> tuple[Rect | None, np.ndarray | None]:
    """Apply a soft brush line segment to a float mask."""
    mask_height, mask_width = mask.shape
    d = brush.size
    if d < 1:
        return None, None

    radius = d / 2.0
    bx0 = max(0, int(min(x0, x1) - radius))
    by0 = max(0, int(min(y0, y1) - radius))
    bx1 = min(mask_width, int(max(x0, x1) + radius) + 1)
    by1 = min(mask_height, int(max(y0, y1) + radius) + 1)
    if bx0 >= bx1 or by0 >= by1:
        return None, None

    sdx = float(x1 - x0)
    sdy = float(y1 - y0)
    seg_len_sq = sdx * sdx + sdy * sdy
    if seg_len_sq < 0.5:
        return apply_dab(mask, x0, y0, brush, erase=erase)

    yy, xx = np.mgrid[by0:by1, bx0:bx1]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    t = ((xx - x0) * sdx + (yy - y0) * sdy) / seg_len_sq
    np.clip(t, 0.0, 1.0, out=t)
    cx = x0 + t * sdx
    cy = y0 + t * sdy
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    stamp = _alpha_from_distance(dist, radius, brush.hardness) * brush.flow
    _apply_stamp(mask, (bx0, by0, bx1, by1), stamp, erase=erase)
    return (bx0, by0, bx1, by1), stamp


def _alpha_from_distance(
        dist: np.ndarray,
        radius: float,
        hardness: float) -> np.ndarray:
    if hardness >= 1.0:
        return (dist <= radius).astype(np.float32)

    inner = radius * hardness
    return np.clip((radius - dist) / max(radius - inner, 0.001), 0, 1)


def _apply_stamp(
        mask: np.ndarray,
        rect: Rect,
        stamp: np.ndarray,
        *,
        erase: bool) -> None:
    x0, y0, x1, y1 = rect
    target = mask[y0:y1, x0:x1]
    if erase:
        target[:] = np.minimum(target, 1.0 - stamp)
    else:
        target[:] = np.maximum(target, stamp)
