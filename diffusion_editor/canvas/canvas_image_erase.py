"""Image alpha eraser operations for canvas brush tools."""

from __future__ import annotations

import numpy as np

from .brush import Brush

Rect = tuple[int, int, int, int]


def erase_dab(layer_image: np.ndarray, brush: Brush, cx: int, cy: int) -> Rect | None:
    stamp = brush._alpha_stamp
    sh, sw = stamp.shape[:2]
    ih, iw = layer_image.shape[:2]

    x0 = cx - sw // 2
    y0 = cy - sh // 2
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = min(sw, iw - x0), min(sh, ih - y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
    if dx0 >= dx1 or dy0 >= dy1:
        return None

    erase = (
        stamp[sy0:sy1, sx0:sx1]
        * (brush.color[3] / 255.0)
        * brush.flow
    )
    _apply_alpha_erase(layer_image, (dx0, dy0, dx1, dy1), erase)
    return (dx0, dy0, dx1, dy1)


def erase_line(
        layer_image: np.ndarray,
        brush: Brush,
        x0: int,
        y0: int,
        x1: int,
        y1: int) -> Rect | None:
    ih, iw = layer_image.shape[:2]
    radius = brush.size / 2.0

    bx0 = max(0, int(min(x0, x1) - radius))
    by0 = max(0, int(min(y0, y1) - radius))
    bx1 = min(iw, int(max(x0, x1) + radius) + 1)
    by1 = min(ih, int(max(y0, y1) + radius) + 1)
    if bx0 >= bx1 or by0 >= by1:
        return None

    sdx = float(x1 - x0)
    sdy = float(y1 - y0)
    seg_len_sq = sdx * sdx + sdy * sdy
    if seg_len_sq < 0.5:
        return erase_dab(layer_image, brush, x0, y0)

    yy, xx = np.mgrid[by0:by1, bx0:bx1]
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)

    t = ((xx - x0) * sdx + (yy - y0) * sdy) / seg_len_sq
    np.clip(t, 0.0, 1.0, out=t)
    cx = x0 + t * sdx
    cy = y0 + t * sdy
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    if brush.hardness >= 1.0:
        erase = (dist <= radius).astype(np.float32)
    else:
        inner = radius * brush.hardness
        erase = np.clip(
            (radius - dist) / max(radius - inner, 0.001),
            0.0,
            1.0,
        )

    erase *= (brush.color[3] / 255.0) * brush.flow
    _apply_alpha_erase(layer_image, (bx0, by0, bx1, by1), erase)
    return (bx0, by0, bx1, by1)


def erase_alpha_region(
        layer_image: np.ndarray,
        rect: Rect,
        erase: np.ndarray) -> None:
    if erase.dtype == np.uint8:
        erase = erase.astype(np.float32) / 255.0
    _apply_alpha_erase(layer_image, rect, erase)


def _apply_alpha_erase(
        layer_image: np.ndarray,
        rect: Rect,
        erase: np.ndarray) -> None:
    x0, y0, x1, y1 = rect
    alpha = layer_image[y0:y1, x0:x1, 3].astype(np.float32)
    layer_image[y0:y1, x0:x1, 3] = np.clip(
        alpha * (1.0 - erase),
        0,
        255,
    ).astype(np.uint8)
