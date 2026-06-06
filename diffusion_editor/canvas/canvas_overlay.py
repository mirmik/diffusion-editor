"""Selection and layer-mask overlay composition for EditorCanvas."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .canvas_geometry import clip_layer_local_rect, visible_layer_rect
from ..document.layer import Layer
from ..document.layer_stack import LayerStack

Rect = tuple[int, int, int, int]

SELECTION_COLOR = (50.0, 50.0, 255.0)
SELECTION_OPACITY = 0.3
MASK_COLOR = (255.0, 50.0, 50.0)
MASK_OPACITY = 0.4


def compose_overlay_pixels(
        selection_alpha: np.ndarray | None,
        mask_alpha: np.ndarray | None) -> np.ndarray | None:
    if selection_alpha is None and mask_alpha is None:
        return None

    if selection_alpha is not None:
        h, w = selection_alpha.shape[:2]
    else:
        h, w = mask_alpha.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    has_base = False
    if selection_alpha is not None:
        sa = selection_alpha / 255.0
        overlay[:, :, 0] = (SELECTION_COLOR[0] * sa).astype(np.uint8)
        overlay[:, :, 1] = (SELECTION_COLOR[1] * sa).astype(np.uint8)
        overlay[:, :, 2] = (SELECTION_COLOR[2] * sa).astype(np.uint8)
        overlay[:, :, 3] = selection_alpha.astype(np.uint8)
        has_base = True

    if mask_alpha is not None:
        ma = mask_alpha / 255.0
        inv = 1.0 - ma
        if has_base:
            overlay[:, :, 0] = (
                MASK_COLOR[0] * ma + overlay[:, :, 0].astype(np.float32) * inv
            ).astype(np.uint8)
            overlay[:, :, 1] = (
                MASK_COLOR[1] * ma + overlay[:, :, 1].astype(np.float32) * inv
            ).astype(np.uint8)
            overlay[:, :, 2] = (
                MASK_COLOR[2] * ma + overlay[:, :, 2].astype(np.float32) * inv
            ).astype(np.uint8)
            overlay[:, :, 3] = np.clip(
                mask_alpha + overlay[:, :, 3].astype(np.float32) * inv,
                0,
                255,
            ).astype(np.uint8)
        else:
            overlay[:, :, 0] = MASK_COLOR[0]
            overlay[:, :, 1] = MASK_COLOR[1]
            overlay[:, :, 2] = MASK_COLOR[2]
            overlay[:, :, 3] = mask_alpha.astype(np.uint8)

    return overlay


class CanvasOverlayBridge:
    def __init__(
            self,
            layer_stack: LayerStack,
            *,
            set_overlay: Callable[[np.ndarray | None], None]):
        self._layer_stack = layer_stack
        self._set_overlay = set_overlay
        self._overlay: np.ndarray | None = None
        self.show_mask = True
        self.show_selection = True

    @property
    def overlay(self) -> np.ndarray | None:
        return self._overlay

    @overlay.setter
    def overlay(self, overlay: np.ndarray | None) -> None:
        self._overlay = overlay

    def clear(self) -> None:
        self._overlay = None

    def rebuild(self) -> None:
        h, w = self._layer_stack.height, self._layer_stack.width
        if h == 0 or w == 0:
            self._overlay = None
            self._set_overlay(None)
            return

        layer = self._layer_stack.active_layer
        selection_alpha = self._selection_alpha((0, 0, w, h))
        mask_alpha = self._full_mask_alpha(layer)
        self._overlay = compose_overlay_pixels(selection_alpha, mask_alpha)
        self._set_overlay(self._overlay)

    def update_mask_region(self, layer: Layer, dirty: Rect | None) -> None:
        if dirty is None or not self.show_mask:
            return
        local_rect, canvas_rect = self._clip_layer_local_rect(layer, dirty)
        if local_rect is None:
            return
        if self._overlay is None:
            self.rebuild()
            return

        x0, y0, x1, y1 = local_rect
        mask_values = layer.mask.data[y0:y1, x0:x1]
        self._replace_canvas_region(canvas_rect, mask_values)

    def update_mask_region_preview(
            self,
            layer: Layer,
            dirty: Rect | None,
            preview_mask: np.ndarray | None) -> None:
        if dirty is None or preview_mask is None or not self.show_mask:
            return
        local_rect, canvas_rect = self._clip_layer_local_rect(layer, dirty)
        if local_rect is None:
            return
        if self._overlay is None:
            self.rebuild()
            if self._overlay is None:
                return

        lx0, ly0, lx1, ly1 = local_rect
        dx0, dy0, _dx1, _dy1 = dirty
        px0 = lx0 - dx0
        py0 = ly0 - dy0
        px1 = px0 + (lx1 - lx0)
        py1 = py0 + (ly1 - ly0)
        self._replace_canvas_region(
            canvas_rect,
            preview_mask[py0:py1, px0:px1],
        )

    def _selection_alpha(self, canvas_rect: Rect) -> np.ndarray | None:
        if not self.show_selection or self._layer_stack.selection.is_empty:
            return None
        x0, y0, x1, y1 = canvas_rect
        return (
            self._layer_stack.selection.data[y0:y1, x0:x1]
            * 255.0
            * SELECTION_OPACITY
        ).astype(np.float32)

    def _full_mask_alpha(self, layer: Layer | None) -> np.ndarray | None:
        if layer is None or not self.show_mask or not layer.has_mask():
            return None

        h, w = self._layer_stack.height, self._layer_stack.width
        mask_alpha = np.zeros((h, w), dtype=np.float32)
        lx0, ly0, lx1, ly1 = visible_layer_rect(layer, w, h)
        if lx1 > lx0 and ly1 > ly0:
            dx0, dy0 = layer.x + lx0, layer.y + ly0
            dx1, dy1 = layer.x + lx1, layer.y + ly1
            mask_alpha[dy0:dy1, dx0:dx1] = (
                layer.mask.data[ly0:ly1, lx0:lx1] * 255.0 * MASK_OPACITY)
        return mask_alpha

    def _clip_layer_local_rect(
            self,
            layer: Layer,
            dirty: Rect) -> tuple[Rect | None, Rect | None]:
        return clip_layer_local_rect(
            layer,
            dirty,
            self._layer_stack.width,
            self._layer_stack.height,
        )

    def _replace_canvas_region(
            self,
            canvas_rect: Rect,
            mask_values: np.ndarray) -> None:
        if self._overlay is None:
            return
        cx0, cy0, cx1, cy1 = canvas_rect
        selection_alpha = self._selection_alpha(canvas_rect)
        mask_alpha = (mask_values * 255.0 * MASK_OPACITY).astype(np.float32)
        region = compose_overlay_pixels(selection_alpha, mask_alpha)
        if region is None:
            self._overlay[cy0:cy1, cx0:cx1] = 0
        else:
            self._overlay[cy0:cy1, cx0:cx1] = region
        self._set_overlay(self._overlay)
