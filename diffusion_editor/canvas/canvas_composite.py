"""CPU/GPU composition bridge for EditorCanvas."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .gpu_compositor import GPUCompositor
from ..document.layer import Layer
from ..document.layer_stack import LayerStack

Rect = tuple[int, int, int, int]


class CanvasCompositeBridge:
    def __init__(
            self,
            layer_stack: LayerStack,
            *,
            gpu_compositing: bool,
            graphics=None,
            set_image: Callable[[np.ndarray], None]):
        self._layer_stack = layer_stack
        self._set_image = set_image
        self._gpu_compositing = gpu_compositing
        self._gpu_compositor = (
            GPUCompositor(layer_stack, graphics=graphics)
            if gpu_compositing else None
        )
        self._composite: np.ndarray | None = None
        self._composite_stale = True

    @property
    def gpu_compositing(self) -> bool:
        return self._gpu_compositing

    @gpu_compositing.setter
    def gpu_compositing(self, enabled: bool) -> None:
        self._gpu_compositing = enabled

    @property
    def gpu_compositor(self):
        return self._gpu_compositor

    @gpu_compositor.setter
    def gpu_compositor(self, compositor) -> None:
        self._gpu_compositor = compositor

    @property
    def using_gpu(self) -> bool:
        return self._gpu_compositing and self._gpu_compositor is not None

    @property
    def composite(self) -> np.ndarray | None:
        return self._composite

    @composite.setter
    def composite(self, composite: np.ndarray | None) -> None:
        self._composite = composite

    @property
    def composite_stale(self) -> bool:
        return self._composite_stale

    @composite_stale.setter
    def composite_stale(self, stale: bool) -> None:
        self._composite_stale = stale

    def rebuild(self) -> None:
        if self._gpu_compositor is not None:
            self._gpu_compositor.rebuild()

    def update_composite(self) -> np.ndarray | None:
        if self.using_gpu:
            self._gpu_compositor.composite()
            self._composite_stale = True
            return None

        self._composite = np.ascontiguousarray(self._layer_stack.composite())
        self._set_image(self._composite)
        return self._composite

    def ensure_cpu_composite(self) -> np.ndarray | None:
        if self.using_gpu and self._composite_stale:
            self._composite = self._gpu_compositor.readback()
            self._composite_stale = False
        return self._composite

    def get_composite(self) -> np.ndarray | None:
        return self.ensure_cpu_composite()

    def get_composite_below(self, layer: Layer) -> np.ndarray | None:
        return np.ascontiguousarray(
            self._layer_stack.composite(exclude_layer=layer))

    def composite_rect_below(
            self,
            target_layer: Layer,
            dy0: int,
            dy1: int,
            dx0: int,
            dx1: int) -> np.ndarray:
        cache = self._layer_stack.get_prefix_below_rect(
            target_layer, dx0, dy0, dx1, dy1)
        return cache.astype(np.float32)

    def refresh_modified_layer_rect(
            self,
            layer: Layer,
            local_rect: Rect,
            canvas_rect: Rect) -> None:
        if not self._layer_stack.is_layer_visible_for_composition(layer):
            self._layer_stack.mark_layer_dirty(layer, canvas_rect)
            if not self.using_gpu:
                self._rebuild_cpu_composite()
            return

        self._layer_stack.mark_layer_dirty(layer, canvas_rect)
        if self.using_gpu:
            self._gpu_compositor.mark_dirty(layer)
            self._gpu_compositor.composite()
            self._composite_stale = True
            return

        if self._composite is None:
            return
        self._blend_layer_rect(layer, local_rect, canvas_rect)

    def refresh_layer_transform(self, layer: Layer, dirty_canvas_rect: Rect) -> None:
        self._layer_stack.mark_layer_dirty(layer, dirty_canvas_rect)
        if self.using_gpu:
            self._gpu_compositor.mark_dirty(layer)
            self._gpu_compositor.composite()
            self._composite_stale = True
            return
        self._rebuild_cpu_composite()

    def preview_erased_layer_rect(
            self,
            layer: Layer,
            local_rect: Rect,
            canvas_rect: Rect,
            erase: np.ndarray) -> None:
        composite = self.ensure_cpu_composite()
        if composite is None:
            return

        x0, y0, x1, y1 = local_rect
        cx0, cy0, cx1, cy1 = canvas_rect
        below = self.composite_rect_below(layer, cy0, cy1, cx0, cx1)
        above = layer.image[y0:y1, x0:x1].astype(np.float32)
        above[:, :, 3] = np.clip(above[:, :, 3] * (1.0 - erase), 0, 255)
        self._blend_into_composite(canvas_rect, above, below)

    @property
    def display_tex(self):
        if self._gpu_compositor is None:
            return None
        return self._gpu_compositor.display_tex

    def display_size(self) -> tuple[int, int]:
        if self._gpu_compositor is None:
            return 0, 0
        return self._gpu_compositor.display_size()

    def dispose(self) -> None:
        if self._gpu_compositor is not None:
            self._gpu_compositor.dispose()
            self._gpu_compositor = None

    def _rebuild_cpu_composite(self) -> None:
        self._composite = np.ascontiguousarray(self._layer_stack.composite())
        self._set_image(self._composite)

    def _blend_layer_rect(
            self,
            layer: Layer,
            local_rect: Rect,
            canvas_rect: Rect) -> None:
        x0, y0, x1, y1 = local_rect
        cx0, cy0, cx1, cy1 = canvas_rect
        below = self.composite_rect_below(layer, cy0, cy1, cx0, cx1)
        above = layer.image[y0:y1, x0:x1].astype(np.float32)
        self._blend_into_composite(canvas_rect, above, below)

    def _blend_into_composite(
            self,
            canvas_rect: Rect,
            above: np.ndarray,
            below: np.ndarray) -> None:
        if self._composite is None:
            return
        cx0, cy0, cx1, cy1 = canvas_rect
        sa = above[:, :, 3:4] / 255.0
        inv_sa = 1.0 - sa
        da = below[:, :, 3:4] / 255.0
        out_a = sa + da * inv_sa
        safe_a = np.maximum(out_a, 1.0 / 255.0)
        out_rgb = (above[:, :, :3] * sa + below[:, :, :3] * da * inv_sa) / safe_a
        self._composite[cy0:cy1, cx0:cx1, :3] = np.clip(
            out_rgb, 0, 255).astype(np.uint8)
        self._composite[cy0:cy1, cx0:cx1, 3:4] = np.clip(
            out_a * 255.0, 0, 255).astype(np.uint8)
        self._set_image(self._composite)
