"""EditorCanvas — extends tcgui Canvas with brush/mask painting and rect modes."""

from __future__ import annotations

import numpy as np

from tcbase import Key, MouseButton, Mods
from tcgui.widgets.canvas import Canvas
from tcgui.widgets.events import KeyEvent

from .layer_stack import LayerStack
from .layer import Layer, DiffusionLayer, LamaLayer, InstructLayer
from .brush import Brush, composite_stroke


class EditorCanvas(Canvas):
    """Zoomable image canvas with brush painting, mask painting, and rect tools."""

    def __init__(self, layer_stack: LayerStack):
        super().__init__()
        self.background_color = (0.08, 0.08, 0.10, 1.0)
        self._layer_stack = layer_stack
        self._composite: np.ndarray | None = None

        self.brush = Brush()
        self._brush_eraser = False
        self._mask_brush_size = 50
        self._mask_brush_hardness = 0.4
        self._mask_brush_flow = 1.0
        self._mask_eraser = False
        self._show_mask = True

        # Rectangle modes
        self._ref_rect_mode = False
        self._ref_rect_dragging = False
        self._ref_rect_start: tuple[int, int] | None = None
        self._ref_rect_end: tuple[int, int] | None = None
        self._show_ref_rect = True

        self._patch_rect_mode = False
        self._patch_rect_dragging = False
        self._patch_rect_start: tuple[int, int] | None = None
        self._patch_rect_end: tuple[int, int] | None = None
        self._show_patch_rect = True

        # Stroke buffer (MAX blending during one stroke)
        self._stroke_mask: np.ndarray | None = None
        self._stroke_color: tuple | None = None
        self._stroke_overlay: np.ndarray | None = None
        self._stroke_is_eraser = False

        self._painting = False
        self._last_paint_pos: tuple[int, int] | None = None
        self._mask_overlay: np.ndarray | None = None
        self._mask_erase_stroke: np.ndarray | None = None
        self._mask_erase_dirty: tuple[int, int, int, int] | None = None
        self._stroke_dirty_rect: tuple[int, int, int, int] | None = None
        self._edit_label: str | None = None
        self._edit_target: str | None = None  # "image" | "mask"
        self._edit_layer: Layer | None = None

        # Callbacks
        self.on_mouse_moved: callable = None
        self.on_color_picked: callable = None
        self.on_ref_rect_drawn: callable = None
        self.on_patch_rect_drawn: callable = None
        self.on_edit_begin: callable = None  # (label: str, layer: Layer, target: str)
        self.on_edit_end: callable = None  # (layer: Layer, target: str, dirty_rect)

        # Wire layer stack
        layer_stack.on_changed = self._on_stack_changed

        # Wire canvas callbacks for painting
        self.on_canvas_mouse_down = self._handle_mouse_down
        self.on_canvas_mouse_move = self._handle_mouse_move
        self.on_canvas_mouse_up = self._handle_mouse_up
        self.on_render_overlay = self._render_overlay

    # ------------------------------------------------------------------
    # Layer stack integration
    # ------------------------------------------------------------------

    def _on_stack_changed(self):
        print("[EditorCanvas] _on_stack_changed called")
        self._update_composite()

    def _update_composite(self):
        self._composite = np.ascontiguousarray(self._layer_stack.composite())
        self.set_image(self._composite)
        self._mask_overlay = None
        self._update_overlay()

    def _update_overlay(self):
        """Rebuild overlay combining stroke preview + mask."""
        layer = self._layer_stack.active_layer
        h, w = (self._layer_stack.height, self._layer_stack.width)
        if h == 0 or w == 0:
            self.set_overlay(None)
            return

        has_stroke = self._stroke_mask is not None and self._stroke_overlay is not None
        has_mask = (self._show_mask
                    and isinstance(layer, (DiffusionLayer, LamaLayer, InstructLayer))
                    and layer.has_mask())

        if not has_stroke and not has_mask:
            self.set_overlay(None)
            return

        if has_stroke and not has_mask:
            # Fast path: stroke only — just copy RGB + alpha, no compositing
            self._stroke_overlay[:, :, 3] = self._stroke_mask
            self.set_overlay(self._stroke_overlay.copy())
            return

        if has_mask and not has_stroke:
            # Fast path: mask only
            overlay = np.empty((h, w, 4), dtype=np.uint8)
            overlay[:, :, 0] = 255
            overlay[:, :, 1] = 50
            overlay[:, :, 2] = 50
            overlay[:, :, 3] = (layer.mask.astype(np.float32) * 0.4).astype(np.uint8)
            self.set_overlay(overlay)
            return

        # Both mask and stroke: composite stroke over mask
        overlay = np.empty((h, w, 4), dtype=np.uint8)
        overlay[:, :, 0] = 255
        overlay[:, :, 1] = 50
        overlay[:, :, 2] = 50
        overlay[:, :, 3] = (layer.mask.astype(np.float32) * 0.4).astype(np.uint8)

        self._stroke_overlay[:, :, 3] = self._stroke_mask
        sa = self._stroke_overlay[:, :, 3:4].astype(np.float32) / 255.0
        inv = 1.0 - sa
        overlay[:, :, :3] = (
            self._stroke_overlay[:, :, :3].astype(np.float32) * sa
            + overlay[:, :, :3].astype(np.float32) * inv
        ).astype(np.uint8)
        overlay[:, :, 3] = np.clip(
            self._stroke_overlay[:, :, 3].astype(np.float32)
            + overlay[:, :, 3].astype(np.float32) * (1.0 - sa[:, :, 0]),
            0, 255
        ).astype(np.uint8)

        self.set_overlay(overlay)

    def _update_stroke_region(self, dirty):
        """Partial overlay update for brush stroke — only copy dirty region alpha."""
        if dirty is None or self._stroke_overlay is None or self._stroke_mask is None:
            return
        x0, y0, x1, y1 = dirty
        self._stroke_overlay[y0:y1, x0:x1, 3] = self._stroke_mask[y0:y1, x0:x1]
        self.mark_overlay_dirty(x0, y0, x1, y1)

    def _update_mask_overlay_region(self, layer, dirty):
        """Partial overlay update for mask painting — only recompute dirty region alpha."""
        if dirty is None:
            return
        x0, y0, x1, y1 = dirty
        if self._mask_overlay is None:
            h, w = self._layer_stack.height, self._layer_stack.width
            self._mask_overlay = np.empty((h, w, 4), dtype=np.uint8)
            self._mask_overlay[:, :, 0] = 255
            self._mask_overlay[:, :, 1] = 50
            self._mask_overlay[:, :, 2] = 50
            self._mask_overlay[:, :, 3] = (
                layer.mask.astype(np.float32) * 0.4).astype(np.uint8)
            self.set_overlay_ref(self._mask_overlay)
            return
        self._mask_overlay[y0:y1, x0:x1, 3] = (
            layer.mask[y0:y1, x0:x1].astype(np.float32) * 0.4).astype(np.uint8)
        self.mark_overlay_dirty(x0, y0, x1, y1)

    def _update_mask_overlay_region_preview(self, layer, dirty, preview_mask):
        """Update overlay alpha from a provided mask preview region."""
        if dirty is None or preview_mask is None:
            return
        x0, y0, x1, y1 = dirty
        if self._mask_overlay is None:
            h, w = self._layer_stack.height, self._layer_stack.width
            self._mask_overlay = np.empty((h, w, 4), dtype=np.uint8)
            self._mask_overlay[:, :, 0] = 255
            self._mask_overlay[:, :, 1] = 50
            self._mask_overlay[:, :, 2] = 50
            self._mask_overlay[:, :, 3] = (
                layer.mask.astype(np.float32) * 0.4).astype(np.uint8)
            self.set_overlay_ref(self._mask_overlay)
        self._mask_overlay[y0:y1, x0:x1, 3] = (
            preview_mask.astype(np.float32) * 0.4).astype(np.uint8)
        self.mark_overlay_dirty(x0, y0, x1, y1)

    def _preview_mask_erase_region(self, layer, dirty):
        """Preview erase on composite without modifying layer.image."""
        if self._composite is None or dirty is None or self._mask_erase_stroke is None:
            return
        x0, y0, x1, y1 = dirty
        if x1 <= x0 or y1 <= y0:
            return
        below = self._composite_rect_below(layer, y0, y1, x0, x1)
        above = layer.image[y0:y1, x0:x1].astype(np.float32)
        erase = self._mask_erase_stroke[y0:y1, x0:x1].astype(np.float32) / 255.0
        # Reduce alpha for preview only
        above[:, :, 3] = np.clip(above[:, :, 3] * (1.0 - erase), 0, 255)
        sa = above[:, :, 3:4] / 255.0
        inv_sa = 1.0 - sa
        da = below[:, :, 3:4] / 255.0
        out_a = sa + da * inv_sa
        safe_a = np.maximum(out_a, 1.0 / 255.0)
        out_rgb = (above[:, :, :3] * sa + below[:, :, :3] * da * inv_sa) / safe_a
        self._composite[y0:y1, x0:x1, :3] = np.clip(
            out_rgb, 0, 255).astype(np.uint8)
        self._composite[y0:y1, x0:x1, 3:4] = np.clip(
            out_a * 255.0, 0, 255).astype(np.uint8)
        self.set_image(self._composite)

    def _erase_layer_rect(self, layer, rect, erase_u8):
        """Erase layer.image alpha by erase_u8 mask within rect, update composite."""
        if rect is None or erase_u8 is None:
            return
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            return
        erase = erase_u8.astype(np.float32) / 255.0
        la = layer.image[y0:y1, x0:x1, 3].astype(np.float32)
        layer.image[y0:y1, x0:x1, 3] = np.clip(
            la * (1.0 - erase), 0, 255).astype(np.uint8)

        if self._composite is not None:
            below = self._composite_rect_below(layer, y0, y1, x0, x1)
            above = layer.image[y0:y1, x0:x1].astype(np.float32)
            sa = above[:, :, 3:4] / 255.0
            inv_sa = 1.0 - sa
            da = below[:, :, 3:4] / 255.0
            out_a = sa + da * inv_sa
            safe_a = np.maximum(out_a, 1.0 / 255.0)
            out_rgb = (above[:, :, :3] * sa + below[:, :, :3] * da * inv_sa) / safe_a
            self._composite[y0:y1, x0:x1, :3] = np.clip(
                out_rgb, 0, 255).astype(np.uint8)
            self._composite[y0:y1, x0:x1, 3:4] = np.clip(
                out_a * 255.0, 0, 255).astype(np.uint8)
            self.set_image(self._composite)

        self._layer_stack.mark_layer_dirty(layer, rect)

    def get_composite(self) -> np.ndarray | None:
        return self._composite

    def get_composite_below(self, layer: Layer) -> np.ndarray | None:
        return np.ascontiguousarray(
            self._layer_stack.composite(exclude_layer=layer))

    def view_center_image(self) -> tuple[int, int]:
        cx = self.x + self.width / 2
        cy = self.y + self.height / 2
        ix, iy = self.widget_to_image(cx, cy)
        return int(ix), int(iy)

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_mask_brush(self, size: int, hardness: float, flow: float = 1.0):
        self._mask_brush_size = size
        self._mask_brush_hardness = hardness
        self._mask_brush_flow = max(0.0, min(flow, 1.0))

    def set_mask_eraser(self, eraser: bool):
        self._mask_eraser = eraser

    def set_brush_eraser(self, eraser: bool):
        self._brush_eraser = eraser

    def set_show_mask(self, show: bool):
        self._show_mask = show
        self._mask_overlay = None
        self._update_overlay()

    def set_ref_rect_mode(self, on: bool):
        self._ref_rect_mode = on
        self.cursor = "cross" if on else ""
        if not on:
            self._ref_rect_dragging = False

    def set_show_ref_rect(self, show: bool):
        self._show_ref_rect = show

    def set_patch_rect_mode(self, on: bool):
        self._patch_rect_mode = on
        self.cursor = "cross" if on else ""
        if not on:
            self._patch_rect_dragging = False

    def set_show_patch_rect(self, show: bool):
        self._show_patch_rect = show

    # ------------------------------------------------------------------
    # Mask painting
    # ------------------------------------------------------------------

    def _is_mask_layer_active(self) -> bool:
        return isinstance(self._layer_stack.active_layer,
                          (DiffusionLayer, LamaLayer, InstructLayer))

    def _dab_mask(self, mask: np.ndarray, cx: int, cy: int, *, erase: bool | None = None):
        """Returns (dirty_rect, stamp_u8) or (None, None)."""
        d = self._mask_brush_size
        if d < 1:
            return None, None
        y, x = np.ogrid[-d / 2:d / 2, -d / 2:d / 2]
        dist = np.sqrt(x * x + y * y)
        radius = d / 2

        if self._mask_brush_hardness >= 1.0:
            alpha_mask = (dist <= radius).astype(np.float32) * 255
        else:
            inner = radius * self._mask_brush_hardness
            alpha_mask = np.clip(
                (radius - dist) / max(radius - inner, 0.001), 0, 1) * 255

        sh, sw = alpha_mask.shape
        ih, iw = mask.shape
        x0 = cx - sw // 2
        y0 = cy - sh // 2
        sx0 = max(0, -x0)
        sy0 = max(0, -y0)
        sx1 = min(sw, iw - x0)
        sy1 = min(sh, ih - y0)
        dx0 = max(0, x0)
        dy0b = max(0, y0)
        dx1 = dx0 + (sx1 - sx0)
        dy1 = dy0b + (sy1 - sy0)
        if dx0 >= dx1 or dy0b >= dy1:
            return None, None
        stamp_slice = (alpha_mask[sy0:sy1, sx0:sx1] * self._mask_brush_flow).astype(np.uint8)
        if erase is None:
            erase = self._mask_eraser
        if erase:
            mask[dy0b:dy1, dx0:dx1] = np.minimum(
                mask[dy0b:dy1, dx0:dx1], 255 - stamp_slice)
        else:
            mask[dy0b:dy1, dx0:dx1] = np.maximum(
                mask[dy0b:dy1, dx0:dx1], stamp_slice)
        return (dx0, dy0b, dx1, dy1), stamp_slice

    def _stroke_mask_line(self, mask: np.ndarray,
                          x0: int, y0: int, x1: int, y1: int,
                          *, erase: bool | None = None):
        """Draw smooth mask stroke segment. Returns (dirty_rect, stamp_u8) or (None, None)."""
        ih, iw = mask.shape
        d = self._mask_brush_size
        if d < 1:
            return None, None
        radius = d / 2.0

        bx0 = max(0, int(min(x0, x1) - radius))
        by0 = max(0, int(min(y0, y1) - radius))
        bx1 = min(iw, int(max(x0, x1) + radius) + 1)
        by1 = min(ih, int(max(y0, y1) + radius) + 1)
        if bx0 >= bx1 or by0 >= by1:
            return None, None

        sdx = float(x1 - x0)
        sdy = float(y1 - y0)
        seg_len_sq = sdx * sdx + sdy * sdy

        if seg_len_sq < 0.5:
            return self._dab_mask(mask, x0, y0, erase=erase)

        yy, xx = np.mgrid[by0:by1, bx0:bx1]
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)

        t = ((xx - x0) * sdx + (yy - y0) * sdy) / seg_len_sq
        np.clip(t, 0.0, 1.0, out=t)
        cx = x0 + t * sdx
        cy = y0 + t * sdy
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        if self._mask_brush_hardness >= 1.0:
            alpha = (dist <= radius).astype(np.float32) * 255
        else:
            inner = radius * self._mask_brush_hardness
            alpha = np.clip(
                (radius - dist) / max(radius - inner, 0.001), 0, 1) * 255

        stamp_u8 = (alpha * self._mask_brush_flow).astype(np.uint8)
        if erase is None:
            erase = self._mask_eraser
        if erase:
            mask[by0:by1, bx0:bx1] = np.minimum(
                mask[by0:by1, bx0:bx1], 255 - stamp_u8)
        else:
            mask[by0:by1, bx0:bx1] = np.maximum(
                mask[by0:by1, bx0:bx1], stamp_u8)
        return (bx0, by0, bx1, by1), stamp_u8

    @staticmethod
    def _union_rect(a, b):
        if a is None:
            return b
        if b is None:
            return a
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return (min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1))

    def _begin_mask_erase(self):
        h, w = self._layer_stack.height, self._layer_stack.width
        if h == 0 or w == 0:
            return
        self._mask_erase_stroke = np.zeros((h, w), dtype=np.uint8)
        self._mask_erase_dirty = None

    # ------------------------------------------------------------------
    # Stroke buffer for brush painting
    # ------------------------------------------------------------------

    def _begin_stroke(self):
        h = self._layer_stack.height
        w = self._layer_stack.width
        if h == 0 or w == 0:
            return
        self._stroke_is_eraser = self._brush_eraser
        self._stroke_color = tuple(self.brush.color)
        if self._stroke_is_eraser:
            self._stroke_mask = None
            self._stroke_overlay = None
        else:
            self._stroke_mask = np.zeros((h, w), dtype=np.uint8)
            self._stroke_overlay = np.zeros((h, w, 4), dtype=np.uint8)
            r, g, b, _a = self._stroke_color
            self._stroke_overlay[:, :, 0] = r
            self._stroke_overlay[:, :, 1] = g
            self._stroke_overlay[:, :, 2] = b
            self.set_overlay_ref(self._stroke_overlay)

    def _end_stroke(self):
        layer = self._layer_stack.active_layer
        if layer is not None and self._stroke_mask is not None:
            composite_stroke(layer.image, self._stroke_mask, self._stroke_color)
            self._layer_stack.mark_layer_dirty(layer)
            self._composite = np.ascontiguousarray(
                self._layer_stack.composite())
            self.set_image(self._composite)
        self._stroke_mask = None
        self._stroke_color = None
        self._stroke_overlay = None
        self._mask_overlay = None
        self._update_overlay()

    # ------------------------------------------------------------------
    # Eraser
    # ------------------------------------------------------------------

    def _composite_rect_below(self, target_layer, dy0, dy1, dx0, dx1):
        cache = self._layer_stack.get_prefix_below_rect(
            target_layer, dx0, dy0, dx1, dy1)
        return cache.astype(np.float32)

    def _erase_dab(self, layer, cx: int, cy: int):
        stamp = self.brush._alpha_stamp
        sh, sw = stamp.shape[:2]
        ih, iw = layer.image.shape[:2]

        x0 = cx - sw // 2
        y0 = cy - sh // 2
        sx0, sy0 = max(0, -x0), max(0, -y0)
        sx1, sy1 = min(sw, iw - x0), min(sh, ih - y0)
        dx0, dy0b = max(0, x0), max(0, y0)
        dx1, dy1 = dx0 + (sx1 - sx0), dy0b + (sy1 - sy0)
        if dx0 >= dx1 or dy0b >= dy1:
            return None

        erase = (stamp[sy0:sy1, sx0:sx1]
                 * (self.brush.color[3] / 255.0)
                 * self.brush.flow)
        la = layer.image[dy0b:dy1, dx0:dx1, 3].astype(np.float32)
        layer.image[dy0b:dy1, dx0:dx1, 3] = np.clip(
            la * (1.0 - erase), 0, 255).astype(np.uint8)

        if self._composite is not None:
            below = self._composite_rect_below(layer, dy0b, dy1, dx0, dx1)
            above = layer.image[dy0b:dy1, dx0:dx1].astype(np.float32)
            sa = above[:, :, 3:4] / 255.0
            inv_sa = 1.0 - sa
            da = below[:, :, 3:4] / 255.0
            out_a = sa + da * inv_sa
            safe_a = np.maximum(out_a, 1.0 / 255.0)
            out_rgb = (above[:, :, :3] * sa + below[:, :, :3] * da * inv_sa) / safe_a
            self._composite[dy0b:dy1, dx0:dx1, :3] = np.clip(
                out_rgb, 0, 255).astype(np.uint8)
            self._composite[dy0b:dy1, dx0:dx1, 3:4] = np.clip(
                out_a * 255.0, 0, 255).astype(np.uint8)
        return (dx0, dy0b, dx1, dy1)

    def _erase_stroke_line(self, layer, x0: int, y0: int, x1: int, y1: int):
        ih, iw = layer.image.shape[:2]
        radius = self.brush.size / 2.0
        stamp = self.brush._alpha_stamp

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
            dirty = self._erase_dab(layer, x0, y0)
            self.set_image(self._composite)
            return dirty

        yy, xx = np.mgrid[by0:by1, bx0:bx1]
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)

        t = ((xx - x0) * sdx + (yy - y0) * sdy) / seg_len_sq
        np.clip(t, 0.0, 1.0, out=t)
        cx = x0 + t * sdx
        cy = y0 + t * sdy
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        if self.brush.hardness >= 1.0:
            erase = (dist <= radius).astype(np.float32)
        else:
            inner = radius * self.brush.hardness
            erase = np.clip(
                (radius - dist) / max(radius - inner, 0.001), 0.0, 1.0)

        erase *= (self.brush.color[3] / 255.0) * self.brush.flow
        la = layer.image[by0:by1, bx0:bx1, 3].astype(np.float32)
        layer.image[by0:by1, bx0:bx1, 3] = np.clip(
            la * (1.0 - erase), 0, 255).astype(np.uint8)

        if self._composite is not None:
            below = self._composite_rect_below(layer, by0, by1, bx0, bx1)
            above = layer.image[by0:by1, bx0:bx1].astype(np.float32)
            sa = above[:, :, 3:4] / 255.0
            inv_sa = 1.0 - sa
            da = below[:, :, 3:4] / 255.0
            out_a = sa + da * inv_sa
            safe_a = np.maximum(out_a, 1.0 / 255.0)
            out_rgb = (above[:, :, :3] * sa + below[:, :, :3] * da * inv_sa) / safe_a
            self._composite[by0:by1, bx0:bx1, :3] = np.clip(
                out_rgb, 0, 255).astype(np.uint8)
            self._composite[by0:by1, bx0:bx1, 3:4] = np.clip(
                out_a * 255.0, 0, 255).astype(np.uint8)

        self.set_image(self._composite)
        return (bx0, by0, bx1, by1)

    # ------------------------------------------------------------------
    # Mouse handlers (called from Canvas callbacks)
    # ------------------------------------------------------------------

    def _handle_mouse_down(self, ix: float, iy: float, button, mods: int = 0):
        from tcbase import Mods
        ix, iy = int(ix), int(iy)

        # Ctrl+LMB or RMB — eyedropper
        if button == MouseButton.RIGHT or (
                button == MouseButton.LEFT and (mods & Mods.CTRL.value)):
            self._pick_color(ix, iy)
            return

        if button == MouseButton.LEFT:
            layer = self._layer_stack.active_layer
            if layer is None:
                return

            # Patch rect mode
            if self._patch_rect_mode and isinstance(
                    layer, (DiffusionLayer, InstructLayer)):
                self._patch_rect_dragging = True
                self._patch_rect_start = (ix, iy)
                self._patch_rect_end = (ix, iy)
                return

            # Ref rect mode
            if self._ref_rect_mode and self._is_mask_layer_active():
                self._ref_rect_dragging = True
                self._ref_rect_start = (ix, iy)
                self._ref_rect_end = (ix, iy)
                return

            # Painting
            self._painting = True
            self._stroke_dirty_rect = None
            self._edit_layer = layer
            if self._is_mask_layer_active():
                self._edit_label = "Mask Stroke"
                self._edit_target = "mask"
                if self.on_edit_begin:
                    self.on_edit_begin(self._edit_label, layer, self._edit_target)
            elif self._brush_eraser:
                self._edit_label = "Erase Stroke"
                self._edit_target = "image"
                if self.on_edit_begin:
                    self.on_edit_begin(self._edit_label, layer, self._edit_target)
            else:
                self._edit_label = "Paint Stroke"
                self._edit_target = "image"
                if self.on_edit_begin:
                    self.on_edit_begin(self._edit_label, layer, self._edit_target)
            if self._is_mask_layer_active():
                if self._mask_eraser:
                    if self._mask_erase_stroke is None:
                        self._begin_mask_erase()
                    dirty, stamp = self._dab_mask(
                        self._mask_erase_stroke, ix, iy, erase=False)
                    self._mask_erase_dirty = self._union_rect(self._mask_erase_dirty, dirty)
                    self._stroke_dirty_rect = self._union_rect(self._stroke_dirty_rect, dirty)
                    if dirty is not None:
                        if self._show_mask:
                            x0, y0, x1, y1 = dirty
                            preview = np.minimum(
                                layer.mask[y0:y1, x0:x1],
                                255 - self._mask_erase_stroke[y0:y1, x0:x1])
                            self._update_mask_overlay_region_preview(layer, dirty, preview)
                        self._preview_mask_erase_region(layer, dirty)
                else:
                    dirty, _stamp = self._dab_mask(layer.mask, ix, iy)
                    self._stroke_dirty_rect = self._union_rect(self._stroke_dirty_rect, dirty)
                    self._update_mask_overlay_region(layer, dirty)
            else:
                self._begin_stroke()
                if self._stroke_is_eraser:
                    dirty = self._erase_dab(layer, ix, iy)
                    self._stroke_dirty_rect = self._union_rect(self._stroke_dirty_rect, dirty)
                    self.set_image(self._composite)
                elif self._stroke_mask is not None:
                    dirty = self.brush.dab_to_mask(self._stroke_mask, ix, iy)
                    self._stroke_dirty_rect = self._union_rect(self._stroke_dirty_rect, dirty)
                    self._update_stroke_region(dirty)
            self._last_paint_pos = (ix, iy)

    def _handle_mouse_move(self, ix: float, iy: float):
        ixi, iyi = int(ix), int(iy)

        if self._patch_rect_dragging:
            self._patch_rect_end = (ixi, iyi)
            return

        if self._ref_rect_dragging:
            self._ref_rect_end = (ixi, iyi)
            return

        if self._painting:
            layer = self._layer_stack.active_layer
            if layer is None:
                return
            if self._is_mask_layer_active():
                if self._mask_eraser:
                    if self._mask_erase_stroke is None:
                        self._begin_mask_erase()
                    if self._last_paint_pos:
                        lx, ly = self._last_paint_pos
                        dirty, _stamp = self._stroke_mask_line(
                            self._mask_erase_stroke, lx, ly, ixi, iyi, erase=False)
                    else:
                        dirty, _stamp = self._dab_mask(
                            self._mask_erase_stroke, ixi, iyi, erase=False)
                    self._mask_erase_dirty = self._union_rect(self._mask_erase_dirty, dirty)
                    if dirty is not None:
                        if self._show_mask:
                            x0, y0, x1, y1 = dirty
                            preview = np.minimum(
                                layer.mask[y0:y1, x0:x1],
                                255 - self._mask_erase_stroke[y0:y1, x0:x1])
                            self._update_mask_overlay_region_preview(layer, dirty, preview)
                        self._preview_mask_erase_region(layer, dirty)
                else:
                    if self._last_paint_pos:
                        lx, ly = self._last_paint_pos
                        dirty, _stamp = self._stroke_mask_line(layer.mask, lx, ly, ixi, iyi)
                    else:
                        dirty, _stamp = self._dab_mask(layer.mask, ixi, iyi)
                    self._stroke_dirty_rect = self._union_rect(self._stroke_dirty_rect, dirty)
                    self._update_mask_overlay_region(layer, dirty)
            else:
                if self._stroke_is_eraser:
                    if self._last_paint_pos:
                        lx, ly = self._last_paint_pos
                        dirty = self._erase_stroke_line(layer, lx, ly, ixi, iyi)
                    else:
                        dirty = self._erase_dab(layer, ixi, iyi)
                        self.set_image(self._composite)
                    self._stroke_dirty_rect = self._union_rect(self._stroke_dirty_rect, dirty)
                elif self._stroke_mask is not None:
                    if self._last_paint_pos:
                        lx, ly = self._last_paint_pos
                        dirty = self.brush.stroke_to_mask(
                            self._stroke_mask, lx, ly, ixi, iyi)
                    else:
                        dirty = self.brush.dab_to_mask(
                            self._stroke_mask, ixi, iyi)
                    self._stroke_dirty_rect = self._union_rect(self._stroke_dirty_rect, dirty)
                    self._update_stroke_region(dirty)
            self._last_paint_pos = (ixi, iyi)

        if self.on_mouse_moved:
            self.on_mouse_moved(ixi, iyi)

    def _handle_mouse_up(self, ix: float, iy: float):
        ixi, iyi = int(ix), int(iy)

        if self._patch_rect_dragging:
            sx, sy = self._patch_rect_start
            x0, y0 = min(sx, ixi), min(sy, iyi)
            x1, y1 = max(sx, ixi), max(sy, iyi)
            self._patch_rect_dragging = False
            self._patch_rect_start = None
            self._patch_rect_end = None
            self._patch_rect_mode = False
            self.cursor = ""
            if x1 - x0 > 2 and y1 - y0 > 2 and self.on_patch_rect_drawn:
                self.on_patch_rect_drawn(x0, y0, x1, y1)
            return

        if self._ref_rect_dragging:
            sx, sy = self._ref_rect_start
            x0, y0 = min(sx, ixi), min(sy, iyi)
            x1, y1 = max(sx, ixi), max(sy, iyi)
            self._ref_rect_dragging = False
            self._ref_rect_start = None
            self._ref_rect_end = None
            self._ref_rect_mode = False
            self.cursor = ""
            if x1 - x0 > 2 and y1 - y0 > 2 and self.on_ref_rect_drawn:
                self.on_ref_rect_drawn(x0, y0, x1, y1)
            return

        if self._painting:
            if self._mask_eraser and self._mask_erase_stroke is not None:
                layer = self._layer_stack.active_layer
                dirty = self._mask_erase_dirty
                if layer is not None and dirty is not None:
                    x0, y0, x1, y1 = dirty
                    layer.mask[y0:y1, x0:x1] = np.minimum(
                        layer.mask[y0:y1, x0:x1],
                        255 - self._mask_erase_stroke[y0:y1, x0:x1])
                    self._erase_layer_rect(
                        layer, dirty, self._mask_erase_stroke[y0:y1, x0:x1])
                    self._update_mask_overlay_region(layer, dirty)
                self._mask_erase_stroke = None
                self._mask_erase_dirty = None
            elif self._stroke_mask is not None:
                self._end_stroke()
            elif self._stroke_is_eraser:
                # Mark prefix caches dirty since layer.image was modified
                # incrementally during eraser dabs
                layer = self._layer_stack.active_layer
                if layer is not None:
                    self._layer_stack.mark_layer_dirty(layer)
                self.set_image(self._composite)
            self._mask_overlay = None
            self._update_overlay()
            if self.on_edit_end:
                self.on_edit_end(self._edit_layer, self._edit_target, self._stroke_dirty_rect)
            self._stroke_dirty_rect = None
            self._edit_label = None
            self._edit_target = None
            self._edit_layer = None
        self._painting = False
        self._last_paint_pos = None

    def _pick_color(self, ix: int, iy: int):
        if self._composite is None:
            return
        h, w = self._composite.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            r, g, b, a = self._composite[iy, ix]
            if self.on_color_picked:
                self.on_color_picked(int(r), int(g), int(b), int(a))

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------

    def on_key_down(self, event: KeyEvent) -> bool:
        if event.key == Key.RBRACKET:
            self.brush.set_size(self.brush.size + 5)
            return True
        elif event.key == Key.LBRACKET:
            self.brush.set_size(self.brush.size - 5)
            return True
        return super().on_key_down(event)

    # ------------------------------------------------------------------
    # Render overlay (rectangles via renderer)
    # ------------------------------------------------------------------

    def _render_overlay(self, canvas, renderer):
        layer = self._layer_stack.active_layer

        # IP-Adapter reference rectangle (blue)
        if self._show_ref_rect and isinstance(layer, DiffusionLayer):
            rect = None
            if (self._ref_rect_dragging
                    and self._ref_rect_start and self._ref_rect_end):
                rect = self._ref_rect_start + self._ref_rect_end
            elif layer.ip_adapter_rect:
                rect = layer.ip_adapter_rect
            if rect:
                ix0, iy0, ix1, iy1 = rect
                wx0, wy0 = canvas.image_to_widget(ix0, iy0)
                wx1, wy1 = canvas.image_to_widget(ix1, iy1)
                renderer.draw_rect(wx0, wy0, wx1 - wx0, wy1 - wy0,
                                   (0.2, 0.47, 1.0, 0.15))
                renderer.draw_rect_outline(wx0, wy0, wx1 - wx0, wy1 - wy0,
                                           (0.2, 0.47, 1.0, 0.8), 2.0)

        # Manual patch rectangle (green)
        if self._show_patch_rect and isinstance(
                layer, (DiffusionLayer, InstructLayer)):
            rect = None
            if (self._patch_rect_dragging
                    and self._patch_rect_start and self._patch_rect_end):
                rect = self._patch_rect_start + self._patch_rect_end
            elif layer.manual_patch_rect:
                rect = layer.manual_patch_rect
            if rect:
                ix0, iy0, ix1, iy1 = rect
                wx0, wy0 = canvas.image_to_widget(ix0, iy0)
                wx1, wy1 = canvas.image_to_widget(ix1, iy1)
                renderer.draw_rect(wx0, wy0, wx1 - wx0, wy1 - wy0,
                                   (0.2, 0.78, 0.31, 0.15))
                renderer.draw_rect_outline(wx0, wy0, wx1 - wx0, wy1 - wy0,
                                           (0.2, 0.78, 0.31, 0.8), 2.0)
