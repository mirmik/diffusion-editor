"""EditorCanvas — extends tcgui Canvas with brush/mask painting and rect modes."""

from __future__ import annotations

import logging

import numpy as np

from tcbase import Key, MouseButton, Mods
from tcgui.widgets.canvas import Canvas
from tcgui.widgets.events import KeyEvent

from ..document.layer_stack import LayerStack
from ..document.layer import Layer
from ..document.tool import LamaTool, InstructTool
from .brush import Brush, BrushToolMode
from .canvas_composite import CanvasCompositeBridge
from .canvas_edit_session import CanvasEditSession
from .canvas_tool_context import CanvasToolContext
from .canvas_mask_erase import MaskEraseStrokeBuffer
from .canvas_mask_paint import CanvasMaskPainter
from .canvas_overlay import CanvasOverlayBridge
from .canvas_paint_stroke import PaintStrokeBuffer
from .canvas_rect_drag import CanvasRectDragController
from .canvas_selection_paint import CanvasSelectionPainter
from .canvas_smudge import SmudgeStrokeBuffer
from .canvas_tools import SelectionPaintTool, create_canvas_tools

logger = logging.getLogger(__name__)


class EditorCanvas(Canvas):
    """Zoomable image canvas with brush painting, mask painting, and rect tools."""

    def __init__(self, layer_stack: LayerStack, *,
                 gpu_compositing: bool = True,
                 ctx=None):
        super().__init__()
        self.background_color = (0.08, 0.08, 0.10, 1.0)
        self._layer_stack = layer_stack
        self._composite_bridge = CanvasCompositeBridge(
            layer_stack,
            gpu_compositing=gpu_compositing,
            graphics=ctx,
            set_image=self.set_image,
        )

        self.brush = Brush()
        self._brush_tool_mode = BrushToolMode.PAINT
        self._mask_painter = CanvasMaskPainter()
        self._overlay_bridge = CanvasOverlayBridge(
            layer_stack,
            set_overlay=self.set_overlay,
        )

        # Selection painting mode
        self._selection_mode = False
        self._selection_painter = CanvasSelectionPainter()

        self._rect_drags = CanvasRectDragController()

        # Stroke buffer (MAX blending during one stroke)
        self._paint_stroke = PaintStrokeBuffer()

        self._edit_session = CanvasEditSession()
        self._mask_erase_stroke = MaskEraseStrokeBuffer()
        self._smudge_stroke = SmudgeStrokeBuffer()
        self._tool_context = CanvasToolContext(
            layer_stack,
            self.brush,
            self._composite_bridge,
            self._overlay_bridge,
            self._paint_stroke,
            self._selection_painter,
            self._mask_painter,
            self._mask_erase_stroke,
            self._smudge_stroke,
        )
        self._stroke_tools = create_canvas_tools()
        self._active_stroke_tool = self._stroke_tools[self._brush_tool_mode]
        self._selection_tool = SelectionPaintTool()
        self._edit_tool = None

        # Callbacks
        self.on_mouse_moved: callable = None
        self.on_color_picked: callable = None
        self.on_patch_rect_drawn: callable = None
        self.on_selection_rect_drawn: callable = None
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

    @property
    def _composite(self) -> np.ndarray | None:
        return self._composite_bridge.composite

    @_composite.setter
    def _composite(self, composite: np.ndarray | None) -> None:
        self._composite_bridge.composite = composite

    @property
    def _gpu_compositing(self) -> bool:
        return self._composite_bridge.gpu_compositing

    @_gpu_compositing.setter
    def _gpu_compositing(self, enabled: bool) -> None:
        self._composite_bridge.gpu_compositing = enabled

    @property
    def _gpu_compositor(self):
        return self._composite_bridge.gpu_compositor

    @_gpu_compositor.setter
    def _gpu_compositor(self, compositor) -> None:
        self._composite_bridge.gpu_compositor = compositor

    @property
    def _composite_stale(self) -> bool:
        return self._composite_bridge.composite_stale

    @_composite_stale.setter
    def _composite_stale(self, stale: bool) -> None:
        self._composite_bridge.composite_stale = stale

    @property
    def _show_mask(self) -> bool:
        return self._overlay_bridge.show_mask

    def _on_stack_changed(self):
        self._composite_bridge.rebuild()
        self._update_composite()

    def _update_composite(self):
        composite = self._composite_bridge.update_composite()
        if composite is None and self._composite_bridge.using_gpu:
            # Keep Canvas.image_size in sync for fit/zoom math in GPU path.
            w, h = self._layer_stack.width, self._layer_stack.height
            if w > 0 and h > 0:
                if self._image_data is None or self._image_data.shape[:2] != (h, w):
                    self._image_data = np.empty((h, w, 4), dtype=np.uint8)
            else:
                self._image_data = None
        self._overlay_bridge.clear()
        self._update_overlay()

    def _update_overlay(self):
        self._overlay_bridge.rebuild()

    def get_composite(self) -> np.ndarray | None:
        return self._composite_bridge.get_composite()

    def get_composite_below(self, layer: Layer) -> np.ndarray | None:
        return self._composite_bridge.get_composite_below(layer)

    def view_center_image(self) -> tuple[int, int]:
        cx = self.x + self.width / 2
        cy = self.y + self.height / 2
        ix, iy = self.widget_to_image(cx, cy)
        return int(ix), int(iy)

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_mask_brush(self, size: int, hardness: float, flow: float = 1.0):
        self._mask_painter.set_brush(size, hardness, flow)
        if self._brush_tool_mode not in (BrushToolMode.MASK, BrushToolMode.MASK_ERASER):
            self.set_brush_tool(BrushToolMode.MASK)

    def set_mask_eraser(self, eraser: bool):
        self._mask_painter.set_eraser(eraser)
        self.set_brush_tool(
            BrushToolMode.MASK_ERASER if eraser else BrushToolMode.MASK)

    def set_brush_eraser(self, eraser: bool):
        self.set_brush_tool(BrushToolMode.ERASER if eraser else BrushToolMode.PAINT)

    def set_brush_tool(self, mode: BrushToolMode | str):
        self._brush_tool_mode = BrushToolMode(mode)
        self._mask_painter.set_eraser(
            self._brush_tool_mode == BrushToolMode.MASK_ERASER)
        self._active_stroke_tool = self._stroke_tools[self._brush_tool_mode]

    @property
    def brush_tool_mode(self) -> BrushToolMode:
        return self._brush_tool_mode

    def set_selection_mode(self, on: bool):
        self._selection_mode = on
        self.cursor = "cross" if on else ""

    def set_selection_brush(self, size: int, hardness: float, flow: float = 1.0):
        self._selection_painter.set_brush(size, hardness, flow)

    def set_selection_eraser(self, eraser: bool):
        self._selection_painter.set_eraser(eraser)

    def set_selection_rect_mode(self, on: bool):
        self._rect_drags.set_selection_rect_mode(on)
        self.cursor = "cross" if on else ""

    def set_show_mask(self, show: bool):
        self._overlay_bridge.show_mask = show
        self._overlay_bridge.clear()
        self._update_overlay()

    def set_show_selection(self, show: bool):
        self._overlay_bridge.show_selection = show
        self._update_overlay()

    def set_patch_rect_mode(self, on: bool):
        self._rect_drags.set_patch_rect_mode(on)
        self.cursor = "cross" if on else ""

    def set_show_patch_rect(self, show: bool):
        self._rect_drags.set_show_patch_rect(show)

    def _can_edit_layer(self, layer) -> bool:
        return (
            layer is not None
            and self._layer_stack.is_layer_visible_for_composition(layer)
        )

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

            if self._rect_drags.begin_selection_rect(ix, iy):
                return

            # Selection painting
            if self._selection_mode:
                self._begin_tool_edit(self._selection_tool, None, ix, iy)
                return

            if not self._can_edit_layer(layer):
                return

            if self._rect_drags.begin_patch_rect(ix, iy):
                return

            # Painting
            self._begin_tool_edit(self._active_stroke_tool, layer, ix, iy)

    def _handle_mouse_move(self, ix: float, iy: float):
        ixi, iyi = int(ix), int(iy)

        if self._rect_drags.move(ixi, iyi):
            return

        if self._edit_session.active:
            self._move_tool_edit(ixi, iyi)

        if self.on_mouse_moved:
            self.on_mouse_moved(ixi, iyi)

    def _handle_mouse_up(self, ix: float, iy: float):
        ixi, iyi = int(ix), int(iy)

        rect_result = self._rect_drags.finish(ixi, iyi)
        if rect_result.handled:
            self.cursor = ""
            if rect_result.rect is not None:
                if (
                        rect_result.target == "selection"
                        and self.on_selection_rect_drawn):
                    self.on_selection_rect_drawn(*rect_result.rect)
                elif rect_result.target == "patch" and self.on_patch_rect_drawn:
                    self.on_patch_rect_drawn(*rect_result.rect)
            return

        if self._edit_session.active:
            self._finish_tool_edit()

    def _begin_tool_edit(self, tool, layer, ix: int, iy: int):
        self._edit_tool = tool
        self._edit_session.begin(
            label=tool.label,
            target=tool.target,
            layer=layer,
            pos=(ix, iy),
        )
        if self.on_edit_begin:
            self.on_edit_begin(tool.label, layer, tool.target)
        dirty = tool.begin(self._tool_context, layer, ix, iy)
        self._edit_session.add_dirty(dirty)

    def _move_tool_edit(self, ix: int, iy: int):
        tool = self._edit_tool
        if tool is None:
            return
        layer = None
        if self._edit_session.target != "selection":
            layer = self._layer_stack.active_layer
            if layer is None:
                return
        dirty = tool.move(
            self._tool_context,
            layer,
            self._edit_session.last_pos,
            ix,
            iy,
        )
        self._edit_session.add_dirty(dirty)
        self._edit_session.move_to((ix, iy))

    def _finish_tool_edit(self):
        tool = self._edit_tool
        layer = None
        if self._edit_session.target != "selection":
            layer = self._layer_stack.active_layer
        if tool is not None:
            tool.end(self._tool_context, layer)
        if self._edit_session.target != "selection":
            self._overlay_bridge.clear()
            self._update_overlay()
        if self.on_edit_end:
            self.on_edit_end(
                self._edit_session.layer,
                self._edit_session.target,
                self._edit_session.dirty_rect,
            )
        self._edit_tool = None
        self._edit_session.clear()

    def _pick_color(self, ix: int, iy: int):
        if self._layer_stack.width == 0 or self._layer_stack.height == 0:
            return
        composite = self._layer_stack.composite()
        h, w = composite.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            r, g, b, a = composite[iy, ix]
            if self.on_color_picked:
                self.on_color_picked(int(r), int(g), int(b), int(a))

    # ------------------------------------------------------------------
    # Rendering (GPU compositing path)
    # ------------------------------------------------------------------

    def render(self, renderer):
        # Borrow the compositor's display tex directly. Since
        # EditorWindow threads the process-global tgfx2 context into
        # both the compositor and the renderer, the handle is valid on
        # both sides — no GL-id wrap, no raw-GL detour that only worked
        # on OpenGL. On Vulkan the raw-GL path returned 0 and left the
        # canvas black; this shared-device path draws a real image on
        # both backends.
        borrowed_display_tex = None
        if self._composite_bridge.using_gpu:
            # During mask erase preview, CanvasToolContext modifies the CPU
            # preview composite and calls set_image(); let base Canvas upload
            # it instead of overriding with the GPU display tex.
            in_mask_erase_preview = self._mask_erase_stroke.mask is not None

            if not in_mask_erase_preview:
                tex = self._composite_bridge.display_tex
                w, h = self._composite_bridge.display_size()
                if tex is not None and w > 0 and h > 0:
                    if self._image_texture is not None and self._image_tex_size is not None:
                        renderer.destroy_texture(self._image_texture)
                    self._image_texture = tex
                    self._image_tex_size = None  # non-owning borrow
                    self._image_dirty = False
                    if self._image_data is None or self._image_data.shape[:2] != (h, w):
                        self._image_data = np.empty((h, w, 4), dtype=np.uint8)
                    borrowed_display_tex = tex
            else:
                if self._image_texture is not None and self._image_tex_size is None:
                    self._image_texture = None
        try:
            super().render(renderer)
        finally:
            # Null out the borrowed reference so Canvas.dispose / texture
            # updates don't try to destroy a texture owned by the compositor.
            if borrowed_display_tex is not None and self._image_texture is borrowed_display_tex:
                self._image_texture = None

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

        # Active layer bounds. Drawn in canvas coordinates, so local-sized
        # layers and moved layers remain visually discoverable.
        if layer is not None and layer.width > 0 and layer.height > 0:
            ix0, iy0, ix1, iy1 = layer.bounds
            wx0, wy0 = canvas.image_to_widget(ix0, iy0)
            wx1, wy1 = canvas.image_to_widget(ix1, iy1)
            w = wx1 - wx0
            h = wy1 - wy0
            if w != 0 and h != 0:
                renderer.draw_rect_outline(wx0, wy0, w, h,
                                           (0.0, 0.0, 0.0, 0.85), 3.0)
                renderer.draw_rect_outline(wx0, wy0, w, h,
                                           (1.0, 1.0, 1.0, 0.95), 1.0)

        # Selection rectangle (cyan)
        rect = self._rect_drags.selection_preview_rect()
        if rect is not None:
            if rect:
                ix0, iy0, ix1, iy1 = rect
                wx0, wy0 = canvas.image_to_widget(ix0, iy0)
                wx1, wy1 = canvas.image_to_widget(ix1, iy1)
                renderer.draw_rect(wx0, wy0, wx1 - wx0, wy1 - wy0,
                                   (0.0, 0.8, 1.0, 0.12))
                renderer.draw_rect_outline(wx0, wy0, wx1 - wx0, wy1 - wy0,
                                           (0.0, 0.8, 1.0, 0.7), 2.0)

        # Patch rectangle (green outline)
        rect = self._rect_drags.patch_preview_rect(layer)
        if rect:
            ix0, iy0, ix1, iy1 = rect
            wx0, wy0 = canvas.image_to_widget(ix0, iy0)
            wx1, wy1 = canvas.image_to_widget(ix1, iy1)
            renderer.draw_rect_outline(wx0, wy0, wx1 - wx0, wy1 - wy0,
                                       (0.2, 0.78, 0.31, 0.8), 2.0)

    def dispose(self):
        """Release GPU resources held by the canvas compositor."""
        self._composite_bridge.dispose()
