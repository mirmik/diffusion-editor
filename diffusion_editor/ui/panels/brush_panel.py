"""BrushPanel — tcgui panel for brush settings and stroke tools."""

from __future__ import annotations

from tcgui.widgets.group_box import GroupBox
from tcgui.widgets.hstack import HStack
from tcgui.widgets.button import Button
from tcgui.widgets.checkbox import Checkbox
from tcgui.widgets.slider_edit import SliderEdit
from tcgui.widgets.color_dialog import ColorDialog
from tcgui.widgets.units import px

from ...canvas.brush import Brush, BrushToolMode


class BrushPanel(GroupBox):
    def __init__(self, brush: Brush):
        super().__init__()
        self.title = "Brush"
        self.preferred_width = px(240)

        self._brush = brush

        self._tool_mode = BrushToolMode.PAINT

        # Callbacks
        self.on_eraser_toggled: callable = None
        self.on_tool_changed: callable = None
        self.on_draw_patch_toggled: callable = None  # (bool)
        self.on_show_patch_toggled: callable = None  # (bool)
        self.on_clear_patch: callable = None

        # Raster tool row
        raster_row = HStack()
        raster_row.spacing = 4

        self._paint_btn = self._make_tool_button("Paint", BrushToolMode.PAINT)
        self._eraser_btn = self._make_tool_button("Erase", BrushToolMode.ERASER)
        self._smudge_btn = self._make_tool_button("Smudge", BrushToolMode.SMUDGE)
        raster_row.add_child(self._paint_btn)
        raster_row.add_child(self._eraser_btn)
        raster_row.add_child(self._smudge_btn)
        self.add_child(raster_row)

        # Mask tool row
        mask_row = HStack()
        mask_row.spacing = 4
        self._mask_btn = self._make_tool_button("Mask", BrushToolMode.MASK)
        self._mask_eraser_btn = self._make_tool_button(
            "Unmask", BrushToolMode.MASK_ERASER)
        self._mask_btn.preferred_width = px(106)
        self._mask_eraser_btn.preferred_width = px(106)
        mask_row.add_child(self._mask_btn)
        mask_row.add_child(self._mask_eraser_btn)
        self.add_child(mask_row)

        # Move tool row
        move_row = HStack()
        move_row.spacing = 4
        self._move_btn = self._make_tool_button("Move", BrushToolMode.MOVE)
        self._move_btn.preferred_width = px(218)
        move_row.add_child(self._move_btn)
        self.add_child(move_row)

        # Patch row
        patch_row = HStack()
        patch_row.spacing = 4

        self._draw_patch_cb = Checkbox()
        self._draw_patch_cb.text = "Draw Patch"
        self._draw_patch_cb.on_changed = lambda v: (
            self.on_draw_patch_toggled and self.on_draw_patch_toggled(v))
        patch_row.add_child(self._draw_patch_cb)

        self._show_patch_cb = Checkbox()
        self._show_patch_cb.text = "Show"
        self._show_patch_cb.checked = True
        self._show_patch_cb.on_changed = lambda v: (
            self.on_show_patch_toggled and self.on_show_patch_toggled(v))
        patch_row.add_child(self._show_patch_cb)

        clear_patch_btn = Button()
        clear_patch_btn.text = "Clear"
        clear_patch_btn.preferred_width = px(50)
        clear_patch_btn.on_click = lambda: (
            self.on_clear_patch and self.on_clear_patch())
        patch_row.add_child(clear_patch_btn)

        self.add_child(patch_row)

        # Color row
        self._color_row = HStack()
        self._color_row.spacing = 8

        self._color_btn = Button()
        self._color_btn.text = "Color"
        self._color_btn.preferred_width = px(60)
        self._color_btn.on_click = self._pick_color
        self._update_color_btn()
        self._color_row.add_child(self._color_btn)

        self.add_child(self._color_row)

        # Size slider
        self._size_slider = SliderEdit()
        self._size_slider.label = "Size"
        self._size_slider.tooltip = "Brush diameter in pixels"
        self._size_slider.min_value = 1
        self._size_slider.max_value = 500
        self._size_slider.value = brush.size
        self._size_slider.decimals = 0
        self._size_slider.on_changed = self._on_size_changed
        self.add_child(self._size_slider)

        # Hardness slider
        self._hard_slider = SliderEdit()
        self._hard_slider.label = "Hardness"
        self._hard_slider.tooltip = "Brush edge softness (0 = soft, 1 = hard)"
        self._hard_slider.min_value = 0.0
        self._hard_slider.max_value = 1.0
        self._hard_slider.value = brush.hardness
        self._hard_slider.decimals = 2
        self._hard_slider.on_changed = self._on_hard_changed
        self.add_child(self._hard_slider)

        # Flow slider
        self._flow_slider = SliderEdit()
        self._flow_slider.label = "Flow"
        self._flow_slider.tooltip = "Paint opacity per stroke (0 = transparent, 1 = opaque)"
        self._flow_slider.min_value = 0.0
        self._flow_slider.max_value = 1.0
        self._flow_slider.value = brush.flow
        self._flow_slider.decimals = 2
        self._flow_slider.on_changed = self._on_flow_changed
        self.add_child(self._flow_slider)

        self._sync_tool_buttons()

    def _update_color_btn(self):
        r, g, b, _ = self._brush.color
        self._color_btn.background_color = (r / 255, g / 255, b / 255, 1.0)

    def _pick_color(self):
        r, g, b, a = self._brush.color
        ColorDialog.pick_color(
            self._ui, initial=(r, g, b, a),
            on_result=self._on_color_result)

    def _on_color_result(self, color: tuple | None):
        if color is not None:
            r, g, b, a = color
            self._brush.set_color(r, g, b, a)
            self._update_color_btn()

    def _make_tool_button(self, text: str, mode: BrushToolMode) -> Button:
        btn = Button()
        btn.text = text
        btn.checkable = True
        btn.preferred_width = px(70)
        btn.on_click = lambda: self._set_tool_mode(mode)
        return btn

    def _sync_tool_buttons(self):
        self._paint_btn.checked = self._tool_mode == BrushToolMode.PAINT
        self._eraser_btn.checked = self._tool_mode == BrushToolMode.ERASER
        self._smudge_btn.checked = self._tool_mode == BrushToolMode.SMUDGE
        self._mask_btn.checked = self._tool_mode == BrushToolMode.MASK
        self._mask_eraser_btn.checked = self._tool_mode == BrushToolMode.MASK_ERASER
        self._move_btn.checked = self._tool_mode == BrushToolMode.MOVE
        # Hide brush-only controls when in Move mode
        is_move = self._tool_mode == BrushToolMode.MOVE
        self._size_slider.visible = not is_move
        self._hard_slider.visible = not is_move
        self._flow_slider.visible = not is_move
        self._color_row.visible = not is_move
        self.title = "Move" if is_move else "Brush"

    def _set_tool_mode(self, mode: BrushToolMode):
        self._tool_mode = mode
        self._sync_tool_buttons()
        if self.on_tool_changed:
            self.on_tool_changed(mode)
        elif self.on_eraser_toggled:
            self.on_eraser_toggled(mode == BrushToolMode.ERASER)

    def set_tool_mode(self, mode: BrushToolMode | str, *, emit: bool = False):
        self._tool_mode = BrushToolMode(mode)
        self._sync_tool_buttons()
        if emit and self.on_tool_changed:
            self.on_tool_changed(self._tool_mode)

    def set_draw_patch_checked(self, value: bool):
        self._draw_patch_cb.checked = value

    def _on_size_changed(self, value: float):
        self._brush.set_size(int(value))

    def _on_hard_changed(self, value: float):
        self._brush.set_hardness(value)

    def _on_flow_changed(self, value: float):
        self._brush.set_flow(value)

    def sync_from_brush(self):
        self._size_slider.value = self._brush.size
        self._hard_slider.value = self._brush.hardness
        self._flow_slider.value = self._brush.flow
        self._update_color_btn()
