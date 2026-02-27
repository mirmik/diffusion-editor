"""BrushPanel — tcgui panel for brush settings (color, size, hardness, eraser)."""

from __future__ import annotations

from tcgui.widgets.group_box import GroupBox
from tcgui.widgets.hstack import HStack
from tcgui.widgets.button import Button
from tcgui.widgets.checkbox import Checkbox
from tcgui.widgets.slider_edit import SliderEdit
from tcgui.widgets.color_dialog import ColorDialog
from tcgui.widgets.units import px

from .brush import Brush


class BrushPanel(GroupBox):
    def __init__(self, brush: Brush):
        super().__init__()
        self.title = "Brush"
        self.preferred_width = px(240)

        self._brush = brush

        # Callbacks
        self.on_eraser_toggled: callable = None

        # Color + eraser row
        row = HStack()
        row.spacing = 8

        self._color_btn = Button()
        self._color_btn.text = "Color"
        self._color_btn.preferred_width = px(60)
        self._color_btn.on_click = self._pick_color
        self._update_color_btn()
        row.add_child(self._color_btn)

        self._eraser_cb = Checkbox()
        self._eraser_cb.text = "Eraser"
        self._eraser_cb.on_changed = self._on_eraser_changed
        row.add_child(self._eraser_cb)

        self.add_child(row)

        # Size slider
        self._size_slider = SliderEdit()
        self._size_slider.label = "Size"
        self._size_slider.min_value = 1
        self._size_slider.max_value = 500
        self._size_slider.value = brush.size
        self._size_slider.decimals = 0
        self._size_slider.on_changed = self._on_size_changed
        self.add_child(self._size_slider)

        # Hardness slider
        self._hard_slider = SliderEdit()
        self._hard_slider.label = "Hardness"
        self._hard_slider.min_value = 0.0
        self._hard_slider.max_value = 1.0
        self._hard_slider.value = brush.hardness
        self._hard_slider.decimals = 2
        self._hard_slider.on_changed = self._on_hard_changed
        self.add_child(self._hard_slider)

        # Flow slider
        self._flow_slider = SliderEdit()
        self._flow_slider.label = "Flow"
        self._flow_slider.min_value = 0.0
        self._flow_slider.max_value = 1.0
        self._flow_slider.value = brush.flow
        self._flow_slider.decimals = 2
        self._flow_slider.on_changed = self._on_flow_changed
        self.add_child(self._flow_slider)

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

    def _on_eraser_changed(self, checked: bool):
        if self.on_eraser_toggled:
            self.on_eraser_toggled(checked)

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
