"""SelectionPanel — brush/eraser controls for document-level selection painting."""

from __future__ import annotations

from tcgui.widgets.group_box import GroupBox
from tcgui.widgets.hstack import HStack
from tcgui.widgets.checkbox import Checkbox
from tcgui.widgets.slider_edit import SliderEdit
from tcgui.widgets.units import px


class SelectionPanel(GroupBox):
    def __init__(self):
        super().__init__()
        self.title = "Selection"
        self.preferred_width = px(240)

        # Callbacks
        self.on_edit_mode_toggled: callable = None  # (active: bool)
        self.on_rect_mode_toggled: callable = None  # (active: bool)
        self.on_brush_changed: callable = None  # (size, hardness)
        self.on_eraser_toggled: callable = None  # (eraser: bool)
        self.on_show_selection_toggled: callable = None  # (show: bool)

        # Edit mode toggle
        self._edit_mode_cb = Checkbox()
        self._edit_mode_cb.text = "Edit Selection"
        self._edit_mode_cb.on_changed = self._on_edit_mode_changed
        self.add_child(self._edit_mode_cb)

        # Rect mode toggle
        self._rect_mode_cb = Checkbox()
        self._rect_mode_cb.text = "Rect"
        self._rect_mode_cb.on_changed = self._on_rect_mode_changed
        self.add_child(self._rect_mode_cb)

        # Eraser toggle
        self._eraser_cb = Checkbox()
        self._eraser_cb.text = "Eraser"
        self._eraser_cb.on_changed = self._on_eraser_changed
        self.add_child(self._eraser_cb)

        # Size slider
        self._size_slider = SliderEdit()
        self._size_slider.label = "Size"
        self._size_slider.tooltip = "Selection brush diameter in pixels"
        self._size_slider.min_value = 1
        self._size_slider.max_value = 500
        self._size_slider.value = 50
        self._size_slider.decimals = 0
        self._size_slider.on_changed = self._on_brush_params_changed
        self.add_child(self._size_slider)

        # Hardness slider
        self._hard_slider = SliderEdit()
        self._hard_slider.label = "Hardness"
        self._hard_slider.tooltip = "Selection brush edge softness (0 = soft, 1 = hard)"
        self._hard_slider.min_value = 0.0
        self._hard_slider.max_value = 1.0
        self._hard_slider.value = 0.4
        self._hard_slider.decimals = 2
        self._hard_slider.on_changed = self._on_brush_params_changed
        self.add_child(self._hard_slider)

        # Show selection toggle
        self._show_cb = Checkbox()
        self._show_cb.text = "Show Selection"
        self._show_cb.checked = True
        self._show_cb.on_changed = self._on_show_changed
        self.add_child(self._show_cb)

    def _on_edit_mode_changed(self, checked: bool):
        if self.on_edit_mode_toggled:
            self.on_edit_mode_toggled(checked)

    def _on_rect_mode_changed(self, checked: bool):
        if self.on_rect_mode_toggled:
            self.on_rect_mode_toggled(checked)

    def _on_eraser_changed(self, checked: bool):
        if self.on_eraser_toggled:
            self.on_eraser_toggled(checked)

    def _on_brush_params_changed(self, _value: float):
        if self.on_brush_changed:
            self.on_brush_changed(
                int(self._size_slider.value),
                self._hard_slider.value,
            )

    def _on_show_changed(self, checked: bool):
        if self.on_show_selection_toggled:
            self.on_show_selection_toggled(checked)
