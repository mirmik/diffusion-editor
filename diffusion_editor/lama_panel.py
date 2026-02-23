"""LamaPanel — tcgui panel for LaMa inpainting settings."""

from __future__ import annotations

from tcgui.widgets.scroll_area import ScrollArea
from tcgui.widgets.vstack import VStack
from tcgui.widgets.hstack import HStack
from tcgui.widgets.group_box import GroupBox
from tcgui.widgets.label import Label
from tcgui.widgets.button import Button
from tcgui.widgets.checkbox import Checkbox
from tcgui.widgets.slider_edit import SliderEdit
from tcgui.widgets.units import px, pct


class LamaPanel(ScrollArea):
    def __init__(self):
        super().__init__()
        self.preferred_width = px(250)

        # Callbacks
        self.on_remove: callable = None
        self.on_clear_mask: callable = None
        self.on_mask_brush_changed: callable = None  # (size, hardness)
        self.on_mask_eraser_toggled: callable = None  # (bool)
        self.on_show_mask_toggled: callable = None  # (bool)
        self.on_select_background: callable = None

        content = VStack()
        content.spacing = 8
        content.preferred_width = pct(100)

        # --- Mask Brush ---
        mask_group = GroupBox()
        mask_group.title = "Mask Brush"

        self._mask_size_slider = SliderEdit()
        self._mask_size_slider.label = "Size"
        self._mask_size_slider.min_value = 1
        self._mask_size_slider.max_value = 500
        self._mask_size_slider.value = 50
        self._mask_size_slider.decimals = 0
        self._mask_size_slider.on_change = self._on_mask_brush_cb
        mask_group.add_child(self._mask_size_slider)

        self._mask_hardness_slider = SliderEdit()
        self._mask_hardness_slider.label = "Hardness"
        self._mask_hardness_slider.min_value = 0.0
        self._mask_hardness_slider.max_value = 1.0
        self._mask_hardness_slider.value = 0.40
        self._mask_hardness_slider.decimals = 2
        self._mask_hardness_slider.on_change = self._on_mask_brush_cb
        mask_group.add_child(self._mask_hardness_slider)

        mask_btn_row = HStack()
        mask_btn_row.spacing = 4

        self._mask_eraser_cb = Checkbox()
        self._mask_eraser_cb.text = "Eraser"
        self._mask_eraser_cb.on_changed = lambda v: (
            self.on_mask_eraser_toggled and self.on_mask_eraser_toggled(v))
        mask_btn_row.add_child(self._mask_eraser_cb)

        self._show_mask_cb = Checkbox()
        self._show_mask_cb.text = "Show Mask"
        self._show_mask_cb.checked = True
        self._show_mask_cb.on_changed = lambda v: (
            self.on_show_mask_toggled and self.on_show_mask_toggled(v))
        mask_btn_row.add_child(self._show_mask_cb)

        mask_group.add_child(mask_btn_row)

        self._select_bg_btn = Button()
        self._select_bg_btn.text = "Select Background"
        self._select_bg_btn.preferred_width = pct(100)
        self._select_bg_btn.on_click = lambda: (
            self.on_select_background and self.on_select_background())
        mask_group.add_child(self._select_bg_btn)

        content.add_child(mask_group)

        # --- LaMa Layer ---
        action_group = GroupBox()
        action_group.title = "LaMa Layer"

        self._layer_info = Label()
        self._layer_info.text = "No LaMa layer selected"
        self._layer_info.font_size = 11
        self._layer_info.color = (0.7, 0.7, 0.7, 1.0)
        action_group.add_child(self._layer_info)

        remove_btn = Button()
        remove_btn.text = "Remove Objects"
        remove_btn.preferred_width = pct(100)
        remove_btn.on_click = lambda: (self.on_remove and self.on_remove())
        action_group.add_child(remove_btn)

        clear_mask_btn = Button()
        clear_mask_btn.text = "Clear Mask"
        clear_mask_btn.preferred_width = pct(100)
        clear_mask_btn.on_click = lambda: (self.on_clear_mask and self.on_clear_mask())
        action_group.add_child(clear_mask_btn)

        content.add_child(action_group)
        self.add_child(content)

    def _on_mask_brush_cb(self, _value=None):
        size = int(self._mask_size_slider.value)
        hardness = self._mask_hardness_slider.value
        if self.on_mask_brush_changed:
            self.on_mask_brush_changed(size, hardness)

    def show_lama_layer(self, layer):
        mask_status = "has mask" if layer.has_mask() else "no mask"
        self._layer_info.text = (
            f"patch: ({layer.patch_x},{layer.patch_y}) "
            f"{layer.patch_w}x{layer.patch_h}\n"
            f"mask: {mask_status}"
        )

    def clear_lama_layer(self):
        self._layer_info.text = "No LaMa layer selected"
