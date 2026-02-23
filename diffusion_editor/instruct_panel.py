"""InstructPanel — tcgui panel for InstructPix2Pix settings."""

from __future__ import annotations

import random

from tcgui.widgets.scroll_area import ScrollArea
from tcgui.widgets.vstack import VStack
from tcgui.widgets.hstack import HStack
from tcgui.widgets.group_box import GroupBox
from tcgui.widgets.label import Label
from tcgui.widgets.button import Button
from tcgui.widgets.checkbox import Checkbox
from tcgui.widgets.text_area import TextArea
from tcgui.widgets.text_input import TextInput
from tcgui.widgets.slider_edit import SliderEdit
from tcgui.widgets.units import px, pct


class InstructPanel(ScrollArea):
    def __init__(self):
        super().__init__()
        self.preferred_width = px(280)

        # Callbacks
        self.on_load_model: callable = None
        self.on_apply: callable = None
        self.on_new_seed: callable = None
        self.on_clear_mask: callable = None
        self.on_mask_brush_changed: callable = None  # (size, hardness)
        self.on_mask_eraser_toggled: callable = None  # (bool)
        self.on_show_mask_toggled: callable = None  # (bool)
        self.on_draw_patch_toggled: callable = None  # (bool)
        self.on_clear_patch: callable = None

        content = VStack()
        content.spacing = 8
        content.preferred_width = pct(100)

        # --- Model ---
        model_group = GroupBox()
        model_group.title = "Model"

        self._load_btn = Button()
        self._load_btn.text = "Load InstructPix2Pix"
        self._load_btn.preferred_width = pct(100)
        self._load_btn.on_click = lambda: (
            self.on_load_model and self.on_load_model())
        model_group.add_child(self._load_btn)

        self._model_status = Label()
        self._model_status.text = "Not loaded"
        self._model_status.font_size = 11
        self._model_status.color = (0.5, 0.5, 0.5, 1.0)
        model_group.add_child(self._model_status)

        content.add_child(model_group)

        # --- Instruction ---
        instr_group = GroupBox()
        instr_group.title = "Instruction"

        self._instruction = TextArea()
        self._instruction.placeholder = "make it snowy"
        self._instruction.preferred_width = pct(100)
        self._instruction.preferred_height = px(50)
        instr_group.add_child(self._instruction)

        content.add_child(instr_group)

        # --- Parameters ---
        params_group = GroupBox()
        params_group.title = "Parameters"

        self._img_guidance_slider = SliderEdit()
        self._img_guidance_slider.label = "Image Guidance"
        self._img_guidance_slider.min_value = 1.0
        self._img_guidance_slider.max_value = 3.0
        self._img_guidance_slider.value = 1.5
        self._img_guidance_slider.decimals = 1
        params_group.add_child(self._img_guidance_slider)

        self._cfg_slider = SliderEdit()
        self._cfg_slider.label = "CFG Scale"
        self._cfg_slider.min_value = 1.0
        self._cfg_slider.max_value = 20.0
        self._cfg_slider.value = 7.0
        self._cfg_slider.decimals = 1
        params_group.add_child(self._cfg_slider)

        self._steps_slider = SliderEdit()
        self._steps_slider.label = "Steps"
        self._steps_slider.min_value = 1
        self._steps_slider.max_value = 50
        self._steps_slider.value = 20
        self._steps_slider.decimals = 0
        params_group.add_child(self._steps_slider)

        # Seed row
        seed_row = HStack()
        seed_row.spacing = 4

        self._seed_edit = TextInput()
        self._seed_edit.text = str(random.randint(0, 2**32 - 1))
        self._seed_edit.placeholder = "seed"
        self._seed_edit.preferred_width = px(160)
        seed_row.add_child(self._seed_edit)

        seed_rnd = Button()
        seed_rnd.text = "Rnd"
        seed_rnd.preferred_width = px(40)
        seed_rnd.on_click = lambda: setattr(
            self._seed_edit, 'text', str(random.randint(0, 2**32 - 1)))
        seed_row.add_child(seed_rnd)

        params_group.add_child(seed_row)
        content.add_child(params_group)

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
        content.add_child(mask_group)

        # --- Instruct Layer ---
        self._layer_group = GroupBox()
        self._layer_group.title = "Instruct Layer"
        self._layer_group.visible = False

        self._layer_info = Label()
        self._layer_info.text = "No instruct layer selected"
        self._layer_info.font_size = 11
        self._layer_info.color = (0.7, 0.7, 0.7, 1.0)
        self._layer_group.add_child(self._layer_info)

        action_row = HStack()
        action_row.spacing = 4

        apply_btn = Button()
        apply_btn.text = "Apply"
        apply_btn.preferred_width = px(80)
        apply_btn.on_click = lambda: (self.on_apply and self.on_apply())
        action_row.add_child(apply_btn)

        new_seed_btn = Button()
        new_seed_btn.text = "New Seed"
        new_seed_btn.preferred_width = px(80)
        new_seed_btn.on_click = lambda: (self.on_new_seed and self.on_new_seed())
        action_row.add_child(new_seed_btn)

        self._layer_group.add_child(action_row)

        clear_mask_btn = Button()
        clear_mask_btn.text = "Clear Mask"
        clear_mask_btn.preferred_width = pct(100)
        clear_mask_btn.on_click = lambda: (self.on_clear_mask and self.on_clear_mask())
        self._layer_group.add_child(clear_mask_btn)

        patch_row = HStack()
        patch_row.spacing = 4

        self._draw_patch_cb = Checkbox()
        self._draw_patch_cb.text = "Draw Patch"
        self._draw_patch_cb.on_changed = lambda v: (
            self.on_draw_patch_toggled and self.on_draw_patch_toggled(v))
        patch_row.add_child(self._draw_patch_cb)

        clear_patch_btn = Button()
        clear_patch_btn.text = "Clear Patch"
        clear_patch_btn.preferred_width = px(80)
        clear_patch_btn.on_click = lambda: (self.on_clear_patch and self.on_clear_patch())
        patch_row.add_child(clear_patch_btn)

        self._layer_group.add_child(patch_row)
        content.add_child(self._layer_group)

        self.add_child(content)

    def _on_mask_brush_cb(self, _value=None):
        size = int(self._mask_size_slider.value)
        hardness = self._mask_hardness_slider.value
        if self.on_mask_brush_changed:
            self.on_mask_brush_changed(size, hardness)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def instruction(self) -> str:
        return self._instruction.text

    @property
    def image_guidance_scale(self) -> float:
        return self._img_guidance_slider.value

    @property
    def guidance_scale(self) -> float:
        return self._cfg_slider.value

    @property
    def steps(self) -> int:
        return int(self._steps_slider.value)

    @property
    def seed(self) -> int:
        try:
            return int(self._seed_edit.text)
        except ValueError:
            return -1

    def set_seed(self, seed: int):
        self._seed_edit.text = str(seed)

    # ------------------------------------------------------------------
    # External updates
    # ------------------------------------------------------------------

    def on_model_loaded(self):
        self._model_status.text = "Loaded: instruct-pix2pix"

    def on_model_load_error(self, error: str):
        self._model_status.text = f"Error: {error[:60]}"

    def show_instruct_layer(self, layer):
        self._layer_group.visible = True
        self._instruction.text = layer.instruction
        self._img_guidance_slider.value = layer.image_guidance_scale
        self._cfg_slider.value = layer.guidance_scale
        self._steps_slider.value = layer.steps
        self._seed_edit.text = str(layer.seed)

        mask_status = "has mask" if layer.has_mask() else "no mask"
        if layer.manual_patch_rect:
            r = layer.manual_patch_rect
            pw, ph = r[2] - r[0], r[3] - r[1]
            patch_info = f"manual {pw}x{ph}"
        else:
            patch_info = f"auto {layer.patch_w}x{layer.patch_h}"

        self._layer_info.text = (
            f"instruction: {layer.instruction[:60]}\n"
            f"patch: {patch_info}  mask: {mask_status}"
        )

    def clear_instruct_layer(self):
        self._layer_group.visible = False
        self._layer_info.text = "No instruct layer selected"
