"""DiffusionPanel — tcgui panel for Stable Diffusion settings."""

from __future__ import annotations

import os
import random

from tcgui.widgets.scroll_area import ScrollArea
from tcgui.widgets.vstack import VStack
from tcgui.widgets.hstack import HStack
from tcgui.widgets.group_box import GroupBox
from tcgui.widgets.label import Label
from tcgui.widgets.button import Button
from tcgui.widgets.checkbox import Checkbox
from tcgui.widgets.combo_box import ComboBox
from tcgui.widgets.text_area import TextArea
from tcgui.widgets.text_input import TextInput
from tcgui.widgets.slider_edit import SliderEdit
from tcgui.widgets.units import px, pct

MODELS_DIR = os.path.expanduser(
    "~/soft/stable-diffusion-webui-forge/models/Stable-diffusion/"
)


class DiffusionPanel(ScrollArea):
    def __init__(self):
        super().__init__()
        self.preferred_width = px(280)

        # Callbacks
        self.on_load_model: callable = None       # (path, prediction_type)
        self.on_regenerate: callable = None
        self.on_new_seed: callable = None
        self.on_clear_mask: callable = None
        self.on_mask_brush_changed: callable = None  # (size, hardness, flow)
        self.on_load_ip_adapter: callable = None
        self.on_draw_rect_toggled: callable = None  # (bool)
        self.on_show_rect_toggled: callable = None  # (bool)
        self.on_clear_rect: callable = None
        self.on_select_background: callable = None
        self.on_draw_patch_toggled: callable = None  # (bool)
        self.on_clear_patch: callable = None
        self.on_mask_eraser_toggled: callable = None  # (bool)
        self.on_show_mask_toggled: callable = None  # (bool)

        content = VStack()
        content.spacing = 8
        content.preferred_width = pct(100)

        # --- Model ---
        model_group = GroupBox()
        model_group.title = "Model"

        self._model_combo = ComboBox()
        self._model_combo.preferred_width = pct(100)
        self._scan_models()
        model_group.add_child(self._model_combo)

        self._prediction_combo = ComboBox()
        self._prediction_combo.items = ["Auto", "epsilon", "v_prediction"]
        self._prediction_combo.selected_index = 0
        self._prediction_combo.preferred_width = pct(100)
        model_group.add_child(self._prediction_combo)

        self._load_btn = Button()
        self._load_btn.text = "Load Model"
        self._load_btn.preferred_width = pct(100)
        self._load_btn.on_click = self._on_load
        model_group.add_child(self._load_btn)

        self._model_status = Label()
        self._model_status.text = "No model loaded"
        self._model_status.font_size = 11
        self._model_status.color = (0.5, 0.5, 0.5, 1.0)
        model_group.add_child(self._model_status)

        self._model_diag = Label()
        self._model_diag.text = ""
        self._model_diag.font_size = 11
        self._model_diag.color = (0.5, 0.5, 0.5, 1.0)
        model_group.add_child(self._model_diag)

        content.add_child(model_group)

        # --- Prompt ---
        prompt_group = GroupBox()
        prompt_group.title = "Prompt"

        pos_lbl = Label()
        pos_lbl.text = "Positive:"
        pos_lbl.font_size = 12
        prompt_group.add_child(pos_lbl)

        self._prompt = TextArea()
        self._prompt.placeholder = "masterpiece, best quality"
        self._prompt.preferred_width = pct(100)
        self._prompt.preferred_height = px(73)
        prompt_group.add_child(self._prompt)

        neg_lbl = Label()
        neg_lbl.text = "Negative:"
        neg_lbl.font_size = 12
        prompt_group.add_child(neg_lbl)

        self._negative_prompt = TextArea()
        self._negative_prompt.placeholder = "worst quality, blurry"
        self._negative_prompt.preferred_width = pct(100)
        self._negative_prompt.preferred_height = px(73)
        prompt_group.add_child(self._negative_prompt)

        content.add_child(prompt_group)

        # --- Parameters ---
        params_group = GroupBox()
        params_group.title = "Parameters"

        mode_lbl = Label()
        mode_lbl.text = "Mode:"
        mode_lbl.font_size = 12
        params_group.add_child(mode_lbl)

        self._mode_combo = ComboBox()
        self._mode_combo.items = ["txt2img", "img2img", "inpaint"]
        self._mode_combo.selected_index = 2
        self._mode_combo.preferred_width = pct(100)
        params_group.add_child(self._mode_combo)

        mc_lbl = Label()
        mc_lbl.text = "Masked Content:"
        mc_lbl.font_size = 12
        params_group.add_child(mc_lbl)

        self._masked_content_combo = ComboBox()
        self._masked_content_combo.items = ["original", "fill", "latent noise", "latent nothing"]
        self._masked_content_combo.selected_index = 0
        self._masked_content_combo.preferred_width = pct(100)
        params_group.add_child(self._masked_content_combo)

        # Strength
        self._strength_slider = SliderEdit()
        self._strength_slider.label = "Strength"
        self._strength_slider.min_value = 0.0
        self._strength_slider.max_value = 1.0
        self._strength_slider.value = 0.30
        self._strength_slider.decimals = 2
        params_group.add_child(self._strength_slider)

        # Steps
        self._steps_slider = SliderEdit()
        self._steps_slider.label = "Steps"
        self._steps_slider.min_value = 1
        self._steps_slider.max_value = 50
        self._steps_slider.value = 20
        self._steps_slider.decimals = 0
        params_group.add_child(self._steps_slider)

        # CFG Scale
        self._cfg_slider = SliderEdit()
        self._cfg_slider.label = "CFG Scale"
        self._cfg_slider.min_value = 1.0
        self._cfg_slider.max_value = 20.0
        self._cfg_slider.value = 7.0
        self._cfg_slider.decimals = 1
        params_group.add_child(self._cfg_slider)

        # Resize checkbox
        self._resize_cb = Checkbox()
        self._resize_cb.text = "Resize to 1024"
        params_group.add_child(self._resize_cb)

        content.add_child(params_group)

        # --- Seed ---
        seed_group = GroupBox()
        seed_group.title = "Seed"

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
        seed_rnd.on_click = lambda: self._set_random_seed()
        seed_row.add_child(seed_rnd)

        seed_group.add_child(seed_row)
        content.add_child(seed_group)

        # --- Mask Brush ---
        mask_group = GroupBox()
        mask_group.title = "Mask Brush"

        self._mask_size_slider = SliderEdit()
        self._mask_size_slider.label = "Size"
        self._mask_size_slider.min_value = 1
        self._mask_size_slider.max_value = 500
        self._mask_size_slider.value = 50
        self._mask_size_slider.decimals = 0
        self._mask_size_slider.on_changed = self._on_mask_brush_cb
        mask_group.add_child(self._mask_size_slider)

        self._mask_hardness_slider = SliderEdit()
        self._mask_hardness_slider.label = "Hardness"
        self._mask_hardness_slider.min_value = 0.0
        self._mask_hardness_slider.max_value = 1.0
        self._mask_hardness_slider.value = 0.40
        self._mask_hardness_slider.decimals = 2
        self._mask_hardness_slider.on_changed = self._on_mask_brush_cb
        mask_group.add_child(self._mask_hardness_slider)

        self._mask_flow_slider = SliderEdit()
        self._mask_flow_slider.label = "Flow"
        self._mask_flow_slider.min_value = 0.0
        self._mask_flow_slider.max_value = 1.0
        self._mask_flow_slider.value = 1.0
        self._mask_flow_slider.decimals = 2
        self._mask_flow_slider.on_changed = self._on_mask_brush_cb
        mask_group.add_child(self._mask_flow_slider)

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

        # --- IP-Adapter ---
        ip_group = GroupBox()
        ip_group.title = "IP-Adapter"

        self._load_ip_btn = Button()
        self._load_ip_btn.text = "Load IP-Adapter"
        self._load_ip_btn.preferred_width = pct(100)
        self._load_ip_btn.on_click = lambda: (
            self.on_load_ip_adapter and self.on_load_ip_adapter())
        ip_group.add_child(self._load_ip_btn)

        self._ip_status = Label()
        self._ip_status.text = "Not loaded"
        self._ip_status.font_size = 11
        self._ip_status.color = (0.5, 0.5, 0.5, 1.0)
        ip_group.add_child(self._ip_status)

        self._ip_scale_slider = SliderEdit()
        self._ip_scale_slider.label = "Scale"
        self._ip_scale_slider.min_value = 0.0
        self._ip_scale_slider.max_value = 1.0
        self._ip_scale_slider.value = 0.60
        self._ip_scale_slider.decimals = 2
        ip_group.add_child(self._ip_scale_slider)

        ip_btn_row = HStack()
        ip_btn_row.spacing = 4

        self._draw_rect_cb = Checkbox()
        self._draw_rect_cb.text = "Draw Rect"
        self._draw_rect_cb.on_changed = lambda v: (
            self.on_draw_rect_toggled and self.on_draw_rect_toggled(v))
        ip_btn_row.add_child(self._draw_rect_cb)

        self._show_rect_cb = Checkbox()
        self._show_rect_cb.text = "Show Rect"
        self._show_rect_cb.checked = True
        self._show_rect_cb.on_changed = lambda v: (
            self.on_show_rect_toggled and self.on_show_rect_toggled(v))
        ip_btn_row.add_child(self._show_rect_cb)

        clear_rect_btn = Button()
        clear_rect_btn.text = "Clear"
        clear_rect_btn.preferred_width = px(50)
        clear_rect_btn.on_click = lambda: (
            self.on_clear_rect and self.on_clear_rect())
        ip_btn_row.add_child(clear_rect_btn)

        ip_group.add_child(ip_btn_row)
        content.add_child(ip_group)

        # --- Diffusion Layer ---
        self._layer_group = GroupBox()
        self._layer_group.title = "Diffusion Layer"
        self._layer_group.visible = False

        self._layer_info = Label()
        self._layer_info.text = "No diffusion layer selected"
        self._layer_info.font_size = 11
        self._layer_info.color = (0.7, 0.7, 0.7, 1.0)
        self._layer_group.add_child(self._layer_info)

        regen_row = HStack()
        regen_row.spacing = 4

        regen_btn = Button()
        regen_btn.text = "Regenerate"
        regen_btn.preferred_width = px(100)
        regen_btn.on_click = lambda: (self.on_regenerate and self.on_regenerate())
        regen_row.add_child(regen_btn)

        new_seed_btn = Button()
        new_seed_btn.text = "New Seed"
        new_seed_btn.preferred_width = px(80)
        new_seed_btn.on_click = lambda: (self.on_new_seed and self.on_new_seed())
        regen_row.add_child(new_seed_btn)

        self._layer_group.add_child(regen_row)

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

    # ------------------------------------------------------------------
    # Model scanning
    # ------------------------------------------------------------------

    def _scan_models(self):
        self._model_paths: list[str] = []
        names = []
        if os.path.isdir(MODELS_DIR):
            for f in sorted(os.listdir(MODELS_DIR)):
                if f.endswith(".safetensors") and "flux" not in f.lower():
                    names.append(f)
                    self._model_paths.append(os.path.join(MODELS_DIR, f))
        self._model_combo.items = names
        if names:
            self._model_combo.selected_index = 0

    def _on_load(self):
        idx = self._model_combo.selected_index
        if idx < 0 or idx >= len(self._model_paths):
            return
        path = self._model_paths[idx]
        self._model_status.text = "Loading..."
        self._model_diag.text = ""
        pred_map = {0: "", 1: "epsilon", 2: "v_prediction"}
        pred = pred_map.get(self._prediction_combo.selected_index, "")
        if self.on_load_model:
            self.on_load_model(path, pred)

    def _on_mask_brush_cb(self, _value=None):
        size = int(self._mask_size_slider.value)
        hardness = self._mask_hardness_slider.value
        flow = self._mask_flow_slider.value
        if self.on_mask_brush_changed:
            self.on_mask_brush_changed(size, hardness, flow)

    def _set_random_seed(self):
        self._seed_edit.text = str(random.randint(0, 2**32 - 1))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def prompt(self) -> str:
        return self._prompt.text

    @property
    def negative_prompt(self) -> str:
        return self._negative_prompt.text

    @property
    def strength(self) -> float:
        return self._strength_slider.value

    @property
    def steps(self) -> int:
        return int(self._steps_slider.value)

    @property
    def guidance_scale(self) -> float:
        return self._cfg_slider.value

    @property
    def seed(self) -> int:
        try:
            return int(self._seed_edit.text)
        except ValueError:
            return -1

    def set_seed(self, seed: int):
        self._seed_edit.text = str(seed)

    @property
    def mode(self) -> str:
        modes = ["txt2img", "img2img", "inpaint"]
        idx = self._mode_combo.selected_index
        return modes[idx] if 0 <= idx < len(modes) else "inpaint"

    @property
    def masked_content(self) -> str:
        items = ["original", "fill", "latent_noise", "latent_nothing"]
        idx = self._masked_content_combo.selected_index
        return items[idx] if 0 <= idx < len(items) else "original"

    @property
    def prediction_type(self) -> str:
        pred_map = {0: "", 1: "epsilon", 2: "v_prediction"}
        return pred_map.get(self._prediction_combo.selected_index, "")

    @property
    def ip_adapter_scale(self) -> float:
        return self._ip_scale_slider.value

    @property
    def resize_to_model_resolution(self) -> bool:
        return self._resize_cb.checked

    # ------------------------------------------------------------------
    # External updates
    # ------------------------------------------------------------------

    def on_model_loaded(self, path: str, model_info: dict):
        name = os.path.basename(path)
        self._model_status.text = f"Loaded: {name}"
        diag = (
            f"scheduler: {model_info.get('scheduler', '?')}\n"
            f"prediction: {model_info.get('prediction_type', '?')}\n"
            f"algorithm: {model_info.get('algorithm_type', '?')}"
        )
        self._model_diag.text = diag

    def on_model_load_error(self, error: str):
        self._model_status.text = f"Error: {error[:80]}"
        self._model_diag.text = ""

    def on_ip_adapter_loaded(self):
        self._ip_status.text = "Loaded"

    def on_ip_adapter_load_error(self, error: str):
        self._ip_status.text = f"Error: {error[:60]}"

    def set_draw_patch_checked(self, value: bool):
        self._draw_patch_cb.checked = value

    def show_diffusion_layer(self, layer):
        self._layer_group.visible = True
        self._prompt.text = layer.prompt
        self._negative_prompt.text = layer.negative_prompt
        self._strength_slider.value = layer.strength
        self._steps_slider.value = layer.steps
        self._cfg_slider.value = layer.guidance_scale
        self._seed_edit.text = str(layer.seed)

        modes = ["txt2img", "img2img", "inpaint"]
        if layer.mode in modes:
            self._mode_combo.selected_index = modes.index(layer.mode)

        mc_items = ["original", "fill", "latent_noise", "latent_nothing"]
        if layer.masked_content in mc_items:
            self._masked_content_combo.selected_index = mc_items.index(layer.masked_content)

        self._ip_scale_slider.value = layer.ip_adapter_scale
        self._resize_cb.checked = layer.resize_to_model_resolution

        model_name = os.path.basename(layer.model_path) if layer.model_path else "?"
        mask_status = "has mask" if layer.has_mask() else "no mask"
        if layer.ip_adapter_rect:
            r = layer.ip_adapter_rect
            ip_info = f"rect ({r[0]},{r[1]})-({r[2]},{r[3]})"
        else:
            ip_info = "none"
        if layer.manual_patch_rect:
            r = layer.manual_patch_rect
            pw, ph = r[2] - r[0], r[3] - r[1]
            patch_info = f"manual {pw}x{ph}"
        else:
            patch_info = f"auto {layer.patch_w}x{layer.patch_h}"

        self._layer_info.text = (
            f"model: {model_name}  mode: {layer.mode}\n"
            f"patch: {patch_info}  mask: {mask_status}\n"
            f"ip_adapter: {ip_info}"
        )

    def clear_diffusion_layer(self):
        self._layer_group.visible = False
        self._layer_info.text = "No diffusion layer selected"
