"""Grounding DINO + SAM 2.1 dialog for object detection and segmentation."""

from __future__ import annotations

from typing import Callable

from tcgui.widgets.checkbox import Checkbox
from tcgui.widgets.combo_box import ComboBox
from tcgui.widgets.dialog import Dialog
from tcgui.widgets.label import Label
from tcgui.widgets.slider_edit import SliderEdit
from tcgui.widgets.text_input import TextInput
from tcgui.widgets.units import px
from tcgui.widgets.vstack import VStack

from ...grounding.types import GROUNDING_MODELS, SAM2_MODELS, GroundingParams


class GroundingDialog:
    """Modal dialog for running Grounding DINO + SAM 2.1 on the canvas."""

    def __init__(
            self,
            *,
            ui,
            on_submit: Callable[[GroundingParams], None],
            gpu_available: bool):
        self._ui = ui
        self._on_submit = on_submit
        self._gpu_available = gpu_available
        self._prompt_input: TextInput | None = None
        self._model_combo: ComboBox | None = None
        self._box_threshold_slider: SliderEdit | None = None
        self._text_threshold_slider: SliderEdit | None = None
        self._sam2_combo: ComboBox | None = None
        self._sam2_checkbox: Checkbox | None = None
        self._sam2_mask_combo: ComboBox | None = None
        self._mask_threshold_slider: SliderEdit | None = None
        self._max_hole_slider: SliderEdit | None = None
        self._max_sprinkle_slider: SliderEdit | None = None
        self._multimask_checkbox: Checkbox | None = None
        self._non_overlap_checkbox: Checkbox | None = None

    def show(self) -> None:
        dlg = Dialog()
        dlg.title = "Detect Objects"
        dlg.buttons = ["OK", "Cancel"]
        dlg.default_button = "OK"
        dlg.cancel_button = "Cancel"
        dlg.min_width = 400

        content = VStack()
        content.spacing = 10

        # ── Grounding DINO ──────────────────────────────────────────────

        prompt_label = Label()
        prompt_label.text = "Objects to find"
        content.add_child(prompt_label)

        self._prompt_input = TextInput()
        self._prompt_input.placeholder = "e.g. cat. red cup. face."
        self._prompt_input.preferred_width = px(400)
        content.add_child(self._prompt_input)

        model_label = Label()
        model_label.text = "Grounding DINO model"
        content.add_child(model_label)

        self._model_combo = ComboBox()
        self._model_combo.preferred_width = px(400)
        for label_text, _model_id in GROUNDING_MODELS:
            self._model_combo.add_item(label_text)
        self._model_combo.selected_index = 1
        content.add_child(self._model_combo)

        box_threshold_label = Label()
        box_threshold_label.text = "Box confidence threshold"
        content.add_child(box_threshold_label)

        self._box_threshold_slider = SliderEdit()
        self._box_threshold_slider.tooltip = "Minimum confidence for object detection boxes"
        self._box_threshold_slider.min_value = 0.05
        self._box_threshold_slider.max_value = 1.0
        self._box_threshold_slider.value = 0.40
        self._box_threshold_slider.step = 0.05
        self._box_threshold_slider.decimals = 2
        self._box_threshold_slider.preferred_width = px(400)
        content.add_child(self._box_threshold_slider)

        text_threshold_label = Label()
        text_threshold_label.text = "Text token threshold"
        content.add_child(text_threshold_label)

        self._text_threshold_slider = SliderEdit()
        self._text_threshold_slider.tooltip = "Minimum confidence for text-to-box matching"
        self._text_threshold_slider.min_value = 0.05
        self._text_threshold_slider.max_value = 1.0
        self._text_threshold_slider.value = 0.30
        self._text_threshold_slider.step = 0.05
        self._text_threshold_slider.decimals = 2
        self._text_threshold_slider.preferred_width = px(400)
        content.add_child(self._text_threshold_slider)

        # ── SAM 2.1 ─────────────────────────────────────────────────────

        sam2_label = Label()
        sam2_label.text = "SAM 2.1 segmentation (experimental)"
        content.add_child(sam2_label)

        self._sam2_checkbox = Checkbox()
        self._sam2_checkbox.text = "Segment with SAM 2.1"
        self._sam2_checkbox.checked = True
        content.add_child(self._sam2_checkbox)

        self._sam2_combo = ComboBox()
        self._sam2_combo.preferred_width = px(400)
        for label_text, _model_id in SAM2_MODELS:
            self._sam2_combo.add_item(label_text)
        self._sam2_combo.selected_index = 0  # Tiny default
        content.add_child(self._sam2_combo)

        self._sam2_mask_combo = ComboBox()
        self._sam2_mask_combo.preferred_width = px(400)
        self._sam2_mask_combo.add_item("whole (full object)")
        self._sam2_mask_combo.add_item("part")
        self._sam2_mask_combo.add_item("subpart")
        self._sam2_mask_combo.selected_index = 0
        content.add_child(self._sam2_mask_combo)

        # ── SAM 2.1 options ─────────────────────────────────────────────

        mask_thresh_label = Label()
        mask_thresh_label.text = "Mask threshold (higher = tighter)"
        content.add_child(mask_thresh_label)

        self._mask_threshold_slider = SliderEdit()
        self._mask_threshold_slider.tooltip = "Higher values produce tighter masks around object edges"
        self._mask_threshold_slider.min_value = 0.0
        self._mask_threshold_slider.max_value = 1.0
        self._mask_threshold_slider.value = 0.0
        self._mask_threshold_slider.step = 0.05
        self._mask_threshold_slider.decimals = 2
        self._mask_threshold_slider.preferred_width = px(400)
        content.add_child(self._mask_threshold_slider)

        max_hole_label = Label()
        max_hole_label.text = "Max hole area (0 = off, px)"
        content.add_child(max_hole_label)

        self._max_hole_slider = SliderEdit()
        self._max_hole_slider.tooltip = "Fill holes smaller than this area in pixels (0 = disabled)"
        self._max_hole_slider.min_value = 0
        self._max_hole_slider.max_value = 10000
        self._max_hole_slider.value = 0
        self._max_hole_slider.step = 100
        self._max_hole_slider.decimals = 0
        self._max_hole_slider.preferred_width = px(400)
        content.add_child(self._max_hole_slider)

        max_sprinkle_label = Label()
        max_sprinkle_label.text = "Max sprinkle area (0 = off, px)"
        content.add_child(max_sprinkle_label)

        self._max_sprinkle_slider = SliderEdit()
        self._max_sprinkle_slider.tooltip = "Remove isolated pixels smaller than this area (0 = disabled)"
        self._max_sprinkle_slider.min_value = 0
        self._max_sprinkle_slider.max_value = 10000
        self._max_sprinkle_slider.value = 0
        self._max_sprinkle_slider.step = 100
        self._max_sprinkle_slider.decimals = 0
        self._max_sprinkle_slider.preferred_width = px(400)
        content.add_child(self._max_sprinkle_slider)

        self._multimask_checkbox = Checkbox()
        self._multimask_checkbox.text = "Multimask output (3 candidates per box)"
        self._multimask_checkbox.checked = True
        content.add_child(self._multimask_checkbox)

        self._non_overlap_checkbox = Checkbox()
        self._non_overlap_checkbox.text = "Non-overlapping masks"
        self._non_overlap_checkbox.checked = False
        content.add_child(self._non_overlap_checkbox)

        # ── GPU ─────────────────────────────────────────────────────────

        self._gpu_checkbox = Checkbox()
        self._gpu_checkbox.text = "Use GPU"
        self._gpu_checkbox.checked = self._gpu_available
        self._gpu_checkbox.enabled = self._gpu_available
        content.add_child(self._gpu_checkbox)

        dlg.content = content
        dlg.on_result = self._on_result
        dlg.show(self._ui)

    def _on_result(self, button: str) -> None:
        if button != "OK" or self._prompt_input is None:
            return

        prompt = self._prompt_input.text.strip()
        if not prompt:
            return
        if not prompt.endswith("."):
            prompt += "."

        box_threshold = (
            self._box_threshold_slider.value if self._box_threshold_slider else 0.40
        )
        text_threshold = (
            self._text_threshold_slider.value if self._text_threshold_slider else 0.30
        )
        model_idx = self._model_combo.selected_index if self._model_combo else 0
        model_id = GROUNDING_MODELS[model_idx][1]

        use_sam2 = self._sam2_checkbox.checked if self._sam2_checkbox else False
        sam2_idx = self._sam2_combo.selected_index if self._sam2_combo else 0
        sam2_id = SAM2_MODELS[sam2_idx][1] if use_sam2 else None
        sam2_mask_channel = (
            self._sam2_mask_combo.selected_index if self._sam2_mask_combo else 0
        )

        mask_threshold = (
            self._mask_threshold_slider.value if self._mask_threshold_slider else 0.0
        )
        max_hole_area = (
            int(self._max_hole_slider.value) if self._max_hole_slider else 0
        )
        max_sprinkle_area = (
            int(self._max_sprinkle_slider.value) if self._max_sprinkle_slider else 0
        )
        multimask = (
            self._multimask_checkbox.checked if self._multimask_checkbox else True
        )
        non_overlap = (
            self._non_overlap_checkbox.checked if self._non_overlap_checkbox else False
        )

        use_gpu = self._gpu_checkbox.checked if self._gpu_checkbox else False

        self._on_submit(
            GroundingParams(
                prompt=prompt,
                model_id=model_id,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                use_gpu=use_gpu,
                sam2_model_id=sam2_id,
                sam2_mask_channel=sam2_mask_channel,
                mask_threshold=mask_threshold,
                max_hole_area=max_hole_area,
                max_sprinkle_area=max_sprinkle_area,
                multimask=multimask,
                non_overlap=non_overlap,
            )
        )
