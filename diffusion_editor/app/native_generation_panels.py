"""termin-gui-native projections for generation tool panels."""

from __future__ import annotations

from typing import Callable

from termin.gui_native import EdgeInsets, TcDocument

from .generation_panels import (
    GenerationAction,
    GenerationIntent,
    GenerationPanelKind,
    GenerationPanelsState,
    GenerationPhase,
)


_PREDICTIONS = ("Auto", "epsilon", "v_prediction")
_PREDICTION_VALUES = ("", "epsilon", "v_prediction")
_MODES = ("txt2img", "img2img", "inpaint")
_MASKED_CONTENT_LABELS = (
    "original", "fill", "latent noise", "latent nothing")
_MASKED_CONTENT_VALUES = (
    "original", "fill", "latent_noise", "latent_nothing")


class NativeGenerationPanels:
    """Generation controls measured by the shell-owned left-panel scroll."""

    def __init__(
            self,
            document: TcDocument,
            state: GenerationPanelsState,
            on_intent: Callable[[GenerationIntent], None],
            request_repaint: Callable[[], None]) -> None:
        self._document = document
        self._on_intent = on_intent
        self._request_repaint = request_repaint
        self._closed = False
        self._syncing = False
        self._connections: list[object] = []
        self._model_paths: list[str] = []
        self._reference_ids: list[str | None] = []

        self.content = document.create_vstack(
            "NativeGenerationPanelsContent")
        self.content.stable_id = "diffusion-editor.generation-panels"
        self.content.set_layout_spacing(6.0)
        self.widget = self.content

        self.diffusion_group, diffusion = self._group(
            "Diffusion", "diffusion")
        self._build_diffusion(diffusion)
        self.content.add_preferred_child(self.diffusion_group.widget)

        self.lama_group, lama = self._group("LaMa", "lama")
        self._build_lama(lama)
        self.content.add_preferred_child(self.lama_group.widget)

        self.instruct_group, instruct = self._group(
            "InstructPix2Pix", "instruct")
        self._build_instruct(instruct)
        self.content.add_preferred_child(self.instruct_group.widget)

        self.mask_group, mask = self._group("Mask Brush", "mask")
        self._build_mask(mask)
        self.content.add_preferred_child(self.mask_group.widget)

        self.empty_label = document.create_label(
            "Attach a Diffusion, LaMa or Instruct tool to the active layer.",
            "NativeGenerationEmptyLabel",
        )
        self.empty_label.stable_id = "diffusion-editor.generation.empty"
        self.content.add_preferred_child(self.empty_label)
        self.apply_generation_panels_state(state)

    def apply_generation_panels_state(
            self, state: GenerationPanelsState) -> None:
        if self._closed:
            return
        self._syncing = True
        try:
            kind = state.active_kind
            self.diffusion_group.widget.visible = (
                kind == GenerationPanelKind.DIFFUSION)
            self.lama_group.widget.visible = (
                kind == GenerationPanelKind.LAMA)
            self.instruct_group.widget.visible = (
                kind == GenerationPanelKind.INSTRUCT)
            self.mask_group.widget.visible = (
                kind != GenerationPanelKind.NONE)
            self.empty_label.visible = kind == GenerationPanelKind.NONE

            diffusion = state.diffusion
            self._sync_combo(
                self.model_combo,
                [choice.name for choice in state.models],
                [choice.stable_id for choice in state.models],
                diffusion.selected_model_path,
                self._set_model_paths,
            )
            self._set_combo_value(
                self.prediction_combo,
                _PREDICTION_VALUES,
                diffusion.prediction_type,
            )
            self.diffusion_prompt.text = diffusion.prompt
            self.diffusion_negative_prompt.text = (
                diffusion.negative_prompt)
            self._set_combo_value(
                self.mode_combo, _MODES, diffusion.mode)
            self._set_combo_value(
                self.masked_content_combo,
                _MASKED_CONTENT_VALUES,
                diffusion.masked_content,
            )
            self.diffusion_strength.value = diffusion.strength
            self.diffusion_steps.value = float(diffusion.steps)
            self.diffusion_guidance.value = diffusion.guidance_scale
            self.diffusion_seed.text = diffusion.seed_text
            self.diffusion_resize.checked = (
                diffusion.resize_to_model_resolution)
            self.diffusion_status.text = (
                f"{diffusion.phase.value}: {diffusion.message}")
            self.diffusion_diagnostics.text = (
                diffusion.model_diagnostics)
            self.diffusion_info.text = diffusion.layer_info
            self.ip_status.text = diffusion.ip_adapter_message
            self.ip_scale.value = diffusion.ip_adapter_scale
            reference_values = [None] + [
                choice.stable_id for choice in state.reference_layers]
            self._sync_combo(
                self.ip_reference_combo,
                ["None"] + [
                    choice.name for choice in state.reference_layers],
                reference_values,
                diffusion.ip_adapter_layer_id,
                self._set_reference_ids,
            )
            diffusion_busy = diffusion.phase in {
                GenerationPhase.LOADING,
                GenerationPhase.RUNNING,
            }
            self.load_model_button.widget.enabled = (
                bool(state.models) and not diffusion_busy)
            self.regenerate_button.widget.enabled = not diffusion_busy
            self.diffusion_new_seed_button.widget.enabled = (
                not diffusion_busy)
            self.load_ip_button.widget.enabled = not diffusion_busy

            self.lama_status.text = (
                f"{state.lama.phase.value}: {state.lama.message}")
            self.lama_info.text = state.lama.layer_info
            lama_busy = state.lama.phase == GenerationPhase.RUNNING
            self.lama_run_button.widget.enabled = not lama_busy
            self.lama_select_background_button.widget.enabled = (
                not lama_busy)

            instruct = state.instruct
            self.instruct_text.text = instruct.instruction
            self.instruct_image_guidance.value = (
                instruct.image_guidance_scale)
            self.instruct_guidance.value = instruct.guidance_scale
            self.instruct_steps.value = float(instruct.steps)
            self.instruct_seed.text = instruct.seed_text
            self.instruct_status.text = (
                f"{instruct.phase.value}: {instruct.message}")
            self.instruct_info.text = instruct.layer_info
            instruct_busy = instruct.phase in {
                GenerationPhase.LOADING,
                GenerationPhase.RUNNING,
            }
            self.instruct_load_button.widget.enabled = (
                not instruct_busy)
            self.instruct_run_button.widget.enabled = not instruct_busy
            self.instruct_new_seed_button.widget.enabled = (
                not instruct_busy)

            self.mask_size.value = float(state.mask.size)
            self.mask_hardness.value = state.mask.hardness
            self.mask_flow.value = state.mask.flow
            self.mask_eraser.checked = state.mask.eraser
            self.show_mask.checked = state.mask.show
        finally:
            self._syncing = False
        self._request_repaint()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._on_intent = lambda _intent: None
        self._connections.clear()

    def _build_diffusion(self, content) -> None:
        self._caption(content, "Model", "diffusion.model.caption")
        self.model_combo = self._combo(
            content,
            "diffusion.model",
            lambda index: self._emit(
                GenerationAction.SELECT_MODEL,
                self._item(self._model_paths, index, ""),
            ),
        )
        self._caption(
            content, "Prediction", "diffusion.prediction.caption")
        self.prediction_combo = self._combo(
            content,
            "diffusion.prediction",
            lambda index: self._emit(
                GenerationAction.SELECT_PREDICTION,
                self._item(_PREDICTION_VALUES, index, ""),
            ),
        )
        self._fill_combo(self.prediction_combo, _PREDICTIONS)
        self.load_model_button = self._button(
            content, "Load Model", "diffusion.load-model",
            lambda: self._emit(GenerationAction.LOAD_MODEL))
        self.diffusion_status = self._label(
            content, "", "diffusion.status")
        self.diffusion_diagnostics = self._label(
            content, "", "diffusion.diagnostics")

        self.diffusion_prompt = self._text_area(
            content,
            "diffusion.prompt",
            "masterpiece, best quality",
            GenerationAction.SET_PROMPT,
        )
        self.diffusion_negative_prompt = self._text_area(
            content,
            "diffusion.negative-prompt",
            "worst quality, blurry",
            GenerationAction.SET_NEGATIVE_PROMPT,
        )
        self._caption(content, "Mode", "diffusion.mode.caption")
        self.mode_combo = self._combo(
            content,
            "diffusion.mode",
            lambda index: self._emit(
                GenerationAction.SET_MODE,
                self._item(_MODES, index, "inpaint"),
            ),
        )
        self._fill_combo(self.mode_combo, _MODES)
        self._caption(
            content,
            "Masked Content",
            "diffusion.masked-content.caption",
        )
        self.masked_content_combo = self._combo(
            content,
            "diffusion.masked-content",
            lambda index: self._emit(
                GenerationAction.SET_MASKED_CONTENT,
                self._item(
                    _MASKED_CONTENT_VALUES, index, "original"),
            ),
        )
        self._fill_combo(
            self.masked_content_combo, _MASKED_CONTENT_LABELS)
        self.diffusion_strength = self._slider(
            content, "Strength", "diffusion.strength",
            0.3, 0.0, 1.0, 0.01, 2,
            GenerationAction.SET_STRENGTH)
        self.diffusion_steps = self._slider(
            content, "Steps", "diffusion.steps",
            20, 1, 50, 1, 0, GenerationAction.SET_STEPS)
        self.diffusion_guidance = self._slider(
            content, "CFG Scale", "diffusion.guidance",
            7, 1, 20, 0.1, 1, GenerationAction.SET_GUIDANCE)
        self.diffusion_resize = self._checkbox(
            content, "Resize to 1024", "diffusion.resize",
            GenerationAction.SET_RESIZE)
        self.diffusion_seed = self._seed_row(
            content, "diffusion")

        self.ip_status = self._label(
            content, "Not loaded", "diffusion.ip.status")
        self.load_ip_button = self._button(
            content, "Load IP-Adapter", "diffusion.ip.load",
            lambda: self._emit(GenerationAction.LOAD_IP_ADAPTER))
        self.ip_scale = self._slider(
            content, "IP-Adapter Scale", "diffusion.ip.scale",
            0.6, 0, 1, 0.01, 2, GenerationAction.SET_IP_SCALE)
        self._caption(
            content, "IP Reference", "diffusion.ip.reference.caption")
        self.ip_reference_combo = self._combo(
            content,
            "diffusion.ip.reference",
            lambda index: self._emit(
                GenerationAction.SELECT_IP_REFERENCE,
                self._item(self._reference_ids, index, None),
            ),
        )
        self.select_background_button = self._button(
            content, "Select Background", "diffusion.select-background",
            lambda: self._emit(GenerationAction.SELECT_BACKGROUND))
        self.diffusion_info = self._label(
            content, "", "diffusion.layer-info")
        actions = self._document.create_hstack(
            "NativeDiffusionActions")
        actions.set_layout_spacing(4.0)
        self.regenerate_button = self._button(
            actions, "Regenerate", "diffusion.run",
            lambda: self._emit(GenerationAction.RUN), add=False)
        self.diffusion_new_seed_button = self._button(
            actions, "New Seed", "diffusion.new-seed",
            self._new_seed_and_run, add=False)
        actions.add_flex_child(self.regenerate_button.widget, 1.0)
        actions.add_flex_child(
            self.diffusion_new_seed_button.widget, 1.0)
        content.add_preferred_child(actions)
        self.diffusion_clear_mask_button = self._button(
            content, "Clear Mask", "diffusion.clear-mask",
            lambda: self._emit(GenerationAction.CLEAR_MASK))

    def _build_lama(self, content) -> None:
        self.lama_status = self._label(content, "", "lama.status")
        self.lama_info = self._label(content, "", "lama.layer-info")
        self.lama_run_button = self._button(
            content, "Remove Objects", "lama.run",
            lambda: self._emit(GenerationAction.RUN))
        self.lama_select_background_button = self._button(
            content, "Select Background", "lama.select-background",
            lambda: self._emit(GenerationAction.SELECT_BACKGROUND))
        self.lama_clear_mask_button = self._button(
            content, "Clear Mask", "lama.clear-mask",
            lambda: self._emit(GenerationAction.CLEAR_MASK))

    def _build_instruct(self, content) -> None:
        self.instruct_status = self._label(
            content, "", "instruct.status")
        self.instruct_load_button = self._button(
            content, "Load InstructPix2Pix", "instruct.load-model",
            lambda: self._emit(GenerationAction.LOAD_MODEL))
        self.instruct_text = self._text_area(
            content,
            "instruct.instruction",
            "make it snowy",
            GenerationAction.SET_INSTRUCTION,
        )
        self.instruct_image_guidance = self._slider(
            content, "Image Guidance", "instruct.image-guidance",
            1.5, 1, 3, 0.1, 1,
            GenerationAction.SET_IMAGE_GUIDANCE)
        self.instruct_guidance = self._slider(
            content, "CFG Scale", "instruct.guidance",
            7, 1, 20, 0.1, 1, GenerationAction.SET_GUIDANCE)
        self.instruct_steps = self._slider(
            content, "Steps", "instruct.steps",
            20, 1, 50, 1, 0, GenerationAction.SET_STEPS)
        self.instruct_seed = self._seed_row(content, "instruct")
        self.instruct_info = self._label(
            content, "", "instruct.layer-info")
        actions = self._document.create_hstack(
            "NativeInstructActions")
        actions.set_layout_spacing(4.0)
        self.instruct_run_button = self._button(
            actions, "Apply", "instruct.run",
            lambda: self._emit(GenerationAction.RUN), add=False)
        self.instruct_new_seed_button = self._button(
            actions, "New Seed", "instruct.new-seed",
            self._new_seed_and_run, add=False)
        actions.add_flex_child(self.instruct_run_button.widget, 1.0)
        actions.add_flex_child(
            self.instruct_new_seed_button.widget, 1.0)
        content.add_preferred_child(actions)
        self.instruct_clear_mask_button = self._button(
            content, "Clear Mask", "instruct.clear-mask",
            lambda: self._emit(GenerationAction.CLEAR_MASK))

    def _build_mask(self, content) -> None:
        self.mask_size = self._slider(
            content, "Size", "mask.size",
            50, 1, 500, 1, 0, GenerationAction.SET_MASK_SIZE)
        self.mask_hardness = self._slider(
            content, "Hardness", "mask.hardness",
            0.4, 0, 1, 0.01, 2,
            GenerationAction.SET_MASK_HARDNESS)
        self.mask_flow = self._slider(
            content, "Flow", "mask.flow",
            1, 0, 1, 0.01, 2, GenerationAction.SET_MASK_FLOW)
        row = self._document.create_hstack("NativeMaskFlags")
        row.set_layout_spacing(4.0)
        self.mask_eraser = self._checkbox(
            row, "Eraser", "mask.eraser",
            GenerationAction.SET_MASK_ERASER, add=False)
        self.show_mask = self._checkbox(
            row, "Show", "mask.show",
            GenerationAction.SET_SHOW_MASK, add=False)
        self._add_inline_checkbox(
            row, self.mask_eraser, "Eraser", "mask.eraser")
        self._add_inline_checkbox(
            row, self.show_mask, "Show", "mask.show")
        content.add_preferred_child(row)

    def _group(self, title: str, suffix: str):
        group = self._document.create_group_box(
            title, f"NativeGeneration{suffix.title()}Group")
        group.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}")
        group.set_padding(EdgeInsets(6.0, 6.0, 6.0, 6.0))
        content = self._document.create_vstack(
            f"NativeGeneration{suffix.title()}Content")
        content.set_layout_spacing(4.0)
        group.set_content(content)
        return group, content

    def _label(self, parent, text: str, suffix: str):
        label = self._document.create_label(
            text, "NativeGenerationLabel")
        label.stable_id = f"diffusion-editor.generation.{suffix}"
        parent.add_preferred_child(label)
        return label

    def _caption(self, parent, text: str, suffix: str):
        return self._label(parent, text, suffix)

    def _button(
            self, parent, text, suffix, callback, *, add=True):
        button = self._document.create_button(text)
        button.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}")
        self._connections.append(button.connect_clicked(callback))
        if add:
            parent.add_preferred_child(button.widget)
        return button

    def _combo(self, parent, suffix, callback):
        combo = self._document.create_combo_box()
        combo.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}")
        self._connections.append(combo.connect_changed(
            lambda index, *_rest: None
            if self._syncing or self._closed
            else callback(index)))
        parent.add_preferred_child(combo.widget)
        return combo

    def _text_area(self, parent, suffix, placeholder, action):
        area = self._document.create_text_area("")
        area.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}")
        area.placeholder = placeholder
        self._connections.append(area.connect_changed(
            lambda text: self._emit(action, text)))
        parent.add_fixed_child(area.widget, 64.0)
        return area

    def _seed_row(self, parent, suffix):
        row = self._document.create_hstack("NativeGenerationSeedRow")
        row.set_layout_spacing(4.0)
        seed = self._document.create_text_input("-1")
        seed.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}.seed")
        seed.placeholder = "seed"
        self._connections.append(seed.connect_changed(
            lambda text: self._emit(GenerationAction.SET_SEED, text)))
        random_button = self._button(
            row, "Rnd", f"{suffix}.random-seed",
            lambda: self._emit(GenerationAction.RANDOM_SEED), add=False)
        row.add_flex_child(seed.widget, 1.0)
        row.add_fixed_child(random_button.widget, 48.0)
        parent.add_preferred_child(row)
        return seed

    def _slider(
            self,
            parent,
            label,
            suffix,
            value,
            minimum,
            maximum,
            step,
            decimals,
            action):
        slider = self._document.create_slider_edit(float(value))
        slider.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}")
        slider.label = label
        slider.set_range(float(minimum), float(maximum))
        slider.set_step(float(step))
        slider.set_decimals(decimals)
        self._connections.append(slider.connect_changed(
            lambda changed: self._emit(action, changed)))
        parent.add_preferred_child(slider.widget)
        return slider

    def _checkbox(
            self, parent, label, suffix, action, *, add=True):
        checkbox = self._document.create_checkbox(False)
        checkbox.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}")
        self._connections.append(checkbox.connect_changed(
            lambda checked: self._emit(action, checked)))
        if add:
            row = self._document.create_hstack(
                "NativeGenerationCheckboxRow")
            row.set_layout_spacing(4.0)
            text = self._document.create_label(
                label, "NativeGenerationCheckboxLabel")
            text.stable_id = (
                f"diffusion-editor.generation.{suffix}.label")
            row.add_preferred_child(checkbox.widget)
            row.add_flex_child(text, 1.0)
            parent.add_preferred_child(row)
        return checkbox

    def _add_inline_checkbox(
            self, parent, checkbox, label: str, suffix: str) -> None:
        cell = self._document.create_hstack(
            "NativeGenerationCheckboxCell")
        cell.set_layout_spacing(2.0)
        text = self._document.create_label(
            label, "NativeGenerationCheckboxLabel")
        text.stable_id = (
            f"diffusion-editor.generation.{suffix}.label")
        cell.add_preferred_child(checkbox.widget)
        cell.add_flex_child(text, 1.0)
        parent.add_flex_child(cell, 1.0)

    def _sync_combo(
            self,
            combo,
            labels,
            values,
            selected,
            store_values) -> None:
        current_labels = [
            combo.item_text(index)
            for index in range(combo.item_count)
        ]
        if current_labels != list(labels):
            combo.clear()
            self._fill_combo(combo, labels)
        store_values(list(values))
        try:
            combo.selected_index = list(values).index(selected)
        except ValueError:
            combo.selected_index = -1

    @staticmethod
    def _fill_combo(combo, labels) -> None:
        for label in labels:
            combo.add_item(str(label))

    @staticmethod
    def _set_combo_value(combo, values, selected) -> None:
        try:
            combo.selected_index = tuple(values).index(selected)
        except ValueError:
            combo.selected_index = -1

    def _set_model_paths(self, values) -> None:
        self._model_paths = values

    def _set_reference_ids(self, values) -> None:
        self._reference_ids = values

    @staticmethod
    def _item(values, index: int, default):
        return values[index] if 0 <= index < len(values) else default

    def _new_seed_and_run(self) -> None:
        self._emit(GenerationAction.RANDOM_SEED)
        self._emit(GenerationAction.RUN)

    def _emit(self, action: GenerationAction, value=None) -> None:
        if not self._syncing and not self._closed:
            self._on_intent(GenerationIntent(action, value))
