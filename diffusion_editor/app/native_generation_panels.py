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
from ..generation.image_edit_profiles import (
    ImageEditLoraAdapter,
    ParameterKind,
    all_image_edit_parameters,
)
from ..generation.text_to_image_profiles import (
    TextToImageLoraAdapter,
    all_text_to_image_parameters,
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
        self._image_edit_reference_ids: list[str | None] = []
        self._image_edit_profile_ids: list[str] = []
        self._image_edit_widgets: dict[str, object] = {}
        self._image_edit_lora_rows: list[dict[str, object]] = []
        self._image_edit_lora_catalog_paths: list[str] = []
        self._image_edit_lora_catalog_labels: list[str] = []
        self._image_edit_choice_values: dict[str, list[str]] = {}
        self._image_edit_specs = {
            parameter.stable_id: parameter
            for parameter in all_image_edit_parameters()
        }
        self._text_to_image_profile_ids: list[str] = []
        self._text_to_image_widgets: dict[str, object] = {}
        self._text_to_image_lora_rows: list[dict[str, object]] = []
        self._text_to_image_lora_catalog_paths: list[str] = []
        self._text_to_image_lora_catalog_labels: list[str] = []
        self._text_to_image_choice_values: dict[str, list[str]] = {}
        self._text_to_image_specs = {
            parameter.stable_id: parameter
            for parameter in all_text_to_image_parameters()
        }

        self.content = document.create_vstack(
            "NativeGenerationPanelsContent")
        self.content.stable_id = "diffusion-editor.generation-panels"
        self.content.set_layout_spacing(6.0)
        self.widget = self.content

        self.text_to_image_group, text_to_image = self._group(
            "Text to Image", "text-to-image")
        self._build_text_to_image(text_to_image)
        self.content.add_preferred_child(self.text_to_image_group.widget)

        self.diffusion_group, diffusion = self._group(
            "Diffusion", "diffusion")
        self._build_diffusion(diffusion)
        self.content.add_preferred_child(self.diffusion_group.widget)

        self.lama_group, lama = self._group("LaMa", "lama")
        self._build_lama(lama)
        self.content.add_preferred_child(self.lama_group.widget)

        self.instruct_group, instruct = self._group(
            "AI Edit", "instruct")
        self._build_instruct(instruct)
        self.content.add_preferred_child(self.instruct_group.widget)

        self.mask_group, mask = self._group("Mask Brush", "mask")
        self._build_mask(mask)
        self.content.add_preferred_child(self.mask_group.widget)

        self.empty_label = document.create_label(
            "Attach a Text to Image, Diffusion, LaMa or AI Edit tool to the "
            "active layer.",
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
            self.text_to_image_group.widget.visible = (
                kind == GenerationPanelKind.TEXT_TO_IMAGE)
            self.diffusion_group.widget.visible = (
                kind == GenerationPanelKind.DIFFUSION)
            self.lama_group.widget.visible = (
                kind == GenerationPanelKind.LAMA)
            self.instruct_group.widget.visible = (
                kind == GenerationPanelKind.INSTRUCT)
            self.mask_group.widget.visible = (
                kind in {
                    GenerationPanelKind.DIFFUSION,
                    GenerationPanelKind.LAMA,
                    GenerationPanelKind.INSTRUCT,
                })
            self.empty_label.visible = kind == GenerationPanelKind.NONE

            text_to_image = state.text_to_image
            self._sync_combo(
                self.text_to_image_profile_combo,
                [choice.name for choice in text_to_image.model_profiles],
                [choice.stable_id for choice in text_to_image.model_profiles],
                text_to_image.model_profile_id,
                self._set_text_to_image_profile_ids,
            )
            self.text_to_image_description.text = (
                text_to_image.profile_description)
            self.text_to_image_output_size.text = text_to_image.output_size
            active_parameters = {
                parameter.stable_id
                for parameter in text_to_image.parameters
            }
            values = text_to_image.parameter_values or {}
            for parameter_id, widget in self._text_to_image_widgets.items():
                supported = parameter_id in active_parameters
                widget.widget.visible = True
                widget.widget.enabled = supported
                if not supported:
                    continue
                spec = self._text_to_image_specs[parameter_id]
                value = values.get(parameter_id, spec.default)
                if parameter_id == "seed":
                    widget.text = str(value)
                elif spec.kind in {ParameterKind.TEXT, ParameterKind.STRING}:
                    widget.text = str(value)
                elif spec.kind in {ParameterKind.INTEGER, ParameterKind.FLOAT}:
                    widget.value = float(value)
                elif spec.kind == ParameterKind.BOOLEAN:
                    widget.checked = bool(value)
                elif spec.kind == ParameterKind.CHOICE:
                    self._set_combo_value(
                        widget,
                        self._text_to_image_choice_values[parameter_id],
                        str(value),
                    )
            self.text_to_image_status.text = (
                f"{text_to_image.phase.value}: {text_to_image.message}")
            self.text_to_image_info.text = text_to_image.layer_info
            text_to_image_busy = text_to_image.phase in {
                GenerationPhase.LOADING,
                GenerationPhase.RUNNING,
            }
            self.text_to_image_add_lora_button.widget.enabled = (
                not text_to_image_busy)
            self._sync_text_to_image_lora_rows(
                text_to_image.lora_adapters,
                text_to_image.lora_catalog,
                enabled=not text_to_image_busy,
            )
            self.text_to_image_load_button.widget.enabled = (
                not text_to_image_busy)
            self.text_to_image_run_button.widget.enabled = (
                not text_to_image_busy)
            self.text_to_image_new_seed_button.widget.enabled = (
                not text_to_image_busy)

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
            self._sync_combo(
                self.image_edit_profile_combo,
                [choice.name for choice in instruct.model_profiles],
                [choice.stable_id for choice in instruct.model_profiles],
                instruct.model_profile_id,
                self._set_image_edit_profile_ids,
            )
            self.image_edit_description.text = instruct.profile_description
            reference_values = [None] + [
                choice.stable_id for choice in state.reference_layers]
            self._sync_combo(
                self.image_edit_reference_combo,
                ["None"] + [
                    choice.name for choice in state.reference_layers],
                reference_values,
                instruct.reference_layer_id,
                self._set_image_edit_reference_ids,
            )
            self.image_edit_reference_label.text = instruct.reference_label
            active_parameters = {
                parameter.stable_id for parameter in instruct.parameters}
            values = instruct.parameter_values or {}
            instruct_busy = instruct.phase in {
                GenerationPhase.LOADING,
                GenerationPhase.RUNNING,
            }
            lora_enabled = (
                instruct.supports_lora_adapters and not instruct_busy)
            self.image_edit_add_lora_button.widget.enabled = lora_enabled
            self._sync_image_edit_lora_rows(
                instruct.lora_adapters,
                instruct.lora_catalog,
                enabled=lora_enabled,
            )
            for parameter_id, widget in self._image_edit_widgets.items():
                supported = parameter_id in active_parameters
                widget.widget.visible = True
                widget.widget.enabled = supported
                if not supported:
                    continue
                spec = self._image_edit_specs[parameter_id]
                value = values.get(parameter_id, spec.default)
                if parameter_id == "seed":
                    widget.text = str(value)
                elif spec.kind in {ParameterKind.TEXT, ParameterKind.STRING}:
                    widget.text = str(value)
                elif spec.kind in {ParameterKind.INTEGER, ParameterKind.FLOAT}:
                    widget.value = float(value)
                elif spec.kind == ParameterKind.BOOLEAN:
                    widget.checked = bool(value)
                elif spec.kind == ParameterKind.CHOICE:
                    self._set_combo_value(
                        widget,
                        self._image_edit_choice_values[parameter_id],
                        str(value),
                    )
            self.instruct_status.text = (
                f"{instruct.phase.value}: {instruct.message}")
            self.instruct_info.text = instruct.layer_info
            self.instruct_load_button.widget.enabled = (
                not instruct_busy)
            self.instruct_run_button.widget.enabled = not instruct_busy
            self.instruct_new_seed_button.widget.enabled = (
                not instruct_busy)
            reference_enabled = (
                instruct.supports_reference_image and not instruct_busy)
            self.image_edit_reference_combo.widget.enabled = reference_enabled
            self.image_edit_reference_browse.widget.enabled = reference_enabled
            self.image_edit_reference_clear.widget.enabled = (
                reference_enabled and instruct.reference_label != "None")

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
        self._image_edit_lora_rows.clear()
        self._text_to_image_lora_rows.clear()
        self._connections.clear()

    def _build_text_to_image(self, content) -> None:
        self._caption(
            content, "Profile", "text-to-image.profile.caption")
        self.text_to_image_profile_combo = self._combo(
            content,
            "text-to-image.profile",
            lambda index: self._emit(
                GenerationAction.SELECT_TEXT_TO_IMAGE_PROFILE,
                self._item(self._text_to_image_profile_ids, index, ""),
            ),
        )
        self.text_to_image_description = self._label(
            content, "", "text-to-image.profile.description")
        self.text_to_image_output_size = self._label(
            content, "", "text-to-image.output-size")
        self.text_to_image_status = self._label(
            content, "", "text-to-image.status")
        self.text_to_image_load_button = self._button(
            content,
            "Load Model",
            "text-to-image.load-model",
            lambda: self._emit(GenerationAction.LOAD_MODEL),
        )
        self._caption(
            content, "LoRA adapters", "text-to-image.lora-adapters.caption")
        self.text_to_image_lora_container = self._document.create_vstack(
            "NativeTextToImageLoraStack")
        self.text_to_image_lora_container.stable_id = (
            "diffusion-editor.generation.text-to-image.lora-adapters")
        self.text_to_image_lora_container.set_layout_spacing(4.0)
        content.add_preferred_child(self.text_to_image_lora_container)
        self.text_to_image_add_lora_button = self._button(
            content,
            "+ Add LoRA",
            "text-to-image.lora-adapters.add",
            lambda: self._emit(GenerationAction.ADD_TEXT_TO_IMAGE_LORA),
        )
        for parameter in all_text_to_image_parameters():
            parameter_id = parameter.stable_id
            suffix = (
                f"text-to-image.parameter."
                f"{parameter_id.replace('_', '-')}")
            emit = lambda changed, key=parameter_id: self._emit(
                GenerationAction.SET_TEXT_TO_IMAGE_PARAMETER,
                (key, changed),
            )
            if parameter.kind == ParameterKind.TEXT:
                self._caption(content, parameter.label, f"{suffix}.caption")
                widget = self._text_area_callback(
                    content, suffix, parameter.placeholder, emit)
            elif parameter.kind == ParameterKind.STRING:
                widget = self._text_input_parameter(
                    content, parameter.label, suffix,
                    parameter.placeholder, emit)
            elif parameter_id == "seed":
                widget = self._seed_parameter_row(content, suffix, emit)
            elif parameter.kind in {
                    ParameterKind.INTEGER, ParameterKind.FLOAT}:
                widget = self._slider_callback(
                    content,
                    parameter.label,
                    suffix,
                    parameter.default,
                    parameter.minimum,
                    parameter.maximum,
                    parameter.step,
                    parameter.decimals,
                    emit,
                )
            elif parameter.kind == ParameterKind.BOOLEAN:
                widget = self._checkbox_callback(
                    content, parameter.label, suffix, emit)
            else:
                self._caption(content, parameter.label, f"{suffix}.caption")
                values = [choice.value for choice in parameter.choices]
                widget = self._combo(
                    content,
                    suffix,
                    lambda index, key=parameter_id: self._emit(
                        GenerationAction.SET_TEXT_TO_IMAGE_PARAMETER,
                        (key, self._item(
                            self._text_to_image_choice_values[key],
                            index,
                            "",
                        )),
                    ),
                )
                self._fill_combo(
                    widget,
                    [choice.label for choice in parameter.choices],
                )
                self._text_to_image_choice_values[parameter_id] = values
            self._text_to_image_widgets[parameter_id] = widget

        self.text_to_image_info = self._label(
            content, "", "text-to-image.layer-info")
        actions = self._document.create_hstack(
            "NativeTextToImageActions")
        actions.set_layout_spacing(4.0)
        self.text_to_image_run_button = self._button(
            actions,
            "Generate",
            "text-to-image.run",
            lambda: self._emit(GenerationAction.RUN),
            add=False,
        )
        self.text_to_image_new_seed_button = self._button(
            actions,
            "New Seed",
            "text-to-image.new-seed",
            self._new_seed_and_run,
            add=False,
        )
        actions.add_flex_child(self.text_to_image_run_button.widget, 1.0)
        actions.add_flex_child(
            self.text_to_image_new_seed_button.widget, 1.0)
        content.add_preferred_child(actions)

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
        self._caption(content, "Model", "instruct.profile.caption")
        self.image_edit_profile_combo = self._combo(
            content,
            "instruct.profile",
            lambda index: self._emit(
                GenerationAction.SELECT_IMAGE_EDIT_PROFILE,
                self._item(self._image_edit_profile_ids, index, ""),
            ),
        )
        self.image_edit_description = self._label(
            content, "", "instruct.profile.description")
        self._caption(
            content, "Second image", "instruct.reference.caption")
        self.image_edit_reference_combo = self._combo(
            content,
            "instruct.reference.layer",
            lambda index: self._emit(
                GenerationAction.SELECT_IMAGE_EDIT_REFERENCE,
                self._item(
                    self._image_edit_reference_ids, index, None),
            ),
        )
        reference_actions = self._document.create_hstack(
            "NativeInstructReferenceActions")
        reference_actions.set_layout_spacing(4.0)
        self.image_edit_reference_browse = self._button(
            reference_actions,
            "Open Image...",
            "instruct.reference.open",
            lambda: self._emit(
                GenerationAction.PICK_IMAGE_EDIT_REFERENCE),
            add=False,
        )
        self.image_edit_reference_clear = self._button(
            reference_actions,
            "Clear",
            "instruct.reference.clear",
            lambda: self._emit(
                GenerationAction.CLEAR_IMAGE_EDIT_REFERENCE),
            add=False,
        )
        reference_actions.add_flex_child(
            self.image_edit_reference_browse.widget, 1.0)
        reference_actions.add_flex_child(
            self.image_edit_reference_clear.widget, 1.0)
        content.add_preferred_child(reference_actions)
        self.image_edit_reference_label = self._label(
            content, "None", "instruct.reference.label")
        self.instruct_status = self._label(
            content, "", "instruct.status")
        self.instruct_load_button = self._button(
            content, "Load Model", "instruct.load-model",
            lambda: self._emit(GenerationAction.LOAD_MODEL))
        self._caption(
            content, "LoRA adapters", "instruct.lora-adapters.caption")
        self.image_edit_lora_container = self._document.create_vstack(
            "NativeImageEditLoraStack")
        self.image_edit_lora_container.stable_id = (
            "diffusion-editor.generation.instruct.lora-adapters")
        self.image_edit_lora_container.set_layout_spacing(4.0)
        content.add_preferred_child(self.image_edit_lora_container)
        self.image_edit_add_lora_button = self._button(
            content,
            "+ Add LoRA",
            "instruct.lora-adapters.add",
            lambda: self._emit(GenerationAction.ADD_IMAGE_EDIT_LORA),
        )
        for parameter in all_image_edit_parameters():
            parameter_id = parameter.stable_id
            suffix = f"instruct.parameter.{parameter_id.replace('_', '-')}"
            emit = lambda changed, key=parameter_id: self._emit(
                GenerationAction.SET_IMAGE_EDIT_PARAMETER,
                (key, changed),
            )
            if parameter.kind == ParameterKind.TEXT:
                self._caption(content, parameter.label, f"{suffix}.caption")
                widget = self._text_area_callback(
                    content, suffix, parameter.placeholder, emit)
            elif parameter.kind == ParameterKind.STRING:
                widget = self._text_input_parameter(
                    content, parameter.label, suffix,
                    parameter.placeholder, emit)
            elif parameter_id == "seed":
                widget = self._seed_parameter_row(
                    content, suffix, emit)
            elif parameter.kind in {
                    ParameterKind.INTEGER, ParameterKind.FLOAT}:
                widget = self._slider_callback(
                    content,
                    parameter.label,
                    suffix,
                    parameter.default,
                    parameter.minimum,
                    parameter.maximum,
                    parameter.step,
                    parameter.decimals,
                    emit,
                )
            elif parameter.kind == ParameterKind.BOOLEAN:
                widget = self._checkbox_callback(
                    content, parameter.label, suffix, emit)
            else:
                self._caption(content, parameter.label, f"{suffix}.caption")
                values = [choice.value for choice in parameter.choices]
                widget = self._combo(
                    content,
                    suffix,
                    lambda index, key=parameter_id: self._emit(
                        GenerationAction.SET_IMAGE_EDIT_PARAMETER,
                        (key, self._item(
                            self._image_edit_choice_values[key], index, "")),
                    ),
                )
                self._fill_combo(
                    widget, [choice.label for choice in parameter.choices])
                self._image_edit_choice_values[parameter_id] = values
            self._image_edit_widgets[parameter_id] = widget

        # Compatibility aliases for integrations which still address the old
        # fixed InstructPix2Pix controls.
        self.instruct_text = self._image_edit_widgets["prompt"]
        self.instruct_image_guidance = self._image_edit_widgets[
            "image_guidance_scale"]
        self.instruct_guidance = self._image_edit_widgets["guidance_scale"]
        self.instruct_steps = self._image_edit_widgets["steps"]
        self.instruct_seed = self._image_edit_widgets["seed"]
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

    def _sync_image_edit_lora_rows(
            self,
            adapters: tuple[ImageEditLoraAdapter, ...],
            catalog,
            *,
            enabled: bool) -> None:
        current_ids = [
            str(row["stable_id"]) for row in self._image_edit_lora_rows]
        next_ids = [adapter.stable_id for adapter in adapters]
        catalog_paths = ["", *(item.stable_id for item in catalog)]
        catalog_labels = [
            "Custom / Hugging Face…", *(item.name for item in catalog)]
        catalog_changed = (
            catalog_paths != self._image_edit_lora_catalog_paths
            or catalog_labels != self._image_edit_lora_catalog_labels
        )
        if current_ids != next_ids or catalog_changed:
            for row in self._image_edit_lora_rows:
                row["connections"].clear()
                self._document.destroy_widget_recursive(
                    row["root"].handle)
            self._image_edit_lora_catalog_paths = catalog_paths
            self._image_edit_lora_catalog_labels = catalog_labels
            self._image_edit_lora_rows = [
                self._create_image_edit_lora_row(
                    adapter, catalog_labels)
                for adapter in adapters
            ]
        for index, (row, adapter) in enumerate(zip(
                self._image_edit_lora_rows, adapters)):
            row["enabled"].checked = adapter.enabled
            row["label"].text = adapter.label
            row["source"].text = adapter.source
            row["weight"].value = adapter.weight
            try:
                row["catalog"].selected_index = catalog_paths.index(
                    adapter.source)
            except ValueError:
                row["catalog"].selected_index = 0
            row["enabled"].widget.enabled = enabled
            row["label"].widget.enabled = enabled
            row["source"].widget.enabled = enabled
            row["catalog"].widget.enabled = enabled
            row["weight"].widget.enabled = enabled
            row["up"].widget.enabled = enabled and index > 0
            row["down"].widget.enabled = (
                enabled and index + 1 < len(adapters))
            row["remove"].widget.enabled = enabled

    def _create_image_edit_lora_row(
            self,
            adapter: ImageEditLoraAdapter,
            catalog_labels: list[str]) -> dict[str, object]:
        adapter_id = adapter.stable_id
        suffix = f"instruct.lora-adapter.{adapter_id}"
        root = self._document.create_vstack("NativeImageEditLoraRow")
        root.stable_id = f"diffusion-editor.generation.{suffix}"
        root.set_layout_spacing(2.0)
        header = self._document.create_hstack("NativeImageEditLoraHeader")
        header.set_layout_spacing(2.0)
        enabled = self._document.create_checkbox(adapter.enabled)
        enabled.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}.enabled")
        label = self._document.create_text_input(adapter.label)
        label.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}.label")
        label.placeholder = "Adapter name"
        weight = self._document.create_slider_edit(adapter.weight)
        weight.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}.weight")
        weight.label = "Weight"
        weight.set_range(-4.0, 4.0)
        weight.set_step(0.05)
        weight.set_decimals(2)
        up = self._document.create_button("↑")
        down = self._document.create_button("↓")
        remove = self._document.create_button("×")
        up.widget.stable_id = f"diffusion-editor.generation.{suffix}.up"
        down.widget.stable_id = f"diffusion-editor.generation.{suffix}.down"
        remove.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}.remove")
        header.add_preferred_child(enabled.widget)
        header.add_flex_child(label.widget, 1.0)
        header.add_fixed_child(weight.widget, 112.0)
        header.add_fixed_child(up.widget, 28.0)
        header.add_fixed_child(down.widget, 28.0)
        header.add_fixed_child(remove.widget, 28.0)
        catalog = self._document.create_combo_box()
        catalog.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}.catalog")
        self._fill_combo(catalog, catalog_labels)
        source = self._document.create_text_input(adapter.source)
        source.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}.source")
        source.placeholder = "Local path or Hugging Face repository"
        root.add_preferred_child(header)
        root.add_preferred_child(catalog.widget)
        root.add_preferred_child(source.widget)
        self.image_edit_lora_container.add_preferred_child(root)
        update = lambda field, value: self._emit(
            GenerationAction.UPDATE_IMAGE_EDIT_LORA,
            (adapter_id, field, value),
        )
        connections = [
            enabled.connect_changed(
                lambda changed: update("enabled", changed)),
            label.connect_changed(
                lambda changed: update("label", changed)),
            source.connect_changed(
                lambda changed: update("source", changed)),
            weight.connect_changed(
                lambda changed: update("weight", changed)),
            catalog.connect_changed(
                lambda index, *_rest: self._select_image_edit_lora_catalog(
                    adapter_id, index)),
            up.connect_clicked(lambda: self._emit(
                GenerationAction.MOVE_IMAGE_EDIT_LORA,
                (adapter_id, -1),
            )),
            down.connect_clicked(lambda: self._emit(
                GenerationAction.MOVE_IMAGE_EDIT_LORA,
                (adapter_id, 1),
            )),
            remove.connect_clicked(lambda: self._emit(
                GenerationAction.REMOVE_IMAGE_EDIT_LORA,
                adapter_id,
            )),
        ]
        return {
            "stable_id": adapter_id,
            "root": root,
            "enabled": enabled,
            "label": label,
            "source": source,
            "catalog": catalog,
            "weight": weight,
            "up": up,
            "down": down,
            "remove": remove,
            "connections": connections,
        }

    def _select_image_edit_lora_catalog(
            self, adapter_id: str, index: int) -> None:
        if self._syncing or self._closed:
            return
        if not 0 < index < len(self._image_edit_lora_catalog_paths):
            return
        self._emit(
            GenerationAction.UPDATE_IMAGE_EDIT_LORA,
            (
                adapter_id,
                "source",
                self._image_edit_lora_catalog_paths[index],
            ),
        )

    def _sync_text_to_image_lora_rows(
            self,
            adapters: tuple[TextToImageLoraAdapter, ...],
            catalog,
            *,
            enabled: bool) -> None:
        current_ids = [
            str(row["stable_id"])
            for row in self._text_to_image_lora_rows]
        next_ids = [adapter.stable_id for adapter in adapters]
        catalog_paths = ["", *(item.stable_id for item in catalog)]
        catalog_labels = [
            "Custom / Hugging Face…", *(item.name for item in catalog)]
        catalog_changed = (
            catalog_paths != self._text_to_image_lora_catalog_paths
            or catalog_labels != self._text_to_image_lora_catalog_labels
        )
        if current_ids != next_ids or catalog_changed:
            for row in self._text_to_image_lora_rows:
                row["connections"].clear()
                self._document.destroy_widget_recursive(
                    row["root"].handle)
            self._text_to_image_lora_catalog_paths = catalog_paths
            self._text_to_image_lora_catalog_labels = catalog_labels
            self._text_to_image_lora_rows = [
                self._create_text_to_image_lora_row(
                    adapter, catalog_labels)
                for adapter in adapters
            ]
        for index, (row, adapter) in enumerate(zip(
                self._text_to_image_lora_rows, adapters)):
            row["enabled"].checked = adapter.enabled
            row["label"].text = adapter.label
            row["source"].text = adapter.source
            row["weight"].value = adapter.weight
            try:
                row["catalog"].selected_index = catalog_paths.index(
                    adapter.source)
            except ValueError:
                row["catalog"].selected_index = 0
            row["enabled"].widget.enabled = enabled
            row["label"].widget.enabled = enabled
            row["source"].widget.enabled = enabled
            row["catalog"].widget.enabled = enabled
            row["weight"].widget.enabled = enabled
            row["up"].widget.enabled = enabled and index > 0
            row["down"].widget.enabled = (
                enabled and index + 1 < len(adapters))
            row["remove"].widget.enabled = enabled

    def _create_text_to_image_lora_row(
            self,
            adapter: TextToImageLoraAdapter,
            catalog_labels: list[str]) -> dict[str, object]:
        adapter_id = adapter.stable_id
        suffix = f"text-to-image.lora-adapter.{adapter_id}"
        root = self._document.create_vstack("NativeTextToImageLoraRow")
        root.stable_id = f"diffusion-editor.generation.{suffix}"
        root.set_layout_spacing(2.0)
        header = self._document.create_hstack(
            "NativeTextToImageLoraHeader")
        header.set_layout_spacing(2.0)
        enabled = self._document.create_checkbox(adapter.enabled)
        enabled.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}.enabled")
        label = self._document.create_text_input(adapter.label)
        label.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}.label")
        label.placeholder = "Adapter name"
        weight = self._document.create_slider_edit(adapter.weight)
        weight.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}.weight")
        weight.label = "Weight"
        weight.set_range(-4.0, 4.0)
        weight.set_step(0.05)
        weight.set_decimals(2)
        up = self._document.create_button("↑")
        down = self._document.create_button("↓")
        remove = self._document.create_button("×")
        up.widget.stable_id = f"diffusion-editor.generation.{suffix}.up"
        down.widget.stable_id = f"diffusion-editor.generation.{suffix}.down"
        remove.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}.remove")
        header.add_preferred_child(enabled.widget)
        header.add_flex_child(label.widget, 1.0)
        header.add_fixed_child(weight.widget, 112.0)
        header.add_fixed_child(up.widget, 28.0)
        header.add_fixed_child(down.widget, 28.0)
        header.add_fixed_child(remove.widget, 28.0)
        catalog = self._document.create_combo_box()
        catalog.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}.catalog")
        self._fill_combo(catalog, catalog_labels)
        source = self._document.create_text_input(adapter.source)
        source.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}.source")
        source.placeholder = "Local path or Hugging Face repository"
        root.add_preferred_child(header)
        root.add_preferred_child(catalog.widget)
        root.add_preferred_child(source.widget)
        self.text_to_image_lora_container.add_preferred_child(root)
        update = lambda field, value: self._emit(
            GenerationAction.UPDATE_TEXT_TO_IMAGE_LORA,
            (adapter_id, field, value),
        )
        connections = [
            enabled.connect_changed(
                lambda changed: update("enabled", changed)),
            label.connect_changed(
                lambda changed: update("label", changed)),
            source.connect_changed(
                lambda changed: update("source", changed)),
            weight.connect_changed(
                lambda changed: update("weight", changed)),
            catalog.connect_changed(
                lambda index, *_rest:
                self._select_text_to_image_lora_catalog(
                    adapter_id, index)),
            up.connect_clicked(lambda: self._emit(
                GenerationAction.MOVE_TEXT_TO_IMAGE_LORA,
                (adapter_id, -1),
            )),
            down.connect_clicked(lambda: self._emit(
                GenerationAction.MOVE_TEXT_TO_IMAGE_LORA,
                (adapter_id, 1),
            )),
            remove.connect_clicked(lambda: self._emit(
                GenerationAction.REMOVE_TEXT_TO_IMAGE_LORA,
                adapter_id,
            )),
        ]
        return {
            "stable_id": adapter_id,
            "root": root,
            "enabled": enabled,
            "label": label,
            "source": source,
            "catalog": catalog,
            "weight": weight,
            "up": up,
            "down": down,
            "remove": remove,
            "connections": connections,
        }

    def _select_text_to_image_lora_catalog(
            self, adapter_id: str, index: int) -> None:
        if self._syncing or self._closed:
            return
        if not 0 < index < len(
                self._text_to_image_lora_catalog_paths):
            return
        self._emit(
            GenerationAction.UPDATE_TEXT_TO_IMAGE_LORA,
            (
                adapter_id,
                "source",
                self._text_to_image_lora_catalog_paths[index],
            ),
        )

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

    def _text_area_callback(self, parent, suffix, placeholder, callback):
        area = self._document.create_text_area("")
        area.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}")
        area.placeholder = placeholder
        self._connections.append(area.connect_changed(
            lambda changed: None
            if self._syncing or self._closed else callback(changed)))
        parent.add_fixed_child(area.widget, 64.0)
        return area

    def _text_input_parameter(
            self, parent, label, suffix, placeholder, callback):
        self._caption(parent, label, f"{suffix}.caption")
        field = self._document.create_text_input("")
        field.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}")
        field.placeholder = placeholder
        self._connections.append(field.connect_changed(
            lambda changed: None
            if self._syncing or self._closed else callback(changed)))
        parent.add_preferred_child(field.widget)
        return field

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

    def _seed_parameter_row(self, parent, suffix, callback):
        row = self._document.create_hstack("NativeGenerationSeedRow")
        row.set_layout_spacing(4.0)
        seed = self._document.create_text_input("-1")
        seed.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}")
        seed.placeholder = "seed"
        self._connections.append(seed.connect_changed(
            lambda changed: None
            if self._syncing or self._closed else callback(changed)))
        random_button = self._button(
            row, "Rnd", f"{suffix}.random",
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

    def _slider_callback(
            self, parent, label, suffix, value, minimum, maximum,
            step, decimals, callback):
        slider = self._document.create_slider_edit(float(value))
        slider.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}")
        slider.label = label
        slider.set_range(float(minimum), float(maximum))
        slider.set_step(float(step))
        slider.set_decimals(decimals)
        self._connections.append(slider.connect_changed(
            lambda changed: None
            if self._syncing or self._closed else callback(changed)))
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

    def _checkbox_callback(
            self, parent, label, suffix, callback):
        checkbox = self._document.create_checkbox(False)
        checkbox.widget.stable_id = (
            f"diffusion-editor.generation.{suffix}")
        self._connections.append(checkbox.connect_changed(
            lambda checked: None
            if self._syncing or self._closed else callback(checked)))
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

    def _set_image_edit_profile_ids(self, values) -> None:
        self._image_edit_profile_ids = values

    def _set_text_to_image_profile_ids(self, values) -> None:
        self._text_to_image_profile_ids = values

    def _set_image_edit_reference_ids(self, values) -> None:
        self._image_edit_reference_ids = values

    @staticmethod
    def _item(values, index: int, default):
        return values[index] if 0 <= index < len(values) else default

    def _new_seed_and_run(self) -> None:
        self._emit(GenerationAction.RANDOM_SEED)
        self._emit(GenerationAction.RUN)

    def _emit(self, action: GenerationAction, value=None) -> None:
        if not self._syncing and not self._closed:
            self._on_intent(GenerationIntent(action, value))
