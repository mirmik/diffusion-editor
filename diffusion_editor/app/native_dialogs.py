"""termin-gui-native application dialog service."""

from __future__ import annotations

from typing import Callable

from termin.gui_native import (
    DialogAction,
    FileDialogMode,
    FileDialogModel,
    MessageBoxKind,
    Rect,
    Size,
    TcDocument,
)

from ..grounding.types import GROUNDING_MODELS, SAM2_MODELS, GroundingParams
from ..document.session import RecoveryRecord
from .application import (
    MAX_HISTORY_MEMORY_LIMIT_GIB,
    MIN_HISTORY_MEMORY_LIMIT_GIB,
)
from .dialogs import (
    FileDialogKind,
    FileDialogSpec,
    SettingsState,
    UnsavedDecision,
)


class NativeApplicationDialogs:
    """Reusable native overlays with app-owned result callbacks."""

    def __init__(
            self,
            document: TcDocument,
            viewport_rect: Callable[[], Rect],
            request_repaint: Callable[[], None]) -> None:
        self._document = document
        self._viewport_rect = viewport_rect
        self._request_repaint = request_repaint
        self._closed = False
        self._connections: list[object] = []
        self._file_callbacks: dict[
            FileDialogKind, Callable[[str | None], None] | None] = {}
        self._file_dialogs = {}
        self._settings_callback = None
        self._grounding_callback = None
        self._message_boxes = []
        self._lifecycle_dialogs = []
        self._build_settings_dialog()
        self._build_grounding_dialog()

    def show_file_dialog(
            self,
            spec: FileDialogSpec,
            on_finished: Callable[[str | None], None]) -> None:
        self._require_open()
        dialog = self._file_dialog(spec.kind)
        if dialog.open:
            dialog.activate("cancel")
        self._file_callbacks[spec.kind] = on_finished
        dialog.set_initial_directory(spec.directory)
        dialog.set_filters(FileDialogModel.parse_filter_string(spec.filters))
        dialog.set_file_name(spec.file_name)
        dialog.show(self._viewport())
        self._request_repaint()

    def show_settings_dialog(
            self,
            state: SettingsState,
            on_finished: Callable[[SettingsState | None], None]) -> None:
        self._require_open()
        if self.settings_dialog.open:
            self.settings_dialog.close()
        self._settings_callback = on_finished
        self.settings_models_dir.text = state.models_dir
        self.settings_history.value = state.history_limit_gib
        self.settings_agent_url.text = state.agent_base_url
        self.settings_agent_key.text = state.agent_api_key
        self.settings_agent_model.text = state.agent_model
        self.settings_temperature.value = state.agent_temperature
        self.settings_max_tokens.value = float(state.agent_max_tokens)
        self.settings_timeout.value = state.agent_timeout_seconds
        self.settings_stream.checked = state.agent_stream
        self.settings_dialog.show(self._viewport())
        self._request_repaint()

    def show_grounding_dialog(
            self,
            gpu_available: bool,
            on_finished: Callable[[GroundingParams | None], None]) -> None:
        self._require_open()
        if self.grounding_dialog.open:
            self.grounding_dialog.close()
        self._grounding_callback = on_finished
        self.grounding_prompt.text = ""
        self.grounding_gpu.checked = gpu_available
        self.grounding_gpu.widget.enabled = gpu_available
        self.grounding_dialog.show(self._viewport())
        self._request_repaint()

    def show_error(self, title: str, message: str) -> None:
        self._require_open()
        box = self._document.create_message_box(
            title, message, MessageBoxKind.Error)
        box.widget.stable_id = "diffusion-editor.dialog.error"
        self._message_boxes.append(box)
        self._connections.append(box.connect_finished(
            lambda _result: self._release_message_box(box)))
        box.show(self._viewport())
        self._request_repaint()

    def show_unsaved_changes(
            self,
            action: str,
            on_finished: Callable[[UnsavedDecision], None]) -> None:
        self._require_open()
        dialog = self._document.create_dialog("Unsaved Changes")
        dialog.widget.stable_id = "diffusion-editor.dialog.unsaved"
        dialog.actions = [
            DialogAction("save", "Save", is_default=True),
            DialogAction("discard", "Discard"),
            DialogAction("cancel", "Cancel", is_cancel=True),
        ]
        label = self._document.create_label(
            f"Save changes before you {action}?")
        label.widget.preferred_size = Size(420.0, 48.0)
        dialog.set_content(label)
        self._lifecycle_dialogs.append(dialog)

        def finished(result) -> None:
            if dialog in self._lifecycle_dialogs:
                self._lifecycle_dialogs.remove(dialog)
            if not self._closed:
                on_finished(UnsavedDecision(result.action_id))

        self._connections.append(dialog.connect_finished(finished))
        dialog.show(self._viewport())
        self._request_repaint()

    def show_recovery(
            self,
            record: RecoveryRecord,
            on_finished: Callable[[bool], None]) -> None:
        self._require_open()
        source = record.project_path or "an unsaved document"
        dialog = self._document.create_dialog("Recover Document")
        dialog.widget.stable_id = "diffusion-editor.dialog.recovery"
        dialog.actions = [
            DialogAction("yes", "Recover", is_default=True),
            DialogAction("no", "Discard", is_cancel=True),
        ]
        label = self._document.create_label(
            f"Recover the last autosave for {source}?")
        label.widget.preferred_size = Size(420.0, 48.0)
        dialog.set_content(label)
        self._lifecycle_dialogs.append(dialog)

        def finished(result) -> None:
            if dialog in self._lifecycle_dialogs:
                self._lifecycle_dialogs.remove(dialog)
            if not self._closed:
                on_finished(result.action_id == "yes")

        self._connections.append(dialog.connect_finished(finished))
        dialog.show(self._viewport())
        self._request_repaint()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for dialog in self._file_dialogs.values():
            if dialog.open:
                dialog.activate("cancel")
        if self.settings_dialog.open:
            self.settings_dialog.close()
        if self.grounding_dialog.open:
            self.grounding_dialog.close()
        for box in tuple(self._message_boxes):
            if box.open:
                self._document.dismiss_overlay(box.handle)
        for dialog in tuple(self._lifecycle_dialogs):
            if dialog.open:
                dialog.close()
        self._file_callbacks.clear()
        self._settings_callback = None
        self._grounding_callback = None
        self._connections.clear()
        self._message_boxes.clear()
        self._lifecycle_dialogs.clear()

    def _file_dialog(self, kind: FileDialogKind):
        dialog = self._file_dialogs.get(kind)
        if dialog is not None:
            return dialog
        mode = {
            FileDialogKind.OPEN_FILE: FileDialogMode.OpenFile,
            FileDialogKind.SAVE_FILE: FileDialogMode.SaveFile,
            FileDialogKind.OPEN_DIRECTORY: FileDialogMode.OpenDirectory,
        }[kind]
        dialog = self._document.create_file_dialog(mode)
        dialog.widget.stable_id = (
            f"diffusion-editor.dialog.file.{kind.value}")
        self._connections.append(dialog.connect_path_finished(
            lambda path, selected_kind=kind:
            self._finish_file(selected_kind, path)))
        self._file_dialogs[kind] = dialog
        return dialog

    def _finish_file(
            self, kind: FileDialogKind, path: str | None) -> None:
        callback = self._file_callbacks.pop(kind, None)
        if callback is not None and not self._closed:
            callback(path)
        self._request_repaint()

    def _build_settings_dialog(self) -> None:
        document = self._document
        dialog = document.create_dialog("Settings")
        dialog.widget.stable_id = "diffusion-editor.dialog.settings"
        dialog.actions = [
            DialogAction("ok", "OK", is_default=True),
            DialogAction("cancel", "Cancel", is_cancel=True),
        ]
        content = document.create_vstack("NativeSettingsDialogContent")
        content.set_layout_spacing(5.0)

        self._caption(content, "Stable Diffusion models directory")
        models_row = document.create_hstack("NativeSettingsModelsRow")
        models_row.set_layout_spacing(4.0)
        self.settings_models_dir = document.create_text_input("")
        self.settings_models_dir.widget.stable_id = (
            "diffusion-editor.settings.models-dir")
        self.settings_models_dir.widget.preferred_size = Size(420.0, 24.0)
        browse = document.create_button("Browse...")
        browse.widget.stable_id = "diffusion-editor.settings.browse"
        self._connections.append(browse.connect_clicked(
            self._browse_models_directory))
        models_row.add_flex_child(self.settings_models_dir.widget, 1.0)
        models_row.add_fixed_child(browse.widget, 82.0)
        content.add_preferred_child(models_row)

        self.settings_history = self._spin(
            content,
            "Undo/Redo memory limit (GiB)",
            "history-limit",
            5.0,
            MIN_HISTORY_MEMORY_LIMIT_GIB,
            MAX_HISTORY_MEMORY_LIMIT_GIB,
            0.25,
            2,
        )
        self.settings_history_note = self._caption(
            content,
            "Older history entries are removed when the limit is exceeded.",
            "diffusion-editor.settings.history-note",
        )
        self._caption(content, "Agent Chat API")
        self.settings_agent_url = self._input(
            content, "Base URL", "settings.agent-url")
        self.settings_agent_key = self._input(
            content, "API key", "settings.agent-key")
        self.settings_agent_model = self._input(
            content, "Model", "settings.agent-model")

        row = document.create_hstack("NativeSettingsAgentNumbers")
        row.set_layout_spacing(4.0)
        self.settings_temperature = self._spin_cell(
            row, "Temperature", "temperature",
            0.7, 0.0, 2.0, 0.05, 2)
        self.settings_max_tokens = self._spin_cell(
            row, "Max tokens", "max-tokens",
            1024, 0, 131072, 128, 0)
        self.settings_timeout = self._spin_cell(
            row, "Timeout sec", "timeout",
            60, 5, 600, 5, 0)
        content.add_preferred_child(row)
        self.settings_stream = self._checkbox(
            content, "Stream responses", "settings.stream")
        # Constrain width to match the legacy dialog, but let the vertical
        # layout derive its height from the actual fields.
        content.preferred_size = Size(520.0, 0.0)
        self.settings_content = content
        dialog.set_content(content)
        self._connections.append(dialog.connect_finished(
            self._finish_settings))
        self.settings_dialog = dialog

    def _browse_models_directory(self) -> None:
        spec = FileDialogSpec(
            FileDialogKind.OPEN_DIRECTORY,
            "Select models directory",
            self.settings_models_dir.text,
        )
        self.show_file_dialog(
            spec,
            lambda path: self._set_models_directory(path),
        )

    def _set_models_directory(self, path: str | None) -> None:
        if path:
            self.settings_models_dir.text = path

    def _finish_settings(self, result) -> None:
        callback, self._settings_callback = self._settings_callback, None
        if callback is None or self._closed:
            return
        if result.action_id != "ok":
            callback(None)
            self._request_repaint()
            return
        callback(SettingsState(
            models_dir=self.settings_models_dir.text,
            history_limit_gib=self.settings_history.value,
            agent_base_url=self.settings_agent_url.text,
            agent_api_key=self.settings_agent_key.text,
            agent_model=self.settings_agent_model.text,
            agent_temperature=self.settings_temperature.value,
            agent_max_tokens=int(self.settings_max_tokens.value),
            agent_timeout_seconds=self.settings_timeout.value,
            agent_stream=self.settings_stream.checked,
        ))
        self._request_repaint()

    def _build_grounding_dialog(self) -> None:
        document = self._document
        dialog = document.create_dialog("Detect Objects")
        dialog.widget.stable_id = "diffusion-editor.dialog.grounding"
        dialog.actions = [
            DialogAction("detect", "Detect", is_default=True),
            DialogAction("cancel", "Cancel", is_cancel=True),
        ]
        scroll = document.create_scroll_area()
        scroll.widget.preferred_size = Size(520.0, 500.0)
        scroll.set_scroll_axes(False, True)
        content = document.create_vstack("NativeGroundingDialogContent")
        content.set_layout_spacing(4.0)
        scroll.set_content(content)

        self.grounding_prompt = self._input(
            content, "Objects to find", "grounding.prompt")
        self.grounding_prompt.placeholder = "e.g. cat. red cup. face."
        self.grounding_model = self._combo(
            content, "Grounding DINO model", "grounding.model",
            [label for label, _model in GROUNDING_MODELS], 1)
        self.grounding_box_threshold = self._slider(
            content, "Box confidence threshold", "grounding.box-threshold",
            0.40, 0.05, 1.0, 0.05, 2)
        self.grounding_text_threshold = self._slider(
            content, "Text token threshold", "grounding.text-threshold",
            0.30, 0.05, 1.0, 0.05, 2)
        self.grounding_sam_caption = self._caption(
            content,
            "SAM 2.1 segmentation (experimental)",
            "diffusion-editor.grounding.sam.caption",
        )
        self.grounding_sam = self._checkbox(
            content, "Segment with SAM 2.1", "grounding.sam", True)
        self.grounding_sam_model = self._combo(
            content, "SAM 2.1 model", "grounding.sam-model",
            [label for label, _model in SAM2_MODELS], 0)
        self.grounding_mask_channel = self._combo(
            content, "Mask channel", "grounding.mask-channel",
            ["whole (full object)", "part", "subpart"], 0)
        self.grounding_mask_threshold = self._slider(
            content, "Mask threshold (higher = tighter)",
            "grounding.mask-threshold",
            0.0, 0.0, 1.0, 0.05, 2)
        self.grounding_max_hole = self._slider(
            content, "Max hole area (0 = off, px)", "grounding.max-hole",
            0, 0, 10000, 100, 0)
        self.grounding_max_sprinkle = self._slider(
            content, "Max sprinkle area (0 = off, px)",
            "grounding.max-sprinkle",
            0, 0, 10000, 100, 0)
        self.grounding_multimask = self._checkbox(
            content, "Multimask output (3 candidates per box)",
            "grounding.multimask", True)
        self.grounding_non_overlap = self._checkbox(
            content, "Non-overlapping masks",
            "grounding.non-overlap", False)
        self.grounding_gpu = self._checkbox(
            content, "Use GPU", "grounding.gpu", False)
        dialog.set_content(scroll.widget)
        self._connections.append(dialog.connect_finished(
            self._finish_grounding))
        self.grounding_dialog = dialog

    def _finish_grounding(self, result) -> None:
        callback, self._grounding_callback = self._grounding_callback, None
        if callback is None or self._closed:
            return
        if result.action_id != "detect":
            callback(None)
            self._request_repaint()
            return
        prompt = self.grounding_prompt.text.strip()
        if not prompt:
            callback(None)
            self.show_error(
                "Detect Objects", "Enter one or more objects to find.")
            return
        if not prompt.endswith("."):
            prompt += "."
        model_index = max(0, self.grounding_model.selected_index)
        sam_index = max(0, self.grounding_sam_model.selected_index)
        callback(GroundingParams(
            prompt=prompt,
            model_id=GROUNDING_MODELS[model_index][1],
            box_threshold=self.grounding_box_threshold.value,
            text_threshold=self.grounding_text_threshold.value,
            use_gpu=self.grounding_gpu.checked,
            sam2_model_id=(
                SAM2_MODELS[sam_index][1]
                if self.grounding_sam.checked else None),
            sam2_mask_channel=max(
                0, self.grounding_mask_channel.selected_index),
            mask_threshold=self.grounding_mask_threshold.value,
            max_hole_area=int(self.grounding_max_hole.value),
            max_sprinkle_area=int(
                self.grounding_max_sprinkle.value),
            multimask=self.grounding_multimask.checked,
            non_overlap=self.grounding_non_overlap.checked,
        ))
        self._request_repaint()

    def _input(self, parent, caption: str, suffix: str):
        self._caption(parent, caption)
        field = self._document.create_text_input("")
        field.widget.stable_id = f"diffusion-editor.{suffix}"
        parent.add_preferred_child(field.widget)
        return field

    def _spin(
            self, parent, caption, suffix, value, minimum, maximum,
            step, decimals, *, add=True):
        field = self._document.create_spin_box(float(value))
        field.widget.stable_id = f"diffusion-editor.settings.{suffix}"
        field.set_range(float(minimum), float(maximum))
        field.step = float(step)
        field.decimals = int(decimals)
        if add:
            self._caption(parent, caption)
            parent.add_preferred_child(field.widget)
        return field

    def _spin_cell(
            self, parent, caption, suffix, value, minimum, maximum,
            step, decimals):
        cell = self._document.create_vstack(
            f"NativeSettings{suffix.title().replace('-', '')}Cell")
        cell.set_layout_spacing(3.0)
        caption_widget = self._caption(
            cell,
            caption,
            f"diffusion-editor.settings.{suffix}.caption",
        )
        setattr(
            self,
            f"settings_{suffix.replace('-', '_')}_caption",
            caption_widget,
        )
        field = self._spin(
            cell, caption, suffix, value, minimum, maximum,
            step, decimals, add=False)
        cell.add_preferred_child(field.widget)
        parent.add_flex_child(cell, 1.0)
        return field

    def _slider(
            self, parent, label, suffix, value, minimum, maximum,
            step, decimals):
        field = self._document.create_slider_edit(float(value))
        field.widget.stable_id = f"diffusion-editor.{suffix}"
        field.label = label
        field.set_range(float(minimum), float(maximum))
        field.set_step(float(step))
        field.set_decimals(int(decimals))
        parent.add_preferred_child(field.widget)
        return field

    def _combo(self, parent, caption, suffix, labels, selected):
        self._caption(parent, caption)
        combo = self._document.create_combo_box()
        combo.widget.stable_id = f"diffusion-editor.{suffix}"
        for label in labels:
            combo.add_item(label)
        combo.selected_index = selected
        parent.add_preferred_child(combo.widget)
        return combo

    def _checkbox(
            self, parent, label, suffix, checked=False):
        row = self._document.create_hstack("NativeDialogCheckboxRow")
        row.set_layout_spacing(4.0)
        checkbox = self._document.create_checkbox(bool(checked))
        checkbox.widget.stable_id = f"diffusion-editor.{suffix}"
        text = self._document.create_label(label, "NativeDialogCheckboxLabel")
        text.stable_id = f"diffusion-editor.{suffix}.label"
        setattr(self, f"{suffix.replace('.', '_').replace('-', '_')}_label", text)
        row.add_preferred_child(checkbox.widget)
        row.add_flex_child(text, 1.0)
        parent.add_preferred_child(row)
        return checkbox

    def _caption(self, parent, text, stable_id=None):
        label = self._document.create_label(text, "NativeDialogCaption")
        if stable_id is not None:
            label.stable_id = stable_id
        parent.add_preferred_child(label)
        return label

    def _release_message_box(self, box) -> None:
        if box in self._message_boxes:
            self._message_boxes.remove(box)

    def _viewport(self) -> Rect:
        viewport = self._viewport_rect()
        if viewport.width <= 0 or viewport.height <= 0:
            return Rect(0.0, 0.0, 800.0, 600.0)
        return viewport

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("native application dialogs are closed")
