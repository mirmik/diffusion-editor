"""Native shell and app-owned command projection for Diffusion Editor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from termin.gui_native import (
    CommandData,
    CommandKind,
    CommandModel,
    MenuBarEntry,
    SrgbColor,
    StyleField,
    TcDocument,
)

from ..generation.types import (
    RECONSTRUCTION_PREVIEW_STAGES,
    RECONSTRUCTION_STAGES,
    RECONSTRUCTION_STAGE_LABELS,
    ReconstructionParameters,
    ReconstructionStage,
    ReconstructionStageStatus,
)
from .presentation import ViewPorts


CommandHandler = Callable[[], None]

_CHROME_BACKGROUND = SrgbColor(0.075, 0.080, 0.095, 1.0)
_WORKSPACE_EDGE = SrgbColor(0.32, 0.35, 0.40, 1.0)
_RECONSTRUCTION_RESOLUTIONS = (1024, 1280, 1536)
_RECONSTRUCTION_LR_CONDITIONING_RESOLUTIONS = (512, 1024)
_RECONSTRUCTION_TEXTURE_SIZES = (1024, 2048, 4096)


def _set_widget_background(widget, color: SrgbColor) -> None:
    style_override = widget.style_override
    style = style_override.value
    style.background = color
    style_override.value = style
    style_override.fields = StyleField.Background.value
    widget.style_override = style_override


@dataclass(frozen=True)
class NativeCommandSpec:
    stable_id: str
    label: str
    shortcut: str = ""
    tooltip: str = ""
    canonical_id: str | None = None
    checkable: bool = False
    checked: bool = False
    enabled: bool = True

    @property
    def action_id(self) -> str:
        return self.canonical_id or self.stable_id


COMMAND_SPECS = (
    NativeCommandSpec("file.new", "New…", "Ctrl+N"),
    NativeCommandSpec("file.new_from_image", "New From Image…"),
    NativeCommandSpec("file.open", "Open…", "Ctrl+O"),
    NativeCommandSpec("file.save", "Save", "Ctrl+S"),
    NativeCommandSpec("file.save_as", "Save As…", "Ctrl+Shift+S"),
    NativeCommandSpec("file.import", "Import Image…", "Ctrl+I"),
    NativeCommandSpec("file.export", "Export Image…", "Ctrl+E"),
    NativeCommandSpec("app.quit", "Quit", "Ctrl+Q"),
    NativeCommandSpec("edit.undo", "Undo", "Ctrl+Z"),
    NativeCommandSpec("edit.redo", "Redo", "Ctrl+Shift+Z"),
    NativeCommandSpec(
        "edit.redo.ctrl_y",
        "Redo",
        "Ctrl+Y",
        canonical_id="edit.redo",
    ),
    NativeCommandSpec("edit.copy", "Copy", "Ctrl+C"),
    NativeCommandSpec("edit.copy_visible", "Copy Visible", "Ctrl+Shift+C"),
    NativeCommandSpec("edit.paste", "Paste", "Ctrl+V"),
    NativeCommandSpec("edit.settings", "Settings…"),
    NativeCommandSpec("selection.all", "Select All", "Ctrl+A"),
    NativeCommandSpec(
        "selection.background",
        "Select Background",
        tooltip="Select the image background using automatic segmentation",
    ),
    NativeCommandSpec("selection.clear", "Clear Selection", "Ctrl+D"),
    NativeCommandSpec("selection.invert", "Invert Selection", "Ctrl+Shift+I"),
    NativeCommandSpec("layer.new", "New Layer", "Ctrl+Shift+N"),
    NativeCommandSpec(
        "layer.new_3d_reconstruction",
        "New 3D Reconstruction",
        tooltip="Create a 3D reconstruction object from the current composite",
    ),
    NativeCommandSpec("layer.remove", "Remove Layer"),
    NativeCommandSpec("layer.flatten", "Flatten"),
    NativeCommandSpec("layer.detect", "Detect Objects…"),
    NativeCommandSpec(
        "generation.3d",
        "Generate to Selected Stage",
        tooltip="Generate a Pixal3D model from the current composite",
    ),
    NativeCommandSpec(
        "generation.3d_cancel",
        "Cancel 3D Generation",
        enabled=False,
    ),
    NativeCommandSpec(
        "view.3d_light_from_camera",
        "Light from Camera",
        tooltip="Set the 3D light direction from the current camera",
        enabled=False,
    ),
    NativeCommandSpec("view.fit", "Fit"),
    NativeCommandSpec(
        "view.agent_panel",
        "Agent Panel",
        checkable=True,
        checked=False,
    ),
)

COMMAND_SPEC_BY_ID = {spec.stable_id: spec for spec in COMMAND_SPECS}

MENU_COMMANDS = (
    (
        "file",
        "File",
        (
            "file.new",
            "file.new_from_image",
            None,
            "file.open",
            "file.save",
            "file.save_as",
            None,
            "file.import",
            "file.export",
            None,
            "app.quit",
        ),
    ),
    (
        "edit",
        "Edit",
        (
            "edit.undo",
            "edit.redo",
            "edit.redo.ctrl_y",
            "edit.copy",
            "edit.copy_visible",
            "edit.paste",
            None,
            "edit.settings",
        ),
    ),
    (
        "selection",
        "Select",
        (
            "selection.all",
            "selection.background",
            "selection.clear",
            None,
            "selection.invert",
        ),
    ),
    (
        "layer",
        "Layer",
        (
            "layer.new",
            "layer.new_3d_reconstruction",
            "layer.remove",
            None,
            "layer.flatten",
            None,
            "layer.detect",
        ),
    ),
    (
        "generation",
        "3D",
        (
            "generation.3d",
            "generation.3d_cancel",
            "view.3d_light_from_camera",
        ),
    ),
    (
        "view",
        "View",
        (
            "view.fit",
            None,
            "view.agent_panel",
        ),
    ),
)

TOOLBAR_COMMANDS = (
    "file.open",
    "file.save",
    None,
    "view.fit",
    "layer.new_3d_reconstruction",
    None,
    "selection.all",
    "selection.clear",
    "selection.invert",
)

RECONSTRUCTION_TOOLBAR_COMMANDS = (
    "generation.3d",
    "generation.3d_cancel",
    None,
    "view.3d_light_from_camera",
)


class NativeEditorView:
    """Native root layout plus presentation ports for the application owner."""

    def __init__(
            self,
            document: TcDocument,
            request_repaint: Callable[[], None],
            set_window_title: Callable[[str], None],
            command_handlers: Mapping[str, CommandHandler]) -> None:
        self._document = document
        self._request_repaint = request_repaint
        self._set_platform_window_title = set_window_title
        self._command_handlers = dict(command_handlers)
        self._command_refs: dict[str, list[tuple[CommandModel, object]]] = {}
        self._command_buttons: dict[str, list[object]] = {}
        self._command_states = {
            spec.action_id: (spec.enabled, spec.checked)
            for spec in COMMAND_SPECS
        }
        self._connections: list[object] = []
        self.window_title = "Diffusion Editor"
        self.closed = False

        self.root = document.create_vstack("DiffusionEditorNativeRoot")
        self.root.stable_id = "diffusion-editor.root"
        self.root.set_layout_spacing(0.0)

        self.menu_bar = document.create_menu_bar()
        self.menu_bar.widget.stable_id = "diffusion-editor.menu-bar"
        self.menu_models: dict[str, CommandModel] = {}
        for menu_id, label, command_ids in MENU_COMMANDS:
            model = self._create_model(command_ids)
            self.menu_models[menu_id] = model
            self.menu_bar.add_menu(MenuBarEntry(menu_id, label, model))
        self._connections.append(
            self.menu_bar.connect_activated(self._on_menu_activated)
        )

        self.toolbar_model = self._create_model(TOOLBAR_COMMANDS)
        self.toolbar = document.create_tool_bar(self.toolbar_model)
        self.toolbar.widget.stable_id = "diffusion-editor.toolbar"
        self.toolbar.item_height = 26.0
        self.toolbar.padding = 4.0
        _set_widget_background(self.toolbar.widget, _CHROME_BACKGROUND)
        self._connections.append(
            self.toolbar.connect_activated(self._on_toolbar_activated)
        )
        self.toolbar_workspace_edge = document.create_separator(True)
        self.toolbar_workspace_edge.stable_id = (
            "diffusion-editor.toolbar-workspace-edge"
        )
        self.toolbar_workspace_edge.set_color(_WORKSPACE_EDGE)
        self.toolbar_workspace_edge.set_thickness(2.0)

        self.left_panel = document.create_vstack("DiffusionEditorLeftPanel")
        self.left_panel.stable_id = "diffusion-editor.left-panel.content"
        self.left_panel.set_layout_spacing(4.0)
        self.left_scroll = document.create_scroll_area()
        self.left_scroll.widget.stable_id = "diffusion-editor.left-panel"
        self.left_scroll.set_scroll_axes(False, True)
        self.left_scroll.set_content(self.left_panel)
        self.left_placeholder = document.create_label(
            "Tool controls", "DiffusionEditorLeftPanelLabel")
        self.left_placeholder.stable_id = "diffusion-editor.left-panel.label"
        self.left_panel.add_preferred_child(self.left_placeholder)
        self.canvas_controls_view = None
        self.generation_panels_view = None
        self.panel_presentation = None
        self.canvas_host = document.create_vstack(
            "DiffusionEditorCanvasHost")
        self.canvas_host.stable_id = "diffusion-editor.canvas-host"
        self.canvas_placeholder = document.create_label(
            "Canvas", "DiffusionEditorCanvasHostLabel")
        self.canvas_placeholder.stable_id = "diffusion-editor.canvas-host.label"
        self.canvas_host.add_preferred_child(self.canvas_placeholder)
        self.canvas_view = None
        self.reconstruction_viewport = None
        self.canvas_reconstruction_splitter = None
        self.reconstruction_toolbar_model = None
        self.reconstruction_toolbar = None
        self.reconstruction_buttons = {}
        self.reconstruction_stage_buttons = {}
        self.reconstruction_stage_checks = {}
        self._reconstruction_stage_handler = None
        self._reconstruction_parameter_handler = None
        self._syncing_reconstruction_parameters = False
        self.reconstruction_parameter_controls = {}
        self.reconstruction_status = None
        self.reconstruction_panel = None
        self.reconstruction_mode = False
        self._canvas_splitter_parking = document.create_vstack(
            "DiffusionEditorCanvasSplitterParking"
        )
        self.layer_panel = document.create_vstack("DiffusionEditorLayerPanel")
        self.layer_panel.stable_id = "diffusion-editor.layer-panel"
        self.layer_panel.set_layout_spacing(4.0)
        self.layer_placeholder = document.create_label(
            "Layers", "DiffusionEditorLayerPanelLabel")
        self.layer_placeholder.stable_id = "diffusion-editor.layer-panel.label"
        self.layer_panel.add_preferred_child(self.layer_placeholder)
        self.layer_panel_view = None
        self.agent_panel = self._placeholder(
            document,
            "diffusion-editor.agent-panel",
            "Agent",
            "DiffusionEditorAgentPanel",
        )
        self.agent_chat_view = None
        self.agent_panel_visible = False
        self._agent_panel_parking = document.create_vstack(
            "DiffusionEditorAgentPanelParking"
        )

        self.right_splitter = document.create_splitter(
            True,
            "DiffusionEditorRightSplitter",
        )
        self.right_splitter.widget.stable_id = "diffusion-editor.right-splitter"
        self.right_splitter.set_first(self._agent_panel_parking)
        self.right_splitter.set_second(self.agent_panel)
        self.right_splitter.set_split_fraction(0.41)
        self.right_splitter.set_min_extents(180.0, 240.0)
        self.right_splitter.widget.visible = False

        self.right_host = document.create_hstack("DiffusionEditorRightHost")
        self.right_host.stable_id = "diffusion-editor.right-host"
        self.right_host.set_layout_spacing(0.0)
        self.right_host.add_flex_child(self.layer_panel, 1.0)
        self.right_host.add_flex_child(self.right_splitter.widget, 1.0)

        self.workspace_splitter = document.create_splitter(
            True,
            "DiffusionEditorWorkspaceSplitter",
        )
        self.workspace_splitter.widget.stable_id = (
            "diffusion-editor.workspace-splitter"
        )
        self.workspace_splitter.set_first(self.canvas_host)
        self.workspace_splitter.set_second(self.right_host)
        self._workspace_fraction_without_agent = 0.72
        self._workspace_fraction_with_agent = 0.46
        self.workspace_splitter.set_split_fraction(
            self._workspace_fraction_without_agent
        )
        self.workspace_splitter.set_min_extents(320.0, 240.0)

        self.main_splitter = document.create_splitter(
            True,
            "DiffusionEditorMainSplitter",
        )
        self.main_splitter.widget.stable_id = "diffusion-editor.main-splitter"
        self.main_splitter.set_first(self.left_scroll.widget)
        self.main_splitter.set_second(self.workspace_splitter.widget)
        self.main_splitter.set_split_fraction(0.20)
        self.main_splitter.set_min_extents(180.0, 480.0)

        self.status_bar = document.create_status_bar("Ready")
        self.status_bar.widget.stable_id = "diffusion-editor.status"
        _set_widget_background(self.status_bar.widget, _CHROME_BACKGROUND)
        self.status_workspace_edge = document.create_separator(True)
        self.status_workspace_edge.stable_id = (
            "diffusion-editor.status-workspace-edge"
        )
        self.status_workspace_edge.set_color(_WORKSPACE_EDGE)
        self.status_workspace_edge.set_thickness(2.0)

        self.root.add_fixed_child(self.menu_bar.widget, 28.0)
        self.root.add_fixed_child(self.toolbar.widget, 34.0)
        self.root.add_fixed_child(self.toolbar_workspace_edge, 2.0)
        self.root.add_flex_child(self.main_splitter.widget, 1.0)
        self.root.add_fixed_child(self.status_workspace_edge, 2.0)
        self.root.add_fixed_child(self.status_bar.widget, 22.0)
        if not document.add_root(self.root.handle):
            raise RuntimeError("failed to add the Diffusion Editor native root")
        self._command_handlers["view.agent_panel"] = (
            self.toggle_agent_panel
        )

    @property
    def command_inventory(self) -> tuple[str, ...]:
        return tuple(spec.stable_id for spec in COMMAND_SPECS)

    def set_status(self, text: str) -> None:
        self._require_open()
        self.status_bar.text = text
        self._request_repaint()

    def set_window_title(self, title: str) -> None:
        self._require_open()
        self.window_title = title
        self._set_platform_window_title(title)

    def set_command_state(
            self,
            command_id: str,
            *,
            enabled: bool,
            checked: bool = False) -> None:
        self._require_open()
        canonical_id = self._canonical_id(command_id)
        refs = self._command_refs.get(canonical_id)
        if not refs:
            raise KeyError(f"unknown native command: {command_id}")
        self._command_states[canonical_id] = (enabled, checked)
        for model, native_id in refs:
            model.set_enabled(native_id, enabled)
            if model.command(native_id).data.checkable:
                model.set_checked(native_id, checked)
        for button in self._command_buttons.get(canonical_id, ()):
            button.widget.enabled = enabled
        self._request_repaint()

    def set_command_handler(
            self,
            command_id: str,
            handler: CommandHandler | None) -> None:
        canonical_id = self._canonical_id(command_id)
        if handler is None:
            self._command_handlers.pop(canonical_id, None)
        else:
            self._command_handlers[canonical_id] = handler

    def activate_command(self, command_id: str) -> bool:
        self._require_open()
        canonical_id = self._canonical_id(command_id)
        enabled, _checked = self._command_states[canonical_id]
        handler = self._command_handlers.get(canonical_id)
        if not enabled or handler is None:
            return False
        handler()
        return True

    def dispatch_shortcut(self, key: int, modifiers: int) -> bool:
        self._require_open()
        if self.menu_bar.dispatch_shortcut(key, modifiers):
            return True
        return bool(
            self.canvas_view is not None
            and self.canvas_view.dispatch_shortcut(key, modifiers)
        )

    def toggle_agent_panel(self) -> None:
        self.set_agent_panel_visible(not self.agent_panel_visible)

    def set_agent_panel_visible(self, visible: bool) -> None:
        self._require_open()
        visible = bool(visible)
        if visible == self.agent_panel_visible:
            return
        if visible:
            self._workspace_fraction_without_agent = (
                self.workspace_splitter.split_fraction
            )
        else:
            self._workspace_fraction_with_agent = (
                self.workspace_splitter.split_fraction
            )
        self.agent_panel_visible = visible
        if visible:
            self.right_splitter.set_first(self.layer_panel)
            self.right_splitter.widget.visible = True
            self.workspace_splitter.set_min_extents(320.0, 420.0)
            self.workspace_splitter.set_split_fraction(
                self._workspace_fraction_with_agent
            )
        else:
            self.right_splitter.set_first(self._agent_panel_parking)
            self.right_host.add_flex_child(self.layer_panel, 1.0)
            self.right_splitter.widget.visible = False
            self.workspace_splitter.set_min_extents(320.0, 240.0)
            self.workspace_splitter.set_split_fraction(
                self._workspace_fraction_without_agent
            )
        self.set_command_state(
            "view.agent_panel",
            enabled=True,
            checked=visible,
        )

    def mount_canvas(self, canvas_view) -> None:
        self._require_open()
        if self.canvas_view is not None:
            raise RuntimeError("native canvas is already mounted")
        self.canvas_host.remove_child(self.canvas_placeholder)
        self.canvas_host.add_flex_child(canvas_view.widget, 1.0)
        capture_widget = getattr(canvas_view, "pointer_capture_widget", None)
        if capture_widget is not None:
            self.canvas_host.add_fixed_child(capture_widget, 0.0)
        self.canvas_view = canvas_view
        self._request_repaint()

    def mount_canvas_controls(self, controls_view) -> None:
        self._require_open()
        if self.canvas_controls_view is not None:
            raise RuntimeError("native canvas controls are already mounted")
        self.left_panel.remove_child(self.left_placeholder)
        self.left_panel.add_preferred_child(controls_view.widget)
        self.canvas_controls_view = controls_view
        self._request_repaint()

    def mount_reconstruction_viewport(self, viewport_view) -> None:
        self._require_open()
        if self.canvas_view is None:
            raise RuntimeError("native canvas must be mounted before the 3D viewport")
        if self.reconstruction_viewport is not None:
            raise RuntimeError("reconstruction viewport is already mounted")
        splitter = self._document.create_splitter(
            True,
            "DiffusionEditorCanvasReconstructionSplitter",
        )
        splitter.widget.stable_id = "diffusion-editor.reconstruction.splitter"
        splitter.set_first(self._canvas_splitter_parking)
        splitter.set_second(viewport_view.widget)
        splitter.set_split_fraction(0.50)
        splitter.set_min_extents(260.0, 260.0)
        toolbar_model = self._create_model(RECONSTRUCTION_TOOLBAR_COMMANDS)
        toolbar = self._document.create_vstack(
            "DiffusionEditorReconstructionActions"
        )
        toolbar.stable_id = "diffusion-editor.reconstruction.toolbar"
        toolbar.set_layout_spacing(4.0)
        reconstruction_buttons = {}
        for command_id in RECONSTRUCTION_TOOLBAR_COMMANDS:
            if command_id is None:
                continue
            canonical_id = self._canonical_id(command_id)
            spec = COMMAND_SPEC_BY_ID[command_id]
            button = self._document.create_button(spec.label)
            button.widget.stable_id = (
                "diffusion-editor.reconstruction."
                + command_id.replace(".", "-")
            )
            button.widget.enabled = self._command_states[canonical_id][0]
            self._connections.append(button.connect_clicked(
                lambda command_id=command_id: self.activate_command(command_id)
            ))
            self._command_buttons.setdefault(canonical_id, []).append(button)
            reconstruction_buttons[command_id] = button
            toolbar.add_preferred_child(button.widget)

        status = self._document.create_label(
            "Empty", "DiffusionEditorReconstructionStatus"
        )
        status.stable_id = "diffusion-editor.reconstruction.status"
        title = self._document.create_label(
            "3D Reconstruction", "DiffusionEditorReconstructionTitle"
        )
        title.stable_id = "diffusion-editor.reconstruction.title"
        stage_list = self._document.create_vstack(
            "DiffusionEditorReconstructionStages"
        )
        stage_list.stable_id = "diffusion-editor.reconstruction.stages"
        stage_list.set_layout_spacing(2.0)
        stage_buttons = {}
        stage_checks = {}
        for stage in RECONSTRUCTION_STAGES:
            row = self._document.create_hstack(
                "DiffusionEditorReconstructionStageRow"
            )
            row.set_layout_spacing(4.0)
            checkbox = self._document.create_checkbox(False)
            checkbox.widget.stable_id = (
                f"diffusion-editor.reconstruction.stage-check.{stage.value}"
            )
            checkbox.widget.enabled = False
            button = self._document.create_button(
                RECONSTRUCTION_STAGE_LABELS[stage]
            )
            button.widget.stable_id = (
                f"diffusion-editor.reconstruction.stage.{stage.value}"
            )
            if stage in RECONSTRUCTION_PREVIEW_STAGES:
                self._connections.append(button.connect_clicked(
                    lambda stage=stage: self._activate_reconstruction_stage(stage)
                ))
            else:
                button.widget.enabled = False
            stage_buttons[stage] = button
            stage_checks[stage] = checkbox
            row.add_preferred_child(checkbox.widget)
            row.add_flex_child(button.widget, 1.0)
            stage_list.add_preferred_child(row)

        parameters_title = self._document.create_label(
            "Generation parameters", "DiffusionEditorReconstructionParametersTitle"
        )
        parameters_title.stable_id = (
            "diffusion-editor.reconstruction.parameters.title"
        )
        parameters_panel = self._document.create_vstack(
            "DiffusionEditorReconstructionParameters"
        )
        parameters_panel.stable_id = "diffusion-editor.reconstruction.parameters"
        parameters_panel.set_layout_spacing(3.0)
        defaults = ReconstructionParameters()
        parameter_controls = {}

        def add_slider(key, label, value, minimum, maximum, step, decimals=0):
            control = self._document.create_slider_edit(float(value))
            control.widget.stable_id = (
                f"diffusion-editor.reconstruction.parameter.{key}"
            )
            control.label = label
            control.set_range(float(minimum), float(maximum))
            control.set_step(float(step))
            control.set_decimals(int(decimals))
            self._connections.append(control.connect_changed(
                lambda changed, key=key: self._change_reconstruction_parameter(
                    key, changed
                )
            ))
            parameter_controls[key] = control
            parameters_panel.add_preferred_child(control.widget)

        def add_spin(key, label, value, minimum, maximum, step, decimals=0):
            row = self._document.create_hstack(
                "DiffusionEditorReconstructionParameterRow"
            )
            row.set_layout_spacing(4.0)
            caption = self._document.create_label(
                label, "DiffusionEditorReconstructionParameterLabel"
            )
            caption.stable_id = (
                f"diffusion-editor.reconstruction.parameter.{key}.label"
            )
            control = self._document.create_spin_box(float(value))
            control.widget.stable_id = (
                f"diffusion-editor.reconstruction.parameter.{key}"
            )
            control.set_range(float(minimum), float(maximum))
            control.step = float(step)
            control.decimals = int(decimals)
            self._connections.append(control.connect_changed(
                lambda changed, key=key: self._change_reconstruction_parameter(
                    key, changed
                )
            ))
            parameter_controls[key] = control
            row.add_flex_child(caption, 1.0)
            row.add_fixed_child(control.widget, 108.0)
            parameters_panel.add_preferred_child(row)

        def add_combo(key, label, values):
            row = self._document.create_hstack(
                "DiffusionEditorReconstructionParameterRow"
            )
            row.set_layout_spacing(4.0)
            caption = self._document.create_label(
                label, "DiffusionEditorReconstructionParameterLabel"
            )
            caption.stable_id = (
                f"diffusion-editor.reconstruction.parameter.{key}.label"
            )
            control = self._document.create_combo_box()
            control.widget.stable_id = (
                f"diffusion-editor.reconstruction.parameter.{key}"
            )
            for item in values:
                control.add_item(str(item))
            control.selected_index = values.index(getattr(defaults, key))
            self._connections.append(control.connect_changed(
                lambda index, *_rest, key=key, values=values:
                self._change_reconstruction_parameter(key, values[index])
            ))
            parameter_controls[key] = control
            row.add_flex_child(caption, 1.0)
            row.add_fixed_child(control.widget, 92.0)
            parameters_panel.add_preferred_child(row)

        add_spin("seed", "Seed", defaults.seed, 0, 2_147_483_647, 1)
        add_slider("steps", "Sampling steps", defaults.steps, 1, 50, 1)
        add_combo("resolution", "HR resolution", _RECONSTRUCTION_RESOLUTIONS)
        add_combo(
            "lr_conditioning_resolution",
            "LR conditioning",
            _RECONSTRUCTION_LR_CONDITIONING_RESOLUTIONS,
        )
        add_slider(
            "manual_fov_degrees", "Camera FOV (0 = Auto)",
            defaults.manual_fov_degrees, 0, 120, 1,
        )
        add_slider(
            "decimation_target", "Final mesh faces",
            defaults.decimation_target, 50_000, 1_000_000, 10_000,
        )
        add_combo("texture_size", "Texture size", _RECONSTRUCTION_TEXTURE_SIZES)
        low_vram_row = self._document.create_hstack(
            "DiffusionEditorReconstructionParameterRow"
        )
        low_vram_row.set_layout_spacing(4.0)
        low_vram = self._document.create_checkbox(defaults.low_vram)
        low_vram.widget.stable_id = (
            "diffusion-editor.reconstruction.parameter.low_vram"
        )
        low_vram_label = self._document.create_label(
            "Low VRAM", "DiffusionEditorReconstructionParameterLabel"
        )
        low_vram_label.stable_id = (
            "diffusion-editor.reconstruction.parameter.low_vram.label"
        )
        self._connections.append(low_vram.connect_changed(
            lambda checked: self._change_reconstruction_parameter(
                "low_vram", checked
            )
        ))
        parameter_controls["low_vram"] = low_vram
        low_vram_row.add_preferred_child(low_vram.widget)
        low_vram_row.add_flex_child(low_vram_label, 1.0)
        parameters_panel.add_preferred_child(low_vram_row)
        panel_content = self._document.create_vstack(
            "DiffusionEditorReconstructionPanelContent"
        )
        panel_content.stable_id = (
            "diffusion-editor.reconstruction.panel-content"
        )
        panel_content.set_layout_spacing(4.0)
        panel_content.add_preferred_child(title)
        panel_content.add_preferred_child(parameters_title)
        panel_content.add_preferred_child(parameters_panel)
        panel_content.add_preferred_child(stage_list)
        panel_content.add_preferred_child(toolbar)
        panel_content.add_preferred_child(status)
        panel = self._document.create_scroll_area()
        panel.widget.stable_id = "diffusion-editor.reconstruction.panel"
        panel.set_scroll_axes(False, True)
        panel.set_content(panel_content)

        self.canvas_reconstruction_splitter = splitter
        self.reconstruction_viewport = viewport_view
        self.reconstruction_toolbar_model = toolbar_model
        self.reconstruction_toolbar = toolbar
        self.reconstruction_buttons = reconstruction_buttons
        self.reconstruction_stage_buttons = stage_buttons
        self.reconstruction_stage_checks = stage_checks
        self.reconstruction_parameter_controls = parameter_controls
        self.reconstruction_status = status
        self.reconstruction_panel = panel
        self._request_repaint()

    def set_reconstruction_stage_handler(self, handler) -> None:
        self._reconstruction_stage_handler = handler

    def set_reconstruction_parameter_handler(self, handler) -> None:
        self._reconstruction_parameter_handler = handler

    def _change_reconstruction_parameter(self, key: str, value) -> None:
        if (
            self._syncing_reconstruction_parameters
            or self._reconstruction_parameter_handler is None
        ):
            return
        if key in {
            "seed",
            "steps",
            "resolution",
            "lr_conditioning_resolution",
            "decimation_target",
            "texture_size",
        }:
            value = int(round(float(value)))
        elif key == "manual_fov_degrees":
            value = float(value)
        elif key == "low_vram":
            value = bool(value)
        self._reconstruction_parameter_handler(key, value)

    def update_reconstruction_parameters(
        self,
        parameters: ReconstructionParameters,
        *,
        busy: bool = False,
    ) -> None:
        controls = self.reconstruction_parameter_controls
        if not controls:
            return
        self._syncing_reconstruction_parameters = True
        try:
            controls["seed"].value = float(parameters.seed)
            controls["steps"].value = float(parameters.steps)
            controls["resolution"].selected_index = (
                _RECONSTRUCTION_RESOLUTIONS.index(parameters.resolution)
            )
            controls["lr_conditioning_resolution"].selected_index = (
                _RECONSTRUCTION_LR_CONDITIONING_RESOLUTIONS.index(
                    parameters.lr_conditioning_resolution
                )
            )
            controls["manual_fov_degrees"].value = (
                parameters.manual_fov_degrees
            )
            controls["decimation_target"].value = float(
                parameters.decimation_target
            )
            controls["texture_size"].selected_index = (
                _RECONSTRUCTION_TEXTURE_SIZES.index(parameters.texture_size)
            )
            controls["low_vram"].checked = parameters.low_vram
            for control in controls.values():
                control.widget.enabled = not busy
        finally:
            self._syncing_reconstruction_parameters = False
        self._request_repaint()

    def _activate_reconstruction_stage(self, stage: ReconstructionStage) -> None:
        if self._reconstruction_stage_handler is not None:
            self._reconstruction_stage_handler(stage)

    def update_reconstruction_stages(
        self,
        statuses,
        progress,
        target: ReconstructionStage,
        selected: ReconstructionStage,
        *,
        busy: bool = False,
    ) -> None:
        for stage, button in self.reconstruction_stage_buttons.items():
            state = statuses.get(stage, ReconstructionStageStatus.PENDING)
            current, total = progress.get(stage, (0, 0))
            details = []
            if stage is selected:
                details.append("preview")
            if stage is target:
                details.append("target")
            if state is ReconstructionStageStatus.RUNNING:
                details.append(
                    f"generating {current}/{total}" if total else "generating"
                )
            elif state is ReconstructionStageStatus.FAILED:
                details.append("failed")
            elif state is ReconstructionStageStatus.SKIPPED:
                details.append("skipped")
            suffix = f" · {' · '.join(details)}" if details else ""
            button.set_text(f"{RECONSTRUCTION_STAGE_LABELS[stage]}{suffix}")
            self.reconstruction_stage_checks[stage].checked = (
                state is ReconstructionStageStatus.READY
            )
            button.widget.enabled = (
                stage in RECONSTRUCTION_PREVIEW_STAGES
                and (not busy or state is ReconstructionStageStatus.READY)
            )
        self._request_repaint()

    def set_reconstruction_context(
            self, visible: bool, status: str = "") -> None:
        self._require_open()
        visible = bool(visible)
        if self.reconstruction_status is not None:
            self.reconstruction_status.text = status.title() if status else ""
        panel = self.reconstruction_panel
        splitter = self.canvas_reconstruction_splitter
        canvas = self.canvas_view
        if (
                panel is None
                or splitter is None
                or canvas is None
                or visible == self.reconstruction_mode):
            self.reconstruction_mode = visible
            self._request_repaint()
            return
        if visible:
            self.main_splitter.set_first(panel.widget)
            self.canvas_host.remove_child(canvas.widget)
            splitter.set_first(canvas.widget)
            self.canvas_host.add_flex_child(splitter.widget, 1.0)
        else:
            self.main_splitter.set_first(self.left_scroll.widget)
            self.canvas_host.remove_child(splitter.widget)
            splitter.set_first(self._canvas_splitter_parking)
            self.canvas_host.add_flex_child(canvas.widget, 1.0)
        self.reconstruction_mode = visible
        self._request_repaint()

    def mount_generation_panels(
            self, panels_view, panel_presentation) -> None:
        self._require_open()
        if self.generation_panels_view is not None:
            raise RuntimeError("native generation panels are already mounted")
        self.left_panel.add_preferred_child(panels_view.widget)
        self.generation_panels_view = panels_view
        self.panel_presentation = panel_presentation
        self._request_repaint()

    def mount_layer_panel(self, layer_panel_view) -> None:
        self._require_open()
        if self.layer_panel_view is not None:
            raise RuntimeError("native layer panel is already mounted")
        self.layer_panel.remove_child(self.layer_placeholder)
        self.layer_panel.add_flex_child(layer_panel_view.widget, 1.0)
        self.layer_panel_view = layer_panel_view
        self._request_repaint()

    def mount_agent_chat(self, agent_chat_view) -> None:
        self._require_open()
        if self.agent_chat_view is not None:
            raise RuntimeError("native agent chat is already mounted")
        self.right_splitter.set_second(agent_chat_view.widget)
        self.agent_chat_view = agent_chat_view
        self._request_repaint()

    def ports(self) -> ViewPorts:
        return ViewPorts(
            status=self,
            commands=self,
            window=self,
            panels=self.panel_presentation,
            canvas=self.canvas_view,
        )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._command_handlers.clear()
        self._connections.clear()

    def _create_model(
            self,
            command_ids: tuple[str | None, ...]) -> CommandModel:
        model = CommandModel()
        separator_index = 0
        for command_id in command_ids:
            if command_id is None:
                model.append(CommandData(
                    f"separator.{separator_index}",
                    kind=CommandKind.Separator,
                ))
                separator_index += 1
                continue
            spec = COMMAND_SPEC_BY_ID[command_id]
            native_id = model.append(CommandData(
                spec.stable_id,
                spec.label,
                shortcut=spec.shortcut,
                tooltip=spec.tooltip,
                enabled=spec.enabled,
                checkable=spec.checkable,
                checked=spec.checked,
            ))
            self._command_refs.setdefault(spec.action_id, []).append(
                (model, native_id)
            )
        return model

    @staticmethod
    def _placeholder(
            document: TcDocument,
            stable_id: str,
            label: str,
            debug_name: str):
        panel = document.create_vstack(debug_name)
        panel.stable_id = stable_id
        panel.set_layout_spacing(4.0)
        title = document.create_label(label, f"{debug_name}Label")
        title.stable_id = f"{stable_id}.label"
        panel.add_preferred_child(title)
        return panel

    def _on_menu_activated(
            self,
            _menu_index: int,
            _native_id,
            command: CommandData) -> None:
        self._on_activated(command)

    def _on_toolbar_activated(
            self,
            _index: int,
            _native_id,
            command: CommandData) -> None:
        self._on_activated(command)

    def _on_activated(self, command: CommandData) -> None:
        canonical_id = self._canonical_id(command.stable_id)
        enabled, checked = self._command_states[canonical_id]
        if command.checkable and command.checked != checked:
            self.set_command_state(
                canonical_id,
                enabled=enabled,
                checked=command.checked,
            )
        self.activate_command(canonical_id)

    @staticmethod
    def _canonical_id(command_id: str) -> str:
        spec = COMMAND_SPEC_BY_ID.get(command_id)
        if spec is not None:
            return spec.action_id
        if command_id in {
                spec.action_id for spec in COMMAND_SPECS
        }:
            return command_id
        raise KeyError(f"unknown native command: {command_id}")

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("native editor view is closed")
