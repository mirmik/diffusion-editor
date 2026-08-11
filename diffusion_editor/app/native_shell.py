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
    RECONSTRUCTION_BACKEND_PARAMETER_KEYS,
    RECONSTRUCTION_BACKEND_STAGES,
    RECONSTRUCTION_BACKEND_LABELS,
    RECONSTRUCTION_PREVIEW_STAGES,
    RECONSTRUCTION_STAGES,
    RECONSTRUCTION_STAGE_LABELS,
    ReconstructionParameters,
    ReconstructionBackend,
    ReconstructionRefineParameters,
    ReconstructionRunKind,
    ReconstructionStage,
    ReconstructionStageStatus,
)
from ..generation.reconstruction_workspace import (
    LEGACY_OPERATION_TARGET_STAGES,
    PIXAL3D_OPERATION_PARAMETER_KEYS,
    PIXAL3D_PIPELINE,
    ReconstructionWorkspace,
    WorkspaceOperationStatus,
    WorkspacePreviewKind,
)
from .native_reconstruction_viewport import (
    RECONSTRUCTION_SHADING_LABELS,
    RECONSTRUCTION_SHADING_MODES,
)
from .presentation import ViewPorts


CommandHandler = Callable[[], None]

_CHROME_BACKGROUND = SrgbColor(0.075, 0.080, 0.095, 1.0)
_WORKSPACE_EDGE = SrgbColor(0.32, 0.35, 0.40, 1.0)
_PIPELINE_GROUP_BACKGROUND = SrgbColor(0.105, 0.115, 0.140, 1.0)
_PIPELINE_OPERATION_BACKGROUND = SrgbColor(0.155, 0.165, 0.190, 1.0)
_PIPELINE_OPERATION_SELECTED = SrgbColor(0.105, 0.285, 0.455, 1.0)
_PIPELINE_OPERATION_READY = SrgbColor(0.105, 0.245, 0.185, 1.0)
_PIPELINE_OPERATION_RUNNING = SrgbColor(0.330, 0.245, 0.095, 1.0)
_PIPELINE_OPERATION_FAILED = SrgbColor(0.355, 0.115, 0.120, 1.0)
_RECONSTRUCTION_RESOLUTIONS = (1024, 1280, 1536)
_RECONSTRUCTION_LR_CONDITIONING_RESOLUTIONS = (512, 1024)
_RECONSTRUCTION_TEXTURE_SIZES = (1024, 2048, 4096)
_HI3DGEN_NORMAL_RESOLUTIONS = (512, 768, 1024)
_HUNYUAN3D21_OCTREE_RESOLUTIONS = (96, 192, 256, 384, 512)

_WORKSPACE_PARAMETER_LABELS = {
    "manual_fov_degrees": "Camera FOV (0 = Auto)",
    "pixal3d_sparse_seed": "Sparse seed",
    "pixal3d_sparse_steps": "Sparse steps",
    "pixal3d_lr_seed": "LR seed",
    "pixal3d_lr_steps": "LR steps",
    "lr_conditioning_resolution": "LR conditioning",
    "resolution": "HR spatial resolution",
    "pixal3d_hr_seed": "HR seed",
    "pixal3d_hr_steps": "HR steps",
    "pixal3d_texture_seed": "Texture seed",
    "pixal3d_texture_steps": "Texture steps",
    "decimation_target": "Final mesh faces",
    "texture_size": "Texture size",
}


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
    NativeCommandSpec(
        "edit.clear_selected_pixels",
        "Clear Selected Pixels",
        "Delete",
        tooltip="Replace selected pixels of the active layer with transparency",
    ),
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
            "edit.clear_selected_pixels",
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
        self.reconstruction_shading_combo = None
        self.reconstruction_buttons = {}
        self.reconstruction_stage_buttons = {}
        self.reconstruction_stage_checks = {}
        self._reconstruction_stage_handler = None
        self._reconstruction_parameter_handler = None
        self._reconstruction_refine_handler = None
        self._syncing_reconstruction_parameters = False
        self._syncing_reconstruction_refine = False
        self.reconstruction_parameter_controls = {}
        self.reconstruction_parameter_widgets = {}
        self.reconstruction_refine_controls = {}
        self.reconstruction_versions_title = None
        self.reconstruction_versions_panel = None
        self.reconstruction_refine_title = None
        self.reconstruction_refine_panel = None
        self.reconstruction_workspace_panel = None
        self.reconstruction_workspace_mode = False
        self.reconstruction_workspace_open = None
        self.reconstruction_workspace_backend = None
        self.reconstruction_workspace_group_buttons = {}
        self.reconstruction_workspace_operation_buttons = {}
        self.reconstruction_workspace_operation_rows = {}
        self.reconstruction_workspace_inspector_title = None
        self.reconstruction_workspace_inspector_description = None
        self.reconstruction_workspace_inspector_inputs = None
        self.reconstruction_workspace_inspector_outputs = None
        self.reconstruction_workspace_variant_combo = None
        self.reconstruction_workspace_artifact_combo = None
        self.reconstruction_workspace_status = None
        self.reconstruction_workspace_actions = {}
        self.reconstruction_workspace_parameter_controls = {}
        self.reconstruction_workspace_parameter_rows = {}
        self.reconstruction_workspace_parameter_reset = None
        self._reconstruction_workspace_parameters = None
        self._reconstruction_workspace_handler = None
        self._reconstruction_workspace_snapshot = None
        self._reconstruction_workspace_operation_ids = []
        self._reconstruction_workspace_artifact_ids = []
        self._selected_reconstruction_workspace_operation_id = None
        self._selected_reconstruction_workspace_artifact_id = None
        self._syncing_reconstruction_workspace = False
        self._reconstruction_workspace_busy = False
        self._expanded_reconstruction_workspace_groups = {"source"}
        self._selected_reconstruction_workspace_operation = None
        self._reconstruction_run_ids = []
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

        shading_row = self._document.create_hstack(
            "DiffusionEditorReconstructionShadingRow"
        )
        shading_row.set_layout_spacing(4.0)
        shading_label = self._document.create_label(
            "Shading", "DiffusionEditorReconstructionShadingLabel"
        )
        shading_combo = self._document.create_combo_box()
        shading_combo.widget.stable_id = (
            "diffusion-editor.reconstruction.shading"
        )
        for mode in RECONSTRUCTION_SHADING_MODES:
            shading_combo.add_item(RECONSTRUCTION_SHADING_LABELS[mode])
        shading_combo.selected_index = 0
        set_shading_mode = getattr(viewport_view, "set_shading_mode", None)

        def activate_shading(index, *_rest):
            if (
                callable(set_shading_mode)
                and 0 <= index < len(RECONSTRUCTION_SHADING_MODES)
            ):
                set_shading_mode(RECONSTRUCTION_SHADING_MODES[index])

        self._connections.append(shading_combo.connect_changed(
            activate_shading
        ))
        shading_row.add_flex_child(shading_label, 1.0)
        shading_row.add_fixed_child(shading_combo.widget, 108.0)
        toolbar.add_preferred_child(shading_row)

        status = self._document.create_label(
            "Empty", "DiffusionEditorReconstructionStatus"
        )
        status.stable_id = "diffusion-editor.reconstruction.status"
        title = self._document.create_label(
            "3D Reconstruction", "DiffusionEditorReconstructionTitle"
        )
        title.stable_id = "diffusion-editor.reconstruction.title"
        open_workspace = self._document.create_button(
            "Try Experimental Pipeline…"
        )
        open_workspace.widget.stable_id = (
            "diffusion-editor.reconstruction.workspace.open"
        )
        self._connections.append(open_workspace.connect_clicked(
            lambda: self._set_reconstruction_workspace_mode(True)
        ))
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
        parameter_widgets = {}

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
            parameter_widgets[key] = control.widget
            parameters_panel.add_preferred_child(control.widget)

        def add_spin(key, label, value, minimum, maximum, step, decimals=0):
            row = self._document.create_hstack(
                "DiffusionEditorReconstructionParameterRow"
            )
            row.stable_id = (
                f"diffusion-editor.reconstruction.parameter.{key}.row"
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
            parameter_widgets[key] = row
            row.add_flex_child(caption, 1.0)
            row.add_fixed_child(control.widget, 108.0)
            parameters_panel.add_preferred_child(row)

        def add_combo(key, label, values, labels=None):
            row = self._document.create_hstack(
                "DiffusionEditorReconstructionParameterRow"
            )
            row.stable_id = (
                f"diffusion-editor.reconstruction.parameter.{key}.row"
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
                control.add_item(
                    labels.get(item, str(item)) if labels else str(item)
                )
            control.selected_index = values.index(getattr(defaults, key))
            self._connections.append(control.connect_changed(
                lambda index, *_rest, key=key, values=values:
                self._change_reconstruction_parameter(key, values[index])
            ))
            parameter_controls[key] = control
            parameter_widgets[key] = row
            row.add_flex_child(caption, 1.0)
            row.add_fixed_child(control.widget, 92.0)
            parameters_panel.add_preferred_child(row)

        add_combo(
            "backend",
            "Backend",
            tuple(ReconstructionBackend),
            RECONSTRUCTION_BACKEND_LABELS,
        )
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
        add_slider(
            "spar3d_guidance_scale", "Point guidance",
            defaults.spar3d_guidance_scale, 0, 20, 0.1, 1,
        )
        add_slider(
            "hi3dgen_slat_steps", "Hi3D latent steps",
            defaults.hi3dgen_slat_steps, 1, 50, 1,
        )
        add_slider(
            "hi3dgen_guidance_scale", "Hi3D guidance",
            defaults.hi3dgen_guidance_scale, 0, 20, 0.1, 1,
        )
        add_combo(
            "hi3dgen_normal_resolution",
            "Normal resolution",
            _HI3DGEN_NORMAL_RESOLUTIONS,
        )
        add_slider(
            "hunyuan3d21_guidance_scale", "HY3D shape guidance",
            defaults.hunyuan3d21_guidance_scale, 0, 20, 0.1, 1,
        )
        add_combo(
            "hunyuan3d21_octree_resolution",
            "HY3D octree resolution",
            _HUNYUAN3D21_OCTREE_RESOLUTIONS,
        )
        add_slider(
            "hunyuan3d21_texture_steps", "HY3D texture steps",
            defaults.hunyuan3d21_texture_steps, 1, 50, 1,
        )
        add_slider(
            "hunyuan3d21_texture_guidance_scale", "HY3D texture guidance",
            defaults.hunyuan3d21_texture_guidance_scale, 0, 20, 0.1, 1,
        )
        add_slider(
            "sam3d_sparse_steps", "SAM sparse steps",
            defaults.sam3d_sparse_steps, 1, 50, 1,
        )
        add_slider(
            "sam3d_slat_steps", "SAM latent steps",
            defaults.sam3d_slat_steps, 1, 50, 1,
        )
        add_slider(
            "sam3d_sparse_guidance_scale", "SAM sparse guidance",
            defaults.sam3d_sparse_guidance_scale, 0, 20, 0.1, 1,
        )
        add_slider(
            "sam3d_slat_guidance_scale", "SAM latent guidance",
            defaults.sam3d_slat_guidance_scale, 0, 20, 0.1, 1,
        )
        add_slider(
            "sam3d_simplify", "SAM simplify",
            defaults.sam3d_simplify, 0, 0.99, 0.01, 2,
        )
        low_vram_row = self._document.create_hstack(
            "DiffusionEditorReconstructionParameterRow"
        )
        low_vram_row.stable_id = (
            "diffusion-editor.reconstruction.parameter.low_vram.row"
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
        parameter_widgets["low_vram"] = low_vram_row
        low_vram_row.add_preferred_child(low_vram.widget)
        low_vram_row.add_flex_child(low_vram_label, 1.0)
        parameters_panel.add_preferred_child(low_vram_row)

        versions_title = self._document.create_label(
            "Versions", "DiffusionEditorReconstructionVersionsTitle"
        )
        versions_title.stable_id = (
            "diffusion-editor.reconstruction.versions.title"
        )
        versions_panel = self._document.create_vstack(
            "DiffusionEditorReconstructionVersions"
        )
        versions_panel.stable_id = "diffusion-editor.reconstruction.versions"
        versions_panel.set_layout_spacing(3.0)

        refine_title = self._document.create_label(
            "Masked refinement", "DiffusionEditorReconstructionRefineTitle"
        )
        refine_title.stable_id = "diffusion-editor.reconstruction.refine.title"
        refine_panel = self._document.create_vstack(
            "DiffusionEditorReconstructionRefine"
        )
        refine_panel.stable_id = "diffusion-editor.reconstruction.refine"
        refine_panel.set_layout_spacing(3.0)
        refine_defaults = ReconstructionRefineParameters()
        refine_controls = {}

        def emit_refine(action, value=None):
            self._activate_reconstruction_refine(action, value)

        def add_refine_checkbox(key, label, checked=False):
            row = self._document.create_hstack(
                "DiffusionEditorReconstructionRefineRow"
            )
            row.set_layout_spacing(4.0)
            control = self._document.create_checkbox(checked)
            control.widget.stable_id = (
                f"diffusion-editor.reconstruction.refine.{key}"
            )
            caption = self._document.create_label(
                label, "DiffusionEditorReconstructionRefineLabel"
            )
            self._connections.append(control.connect_changed(
                lambda value, key=key: emit_refine(key, bool(value))
            ))
            refine_controls[key] = control
            row.add_preferred_child(control.widget)
            row.add_flex_child(caption, 1.0)
            refine_panel.add_preferred_child(row)

        def add_refine_slider(
            key, label, value, minimum, maximum, step, decimals=0
        ):
            control = self._document.create_slider_edit(float(value))
            control.widget.stable_id = (
                f"diffusion-editor.reconstruction.refine.{key}"
            )
            control.label = label
            control.set_range(float(minimum), float(maximum))
            control.set_step(float(step))
            control.set_decimals(int(decimals))
            self._connections.append(control.connect_changed(
                lambda changed, key=key: emit_refine(key, changed)
            ))
            refine_controls[key] = control
            refine_panel.add_preferred_child(control.widget)

        def add_refine_spin(key, label, value, minimum, maximum):
            row = self._document.create_hstack(
                "DiffusionEditorReconstructionRefineRow"
            )
            row.set_layout_spacing(4.0)
            caption = self._document.create_label(
                label, "DiffusionEditorReconstructionRefineLabel"
            )
            control = self._document.create_spin_box(float(value))
            control.widget.stable_id = (
                f"diffusion-editor.reconstruction.refine.{key}"
            )
            control.set_range(float(minimum), float(maximum))
            control.step = 1.0
            control.decimals = 0
            self._connections.append(control.connect_changed(
                lambda changed, key=key: emit_refine(key, changed)
            ))
            refine_controls[key] = control
            row.add_flex_child(caption, 1.0)
            row.add_fixed_child(control.widget, 108.0)
            refine_panel.add_preferred_child(row)

        add_refine_checkbox("paint", "Paint refine mask")
        add_refine_checkbox("erase", "Erase mask")
        add_refine_checkbox(
            "resize_detail_to_1024", "Resize masked detail to 1024", True
        )
        add_refine_slider("brush_size", "Brush size", 50, 1, 500, 1)
        add_refine_slider("brush_hardness", "Hardness", 0.4, 0, 1, 0.05, 2)
        add_refine_slider("brush_flow", "Flow", 1.0, 0, 1, 0.05, 2)
        add_refine_slider(
            "strength", "Strength", refine_defaults.strength,
            0.05, 1.0, 0.05, 2,
        )
        add_refine_spin("steps", "Refine steps", refine_defaults.steps, 1, 50)
        add_refine_spin(
            "seed", "Refine seed", refine_defaults.seed, 0, 2_147_483_647
        )

        version_row = self._document.create_hstack(
            "DiffusionEditorReconstructionRefineRow"
        )
        version_row.set_layout_spacing(4.0)
        version_label = self._document.create_label(
            "Version", "DiffusionEditorReconstructionRefineLabel"
        )
        version_combo = self._document.create_combo_box()
        version_combo.widget.stable_id = (
            "diffusion-editor.reconstruction.refine.version"
        )
        self._connections.append(version_combo.connect_changed(
            lambda index, *_rest: emit_refine(
                "select_run",
                self._reconstruction_run_ids[index]
                if 0 <= index < len(self._reconstruction_run_ids)
                else None,
            )
        ))
        refine_controls["version"] = version_combo
        version_row.add_flex_child(version_label, 1.0)
        version_row.add_fixed_child(version_combo.widget, 120.0)
        versions_panel.add_preferred_child(version_row)

        clear_mask = self._document.create_button("Clear refine mask")
        clear_mask.widget.stable_id = (
            "diffusion-editor.reconstruction.refine.clear"
        )
        self._connections.append(clear_mask.connect_clicked(
            lambda: emit_refine("clear")
        ))
        refine_controls["clear"] = clear_mask
        refine_panel.add_preferred_child(clear_mask.widget)

        run_refine = self._document.create_button("Refine geometry in mask")
        run_refine.widget.stable_id = (
            "diffusion-editor.reconstruction.refine.run"
        )
        self._connections.append(run_refine.connect_clicked(
            lambda: emit_refine("run")
        ))
        refine_controls["run"] = run_refine
        refine_panel.add_preferred_child(run_refine.widget)

        run_texture_refine = self._document.create_button(
            "Refine texture in mask"
        )
        run_texture_refine.widget.stable_id = (
            "diffusion-editor.reconstruction.refine.run_texture"
        )
        self._connections.append(run_texture_refine.connect_clicked(
            lambda: emit_refine("run_texture")
        ))
        refine_controls["run_texture"] = run_texture_refine
        refine_panel.add_preferred_child(run_texture_refine.widget)
        panel_content = self._document.create_vstack(
            "DiffusionEditorReconstructionPanelContent"
        )
        panel_content.stable_id = (
            "diffusion-editor.reconstruction.panel-content"
        )
        panel_content.set_layout_spacing(4.0)
        panel_content.add_preferred_child(title)
        panel_content.add_preferred_child(open_workspace.widget)
        panel_content.add_preferred_child(parameters_title)
        panel_content.add_preferred_child(parameters_panel)
        panel_content.add_preferred_child(versions_title)
        panel_content.add_preferred_child(versions_panel)
        panel_content.add_preferred_child(refine_title)
        panel_content.add_preferred_child(refine_panel)
        panel_content.add_preferred_child(stage_list)
        panel_content.add_preferred_child(toolbar)
        panel_content.add_preferred_child(status)
        panel = self._document.create_scroll_area()
        panel.widget.stable_id = "diffusion-editor.reconstruction.panel"
        panel.set_scroll_axes(False, True)
        panel.set_content(panel_content)

        workspace_content = self._document.create_vstack(
            "DiffusionEditorReconstructionWorkspaceContent"
        )
        workspace_content.stable_id = (
            "diffusion-editor.reconstruction.workspace.content"
        )
        workspace_content.set_layout_spacing(4.0)
        workspace_title = self._document.create_label(
            "3D Pipeline · Experimental",
            "DiffusionEditorReconstructionWorkspaceTitle",
        )
        workspace_title.stable_id = (
            "diffusion-editor.reconstruction.workspace.title"
        )
        workspace_notice = self._document.create_label(
            "Isolated preview: legacy generation remains unchanged.",
            "DiffusionEditorReconstructionWorkspaceNotice",
        )
        workspace_notice.stable_id = (
            "diffusion-editor.reconstruction.workspace.notice"
        )
        back_to_legacy = self._document.create_button("← Legacy Tools")
        back_to_legacy.widget.stable_id = (
            "diffusion-editor.reconstruction.workspace.back"
        )
        self._connections.append(back_to_legacy.connect_clicked(
            lambda: self._set_reconstruction_workspace_mode(False)
        ))
        workspace_backend = self._document.create_label(
            "Backend: Pixal3D",
            "DiffusionEditorReconstructionWorkspaceBackend",
        )
        workspace_backend.stable_id = (
            "diffusion-editor.reconstruction.workspace.backend"
        )
        workspace_content.add_preferred_child(workspace_title)
        workspace_content.add_preferred_child(workspace_notice)
        workspace_content.add_preferred_child(back_to_legacy.widget)
        workspace_content.add_preferred_child(workspace_backend)

        workspace_group_buttons = {}
        workspace_operation_buttons = {}
        workspace_operation_rows = {}
        for group in PIXAL3D_PIPELINE.groups:
            group_row = self._document.create_hstack(
                "DiffusionEditorReconstructionWorkspaceGroup"
            )
            group_row.stable_id = (
                f"diffusion-editor.reconstruction.workspace.group.{group.key}"
            )
            group_row.set_layout_spacing(6.0)
            _set_widget_background(group_row, _PIPELINE_GROUP_BACKGROUND)
            group_button = self._document.create_button(
                "-" if group.key == "source" else "+"
            )
            group_button.widget.stable_id = (
                "diffusion-editor.reconstruction.workspace.group."
                f"{group.key}.toggle"
            )
            group_label = self._document.create_label(
                group.label,
                "DiffusionEditorReconstructionWorkspaceGroupLabel",
            )
            group_label.stable_id = (
                "diffusion-editor.reconstruction.workspace.group."
                f"{group.key}.label"
            )
            self._connections.append(group_button.connect_clicked(
                lambda group_key=group.key:
                self._toggle_reconstruction_workspace_group(group_key)
            ))
            workspace_group_buttons[group.key] = group_button
            group_row.add_fixed_child(group_button.widget, 30.0)
            group_row.add_flex_child(group_label, 1.0)
            workspace_content.add_fixed_child(group_row, 28.0)
            for operation in PIXAL3D_PIPELINE.operations_in_group(group.key):
                operation_row = self._document.create_hstack(
                    "DiffusionEditorReconstructionWorkspaceOperationRow"
                )
                operation_row.stable_id = (
                    "diffusion-editor.reconstruction.workspace.operation."
                    f"{operation.key}.row"
                )
                operation_indent = self._document.create_hstack(
                    "DiffusionEditorReconstructionWorkspaceOperationIndent"
                )
                button = self._document.create_button(operation.label)
                button.widget.stable_id = (
                    "diffusion-editor.reconstruction.workspace.operation."
                    f"{operation.key}"
                )
                _set_widget_background(
                    button.widget, _PIPELINE_OPERATION_BACKGROUND
                )
                self._connections.append(button.connect_clicked(
                    lambda operation_key=operation.key:
                    self._select_reconstruction_workspace_operation(
                        operation_key
                    )
                ))
                operation_row.add_fixed_child(operation_indent, 18.0)
                operation_row.add_flex_child(button.widget, 1.0)
                operation_row.visible = group.key == "source"
                workspace_operation_buttons[operation.key] = button
                workspace_operation_rows[operation.key] = operation_row
                workspace_content.add_fixed_child(operation_row, 28.0)

        inspector_title = self._document.create_label(
            "Operation inspector",
            "DiffusionEditorReconstructionWorkspaceInspectorTitle",
        )
        inspector_title.stable_id = (
            "diffusion-editor.reconstruction.workspace.inspector.title"
        )
        inspector_description = self._document.create_label(
            "Select an operation.",
            "DiffusionEditorReconstructionWorkspaceInspectorText",
        )
        inspector_description.stable_id = (
            "diffusion-editor.reconstruction.workspace.inspector.description"
        )
        inspector_inputs = self._document.create_label(
            "Inputs: —", "DiffusionEditorReconstructionWorkspaceInspectorText"
        )
        inspector_inputs.stable_id = (
            "diffusion-editor.reconstruction.workspace.inspector.inputs"
        )
        inspector_outputs = self._document.create_label(
            "Outputs: —", "DiffusionEditorReconstructionWorkspaceInspectorText"
        )
        inspector_outputs.stable_id = (
            "diffusion-editor.reconstruction.workspace.inspector.outputs"
        )
        workspace_content.add_preferred_child(inspector_title)
        workspace_content.add_preferred_child(inspector_description)
        workspace_content.add_preferred_child(inspector_inputs)
        workspace_content.add_preferred_child(inspector_outputs)

        parameter_caption = self._document.create_label(
            "Operation parameters",
            "DiffusionEditorReconstructionWorkspaceInspectorTitle",
        )
        parameter_caption.stable_id = (
            "diffusion-editor.reconstruction.workspace.parameters.title"
        )
        parameter_hint = self._document.create_label(
            "Stage overrides inherit Legacy seed/steps until edited.",
            "DiffusionEditorReconstructionWorkspaceInspectorText",
        )
        parameter_hint.stable_id = (
            "diffusion-editor.reconstruction.workspace.parameters.hint"
        )
        workspace_content.add_preferred_child(parameter_caption)
        workspace_content.add_preferred_child(parameter_hint)
        workspace_parameter_controls = {}
        workspace_parameter_rows = {}

        def add_workspace_spin(
                key, minimum, maximum, step=1, decimals=0) -> None:
            row = self._document.create_hstack(
                "DiffusionEditorReconstructionWorkspaceInspectorRow"
            )
            row.stable_id = (
                f"diffusion-editor.reconstruction.workspace.parameter.{key}.row"
            )
            row.set_layout_spacing(4.0)
            label = self._document.create_label(
                _WORKSPACE_PARAMETER_LABELS[key],
                "DiffusionEditorReconstructionWorkspaceInspectorText",
            )
            control = self._document.create_spin_box(float(minimum))
            control.widget.stable_id = (
                f"diffusion-editor.reconstruction.workspace.parameter.{key}"
            )
            control.set_range(float(minimum), float(maximum))
            control.step = float(step)
            control.decimals = int(decimals)
            self._connections.append(control.connect_changed(
                lambda value, key=key:
                self._change_reconstruction_workspace_parameter(key, value)
            ))
            row.add_flex_child(label, 1.0)
            row.add_fixed_child(control.widget, 108.0)
            row.visible = False
            workspace_parameter_controls[key] = control
            workspace_parameter_rows[key] = row
            workspace_content.add_preferred_child(row)

        def add_workspace_combo(key, values) -> None:
            row = self._document.create_hstack(
                "DiffusionEditorReconstructionWorkspaceInspectorRow"
            )
            row.stable_id = (
                f"diffusion-editor.reconstruction.workspace.parameter.{key}.row"
            )
            row.set_layout_spacing(4.0)
            label = self._document.create_label(
                _WORKSPACE_PARAMETER_LABELS[key],
                "DiffusionEditorReconstructionWorkspaceInspectorText",
            )
            control = self._document.create_combo_box()
            control.widget.stable_id = (
                f"diffusion-editor.reconstruction.workspace.parameter.{key}"
            )
            for value in values:
                control.add_item(str(value))
            self._connections.append(control.connect_changed(
                lambda index, *_rest, key=key, values=values:
                self._change_reconstruction_workspace_parameter(
                    key, values[index]
                )
            ))
            row.add_flex_child(label, 1.0)
            row.add_fixed_child(control.widget, 108.0)
            row.visible = False
            workspace_parameter_controls[key] = control
            workspace_parameter_rows[key] = row
            workspace_content.add_preferred_child(row)

        for phase in ("sparse", "lr", "hr", "texture"):
            add_workspace_spin(
                f"pixal3d_{phase}_seed", 0, 2_147_483_647
            )
            add_workspace_spin(f"pixal3d_{phase}_steps", 1, 50)
        add_workspace_spin("manual_fov_degrees", 0, 120, decimals=1)
        add_workspace_combo(
            "lr_conditioning_resolution",
            _RECONSTRUCTION_LR_CONDITIONING_RESOLUTIONS,
        )
        add_workspace_combo("resolution", _RECONSTRUCTION_RESOLUTIONS)
        add_workspace_spin(
            "decimation_target", 50_000, 1_000_000, step=10_000
        )
        add_workspace_combo("texture_size", _RECONSTRUCTION_TEXTURE_SIZES)
        parameter_reset = self._document.create_button(
            "Use Legacy Defaults for This Operation"
        )
        parameter_reset.widget.stable_id = (
            "diffusion-editor.reconstruction.workspace.parameters.reset"
        )
        self._connections.append(parameter_reset.connect_clicked(
            self._reset_reconstruction_workspace_parameters
        ))
        workspace_content.add_preferred_child(parameter_reset.widget)

        variant_row = self._document.create_hstack(
            "DiffusionEditorReconstructionWorkspaceInspectorRow"
        )
        variant_row.set_layout_spacing(4.0)
        variant_label = self._document.create_label(
            "Variant", "DiffusionEditorReconstructionWorkspaceInspectorText"
        )
        variant_combo = self._document.create_combo_box()
        variant_combo.widget.stable_id = (
            "diffusion-editor.reconstruction.workspace.variant"
        )
        self._connections.append(variant_combo.connect_changed(
            self._change_reconstruction_workspace_variant
        ))
        variant_row.add_flex_child(variant_label, 1.0)
        variant_row.add_fixed_child(variant_combo.widget, 170.0)
        workspace_content.add_preferred_child(variant_row)

        artifact_row = self._document.create_hstack(
            "DiffusionEditorReconstructionWorkspaceInspectorRow"
        )
        artifact_row.set_layout_spacing(4.0)
        artifact_label = self._document.create_label(
            "Artifact", "DiffusionEditorReconstructionWorkspaceInspectorText"
        )
        artifact_combo = self._document.create_combo_box()
        artifact_combo.widget.stable_id = (
            "diffusion-editor.reconstruction.workspace.artifact"
        )
        self._connections.append(artifact_combo.connect_changed(
            self._change_reconstruction_workspace_artifact
        ))
        artifact_row.add_flex_child(artifact_label, 1.0)
        artifact_row.add_fixed_child(artifact_combo.widget, 170.0)
        workspace_content.add_preferred_child(artifact_row)

        workspace_actions = {}
        for key, label in (
            ("preview", "Preview Artifact"),
            ("generate", "Run / Resume Through Selected"),
            ("refine", "Refine Selected"),
            ("accept", "Accept and Continue"),
        ):
            button = self._document.create_button(label)
            button.widget.stable_id = (
                f"diffusion-editor.reconstruction.workspace.action.{key}"
            )
            button.widget.enabled = False
            if key == "preview":
                self._connections.append(button.connect_clicked(
                    self._preview_reconstruction_workspace_artifact
                ))
            elif key == "generate":
                self._connections.append(button.connect_clicked(
                    self._run_reconstruction_workspace_operation
                ))
            workspace_actions[key] = button
            workspace_content.add_preferred_child(button.widget)
        workspace_status = self._document.create_label(
            "Base Pixal3D stages resume from the latest compatible checkpoint.",
            "DiffusionEditorReconstructionWorkspaceNotice",
        )
        workspace_status.stable_id = (
            "diffusion-editor.reconstruction.workspace.status"
        )
        workspace_content.add_preferred_child(workspace_status)
        workspace_panel = self._document.create_scroll_area()
        workspace_panel.widget.stable_id = (
            "diffusion-editor.reconstruction.workspace.panel"
        )
        workspace_panel.set_scroll_axes(False, True)
        workspace_panel.set_content(workspace_content)

        self.canvas_reconstruction_splitter = splitter
        self.reconstruction_viewport = viewport_view
        self.reconstruction_toolbar_model = toolbar_model
        self.reconstruction_toolbar = toolbar
        self.reconstruction_shading_combo = shading_combo
        self.reconstruction_buttons = reconstruction_buttons
        self.reconstruction_stage_buttons = stage_buttons
        self.reconstruction_stage_checks = stage_checks
        self.reconstruction_parameter_controls = parameter_controls
        self.reconstruction_parameter_widgets = parameter_widgets
        self.reconstruction_refine_controls = refine_controls
        self.reconstruction_versions_title = versions_title
        self.reconstruction_versions_panel = versions_panel
        self.reconstruction_refine_title = refine_title
        self.reconstruction_refine_panel = refine_panel
        self.reconstruction_workspace_panel = workspace_panel
        self.reconstruction_workspace_open = open_workspace
        self.reconstruction_workspace_backend = workspace_backend
        self.reconstruction_workspace_group_buttons = workspace_group_buttons
        self.reconstruction_workspace_operation_buttons = (
            workspace_operation_buttons
        )
        self.reconstruction_workspace_operation_rows = (
            workspace_operation_rows
        )
        self.reconstruction_workspace_inspector_title = inspector_title
        self.reconstruction_workspace_inspector_description = (
            inspector_description
        )
        self.reconstruction_workspace_inspector_inputs = inspector_inputs
        self.reconstruction_workspace_inspector_outputs = inspector_outputs
        self.reconstruction_workspace_variant_combo = variant_combo
        self.reconstruction_workspace_artifact_combo = artifact_combo
        self.reconstruction_workspace_status = workspace_status
        self.reconstruction_workspace_actions = workspace_actions
        self.reconstruction_workspace_parameter_controls = (
            workspace_parameter_controls
        )
        self.reconstruction_workspace_parameter_rows = workspace_parameter_rows
        self.reconstruction_workspace_parameter_reset = parameter_reset
        self.reconstruction_status = status
        self.reconstruction_panel = panel
        self._select_reconstruction_workspace_operation("source.prepare")
        self._request_repaint()

    def _toggle_reconstruction_workspace_group(self, key: str) -> None:
        if key in self._expanded_reconstruction_workspace_groups:
            self._expanded_reconstruction_workspace_groups.remove(key)
        else:
            self._expanded_reconstruction_workspace_groups.add(key)
        self._refresh_reconstruction_workspace_groups()

    def _refresh_reconstruction_workspace_groups(self) -> None:
        for group in PIXAL3D_PIPELINE.groups:
            expanded = (
                group.key in self._expanded_reconstruction_workspace_groups
            )
            button = self.reconstruction_workspace_group_buttons.get(group.key)
            if button is not None:
                button.set_text("-" if expanded else "+")
            for operation in PIXAL3D_PIPELINE.operations_in_group(group.key):
                operation_row = (
                    self.reconstruction_workspace_operation_rows.get(
                        operation.key
                    )
                )
                if operation_row is not None:
                    operation_row.visible = expanded
        self._request_repaint()

    def _set_reconstruction_workspace_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if (
                enabled
                and self.reconstruction_workspace_open is not None
                and not self.reconstruction_workspace_open.widget.enabled):
            return
        self.reconstruction_workspace_mode = enabled
        if self.reconstruction_mode:
            active = (
                self.reconstruction_workspace_panel
                if enabled else self.reconstruction_panel
            )
            if active is not None:
                self.main_splitter.set_first(active.widget)
        self._request_repaint()

    def _select_reconstruction_workspace_operation(self, key: str) -> None:
        try:
            operation = PIXAL3D_PIPELINE.operation(key)
        except KeyError:
            return
        self._expanded_reconstruction_workspace_groups.add(
            operation.group_key
        )
        self._refresh_reconstruction_workspace_groups()
        self._selected_reconstruction_workspace_operation = key
        self._refresh_reconstruction_workspace_operation_buttons()
        if self.reconstruction_workspace_inspector_title is not None:
            self.reconstruction_workspace_inspector_title.text = operation.label
            self.reconstruction_workspace_inspector_description.text = (
                operation.description
            )
            inputs = ", ".join(operation.input_roles) or "none"
            outputs = ", ".join(
                output.label for output in operation.outputs
            ) or "none"
            self.reconstruction_workspace_inspector_inputs.text = (
                f"Inputs: {inputs}"
            )
            self.reconstruction_workspace_inspector_outputs.text = (
                f"Outputs: {outputs}"
            )
        self._refresh_reconstruction_workspace_variants()
        self._refresh_reconstruction_workspace_parameters()
        self._refresh_reconstruction_workspace_actions()
        self._request_repaint()

    def set_reconstruction_workspace_handler(self, handler) -> None:
        self._reconstruction_workspace_handler = handler

    def _change_reconstruction_workspace_parameter(
            self, key: str, value) -> None:
        if (
            self._syncing_reconstruction_workspace
            or self._reconstruction_workspace_handler is None
        ):
            return
        if key == "manual_fov_degrees":
            value = float(value)
        else:
            value = int(round(float(value)))
        self._reconstruction_workspace_handler(
            "set_operation_parameter", (key, value)
        )

    def _reset_reconstruction_workspace_parameters(self) -> None:
        if (
            self._reconstruction_workspace_handler is None
            or self._selected_reconstruction_workspace_operation is None
        ):
            return
        self._reconstruction_workspace_handler(
            "reset_operation_parameters",
            self._selected_reconstruction_workspace_operation,
        )

    def _refresh_reconstruction_workspace_parameters(self) -> None:
        parameters = self._reconstruction_workspace_parameters
        operation = self._selected_reconstruction_workspace_operation
        visible_keys = set(PIXAL3D_OPERATION_PARAMETER_KEYS.get(
            operation, ()
        ))
        for key, row in self.reconstruction_workspace_parameter_rows.items():
            row.visible = key in visible_keys
        reset = self.reconstruction_workspace_parameter_reset
        if reset is not None:
            reset.widget.visible = bool(visible_keys)
            reset.widget.enabled = bool(
                visible_keys and not self._reconstruction_workspace_busy
            )
        if parameters is None:
            return
        self._syncing_reconstruction_workspace = True
        try:
            for key in visible_keys:
                control = self.reconstruction_workspace_parameter_controls[key]
                if key.startswith("pixal3d_") and key.endswith("_seed"):
                    phase = key.removeprefix("pixal3d_").removesuffix("_seed")
                    control.value = float(parameters.pixal3d_seed_for(phase))
                elif key.startswith("pixal3d_") and key.endswith("_steps"):
                    phase = key.removeprefix("pixal3d_").removesuffix("_steps")
                    control.value = float(parameters.pixal3d_steps_for(phase))
                elif key == "lr_conditioning_resolution":
                    control.selected_index = (
                        _RECONSTRUCTION_LR_CONDITIONING_RESOLUTIONS.index(
                            parameters.lr_conditioning_resolution
                        )
                    )
                elif key == "resolution":
                    control.selected_index = _RECONSTRUCTION_RESOLUTIONS.index(
                        parameters.resolution
                    )
                elif key == "texture_size":
                    control.selected_index = _RECONSTRUCTION_TEXTURE_SIZES.index(
                        parameters.texture_size
                    )
                else:
                    control.value = float(getattr(parameters, key))
                control.widget.enabled = not self._reconstruction_workspace_busy
        finally:
            self._syncing_reconstruction_workspace = False

    def _change_reconstruction_workspace_variant(
            self, index: int, *_rest) -> None:
        if self._syncing_reconstruction_workspace:
            return
        if 0 <= index < len(self._reconstruction_workspace_operation_ids):
            self._selected_reconstruction_workspace_operation_id = (
                self._reconstruction_workspace_operation_ids[index]
            )
        else:
            self._selected_reconstruction_workspace_operation_id = None
        self._refresh_reconstruction_workspace_artifacts()

    def _change_reconstruction_workspace_artifact(
            self, index: int, *_rest) -> None:
        if self._syncing_reconstruction_workspace:
            return
        preview = self.reconstruction_workspace_actions.get("preview")
        artifact = self._selected_reconstruction_workspace_artifact(index)
        self._selected_reconstruction_workspace_artifact_id = (
            artifact.artifact_id if artifact is not None else None
        )
        if preview is not None:
            preview.widget.enabled = bool(
                artifact is not None
                and artifact.preview_kind in {
                    WorkspacePreviewKind.IMAGE,
                    WorkspacePreviewKind.MESH,
                    WorkspacePreviewKind.POINTS,
                    WorkspacePreviewKind.OVERLAY,
                }
            )
        self._request_repaint()

    def _selected_reconstruction_workspace_artifact(self, index=None):
        workspace = self._reconstruction_workspace_snapshot
        combo = self.reconstruction_workspace_artifact_combo
        if workspace is None or combo is None:
            return None
        if index is None:
            index = combo.selected_index
        if not 0 <= index < len(self._reconstruction_workspace_artifact_ids):
            return None
        try:
            return workspace.artifact(
                self._reconstruction_workspace_artifact_ids[index]
            )
        except KeyError:
            return None

    def _preview_reconstruction_workspace_artifact(self) -> None:
        artifact = self._selected_reconstruction_workspace_artifact()
        if artifact is None or self._reconstruction_workspace_handler is None:
            return
        self._reconstruction_workspace_handler(
            "preview_artifact", artifact.artifact_id
        )

    def _run_reconstruction_workspace_operation(self) -> None:
        key = self._selected_reconstruction_workspace_operation
        if (
                key not in LEGACY_OPERATION_TARGET_STAGES
                or self._reconstruction_workspace_handler is None
                or self._reconstruction_workspace_busy):
            return
        self._reconstruction_workspace_handler("generate_to_operation", key)

    def _refresh_reconstruction_workspace_actions(self) -> None:
        generate = self.reconstruction_workspace_actions.get("generate")
        if generate is not None:
            generate.widget.enabled = bool(
                self._reconstruction_workspace_snapshot is not None
                and self._selected_reconstruction_workspace_operation
                in LEGACY_OPERATION_TARGET_STAGES
                and not self._reconstruction_workspace_busy
            )

    def _refresh_reconstruction_workspace_operation_buttons(self) -> None:
        workspace = self._reconstruction_workspace_snapshot
        for spec in PIXAL3D_PIPELINE.operations:
            operations = (
                workspace.operations_for_spec(spec.key)
                if workspace is not None else ()
            )
            ready = sum(
                operation.status in {
                    WorkspaceOperationStatus.READY,
                    WorkspaceOperationStatus.CACHED,
                }
                for operation in operations
            )
            running = any(
                operation.status is WorkspaceOperationStatus.RUNNING
                for operation in operations
            )
            failed = any(
                operation.status is WorkspaceOperationStatus.FAILED
                for operation in operations
            )
            suffix = ""
            background = _PIPELINE_OPERATION_BACKGROUND
            if running:
                suffix = " · running"
                background = _PIPELINE_OPERATION_RUNNING
            elif ready:
                variant_suffix = (
                    f" · {len(operations)} variants"
                    if len(operations) > 1 else ""
                )
                suffix = f" · ready{variant_suffix}"
                background = _PIPELINE_OPERATION_READY
            elif failed:
                suffix = " · failed"
                background = _PIPELINE_OPERATION_FAILED
            button = self.reconstruction_workspace_operation_buttons.get(
                spec.key
            )
            if button is None:
                continue
            button.set_text(f"{spec.label}{suffix}")
            if spec.key == self._selected_reconstruction_workspace_operation:
                background = _PIPELINE_OPERATION_SELECTED
            _set_widget_background(button.widget, background)

    def _refresh_reconstruction_workspace_variants(self) -> None:
        combo = self.reconstruction_workspace_variant_combo
        workspace = self._reconstruction_workspace_snapshot
        if combo is None:
            return
        previous = self._selected_reconstruction_workspace_operation_id
        operations = (
            workspace.operations_for_spec(
                self._selected_reconstruction_workspace_operation
            )
            if workspace is not None
            and self._selected_reconstruction_workspace_operation is not None
            else ()
        )
        self._syncing_reconstruction_workspace = True
        try:
            combo.clear()
            self._reconstruction_workspace_operation_ids = [
                operation.operation_id for operation in operations
            ]
            for operation in operations:
                combo.add_item(
                    f"{operation.variant_label} · {operation.status.value}"
                )
            selected = next(
                (index for index, operation in enumerate(operations)
                 if operation.operation_id == previous),
                -1,
            )
            if selected < 0 and workspace is not None:
                selected_artifact = workspace.selected_artifact_id
                if selected_artifact:
                    try:
                        producer = workspace.artifact(
                            selected_artifact
                        ).producer_operation_id
                    except KeyError:
                        producer = None
                    selected = next(
                        (index for index, operation in enumerate(operations)
                         if operation.operation_id == producer),
                        -1,
                    )
            if selected < 0 and operations:
                selected = next(
                    (index for index, operation in enumerate(operations)
                     if operation.status is WorkspaceOperationStatus.RUNNING),
                    len(operations) - 1,
                )
            combo.selected_index = selected
            combo.widget.enabled = bool(operations)
            self._selected_reconstruction_workspace_operation_id = (
                operations[selected].operation_id if selected >= 0 else None
            )
        finally:
            self._syncing_reconstruction_workspace = False
        self._refresh_reconstruction_workspace_artifacts()

    def _refresh_reconstruction_workspace_artifacts(self) -> None:
        combo = self.reconstruction_workspace_artifact_combo
        workspace = self._reconstruction_workspace_snapshot
        if combo is None:
            return
        operation_id = self._selected_reconstruction_workspace_operation_id
        artifacts = (
            workspace.artifacts_for_operation(operation_id)
            if workspace is not None and operation_id is not None
            else ()
        )
        self._syncing_reconstruction_workspace = True
        try:
            previous_artifact_id = (
                self._selected_reconstruction_workspace_artifact_id
            )
            combo.clear()
            self._reconstruction_workspace_artifact_ids = [
                artifact.artifact_id for artifact in artifacts
            ]
            operation = (
                workspace.operation(operation_id)
                if workspace is not None and operation_id is not None
                else None
            )
            spec = (
                PIXAL3D_PIPELINE.operation(operation.spec_key)
                if operation is not None else None
            )
            port_labels = {
                output.role: output.label for output in spec.outputs
            } if spec is not None else {}
            for artifact in artifacts:
                combo.add_item(
                    f"{port_labels.get(artifact.role, artifact.role)}"
                    f" · {artifact.kind.value}"
                )
            selected = next(
                (index for index, artifact in enumerate(artifacts)
                 if artifact.artifact_id == previous_artifact_id),
                -1,
            )
            if selected < 0:
                selected = next(
                    (index for index, artifact in enumerate(artifacts)
                     if workspace is not None
                     and artifact.artifact_id == workspace.selected_artifact_id),
                    0 if artifacts else -1,
                )
            combo.selected_index = selected
            combo.widget.enabled = bool(artifacts)
        finally:
            self._syncing_reconstruction_workspace = False
        self._change_reconstruction_workspace_artifact(
            combo.selected_index
        )

    def update_reconstruction_workspace(
            self,
            backend: ReconstructionBackend,
            workspace: ReconstructionWorkspace | None = None,
            parameters: ReconstructionParameters | None = None,
            *,
            busy: bool = False,
    ) -> None:
        available = backend is ReconstructionBackend.PIXAL3D
        self._reconstruction_workspace_busy = bool(busy)
        self._reconstruction_workspace_snapshot = workspace
        self._reconstruction_workspace_parameters = parameters
        if self.reconstruction_workspace_open is not None:
            self.reconstruction_workspace_open.widget.enabled = available
        if self.reconstruction_workspace_backend is not None:
            label = RECONSTRUCTION_BACKEND_LABELS[backend]
            suffix = "" if available else " · use Legacy Tools"
            self.reconstruction_workspace_backend.text = (
                f"Backend: {label}{suffix}"
            )
        for button in self.reconstruction_workspace_operation_buttons.values():
            button.widget.enabled = available
        if workspace is not None:
            if self.reconstruction_workspace_status is not None:
                state = "generating" if busy else "read-through"
                self.reconstruction_workspace_status.text = (
                    f"Legacy graph adapter · {state} · "
                    f"{len(workspace.operations)} operations · "
                    f"{len(workspace.artifacts)} artifacts"
                )
        elif self.reconstruction_workspace_status is not None:
            self.reconstruction_workspace_status.text = (
                "No Pixal3D graph snapshot is available."
            )
        self._refresh_reconstruction_workspace_operation_buttons()
        self._refresh_reconstruction_workspace_variants()
        self._refresh_reconstruction_workspace_parameters()
        self._refresh_reconstruction_workspace_actions()
        if not available and self.reconstruction_workspace_mode:
            self._set_reconstruction_workspace_mode(False)
        self._request_repaint()

    def set_reconstruction_stage_handler(self, handler) -> None:
        self._reconstruction_stage_handler = handler

    def set_reconstruction_parameter_handler(self, handler) -> None:
        self._reconstruction_parameter_handler = handler

    def set_reconstruction_refine_handler(self, handler) -> None:
        self._reconstruction_refine_handler = handler

    def _activate_reconstruction_refine(self, action: str, value=None) -> None:
        if (
            self._syncing_reconstruction_refine
            or self._reconstruction_refine_handler is None
        ):
            return
        self._reconstruction_refine_handler(action, value)

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
            "hi3dgen_slat_steps",
            "hi3dgen_normal_resolution",
            "hunyuan3d21_octree_resolution",
            "hunyuan3d21_texture_steps",
            "sam3d_sparse_steps",
            "sam3d_slat_steps",
        }:
            value = int(round(float(value)))
        elif key == "manual_fov_degrees":
            value = float(value)
        elif key in {
            "spar3d_guidance_scale",
            "hi3dgen_guidance_scale",
            "hunyuan3d21_guidance_scale",
            "hunyuan3d21_texture_guidance_scale",
            "sam3d_sparse_guidance_scale",
            "sam3d_slat_guidance_scale",
            "sam3d_simplify",
        }:
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
            controls["backend"].selected_index = tuple(
                ReconstructionBackend
            ).index(parameters.backend)
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
            controls["spar3d_guidance_scale"].value = (
                parameters.spar3d_guidance_scale
            )
            controls["hi3dgen_slat_steps"].value = float(
                parameters.hi3dgen_slat_steps
            )
            controls["hi3dgen_guidance_scale"].value = (
                parameters.hi3dgen_guidance_scale
            )
            controls["hi3dgen_normal_resolution"].selected_index = (
                _HI3DGEN_NORMAL_RESOLUTIONS.index(
                    parameters.hi3dgen_normal_resolution
                )
            )
            controls["hunyuan3d21_guidance_scale"].value = (
                parameters.hunyuan3d21_guidance_scale
            )
            controls["hunyuan3d21_octree_resolution"].selected_index = (
                _HUNYUAN3D21_OCTREE_RESOLUTIONS.index(
                    parameters.hunyuan3d21_octree_resolution
                )
            )
            controls["hunyuan3d21_texture_steps"].value = float(
                parameters.hunyuan3d21_texture_steps
            )
            controls["hunyuan3d21_texture_guidance_scale"].value = (
                parameters.hunyuan3d21_texture_guidance_scale
            )
            controls["sam3d_sparse_steps"].value = float(
                parameters.sam3d_sparse_steps
            )
            controls["sam3d_slat_steps"].value = float(
                parameters.sam3d_slat_steps
            )
            controls["sam3d_sparse_guidance_scale"].value = (
                parameters.sam3d_sparse_guidance_scale
            )
            controls["sam3d_slat_guidance_scale"].value = (
                parameters.sam3d_slat_guidance_scale
            )
            controls["sam3d_simplify"].value = parameters.sam3d_simplify
            controls["low_vram"].checked = parameters.low_vram
            applicable = RECONSTRUCTION_BACKEND_PARAMETER_KEYS[
                parameters.backend
            ]
            for key, control in controls.items():
                visible = key in applicable
                self.reconstruction_parameter_widgets[key].visible = visible
                control.widget.enabled = visible and not busy
        finally:
            self._syncing_reconstruction_parameters = False
        self._request_repaint()

    def _activate_reconstruction_stage(self, stage: ReconstructionStage) -> None:
        if self._reconstruction_stage_handler is not None:
            self._reconstruction_stage_handler(stage)

    def update_reconstruction_refine(
        self,
        parameters: ReconstructionRefineParameters,
        runs,
        active_run_id,
        *,
        mask_ready: bool,
        refine_supported: bool = True,
        can_refine: bool,
        can_texture_refine: bool = False,
        paint_active: bool = False,
        erase_active: bool = False,
        brush_size: int = 50,
        brush_hardness: float = 0.4,
        brush_flow: float = 1.0,
        busy: bool = False,
    ) -> None:
        controls = self.reconstruction_refine_controls
        if not controls:
            return
        self._syncing_reconstruction_refine = True
        try:
            controls["paint"].checked = paint_active
            controls["erase"].checked = erase_active
            controls["resize_detail_to_1024"].checked = (
                parameters.resize_detail_to_1024
            )
            controls["brush_size"].value = float(brush_size)
            controls["brush_hardness"].value = brush_hardness
            controls["brush_flow"].value = brush_flow
            controls["strength"].value = parameters.strength
            controls["steps"].value = float(parameters.steps)
            controls["seed"].value = float(parameters.seed)
            labels = []
            refined_index = 0
            for run in runs:
                if run.kind is ReconstructionRunKind.BASE:
                    labels.append(
                        f"Base ({RECONSTRUCTION_BACKEND_LABELS[run.backend]})"
                    )
                else:
                    refined_index += 1
                    labels.append(f"Refined {refined_index}")
            version = controls["version"]
            current_labels = [
                version.item_text(index) for index in range(version.item_count)
            ]
            if current_labels != labels:
                version.clear()
                for label in labels:
                    version.add_item(label)
            self._reconstruction_run_ids = [run.run_id for run in runs]
            has_versions = bool(runs)
            self.reconstruction_versions_title.visible = has_versions
            self.reconstruction_versions_panel.visible = has_versions
            self.reconstruction_refine_title.visible = refine_supported
            self.reconstruction_refine_panel.visible = refine_supported
            try:
                version.selected_index = self._reconstruction_run_ids.index(
                    active_run_id
                )
            except ValueError:
                version.selected_index = -1
            for key in (
                "paint", "erase", "resize_detail_to_1024",
                "brush_size", "brush_hardness",
                "brush_flow", "strength", "steps", "seed",
            ):
                controls[key].widget.enabled = refine_supported and not busy
            controls["clear"].widget.enabled = (
                refine_supported and mask_ready and not busy
            )
            controls["run"].widget.enabled = (
                refine_supported and can_refine and mask_ready and not busy
            )
            controls["run_texture"].widget.enabled = (
                refine_supported
                and can_texture_refine and mask_ready and not busy
            )
            controls["version"].widget.enabled = bool(runs) and not busy
        finally:
            self._syncing_reconstruction_refine = False
        self._request_repaint()

    def update_reconstruction_stages(
        self,
        statuses,
        progress,
        target: ReconstructionStage,
        selected: ReconstructionStage,
        *,
        busy: bool = False,
        backend: ReconstructionBackend | None = None,
    ) -> None:
        supported = (
            RECONSTRUCTION_BACKEND_STAGES[backend]
            if backend is not None else RECONSTRUCTION_STAGES
        )
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
            previewable = (
                stage in RECONSTRUCTION_PREVIEW_STAGES
                and (
                    stage is not ReconstructionStage.TEXTURE_LATENT
                    or backend in {
                        ReconstructionBackend.HUNYUAN3D21,
                        ReconstructionBackend.SAM3D_OBJECTS,
                    }
                )
            )
            button.widget.enabled = (
                stage in supported
                and
                previewable
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
            active_panel = (
                self.reconstruction_workspace_panel
                if self.reconstruction_workspace_mode
                else panel
            )
            self.main_splitter.set_first(active_panel.widget)
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
