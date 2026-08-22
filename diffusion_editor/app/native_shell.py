"""Native shell and app-owned command projection for Diffusion Editor."""

from __future__ import annotations

from dataclasses import dataclass
import json
import secrets
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
    ReconstructionRefinePlacement,
    ReconstructionRefineParameters,
    ReconstructionRunKind,
    ReconstructionStage,
    ReconstructionStageStatus,
)
from ..generation.local_detail_geometry import (
    euler_degrees_from_quaternion,
)
from ..generation.reconstruction_workspace import (
    LEGACY_OPERATION_TARGET_STAGES,
    PIXAL3D_OPERATION_PARAMETER_KEYS,
    PIXAL3D_PIPELINE,
    PIXAL3D_PRESENTED_OPERATION_KEYS,
    ReconstructionWorkspace,
    WorkspaceOperationStatus,
    WorkspacePreviewKind,
    pixal3d_presented_operations,
)
from .native_reconstruction_viewport import (
    POINT_CLOUD_COLOR_LABELS,
    POINT_CLOUD_COLOR_MODES,
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
_MAX_RECONSTRUCTION_SEED = 2_147_483_647

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

_WORKSPACE_REFINE_PARAMETER_SCOPES = {
    "hr.refine": ("hr", "HR geometry refine parameters"),
    "texture.refine": ("texture", "Texture refine parameters"),
}

def _set_widget_background(widget, color: SrgbColor) -> None:
    style_override = widget.style_override
    style = style_override.value
    style.background = color
    style_override.value = style
    style_override.fields = StyleField.Background.value
    widget.style_override = style_override


def _random_reconstruction_seed() -> int:
    return secrets.randbelow(_MAX_RECONSTRUCTION_SEED + 1)


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
        "ai.depth_map",
        "Depth Map — DA3 Nested Giant 1.1",
        tooltip=(
            "Create metric depth with a predicted camera matrix using the "
            "highest-quality Depth Anything 3 profile"
        ),
    ),
    NativeCommandSpec(
        "ai.depth_map.subject",
        "Depth Map — Character Only (DA3)",
        tooltip=(
            "Automatically isolate the foreground character and create a "
            "depth map with a transparent background"
        ),
    ),
    NativeCommandSpec(
        "ai.depth_map.da3_mono",
        "Depth Map — DA3 Mono Large",
        tooltip=(
            "Create high-quality direct relative depth without camera "
            "calibration using Depth Anything 3 Mono Large"
        ),
    ),
    NativeCommandSpec(
        "ai.depth_map.depth_pro",
        "Depth Map — Apple Depth Pro",
        tooltip="Create a metric depth layer using Apple Depth Pro",
    ),
    NativeCommandSpec(
        "ai.depth_map.v2_large",
        "Depth Map — V2 Large",
        tooltip="Create a relative depth layer using Depth Anything V2 Large",
    ),
    NativeCommandSpec(
        "ai.depth_map.v2_small",
        "Depth Map — V2 Small",
        tooltip="Create a fast relative depth layer using Depth Anything V2 Small",
    ),
    NativeCommandSpec(
        "ai.depth_point_cloud",
        "View Depth as Point Cloud",
        tooltip=(
            "Project the latest depth result into a colored Termin point "
            "cloud with orbit controls"
        ),
    ),
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
        "ai",
        "AI",
        (
            "ai.depth_map",
            "ai.depth_map.subject",
            None,
            "ai.depth_map.da3_mono",
            "ai.depth_map.depth_pro",
            "ai.depth_map.v2_large",
            "ai.depth_map.v2_small",
            None,
            "ai.depth_point_cloud",
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
    "ai.depth_map",
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
        self.reconstruction_refine_viewport = None
        self.reconstruction_refine_placement_panel = None
        self.reconstruction_refine_placement_controls = {}
        self.reconstruction_refine_placement_actions = {}
        self.reconstruction_views_splitter = None
        self.reconstruction_refine_view_container = None
        self._reconstruction_refine_view_visible = False
        self._reconstruction_primary_view_parking = document.create_vstack(
            "DiffusionEditorReconstructionPrimaryViewParking"
        )
        self._reconstruction_outer_view_parking = document.create_vstack(
            "DiffusionEditorReconstructionOuterViewParking"
        )
        self.canvas_reconstruction_splitter = None
        self.reconstruction_toolbar_model = None
        self.reconstruction_toolbar = None
        self.reconstruction_shading_combo = None
        self.reconstruction_point_color_combo = None
        self.reconstruction_point_color_legend = None
        self.reconstruction_buttons = {}
        self.reconstruction_stage_buttons = {}
        self.reconstruction_stage_checks = {}
        self._reconstruction_stage_handler = None
        self._reconstruction_parameter_handler = None
        self._reconstruction_refine_handler = None
        self._reconstruction_refine_placement_handler = None
        self._syncing_reconstruction_parameters = False
        self._syncing_reconstruction_refine = False
        self._syncing_reconstruction_refine_placement = False
        self._syncing_reconstruction_point_color = False
        self.reconstruction_parameter_controls = {}
        self.reconstruction_parameter_widgets = {}
        self.reconstruction_seed_buttons = {}
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
        self.reconstruction_workspace_refine_source_row = None
        self.reconstruction_workspace_refine_source_combo = None
        self.reconstruction_workspace_status = None
        self.reconstruction_workspace_actions = {}
        self.reconstruction_workspace_parameter_controls = {}
        self.reconstruction_workspace_parameter_rows = {}
        self.reconstruction_workspace_parameter_reset = None
        self.reconstruction_workspace_seed_buttons = {}
        self.reconstruction_workspace_mask_panel = None
        self.reconstruction_workspace_mask_controls = {}
        self.reconstruction_workspace_refine_parameters_panel = None
        self.reconstruction_workspace_refine_parameters_title = None
        self.reconstruction_workspace_refine_parameter_controls = {}
        self.reconstruction_workspace_refine_parameter_rows = {}
        self._reconstruction_workspace_refine_parameters = {}
        self._reconstruction_workspace_parameters = None
        self._reconstruction_workspace_handler = None
        self._reconstruction_workspace_snapshot = None
        self._reconstruction_workspace_operation_ids = []
        self._reconstruction_workspace_artifact_ids = []
        self._reconstruction_workspace_refine_source_artifact_ids = []
        self._selected_reconstruction_workspace_operation_id = None
        self._selected_reconstruction_workspace_artifact_id = None
        self._syncing_reconstruction_workspace = False
        self._reconstruction_workspace_busy = False
        self._reconstruction_workspace_mask_ready = False
        self._reconstruction_workspace_can_lr_refine = False
        self._reconstruction_workspace_can_refine = False
        self._reconstruction_workspace_can_texture_refine = False
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
            if not 0 <= index < len(RECONSTRUCTION_SHADING_MODES):
                return
            mode = RECONSTRUCTION_SHADING_MODES[index]
            if callable(set_shading_mode):
                set_shading_mode(mode)
            refine_set_shading = getattr(
                self.reconstruction_refine_viewport,
                "set_shading_mode",
                None,
            )
            if callable(refine_set_shading):
                refine_set_shading(mode)

        self._connections.append(shading_combo.connect_changed(
            activate_shading
        ))
        shading_row.add_flex_child(shading_label, 1.0)
        shading_row.add_fixed_child(shading_combo.widget, 108.0)
        toolbar.add_preferred_child(shading_row)

        point_color_row = self._document.create_hstack(
            "DiffusionEditorReconstructionPointColorRow"
        )
        point_color_row.set_layout_spacing(4.0)
        point_color_label = self._document.create_label(
            "Point color", "DiffusionEditorReconstructionPointColorLabel"
        )
        point_color_combo = self._document.create_combo_box()
        point_color_combo.widget.stable_id = (
            "diffusion-editor.reconstruction.point-color"
        )
        for mode in POINT_CLOUD_COLOR_MODES:
            point_color_combo.add_item(POINT_CLOUD_COLOR_LABELS[mode])
        point_color_combo.selected_index = 0
        point_color_combo.widget.enabled = False
        set_point_color_mode = getattr(
            viewport_view, "set_point_cloud_color_mode", None)

        def activate_point_color(index, *_rest):
            if self._syncing_reconstruction_point_color:
                return
            if not 0 <= index < len(POINT_CLOUD_COLOR_MODES):
                return
            if callable(set_point_color_mode):
                set_point_color_mode(POINT_CLOUD_COLOR_MODES[index])

        self._connections.append(point_color_combo.connect_changed(
            activate_point_color
        ))
        point_color_row.add_flex_child(point_color_label, 1.0)
        point_color_row.add_fixed_child(point_color_combo.widget, 108.0)
        toolbar.add_preferred_child(point_color_row)
        point_color_legend = self._document.create_label(
            "", "DiffusionEditorReconstructionPointColorLegend"
        )
        point_color_legend.stable_id = (
            "diffusion-editor.reconstruction.point-color-legend"
        )
        point_color_legend.visible = False
        toolbar.add_preferred_child(point_color_legend)

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
        parameter_seed_buttons = {}

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
            if key == "seed":
                random_button = self._document.create_button("Random")
                random_button.widget.stable_id = (
                    "diffusion-editor.reconstruction.parameter.seed.random"
                )
                self._connections.append(random_button.connect_clicked(
                    self._randomize_reconstruction_seed
                ))
                parameter_seed_buttons[key] = random_button
                row.add_fixed_child(random_button.widget, 68.0)
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
        add_spin("seed", "Seed", defaults.seed, 0, _MAX_RECONSTRUCTION_SEED, 1)
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
            if key == "seed":
                random_button = self._document.create_button("Random")
                random_button.widget.stable_id = (
                    "diffusion-editor.reconstruction.refine.seed.random"
                )
                self._connections.append(random_button.connect_clicked(
                    self._randomize_reconstruction_refine_seed
                ))
                refine_controls["seed_random"] = random_button
                row.add_fixed_child(random_button.widget, 68.0)
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
            "seed", "Refine seed", refine_defaults.seed,
            0, _MAX_RECONSTRUCTION_SEED
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
        workspace_common_tools = self._document.create_vstack(
            "DiffusionEditorReconstructionWorkspaceCommonTools"
        )
        workspace_common_tools.stable_id = (
            "diffusion-editor.reconstruction.workspace.common-tools"
        )
        workspace_common_tools.set_layout_spacing(3.0)
        workspace_content.add_preferred_child(workspace_common_tools)

        workspace_group_buttons = {}
        workspace_operation_buttons = {}
        workspace_operation_rows = {}
        for group in PIXAL3D_PIPELINE.groups:
            presented_operations = pixal3d_presented_operations(group.key)
            if not presented_operations:
                continue
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
            for operation in presented_operations:
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
        inspector_inputs.visible = False
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
        workspace_parameter_seed_buttons = {}

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
            if key.startswith("pixal3d_") and key.endswith("_seed"):
                random_button = self._document.create_button("Random")
                random_button.widget.stable_id = (
                    "diffusion-editor.reconstruction.workspace.parameter."
                    f"{key}.random"
                )
                self._connections.append(random_button.connect_clicked(
                    lambda key=key: self._randomize_reconstruction_workspace_seed(
                        key
                    )
                ))
                workspace_parameter_seed_buttons[key] = random_button
                row.add_fixed_child(random_button.widget, 68.0)
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
                f"pixal3d_{phase}_seed", 0, _MAX_RECONSTRUCTION_SEED
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

        workspace_mask_panel = self._document.create_vstack(
            "DiffusionEditorReconstructionWorkspaceMaskTools"
        )
        workspace_mask_panel.stable_id = (
            "diffusion-editor.reconstruction.workspace.mask"
        )
        workspace_mask_panel.set_layout_spacing(3.0)
        mask_caption = self._document.create_label(
            "Mask tools",
            "DiffusionEditorReconstructionWorkspaceInspectorTitle",
        )
        mask_caption.stable_id = (
            "diffusion-editor.reconstruction.workspace.mask.title"
        )
        mask_hint = self._document.create_label(
            "Paint the shared red selection mask on the source image.",
            "DiffusionEditorReconstructionWorkspaceInspectorText",
        )
        mask_hint.stable_id = (
            "diffusion-editor.reconstruction.workspace.mask.hint"
        )
        workspace_mask_panel.add_preferred_child(mask_caption)
        workspace_mask_panel.add_preferred_child(mask_hint)
        workspace_mask_controls = {}

        def add_workspace_mask_checkbox(key, label):
            row = self._document.create_hstack(
                "DiffusionEditorReconstructionWorkspaceInspectorRow"
            )
            row.set_layout_spacing(4.0)
            control = self._document.create_checkbox(False)
            control.widget.stable_id = (
                f"diffusion-editor.reconstruction.workspace.mask.{key}"
            )
            caption = self._document.create_label(
                label, "DiffusionEditorReconstructionWorkspaceInspectorText"
            )
            self._connections.append(control.connect_changed(
                lambda value, key=key:
                self._activate_reconstruction_refine(key, bool(value))
            ))
            row.add_preferred_child(control.widget)
            row.add_flex_child(caption, 1.0)
            workspace_mask_controls[key] = control
            workspace_mask_panel.add_preferred_child(row)

        def add_workspace_mask_slider(
                key, label, value, minimum, maximum, step, decimals=0):
            control = self._document.create_slider_edit(float(value))
            control.widget.stable_id = (
                f"diffusion-editor.reconstruction.workspace.mask.{key}"
            )
            control.label = label
            control.set_range(float(minimum), float(maximum))
            control.set_step(float(step))
            control.set_decimals(int(decimals))
            self._connections.append(control.connect_changed(
                lambda value, key=key:
                self._activate_reconstruction_refine(key, value)
            ))
            workspace_mask_controls[key] = control
            workspace_mask_panel.add_preferred_child(control.widget)

        add_workspace_mask_checkbox("paint", "Paint refine mask")
        add_workspace_mask_checkbox("erase", "Erase mask")
        add_workspace_mask_slider("brush_size", "Brush size", 50, 1, 500, 1)
        add_workspace_mask_slider(
            "brush_hardness", "Hardness", 0.4, 0, 1, 0.05, 2
        )
        add_workspace_mask_slider(
            "brush_flow", "Flow", 1.0, 0, 1, 0.05, 2
        )
        clear_workspace_mask = self._document.create_button(
            "Clear refine mask"
        )
        clear_workspace_mask.widget.stable_id = (
            "diffusion-editor.reconstruction.workspace.mask.clear"
        )
        self._connections.append(clear_workspace_mask.connect_clicked(
            lambda: self._activate_reconstruction_refine("clear")
        ))
        workspace_mask_controls["clear"] = clear_workspace_mask
        workspace_mask_panel.add_preferred_child(clear_workspace_mask.widget)
        workspace_mask_panel.visible = True
        workspace_common_tools.add_preferred_child(workspace_mask_panel)

        workspace_refine_parameters_panel = self._document.create_vstack(
            "DiffusionEditorReconstructionWorkspaceRefineParameters"
        )
        workspace_refine_parameters_panel.stable_id = (
            "diffusion-editor.reconstruction.workspace.refine-parameters"
        )
        workspace_refine_parameters_panel.set_layout_spacing(3.0)
        workspace_refine_parameters_title = self._document.create_label(
            "Refine parameters",
            "DiffusionEditorReconstructionWorkspaceInspectorTitle",
        )
        workspace_refine_parameters_panel.add_preferred_child(
            workspace_refine_parameters_title
        )
        workspace_refine_parameter_controls = {}
        workspace_refine_parameter_rows = {}

        def add_workspace_refine_checkbox(key, label, checked=False):
            row = self._document.create_hstack(
                "DiffusionEditorReconstructionWorkspaceInspectorRow"
            )
            row.set_layout_spacing(4.0)
            control = self._document.create_checkbox(checked)
            control.widget.stable_id = (
                "diffusion-editor.reconstruction.workspace.refine."
                f"{key}"
            )
            caption = self._document.create_label(
                label, "DiffusionEditorReconstructionWorkspaceInspectorText"
            )
            self._connections.append(control.connect_changed(
                lambda value, key=key:
                self._change_reconstruction_workspace_refine_parameter(
                    key, bool(value)
                )
            ))
            row.add_preferred_child(control.widget)
            row.add_flex_child(caption, 1.0)
            workspace_refine_parameter_controls[key] = control
            workspace_refine_parameter_rows[key] = row
            workspace_refine_parameters_panel.add_preferred_child(row)

        def add_workspace_refine_slider(
                key, label, value, minimum, maximum, step, decimals=0):
            control = self._document.create_slider_edit(float(value))
            control.widget.stable_id = (
                "diffusion-editor.reconstruction.workspace.refine."
                f"{key}"
            )
            control.label = label
            control.set_range(float(minimum), float(maximum))
            control.set_step(float(step))
            control.set_decimals(int(decimals))
            self._connections.append(control.connect_changed(
                lambda value, key=key:
                self._change_reconstruction_workspace_refine_parameter(
                    key, value
                )
            ))
            workspace_refine_parameter_controls[key] = control
            workspace_refine_parameter_rows[key] = control.widget
            workspace_refine_parameters_panel.add_preferred_child(
                control.widget
            )

        def add_workspace_refine_spin(key, label, value, minimum, maximum):
            row = self._document.create_hstack(
                "DiffusionEditorReconstructionWorkspaceInspectorRow"
            )
            row.set_layout_spacing(4.0)
            caption = self._document.create_label(
                label, "DiffusionEditorReconstructionWorkspaceInspectorText"
            )
            control = self._document.create_spin_box(float(value))
            control.widget.stable_id = (
                "diffusion-editor.reconstruction.workspace.refine."
                f"{key}"
            )
            control.set_range(float(minimum), float(maximum))
            control.step = 1.0
            control.decimals = 0
            self._connections.append(control.connect_changed(
                lambda value, key=key:
                self._change_reconstruction_workspace_refine_parameter(
                    key, value
                )
            ))
            row.add_flex_child(caption, 1.0)
            row.add_fixed_child(control.widget, 108.0)
            if key == "seed":
                random_button = self._document.create_button("Random")
                random_button.widget.stable_id = (
                    "diffusion-editor.reconstruction.workspace.refine."
                    "seed.random"
                )
                self._connections.append(random_button.connect_clicked(
                    self._randomize_reconstruction_workspace_refine_seed
                ))
                workspace_refine_parameter_controls["seed_random"] = (
                    random_button
                )
                row.add_fixed_child(random_button.widget, 68.0)
            workspace_refine_parameter_controls[key] = control
            workspace_refine_parameter_rows[key] = row
            workspace_refine_parameters_panel.add_preferred_child(row)

        def add_workspace_refine_combo(key, label, values, labels=None):
            row = self._document.create_hstack(
                "DiffusionEditorReconstructionWorkspaceInspectorRow"
            )
            row.set_layout_spacing(4.0)
            caption = self._document.create_label(
                label, "DiffusionEditorReconstructionWorkspaceInspectorText"
            )
            control = self._document.create_combo_box()
            control.widget.stable_id = (
                "diffusion-editor.reconstruction.workspace.refine."
                f"{key}"
            )
            shown = labels or tuple(map(str, values))
            for item in shown:
                control.add_item(str(item))
            self._connections.append(control.connect_changed(
                lambda index, *_rest, key=key, values=values:
                self._change_reconstruction_workspace_refine_parameter(
                    key, values[index]
                )
            ))
            row.add_flex_child(caption, 1.0)
            row.add_fixed_child(control.widget, 108.0)
            workspace_refine_parameter_controls[key] = control
            workspace_refine_parameter_rows[key] = row
            workspace_refine_parameters_panel.add_preferred_child(row)

        add_workspace_refine_checkbox(
            "resize_detail_to_1024", "Resize masked detail to 1024",
            refine_defaults.resize_detail_to_1024,
        )
        add_workspace_refine_slider(
            "strength", "Strength", refine_defaults.strength,
            0.05, 1.0, 0.05, 2,
        )
        add_workspace_refine_spin(
            "steps", "Refine steps", refine_defaults.steps, 1, 50
        )
        add_workspace_refine_spin(
            "seed", "Refine seed", refine_defaults.seed,
            0, _MAX_RECONSTRUCTION_SEED,
        )
        add_workspace_refine_combo(
            "local_resolution", "Local HR resolution",
            (None, *_RECONSTRUCTION_RESOLUTIONS),
            ("Same as base", *_RECONSTRUCTION_RESOLUTIONS),
        )
        workspace_refine_parameters_panel.visible = False
        workspace_content.add_preferred_child(
            workspace_refine_parameters_panel
        )

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

        refine_source_row = self._document.create_hstack(
            "DiffusionEditorReconstructionWorkspaceInspectorRow"
        )
        refine_source_row.set_layout_spacing(4.0)
        refine_source_label = self._document.create_label(
            "Refine source",
            "DiffusionEditorReconstructionWorkspaceInspectorText",
        )
        refine_source_combo = self._document.create_combo_box()
        refine_source_combo.widget.stable_id = (
            "diffusion-editor.reconstruction.workspace.refine-source"
        )
        self._connections.append(refine_source_combo.connect_changed(
            self._change_reconstruction_workspace_refine_source
        ))
        refine_source_row.add_flex_child(refine_source_label, 1.0)
        refine_source_row.add_fixed_child(refine_source_combo.widget, 170.0)
        refine_source_row.visible = False
        workspace_content.add_preferred_child(refine_source_row)

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
            elif key == "refine":
                self._connections.append(button.connect_clicked(
                    self._run_reconstruction_workspace_refine
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
        self.reconstruction_point_color_combo = point_color_combo
        self.reconstruction_point_color_legend = point_color_legend
        self.reconstruction_buttons = reconstruction_buttons
        self.reconstruction_stage_buttons = stage_buttons
        self.reconstruction_stage_checks = stage_checks
        self.reconstruction_parameter_controls = parameter_controls
        self.reconstruction_parameter_widgets = parameter_widgets
        self.reconstruction_seed_buttons = parameter_seed_buttons
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
        self.reconstruction_workspace_refine_source_row = refine_source_row
        self.reconstruction_workspace_refine_source_combo = (
            refine_source_combo
        )
        self.reconstruction_workspace_status = workspace_status
        self.reconstruction_workspace_actions = workspace_actions
        self.reconstruction_workspace_parameter_controls = (
            workspace_parameter_controls
        )
        self.reconstruction_workspace_parameter_rows = workspace_parameter_rows
        self.reconstruction_workspace_seed_buttons = (
            workspace_parameter_seed_buttons
        )
        self.reconstruction_workspace_parameter_reset = parameter_reset
        self.reconstruction_workspace_mask_panel = workspace_mask_panel
        self.reconstruction_workspace_mask_controls = workspace_mask_controls
        self.reconstruction_workspace_refine_parameters_panel = (
            workspace_refine_parameters_panel
        )
        self.reconstruction_workspace_refine_parameters_title = (
            workspace_refine_parameters_title
        )
        self.reconstruction_workspace_refine_parameter_controls = (
            workspace_refine_parameter_controls
        )
        self.reconstruction_workspace_refine_parameter_rows = (
            workspace_refine_parameter_rows
        )
        self.reconstruction_status = status
        self.reconstruction_panel = panel
        self._select_reconstruction_workspace_operation("source.prepare")
        self._request_repaint()

    def set_reconstruction_point_color_state(
            self,
            has_confidence: bool,
            mode: str = "image",
            legend: str = "",
    ) -> None:
        combo = self.reconstruction_point_color_combo
        label = self.reconstruction_point_color_legend
        if combo is None or label is None:
            return
        normalized = str(mode).strip().lower()
        if normalized not in POINT_CLOUD_COLOR_MODES:
            normalized = "image"
        if normalized == "confidence" and not has_confidence:
            normalized = "image"
        self._syncing_reconstruction_point_color = True
        try:
            combo.selected_index = POINT_CLOUD_COLOR_MODES.index(normalized)
            combo.widget.enabled = bool(has_confidence)
            label.text = str(legend) if has_confidence else ""
            label.visible = bool(has_confidence)
        finally:
            self._syncing_reconstruction_point_color = False
        self._request_repaint()

    def mount_reconstruction_refine_viewport(self, viewport_view) -> None:
        self._require_open()
        if self.reconstruction_viewport is None:
            raise RuntimeError("primary reconstruction viewport is not mounted")
        if self.reconstruction_refine_viewport is not None:
            raise RuntimeError("refine viewport is already mounted")
        container = self._document.create_vstack(
            "DiffusionEditorReconstructionRefineView"
        )
        container.stable_id = (
            "diffusion-editor.reconstruction.refine-view"
        )
        title = self._document.create_label(
            "Refine output · before merge",
            "DiffusionEditorReconstructionWorkspaceInspectorTitle",
        )
        title.stable_id = (
            "diffusion-editor.reconstruction.refine-view.title"
        )
        container.add_preferred_child(title)
        container.add_flex_child(viewport_view.widget, 1.0)
        placement_panel = self._document.create_vstack(
            "DiffusionEditorReconstructionRefinePlacement"
        )
        placement_panel.stable_id = (
            "diffusion-editor.reconstruction.refine-placement"
        )
        placement_panel.set_layout_spacing(3.0)
        placement_title = self._document.create_label(
            "Place local fragment",
            "DiffusionEditorReconstructionWorkspaceInspectorTitle",
        )
        placement_panel.add_preferred_child(placement_title)
        placement_controls = {}

        def add_placement_row(specs):
            row = self._document.create_hstack(
                "DiffusionEditorReconstructionRefinePlacementRow"
            )
            row.set_layout_spacing(4.0)
            for key, label, value, minimum, maximum, step, decimals in specs:
                caption = self._document.create_label(
                    label,
                    "DiffusionEditorReconstructionWorkspaceInspectorText",
                )
                control = self._document.create_spin_box(float(value))
                control.widget.stable_id = (
                    "diffusion-editor.reconstruction.refine-placement."
                    f"{key}"
                )
                control.set_range(float(minimum), float(maximum))
                control.step = float(step)
                control.decimals = int(decimals)
                self._connections.append(control.connect_changed(
                    lambda changed, key=key:
                    self._change_reconstruction_refine_placement(
                        key, float(changed)
                    )
                ))
                placement_controls[key] = control
                row.add_preferred_child(caption)
                row.add_flex_child(control.widget, 1.0)
            placement_panel.add_preferred_child(row)

        add_placement_row((
            ("x", "X", 0.0, -10.0, 10.0, 0.005, 4),
            ("y", "Y", 0.0, -10.0, 10.0, 0.005, 4),
            ("z", "Z", 0.0, -10.0, 10.0, 0.005, 4),
        ))
        add_placement_row((
            ("rx", "RX°", 0.0, -180.0, 180.0, 0.5, 2),
            ("ry", "RY°", 0.0, -180.0, 180.0, 0.5, 2),
            ("rz", "RZ°", 0.0, -180.0, 180.0, 0.5, 2),
            ("scale", "S", 1.0, 0.1, 10.0, 0.005, 4),
        ))
        action_row = self._document.create_hstack(
            "DiffusionEditorReconstructionRefinePlacementActions"
        )
        action_row.set_layout_spacing(4.0)
        reset = self._document.create_button("Reset")
        reset.widget.stable_id = (
            "diffusion-editor.reconstruction.refine-placement.reset"
        )
        accept = self._document.create_button("Accept placement")
        accept.widget.stable_id = (
            "diffusion-editor.reconstruction.refine-placement.accept"
        )
        self._connections.append(reset.connect_clicked(
            lambda: self._activate_reconstruction_refine_placement("reset")
        ))
        self._connections.append(accept.connect_clicked(
            lambda: self._activate_reconstruction_refine_placement("accept")
        ))
        action_row.add_flex_child(reset.widget, 1.0)
        action_row.add_flex_child(accept.widget, 1.0)
        placement_panel.add_preferred_child(action_row)
        placement_panel.visible = False
        container.add_preferred_child(placement_panel)
        splitter = self._document.create_splitter(
            True,
            "DiffusionEditorReconstructionViewsSplitter",
        )
        splitter.widget.stable_id = (
            "diffusion-editor.reconstruction.views-splitter"
        )
        splitter.set_first(self._reconstruction_primary_view_parking)
        splitter.set_second(container)
        splitter.set_split_fraction(0.62)
        splitter.set_min_extents(260.0, 220.0)
        self.reconstruction_refine_viewport = viewport_view
        self.reconstruction_views_splitter = splitter
        self.reconstruction_refine_view_container = container
        self.reconstruction_refine_placement_panel = placement_panel
        self.reconstruction_refine_placement_controls = placement_controls
        self.reconstruction_refine_placement_actions = {
            "reset": reset,
            "accept": accept,
        }
        if self.reconstruction_shading_combo is not None:
            index = self.reconstruction_shading_combo.selected_index
            if 0 <= index < len(RECONSTRUCTION_SHADING_MODES):
                viewport_view.set_shading_mode(
                    RECONSTRUCTION_SHADING_MODES[index]
                )
        self._request_repaint()

    def set_reconstruction_refine_placement_handler(self, handler) -> None:
        self._reconstruction_refine_placement_handler = handler

    def _change_reconstruction_refine_placement(
            self, key: str, value: float) -> None:
        if (
                self._syncing_reconstruction_refine_placement
                or self._reconstruction_refine_placement_handler is None):
            return
        self._reconstruction_refine_placement_handler(key, float(value))

    def _activate_reconstruction_refine_placement(self, action: str) -> None:
        if self._reconstruction_refine_placement_handler is not None:
            self._reconstruction_refine_placement_handler(action, None)

    def update_reconstruction_refine_placement(
        self,
        placement: ReconstructionRefinePlacement | None,
        *,
        busy: bool = False,
        accepted: bool = False,
    ) -> None:
        panel = self.reconstruction_refine_placement_panel
        controls = self.reconstruction_refine_placement_controls
        if panel is None:
            return
        panel.visible = placement is not None
        if placement is None:
            self._request_repaint()
            return
        rotations = euler_degrees_from_quaternion(placement.orientation)
        values = {
            "x": placement.translation[0],
            "y": placement.translation[1],
            "z": placement.translation[2],
            "rx": rotations[0],
            "ry": rotations[1],
            "rz": rotations[2],
            "scale": placement.scale,
        }
        self._syncing_reconstruction_refine_placement = True
        try:
            for key, value in values.items():
                controls[key].value = float(value)
                controls[key].widget.enabled = not busy
            self.reconstruction_refine_placement_actions[
                "reset"
            ].widget.enabled = (
                not busy and placement != ReconstructionRefinePlacement()
            )
            self.reconstruction_refine_placement_actions[
                "accept"
            ].widget.enabled = not busy and not accepted
        finally:
            self._syncing_reconstruction_refine_placement = False
        self._request_repaint()

    def set_reconstruction_refine_view_visible(self, visible: bool) -> None:
        visible = bool(visible)
        outer = self.canvas_reconstruction_splitter
        inner = self.reconstruction_views_splitter
        primary = self.reconstruction_viewport
        if (
                outer is None or inner is None or primary is None
                or visible == self._reconstruction_refine_view_visible):
            return
        if visible:
            outer.set_second(self._reconstruction_outer_view_parking)
            inner.set_first(primary.widget)
            outer.set_second(inner.widget)
        else:
            inner.set_first(self._reconstruction_primary_view_parking)
            outer.set_second(primary.widget)
        self._reconstruction_refine_view_visible = visible
        self._request_repaint()

    def _toggle_reconstruction_workspace_group(self, key: str) -> None:
        if key in self._expanded_reconstruction_workspace_groups:
            self._expanded_reconstruction_workspace_groups.remove(key)
        else:
            self._expanded_reconstruction_workspace_groups.add(key)
        self._refresh_reconstruction_workspace_groups()

    def _refresh_reconstruction_workspace_groups(self) -> None:
        for group in PIXAL3D_PIPELINE.groups:
            presented_operations = pixal3d_presented_operations(group.key)
            if not presented_operations:
                continue
            expanded = (
                group.key in self._expanded_reconstruction_workspace_groups
            )
            button = self.reconstruction_workspace_group_buttons.get(group.key)
            if button is not None:
                button.set_text("-" if expanded else "+")
            for operation in presented_operations:
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
        if self._reconstruction_workspace_handler is not None:
            self._reconstruction_workspace_handler(
                "set_workspace_mode", enabled
            )
        self._request_repaint()

    def _select_reconstruction_workspace_operation(self, key: str) -> None:
        if key not in PIXAL3D_PRESENTED_OPERATION_KEYS:
            return
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
            outputs = ", ".join(
                output.label for output in operation.outputs
                if output.preview_kind is not WorkspacePreviewKind.NONE
            ) or "no direct preview"
            self.reconstruction_workspace_inspector_outputs.text = (
                f"Preview: {outputs}"
            )
        self._refresh_reconstruction_workspace_variants()
        if self.reconstruction_workspace_refine_source_row is not None:
            self.reconstruction_workspace_refine_source_row.visible = (
                key == "lr.refine"
            )
        self._refresh_reconstruction_workspace_refine_sources()
        self._refresh_reconstruction_workspace_parameters()
        self._refresh_reconstruction_workspace_refine_parameters()
        self._refresh_reconstruction_workspace_actions()
        if self._reconstruction_workspace_handler is not None:
            self._reconstruction_workspace_handler("select_operation", key)
            self._preview_reconstruction_workspace_artifact()
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

    def _change_reconstruction_workspace_refine_parameter(
            self, key: str, value) -> None:
        if self._syncing_reconstruction_refine:
            return
        scope = _WORKSPACE_REFINE_PARAMETER_SCOPES.get(
            self._selected_reconstruction_workspace_operation
        )
        if scope is None:
            return
        self._activate_reconstruction_refine(f"{scope[0]}_{key}", value)

    def _refresh_reconstruction_workspace_refine_parameters(self) -> None:
        panel = self.reconstruction_workspace_refine_parameters_panel
        operation = self._selected_reconstruction_workspace_operation
        scope = _WORKSPACE_REFINE_PARAMETER_SCOPES.get(operation)
        if panel is None:
            return
        panel.visible = scope is not None
        if scope is None:
            return
        if self.reconstruction_workspace_refine_parameters_title is not None:
            self.reconstruction_workspace_refine_parameters_title.text = (
                scope[1]
            )
        resize_row = self.reconstruction_workspace_refine_parameter_rows.get(
            "resize_detail_to_1024"
        )
        if resize_row is not None:
            resize_row.visible = operation == "texture.refine"
        resolution_row = self.reconstruction_workspace_refine_parameter_rows.get(
            "local_resolution"
        )
        if resolution_row is not None:
            resolution_row.visible = operation == "hr.refine"
        parameters = self._reconstruction_workspace_refine_parameters.get(
            operation
        )
        if parameters is None:
            return
        controls = self.reconstruction_workspace_refine_parameter_controls
        self._syncing_reconstruction_refine = True
        try:
            controls["strength"].label = (
                "Replacement extent"
                if operation == "hr.refine" else "Strength"
            )
            controls["resize_detail_to_1024"].checked = (
                parameters.resize_detail_to_1024
            )
            controls["strength"].value = parameters.strength
            controls["steps"].value = float(parameters.steps)
            controls["seed"].value = float(parameters.seed)
            controls["local_resolution"].selected_index = (
                (None, *_RECONSTRUCTION_RESOLUTIONS).index(
                    parameters.local_resolution
                )
            )
            for key in (
                    "resize_detail_to_1024", "strength", "steps", "seed",
                    "local_resolution"):
                controls[key].widget.enabled = bool(
                    not self._reconstruction_workspace_busy
                    and (key != "resize_detail_to_1024"
                         or operation == "texture.refine")
                    and (key != "local_resolution"
                         or operation == "hr.refine")
                )
            seed_random = controls.get("seed_random")
            if seed_random is not None:
                seed_random.widget.enabled = (
                    not self._reconstruction_workspace_busy
                )
        finally:
            self._syncing_reconstruction_refine = False

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
        for key, button in getattr(
                self, "reconstruction_workspace_seed_buttons", {}
        ).items():
            button.widget.enabled = bool(
                key in visible_keys and not self._reconstruction_workspace_busy
            )
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
        if self._reconstruction_workspace_handler is not None:
            self._reconstruction_workspace_handler(
                "select_operation_variant",
                self._selected_reconstruction_workspace_operation_id,
            )

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

    def _change_reconstruction_workspace_refine_source(
            self, index: int, *_rest) -> None:
        if (
                self._syncing_reconstruction_workspace
                or self._reconstruction_workspace_handler is None
                or not 0 <= index < len(
                    self._reconstruction_workspace_refine_source_artifact_ids
                )):
            return
        self._reconstruction_workspace_handler(
            "select_refine_source",
            self._reconstruction_workspace_refine_source_artifact_ids[index],
        )

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

    def _run_reconstruction_workspace_refine(self) -> None:
        key = self._selected_reconstruction_workspace_operation
        if self._reconstruction_workspace_busy:
            return
        if (
                key == "lr.refine"
                and self._reconstruction_workspace_can_lr_refine):
            self._activate_reconstruction_refine("run_lr")
        elif key == "hr.refine" and self._reconstruction_workspace_can_refine:
            self._activate_reconstruction_refine("run")
        elif (
                key == "texture.refine"
                and self._reconstruction_workspace_can_texture_refine):
            self._activate_reconstruction_refine("run_texture_workspace")

    def _refresh_reconstruction_workspace_actions(self) -> None:
        generate = self.reconstruction_workspace_actions.get("generate")
        if generate is not None:
            generate.widget.enabled = bool(
                self._reconstruction_workspace_snapshot is not None
                and self._selected_reconstruction_workspace_operation
                in LEGACY_OPERATION_TARGET_STAGES
                and not self._reconstruction_workspace_busy
            )
        refine = self.reconstruction_workspace_actions.get("refine")
        operation = self._selected_reconstruction_workspace_operation
        if refine is not None:
            refine.widget.enabled = bool(
                self._reconstruction_workspace_mask_ready
                and not self._reconstruction_workspace_busy
                and (
                    operation == "lr.refine"
                    and self._reconstruction_workspace_can_lr_refine
                    or operation == "hr.refine"
                    and self._reconstruction_workspace_can_refine
                    or operation == "texture.refine"
                    and self._reconstruction_workspace_can_texture_refine
                )
            )
        status = self.reconstruction_workspace_status
        if status is not None:
            if operation in {"lr.refine", "hr.refine", "texture.refine"}:
                can_run = bool(refine and refine.widget.enabled)
                if can_run:
                    status.text = "Ready to refine the selected mask."
                elif not self._reconstruction_workspace_mask_ready:
                    status.text = "Paint a non-empty refine mask to continue."
                else:
                    status.text = (
                        "The selected version has no compatible checkpoint "
                        "for this refine operation."
                    )
            else:
                status.text = (
                    "Base Pixal3D stages resume from the latest compatible "
                    "checkpoint."
                )

    def _refresh_reconstruction_workspace_operation_buttons(self) -> None:
        workspace = self._reconstruction_workspace_snapshot
        for spec in PIXAL3D_PIPELINE.operations:
            if spec.key not in PIXAL3D_PRESENTED_OPERATION_KEYS:
                continue
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
            tuple(
                artifact
                for artifact in workspace.artifacts_for_operation(operation_id)
                if artifact.preview_kind is not WorkspacePreviewKind.NONE
            )
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

    def _refresh_reconstruction_workspace_refine_sources(self) -> None:
        combo = self.reconstruction_workspace_refine_source_combo
        workspace = self._reconstruction_workspace_snapshot
        if combo is None:
            return
        candidates = []
        if workspace is not None:
            for spec_key in ("lr.generate", "lr.refine"):
                for operation in workspace.operations_for_spec(spec_key):
                    for artifact in workspace.artifacts_for_operation(
                            operation.operation_id):
                        if artifact.role != "lr_checkpoint":
                            continue
                        try:
                            metadata = json.loads(artifact.metadata_json)
                        except (TypeError, json.JSONDecodeError):
                            continue
                        if not metadata.get("lr_variant_id"):
                            continue
                        label = metadata.get(
                            "lr_variant_label", operation.variant_label
                        )
                        candidates.append((artifact, str(label), metadata))
        selected = next(
            (index for index, (_artifact, _label, metadata)
             in enumerate(candidates)
             if metadata.get("selected_refine_source")),
            0 if candidates else -1,
        )
        self._syncing_reconstruction_workspace = True
        try:
            combo.clear()
            self._reconstruction_workspace_refine_source_artifact_ids = [
                artifact.artifact_id
                for artifact, _label, _metadata in candidates
            ]
            for _artifact, label, _metadata in candidates:
                combo.add_item(label)
            combo.selected_index = selected
            combo.widget.enabled = bool(
                candidates and not self._reconstruction_workspace_busy
            )
        finally:
            self._syncing_reconstruction_workspace = False

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
                decision_points = sum(
                    operation.spec_key in PIXAL3D_PRESENTED_OPERATION_KEYS
                    for operation in workspace.operations
                )
                previews = sum(
                    artifact.preview_kind is not WorkspacePreviewKind.NONE
                    for artifact in workspace.artifacts
                    if workspace.operation(
                        artifact.producer_operation_id
                    ).spec_key in PIXAL3D_PRESENTED_OPERATION_KEYS
                )
                self.reconstruction_workspace_status.text = (
                    f"Legacy graph adapter · {state} · "
                    f"{decision_points} decision points · "
                    f"{previews} previews"
                )
        elif self.reconstruction_workspace_status is not None:
            self.reconstruction_workspace_status.text = (
                "No Pixal3D graph snapshot is available."
            )
        self._refresh_reconstruction_workspace_operation_buttons()
        self._refresh_reconstruction_workspace_variants()
        self._refresh_reconstruction_workspace_refine_sources()
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

    def _randomize_reconstruction_seed(self) -> None:
        self._change_reconstruction_parameter(
            "seed", _random_reconstruction_seed()
        )

    def _randomize_reconstruction_workspace_seed(self, key: str) -> None:
        self._change_reconstruction_workspace_parameter(
            key, _random_reconstruction_seed()
        )

    def _randomize_reconstruction_refine_seed(self) -> None:
        self._activate_reconstruction_refine(
            "seed", _random_reconstruction_seed()
        )

    def _randomize_reconstruction_workspace_refine_seed(self) -> None:
        self._change_reconstruction_workspace_refine_parameter(
            "seed", _random_reconstruction_seed()
        )

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
            for key, button in getattr(
                    self, "reconstruction_seed_buttons", {}
            ).items():
                button.widget.enabled = key in applicable and not busy
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
        lr_parameters: ReconstructionRefineParameters | None = None,
        texture_parameters: ReconstructionRefineParameters | None = None,
        mask_ready: bool,
        refine_supported: bool = True,
        can_refine: bool,
        can_texture_refine: bool = False,
        can_lr_refine: bool = False,
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
            seed_random = controls.get("seed_random")
            if seed_random is not None:
                seed_random.widget.enabled = refine_supported and not busy
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
            workspace_controls = self.reconstruction_workspace_mask_controls
            if workspace_controls:
                workspace_controls["paint"].checked = paint_active
                workspace_controls["erase"].checked = erase_active
                workspace_controls["brush_size"].value = float(brush_size)
                workspace_controls["brush_hardness"].value = brush_hardness
                workspace_controls["brush_flow"].value = brush_flow
                for key in (
                    "paint", "erase", "brush_size",
                    "brush_hardness", "brush_flow",
                ):
                    workspace_controls[key].widget.enabled = (
                        refine_supported and not busy
                    )
                workspace_controls["clear"].widget.enabled = (
                    refine_supported and mask_ready and not busy
                )
            self._reconstruction_workspace_refine_parameters = {
                "lr.refine": lr_parameters or parameters,
                "hr.refine": parameters,
                "texture.refine": texture_parameters or parameters,
            }
            self._reconstruction_workspace_mask_ready = bool(mask_ready)
            self._reconstruction_workspace_can_lr_refine = bool(
                refine_supported and can_lr_refine
            )
            self._reconstruction_workspace_can_refine = bool(
                refine_supported and can_refine
            )
            self._reconstruction_workspace_can_texture_refine = bool(
                refine_supported and can_texture_refine
            )
        finally:
            self._syncing_reconstruction_refine = False
        self._refresh_reconstruction_workspace_refine_parameters()
        self._refresh_reconstruction_workspace_actions()
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
