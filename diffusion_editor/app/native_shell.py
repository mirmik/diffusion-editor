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

from .presentation import ViewPorts


CommandHandler = Callable[[], None]

_CHROME_BACKGROUND = SrgbColor(0.075, 0.080, 0.095, 1.0)
_WORKSPACE_EDGE = SrgbColor(0.32, 0.35, 0.40, 1.0)


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
    NativeCommandSpec("selection.clear", "Clear Selection", "Ctrl+D"),
    NativeCommandSpec("selection.invert", "Invert Selection", "Ctrl+Shift+I"),
    NativeCommandSpec("layer.new", "New Layer", "Ctrl+Shift+N"),
    NativeCommandSpec("layer.remove", "Remove Layer"),
    NativeCommandSpec("layer.flatten", "Flatten"),
    NativeCommandSpec("layer.detect", "Detect Objects…"),
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
            "layer.remove",
            None,
            "layer.flatten",
            None,
            "layer.detect",
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
    None,
    "selection.all",
    "selection.clear",
    "selection.invert",
)


class NativeEditorView:
    """Native root layout plus presentation ports for the application owner."""

    def __init__(
            self,
            document: TcDocument,
            request_repaint: Callable[[], None],
            set_window_title: Callable[[str], None],
            command_handlers: Mapping[str, CommandHandler]) -> None:
        self._request_repaint = request_repaint
        self._set_platform_window_title = set_window_title
        self._command_handlers = dict(command_handlers)
        self._command_refs: dict[str, list[tuple[CommandModel, object]]] = {}
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
