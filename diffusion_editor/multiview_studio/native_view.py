"""termin-gui-native projection of the multiview studio project."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from PIL import Image
from termin.gui_native import (
    CollectionItem,
    CollectionModel,
    CommandData,
    CommandKind,
    CommandModel,
    EdgeInsets,
    ImageFit,
    MenuBarEntry,
    Point,
    Size,
    TcDocument,
)
from tgfx import TextureEncoding

from .model import MultiviewProject, ViewKey, all_view_keys


_FRONT_KEY = ViewKey("eye", 0)
_BACK_KEY = ViewKey("eye", 180)
_SOURCE_KEYS = {_FRONT_KEY: "front", _BACK_KEY: "back"}
_ELEVATION_LABELS = {"low": "Low", "eye": "Eye", "elevated": "High"}


class StudioActions(Protocol):
    def new_project(self) -> None: ...
    def open_project(self) -> None: ...
    def save_project(self) -> None: ...
    def save_project_as(self) -> None: ...
    def open_recent_project(self, path: str) -> None: ...
    def clear_recent_projects(self) -> None: ...
    def quit(self) -> None: ...
    def pick_source(self, source: str) -> None: ...
    def pick_slot(self, key: ViewKey) -> None: ...
    def clear_slot(self, key: ViewKey) -> None: ...
    def set_qwen_seed(self, seed: int) -> None: ...
    def set_trellis_setting(self, field: str, value: int) -> None: ...
    def set_texture_setting(self, field: str, value: int) -> None: ...
    def set_mesh_postprocess(self, field: str, value: bool | float) -> None: ...
    def generate_view(self, key: ViewKey) -> None: ...
    def generate_four(self) -> None: ...
    def generate_missing(self) -> None: ...
    def generate_all(self) -> None: ...
    def build_shape(self) -> None: ...
    def reprocess_shape(self) -> None: ...
    def texture_model(self) -> None: ...
    def open_shape(self) -> None: ...
    def cancel_job(self) -> None: ...
    def begin_refine_cube(self) -> None: ...
    def set_refine_cube_value(self, field: str, value: float) -> None: ...
    def confirm_refine_cube(self) -> None: ...
    def cancel_refine_cube(self) -> None: ...
    def clear_refine_cube(self) -> None: ...


class NativeMultiviewStudioView:
    def __init__(
        self,
        document: TcDocument,
        actions: StudioActions,
        *,
        request_repaint: Callable[[], None],
        texture_lease_factory: Callable[[], object],
    ) -> None:
        self._document = document
        self._actions = actions
        self._request_repaint = request_repaint
        self._texture_lease_factory = texture_lease_factory
        self._connections: list[object] = []
        self._leases: dict[str, object] = {}
        self._preview_signatures: dict[str, tuple[str, int, int]] = {}
        self._preview_sizes: dict[str, Size] = {}
        self._syncing = False
        self._closed = False
        self._busy = False
        self._project = MultiviewProject()
        self._project_path: Path | None = None
        self._slot_keys = all_view_keys()
        self._selected_key = _FRONT_KEY
        self._context_key: ViewKey | None = None
        self._model_viewport = None
        self._refine_editing = False
        self._refine_picking = False
        self._menu_command_ids: dict[str, tuple[CommandModel, object]] = {}
        self._recent_command_paths: dict[str, Path] = {}
        self.slot_widgets: dict[ViewKey, dict[str, object]] = {}
        self.setting_controls: dict[str, object] = {}

        self.root = document.create_vstack("MultiviewStudioRoot")
        self.root.stable_id = "multiview-studio.root"
        self.root.set_layout_spacing(0.0)

        self._build_menu_bar()

        self.toolbar = document.create_hstack("MultiviewStudioToolbar")
        self.toolbar.stable_id = "multiview-studio.toolbar"
        self.toolbar.set_layout_spacing(5.0)
        self.generate_four_button = self._toolbar_button(
            "Generate four", actions.generate_four
        )
        self.generate_missing_button = self._toolbar_button(
            "Generate missing", actions.generate_missing
        )
        self.generate_all_button = self._toolbar_button(
            "Generate all", actions.generate_all
        )
        self.reprocess_shape_button = self._toolbar_button(
            "Reprocess cached mesh", actions.reprocess_shape
        )
        self.reprocess_shape_button.widget.enabled = False
        self.cancel_button = self._toolbar_button("Cancel", actions.cancel_job)
        self.cancel_button.widget.enabled = False

        self.left_content = document.create_vstack("MultiviewStudioLeft")
        self.left_content.stable_id = "multiview-studio.left-content"
        self.left_content.set_layout_spacing(6.0)
        self.left_content.set_layout_padding(EdgeInsets(6, 6, 6, 6))
        self._build_settings()
        self._build_texture_settings()
        self._build_postprocess_settings()
        self._build_shape_output()
        self.left_scroll = document.create_scroll_area()
        self.left_scroll.widget.stable_id = "multiview-studio.left-scroll"
        self.left_scroll.set_scroll_axes(False, True)
        self.left_scroll.set_content(self.left_content)

        self.content = document.create_vstack("MultiviewStudioContent")
        self.content.stable_id = "multiview-studio.content"
        self.content.set_layout_spacing(5.0)
        self.content.set_layout_padding(EdgeInsets(6, 6, 6, 6))

        navigator_label = document.create_label(
            "Views — select to inspect, right-click for actions"
        )
        navigator_label.stable_id = "multiview-studio.navigator.title"
        self.content.add_fixed_child(navigator_label, 22.0)

        self.slot_model = CollectionModel()
        self.slot_grid = document.create_file_grid_widget(self.slot_model)
        self.slot_grid.widget.stable_id = "multiview-studio.view-grid"
        # Eight fixed-width cards preserve the semantic 3x8 camera matrix even
        # when the application window is maximized.  The grid itself remains
        # a native collection view, so selection and context menus stay native.
        self.slot_grid.set_tile_size(166.0, 116.0)
        self.slot_grid.set_icon_size(78.0)
        self.slot_grid.set_tile_spacing(4.0)
        self.slot_grid.set_padding(4.0)
        self.slot_grid.empty_text = "No views"
        self._connections.extend(
            (
                self.slot_grid.connect_selection_changed(
                    self._on_slot_selection_changed
                ),
                self.slot_grid.connect_activated(self._on_slot_activated),
                self.slot_grid.connect_context_menu_requested(
                    self._on_slot_context_menu
                ),
            )
        )
        navigator_row = document.create_hstack(
            "MultiviewStudioNavigatorRow"
        )
        navigator_row.stable_id = "multiview-studio.navigator.row"
        navigator_row.add_fixed_child(self.slot_grid.widget, 1368.0)
        navigator_row.add_flex_child(document.create_spacer(Size(1, 1)), 1.0)
        self.content.add_fixed_child(navigator_row, 364.0)
        for index, key in enumerate(self._slot_keys):
            self.slot_widgets[key] = {"index": index}

        self.context_model = CommandModel()
        self.context_menu = document.create_menu(self.context_model)
        self.context_menu.widget.stable_id = (
            "multiview-studio.slot.context-menu"
        )
        self._connections.append(
            self.context_menu.connect_activated(self._on_context_activated)
        )

        self.selected_panel = document.create_group_box("Selected view")
        self.selected_panel.widget.stable_id = (
            "multiview-studio.workspace.image"
        )
        selected_content = document.create_vstack(
            "MultiviewStudioSelectedView"
        )
        selected_content.set_layout_spacing(4.0)
        self.selected_title = document.create_label("Front · 000°")
        self.selected_title.stable_id = (
            "multiview-studio.workspace.image.title"
        )
        self.selected_path = document.create_label("Empty")
        self.selected_path.stable_id = (
            "multiview-studio.workspace.image.path"
        )
        self.selected_image = document.create_image_widget()
        self.selected_image.widget.stable_id = (
            "multiview-studio.workspace.image.preview"
        )
        self.selected_image.fit = ImageFit.Contain
        selected_content.add_fixed_child(self.selected_title, 22.0)
        selected_content.add_fixed_child(self.selected_path, 20.0)
        selected_content.add_flex_child(self.selected_image.widget, 1.0)
        self.selected_panel.set_content(selected_content)

        self.model_panel = document.create_group_box("3D model")
        self.model_panel.widget.stable_id = "multiview-studio.workspace.model"
        self.model_content = document.create_vstack(
            "MultiviewStudioModelViewport"
        )
        self.model_content.set_layout_spacing(4.0)
        shading_row = document.create_hstack(
            "MultiviewStudioModelShading"
        )
        shading_row.stable_id = "multiview-studio.workspace.model.shading"
        shading_row.set_layout_spacing(4.0)
        shading_label = document.create_label("Shading")
        shading_row.add_flex_child(shading_label, 1.0)
        self.model_shading_buttons = {}
        self._model_shading = "flat"
        for mode, label in (
            ("flat", "Flat"),
            ("smooth", "Smooth"),
            ("wireframe", "Wireframe"),
        ):
            button = document.create_button(label)
            button.widget.stable_id = (
                f"multiview-studio.workspace.model.shading.{mode}"
            )
            self._connections.append(
                button.connect_clicked(
                    lambda selected=mode: self._set_model_shading(selected)
                )
            )
            shading_row.add_preferred_child(button.widget)
            self.model_shading_buttons[mode] = button
        self.model_content.add_preferred_child(shading_row)
        self._update_model_shading_buttons()

        refine_actions = document.create_hstack("MultiviewStudioRefineActions")
        refine_actions.stable_id = "multiview-studio.workspace.model.refine"
        refine_actions.set_layout_spacing(4.0)
        self.refine_cube_status = document.create_label("No refine region")
        self.refine_cube_status.stable_id = (
            "multiview-studio.workspace.model.refine.status"
        )
        refine_actions.add_flex_child(self.refine_cube_status, 1.0)
        self.pick_refine_cube_button = document.create_button("Pick cube")
        self.confirm_refine_cube_button = document.create_button("Confirm")
        self.cancel_refine_cube_button = document.create_button("Cancel")
        self.clear_refine_cube_button = document.create_button("Clear")
        for name, button, callback in (
            ("pick", self.pick_refine_cube_button, actions.begin_refine_cube),
            ("confirm", self.confirm_refine_cube_button, actions.confirm_refine_cube),
            ("cancel", self.cancel_refine_cube_button, actions.cancel_refine_cube),
            ("clear", self.clear_refine_cube_button, actions.clear_refine_cube),
        ):
            button.widget.stable_id = (
                f"multiview-studio.workspace.model.refine.{name}"
            )
            self._connections.append(button.connect_clicked(callback))
            refine_actions.add_preferred_child(button.widget)
        self.model_content.add_preferred_child(refine_actions)

        self.refine_cube_controls = {}
        self.refine_cube_value_rows = []
        for fields in (("x", "y"), ("z", "side")):
            row = document.create_hstack("MultiviewStudioRefineValues")
            row.set_layout_spacing(4.0)
            row.visible = False
            self.refine_cube_value_rows.append(row)
            for field in fields:
                label = document.create_label(
                    "Size" if field == "side" else field.upper()
                )
                control = document.create_spin_box(
                    1.0 if field == "side" else 0.0
                )
                control.widget.stable_id = (
                    f"multiview-studio.workspace.model.refine.{field}"
                )
                control.set_range(
                    -1_000_000.0 if field != "side" else 0.0001,
                    1_000_000.0,
                )
                control.step = 0.01
                control.decimals = 4
                self._connections.append(control.connect_changed(
                    lambda value, selected=field: None
                    if self._syncing
                    else actions.set_refine_cube_value(
                        selected, float(value)
                    )
                ))
                row.add_preferred_child(label)
                row.add_flex_child(control.widget, 1.0)
                self.refine_cube_controls[field] = control
            self.model_content.add_preferred_child(row)

        self.model_hint = document.create_label(
            "Build a TRELLIS.2 shape to inspect it here"
        )
        self.model_hint.stable_id = "multiview-studio.workspace.model.empty"
        self.model_content.add_flex_child(self.model_hint, 1.0)
        self.model_panel.set_content(self.model_content)

        self.workspace_splitter = document.create_splitter(
            True, "MultiviewStudioWorkspaceSplitter"
        )
        self.workspace_splitter.widget.stable_id = (
            "multiview-studio.workspace.splitter"
        )
        self.workspace_splitter.set_first(self.selected_panel.widget)
        self.workspace_splitter.set_second(self.model_panel.widget)
        self.workspace_splitter.set_split_fraction(0.50)
        self.workspace_splitter.set_min_extents(300.0, 300.0)
        self.content.add_flex_child(self.workspace_splitter.widget, 1.0)

        self.main_splitter = document.create_splitter(
            True, "MultiviewStudioMainSplitter"
        )
        self.main_splitter.widget.stable_id = "multiview-studio.main-splitter"
        self.main_splitter.set_first(self.left_scroll.widget)
        self.main_splitter.set_second(self.content)
        self.main_splitter.set_split_fraction(0.17)
        self.main_splitter.set_min_extents(240.0, 640.0)

        self.status = document.create_status_bar("Ready")
        self.status.widget.stable_id = "multiview-studio.status"
        self.root.add_fixed_child(self.menu_bar.widget, 28.0)
        self.root.add_fixed_child(self.toolbar, 38.0)
        self.root.add_flex_child(self.main_splitter.widget, 1.0)
        self.root.add_fixed_child(self.status.widget, 24.0)
        if not document.add_root(self.root.handle):
            raise RuntimeError("failed to add multiview studio root")

    def _build_menu_bar(self) -> None:
        self.menu_bar = self._document.create_menu_bar()
        self.menu_bar.widget.stable_id = "multiview-studio.menu-bar"
        self.menu_models: dict[str, CommandModel] = {}

        self.recent_model = CommandModel()
        self.set_recent_projects(())

        file_model = CommandModel()
        self._append_menu_command(
            file_model, CommandData("file.new", "New", shortcut="Ctrl+N")
        )
        self._append_menu_command(
            file_model, CommandData("file.open", "Open…", shortcut="Ctrl+O")
        )
        self._append_menu_command(
            file_model,
            CommandData(
                "file.open_recent",
                "Open Recent",
                submenu=self.recent_model,
            ),
        )
        self._append_separator(file_model, "file.separator.save")
        self._append_menu_command(
            file_model, CommandData("file.save", "Save", shortcut="Ctrl+S")
        )
        self._append_menu_command(
            file_model,
            CommandData("file.save_as", "Save As…", shortcut="Ctrl+Shift+S"),
        )
        self._append_separator(file_model, "file.separator.quit")
        self._append_menu_command(
            file_model, CommandData("app.quit", "Quit", shortcut="Ctrl+Q")
        )
        self.menu_models["file"] = file_model

        generate_model = CommandModel()
        self._append_menu_command(
            generate_model, CommandData("generate.selected", "Generate Selected View")
        )
        self._append_menu_command(
            generate_model, CommandData("generate.four", "Generate Four Views")
        )
        self._append_menu_command(
            generate_model, CommandData("generate.missing", "Generate Missing Views")
        )
        self._append_menu_command(
            generate_model, CommandData("generate.all", "Generate All Views")
        )
        self._append_separator(generate_model, "generate.separator.shape")
        self._append_menu_command(
            generate_model, CommandData("generate.shape", "Build TRELLIS.2 Shape")
        )
        self._append_menu_command(
            generate_model,
            CommandData(
                "generate.reprocess",
                "Reprocess Cached Mesh",
                enabled=False,
            ),
        )
        self._append_menu_command(
            generate_model,
            CommandData("generate.texture", "Texture Model", enabled=False),
        )
        self._append_separator(generate_model, "generate.separator.cancel")
        self._append_menu_command(
            generate_model, CommandData("generate.cancel", "Cancel", enabled=False)
        )
        self.menu_models["generate"] = generate_model

        view_model = CommandModel()
        self._append_menu_command(
            view_model,
            CommandData("view.show_shape", "Show Generated Model", enabled=False),
        )
        self._append_separator(view_model, "view.separator.shading")
        for mode, label in (
            ("flat", "Flat Shading"),
            ("smooth", "Smooth Shading"),
            ("wireframe", "Wireframe"),
        ):
            self._append_menu_command(
                view_model,
                CommandData(
                    f"view.shading.{mode}",
                    label,
                    checkable=True,
                    checked=mode == "flat",
                ),
            )
        self.menu_models["view"] = view_model

        for menu_id, label in (
            ("file", "File"),
            ("generate", "Generate"),
            ("view", "View"),
        ):
            self.menu_bar.add_menu(
                MenuBarEntry(menu_id, label, self.menu_models[menu_id])
            )
        self._connections.append(
            self.menu_bar.connect_activated(self._on_menu_activated)
        )

    def _append_menu_command(self, model: CommandModel, data: CommandData) -> None:
        native_id = model.append(data)
        self._menu_command_ids[data.stable_id] = (model, native_id)

    @staticmethod
    def _append_separator(model: CommandModel, stable_id: str) -> None:
        model.append(CommandData(stable_id, kind=CommandKind.Separator))

    def set_recent_projects(self, projects: tuple[Path, ...]) -> None:
        self._recent_command_paths.clear()
        commands: list[CommandData] = []
        for index, path in enumerate(projects):
            resolved = Path(path).expanduser().resolve()
            stable_id = f"file.recent.{index}"
            self._recent_command_paths[stable_id] = resolved
            commands.append(
                CommandData(
                    stable_id,
                    f"{index + 1}. {resolved.name} — {resolved.parent}",
                    tooltip=str(resolved),
                )
            )
        if commands:
            commands.append(
                CommandData("file.recent.separator", kind=CommandKind.Separator)
            )
            commands.append(CommandData("file.recent.clear", "Clear Recent Projects"))
        else:
            commands.append(CommandData("file.recent.empty", "No Recent Projects", enabled=False))
        self.recent_model.set_commands(commands)
        self._request_repaint()

    def dispatch_shortcut(self, key: int, modifiers: int) -> bool:
        return bool(self.menu_bar.dispatch_shortcut(key, modifiers))

    def _set_menu_command_enabled(self, stable_id: str, enabled: bool) -> None:
        model, native_id = self._menu_command_ids[stable_id]
        model.set_enabled(native_id, enabled)

    def _set_menu_command_checked(self, stable_id: str, checked: bool) -> None:
        model, native_id = self._menu_command_ids[stable_id]
        model.set_checked(native_id, checked)

    def _on_menu_activated(
        self,
        _menu_index: int,
        _native_id,
        command: CommandData,
    ) -> None:
        action = command.stable_id
        recent_path = self._recent_command_paths.get(action)
        if recent_path is not None:
            self._actions.open_recent_project(str(recent_path))
            return
        handlers = {
            "file.new": self._actions.new_project,
            "file.open": self._actions.open_project,
            "file.save": self._actions.save_project,
            "file.save_as": self._actions.save_project_as,
            "file.recent.clear": self._actions.clear_recent_projects,
            "app.quit": self._actions.quit,
            "generate.selected": lambda: self._actions.generate_view(
                self._selected_key
            ),
            "generate.four": self._actions.generate_four,
            "generate.missing": self._actions.generate_missing,
            "generate.all": self._actions.generate_all,
            "generate.shape": self._actions.build_shape,
            "generate.reprocess": self._actions.reprocess_shape,
            "generate.texture": self._actions.texture_model,
            "generate.cancel": self._actions.cancel_job,
            "view.show_shape": self._actions.open_shape,
        }
        handler = handlers.get(action)
        if handler is not None:
            handler()
            return
        prefix = "view.shading."
        if action.startswith(prefix):
            self._set_model_shading(action.removeprefix(prefix))

    def mount_model_viewport(self, viewport) -> None:
        """Replace the empty model panel with a live native 3D viewport."""
        if self._model_viewport is not None:
            raise RuntimeError("multiview model viewport is already mounted")
        self.model_content.remove_child(self.model_hint)
        self.model_content.add_flex_child(viewport.widget, 1.0)
        self._model_viewport = viewport
        self._set_model_shading(self._model_shading)
        self._request_repaint()

    def _set_model_shading(self, mode: str) -> None:
        if mode not in self.model_shading_buttons:
            raise ValueError(f"unsupported model shading mode: {mode}")
        self._model_shading = mode
        if self._model_viewport is not None:
            self._model_viewport.set_shading_mode(mode)
        self._update_model_shading_buttons()
        self._request_repaint()

    def _update_model_shading_buttons(self) -> None:
        for mode, button in self.model_shading_buttons.items():
            label = mode.capitalize() if mode != "wireframe" else "Wireframe"
            button.set_text(f"{label} •" if mode == self._model_shading else label)
            command_id = f"view.shading.{mode}"
            if command_id in self._menu_command_ids:
                self._set_menu_command_checked(
                    command_id, mode == self._model_shading
                )

    def apply_project(
        self,
        project: MultiviewProject,
        project_path: Path | None,
        dirty: bool,
    ) -> None:
        if self._closed:
            return
        self._project = project
        self._project_path = project_path
        self._syncing = True
        try:
            self.slot_model.set_items(
                [self._slot_item(slot) for slot in project.slots]
            )
            selected_index = self._slot_keys.index(self._selected_key)
            self.slot_grid.select(selected_index)
            self._apply_selected_view()

            self.setting_controls["qwen_seed"].text = str(project.qwen_seed)
            self.setting_controls["seed"].text = str(project.trellis.seed)
            for field in (
                "total_steps",
                "warmup_steps",
                "resolution",
                "decimation_target",
            ):
                self.setting_controls[field].value = float(
                    getattr(project.trellis, field)
                )
            self.setting_controls["texture.seed"].text = str(
                project.texture.seed
            )
            for field in (
                "total_steps",
                "warmup_steps",
                "resolution",
                "texture_size",
            ):
                self.setting_controls[f"texture.{field}"].value = float(
                    getattr(project.texture, field)
                )
            postprocess = project.trellis.postprocess
            for field in (
                "fill_holes",
                "remesh",
                "simplify",
                "cleanup",
                "final_repair",
                "remove_isolated_double_faces",
                "remove_degenerate_faces",
            ):
                self.setting_controls[f"postprocess.{field}"].checked = bool(
                    getattr(postprocess, field)
                )
            self.setting_controls[
                "postprocess.fill_hole_perimeter"
            ].value = postprocess.fill_hole_perimeter

            has_front = bool(project.front_path)
            has_both = has_front and bool(project.back_path)
            self.generate_four_button.widget.enabled = has_both and not self._busy
            self.generate_missing_button.widget.enabled = has_both and not self._busy
            self.generate_all_button.widget.enabled = has_both and not self._busy
            self._set_menu_command_enabled(
                "generate.selected",
                has_both and not self._busy,
            )
            self._set_menu_command_enabled(
                "generate.four", has_both and not self._busy
            )
            self._set_menu_command_enabled(
                "generate.missing", has_both and not self._busy
            )
            self._set_menu_command_enabled(
                "generate.all", has_both and not self._busy
            )
            errors = project.validate_shape_request()
            self.build_shape_button.widget.enabled = not errors and not self._busy
            self._set_menu_command_enabled(
                "generate.shape", not errors and not self._busy
            )
            cache_available = bool(project.geometry_path) and (
                Path(project.geometry_path).parent / "decoded-mesh-z-up.npz"
            ).is_file()
            can_reprocess = cache_available and not self._busy
            self.reprocess_shape_button.widget.enabled = can_reprocess
            self._set_menu_command_enabled(
                "generate.reprocess", can_reprocess
            )
            self._set_menu_command_enabled("generate.cancel", self._busy)
            self.cancel_button.widget.enabled = self._busy
            self.shape_path_label.text = self._path_label(project.shape_path)
            cube = project.refine_cube
            for index, field in enumerate(("x", "y", "z")):
                self.refine_cube_controls[field].value = (
                    cube.center[index] if cube is not None else 0.0
                )
            self.refine_cube_controls["side"].value = (
                cube.side if cube is not None else 1.0
            )
            geometry_exists = bool(project.geometry_path) and Path(
                project.geometry_path
            ).is_file()
            shape_exists = bool(project.shape_path) and Path(
                project.shape_path
            ).is_file()
            texture_errors = project.validate_texture_request()
            can_texture = geometry_exists and not texture_errors and not self._busy
            self.texture_model_button.widget.enabled = can_texture
            self.texture_model_button.set_text(
                "Retexture model"
                if shape_exists and project.shape_path != project.geometry_path
                else "Texture model"
            )
            self._set_menu_command_enabled("generate.texture", can_texture)
            can_open_shape = shape_exists and not self._busy
            self.build_shape_button.set_text(
                "Rebuild model" if geometry_exists else "Build model"
            )
            self._set_menu_command_enabled("view.show_shape", can_open_shape)
            populated = sum(slot.populated for slot in project.slots)
            name = project_path.name if project_path else "Untitled"
            marker = " *" if dirty else ""
            detail = errors[0] if errors else "Ready to build shape"
            self.status.text = (
                f"{name}{marker} · {populated}/24 populated · {detail}"
            )
            self._update_refine_controls()
        finally:
            self._syncing = False
        self._request_repaint()

    def set_status(self, text: str) -> None:
        self.status.text = text
        self._request_repaint()

    def set_reprocess_available(self, available: bool) -> None:
        enabled = bool(available) and not self._busy
        self.reprocess_shape_button.widget.enabled = enabled
        self._set_menu_command_enabled("generate.reprocess", enabled)
        self._request_repaint()

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        for control in self.setting_controls.values():
            control.widget.enabled = not self._busy
        self.cancel_button.widget.enabled = self._busy
        self._set_menu_command_enabled("generate.cancel", self._busy)
        if self._busy:
            self.build_shape_button.widget.enabled = False
            self.texture_model_button.widget.enabled = False
            self.reprocess_shape_button.widget.enabled = False
            self._set_menu_command_enabled("generate.texture", False)
            self._set_menu_command_enabled("generate.reprocess", False)
            self._set_menu_command_enabled("view.show_shape", False)
        self._update_refine_controls()

    def set_refine_editing(self, active: bool, *, picking: bool = False) -> None:
        self._refine_editing = bool(active)
        self._refine_picking = bool(picking) and self._refine_editing
        self._update_refine_controls()
        self._request_repaint()

    def _update_refine_controls(self) -> None:
        cube = self._project.refine_cube
        has_geometry = bool(self._project.geometry_path) and Path(
            self._project.geometry_path
        ).is_file()
        self.pick_refine_cube_button.widget.enabled = has_geometry and not self._busy
        for row in self.refine_cube_value_rows:
            row.visible = cube is not None
        for control in self.refine_cube_controls.values():
            control.widget.enabled = cube is not None and not self._busy
        self.confirm_refine_cube_button.widget.enabled = (
            cube is not None and not cube.confirmed and not self._busy
        )
        self.cancel_refine_cube_button.widget.enabled = (
            self._refine_editing and not self._busy
        )
        self.clear_refine_cube_button.widget.enabled = (
            cube is not None and not self._busy
        )
        if self._refine_picking:
            self.refine_cube_status.text = "Click the model"
        elif cube is None:
            self.refine_cube_status.text = "No refine region"
        elif cube.confirmed:
            self.refine_cube_status.text = "Refine cube confirmed"
        else:
            self.refine_cube_status.text = "Draft refine cube"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for lease in self._leases.values():
            lease.close()
        self._leases.clear()
        self._preview_signatures.clear()
        self._preview_sizes.clear()
        self._connections.clear()

    def _build_settings(self) -> None:
        group = self._document.create_group_box("Generation settings")
        group.widget.stable_id = "multiview-studio.settings"
        content = self._document.create_vstack("MultiviewStudioSettings")
        content.set_layout_spacing(4.0)
        self.setting_controls["qwen_seed"] = self._seed_input(
            content,
            "Qwen seed",
            "qwen-seed",
            20_260_822,
            lambda value: self._actions.set_qwen_seed(value),
        )
        self.setting_controls["seed"] = self._seed_input(
            content,
            "TRELLIS.2 seed",
            "trellis-seed",
            42,
            lambda value: self._actions.set_trellis_setting("seed", value),
        )
        self.setting_controls["total_steps"] = self._spin(
            content,
            "Steps per stage",
            "total-steps",
            39,
            1,
            200,
            1,
            lambda value: self._actions.set_trellis_setting(
                "total_steps", value
            ),
        )
        self.setting_controls["warmup_steps"] = self._spin(
            content,
            "Front warmup / stage",
            "warmup-steps",
            15,
            0,
            200,
            1,
            lambda value: self._actions.set_trellis_setting(
                "warmup_steps", value
            ),
        )
        self.setting_controls["resolution"] = self._spin(
            content,
            "Shape resolution",
            "resolution",
            1024,
            1024,
            2048,
            128,
            lambda value: self._actions.set_trellis_setting(
                "resolution", value
            ),
        )
        self.setting_controls["decimation_target"] = self._spin(
            content,
            "Target faces",
            "decimation-target",
            250_000,
            10_000,
            4_000_000,
            10_000,
            lambda value: self._actions.set_trellis_setting(
                "decimation_target", value
            ),
        )
        group.set_content(content)
        self.left_content.add_preferred_child(group.widget)

    def _build_postprocess_settings(self) -> None:
        group = self._document.create_group_box("Mesh postprocess (cached)")
        group.widget.stable_id = "multiview-studio.postprocess"
        content = self._document.create_vstack("MultiviewStudioPostprocess")
        content.set_layout_spacing(4.0)
        stages = (
            ("fill_holes", "Initial fill holes"),
            ("remesh", "Narrow-band DC remesh"),
            ("simplify", "CuMesh simplify"),
            ("cleanup", "CuMesh topology cleanup"),
            ("final_repair", "Final MeshLib repair"),
            (
                "remove_isolated_double_faces",
                "Remove isolated double faces",
            ),
            ("remove_degenerate_faces", "Remove zero-area faces"),
        )
        for field, label in stages:
            self.setting_controls[f"postprocess.{field}"] = self._checkbox(
                content,
                label,
                f"postprocess.{field.replace('_', '-')}",
                lambda checked, selected=field: self._actions.set_mesh_postprocess(
                    selected, checked
                ),
            )
        self.setting_controls["postprocess.fill_hole_perimeter"] = (
            self._float_spin(
                content,
                "Fill max perimeter",
                "postprocess.fill-hole-perimeter",
                0.03,
                0.0001,
                1.0,
                0.001,
                lambda value: self._actions.set_mesh_postprocess(
                    "fill_hole_perimeter", value
                ),
            )
        )
        hint = self._document.create_label(
            "Changing these stages does not rerun TRELLIS; use Reprocess cached mesh."
        )
        hint.stable_id = "multiview-studio.postprocess.hint"
        content.add_preferred_child(hint)
        group.set_content(content)
        self.left_content.add_preferred_child(group.widget)

    def _build_texture_settings(self) -> None:
        group = self._document.create_group_box("Texture generation")
        group.widget.stable_id = "multiview-studio.texture-settings"
        content = self._document.create_vstack("MultiviewStudioTextureSettings")
        content.set_layout_spacing(4.0)
        self.setting_controls["texture.seed"] = self._seed_input(
            content,
            "Texture seed",
            "texture-seed",
            43,
            lambda value: self._actions.set_texture_setting("seed", value),
        )
        for field, label, value, minimum, maximum, step in (
            ("total_steps", "Texture steps", 39, 1, 200, 1),
            ("warmup_steps", "Front warmup", 15, 0, 200, 1),
            ("resolution", "Shape encode resolution", 1024, 512, 1024, 512),
            ("texture_size", "Texture size", 2048, 1024, 4096, 512),
        ):
            self.setting_controls[f"texture.{field}"] = self._spin(
                content,
                label,
                f"texture-{field.replace('_', '-')}",
                value,
                minimum,
                maximum,
                step,
                lambda selected, name=field: self._actions.set_texture_setting(
                    name, selected
                ),
            )
        hint = self._document.create_label(
            "Re-encodes the current repaired mesh; all populated views are used."
        )
        hint.stable_id = "multiview-studio.texture-settings.hint"
        content.add_preferred_child(hint)
        group.set_content(content)
        self.left_content.add_preferred_child(group.widget)

    def _build_shape_output(self) -> None:
        group = self._document.create_group_box("Shape output")
        group.widget.stable_id = "multiview-studio.shape-output"
        content = self._document.create_vstack("MultiviewStudioShapeOutput")
        content.set_layout_spacing(4.0)
        self.shape_path_label = self._document.create_label("Empty")
        self.shape_path_label.stable_id = "multiview-studio.shape-output.path"
        self.build_shape_button = self._document.create_button("Build model")
        self.build_shape_button.widget.stable_id = (
            "multiview-studio.shape-output.build"
        )
        self._connections.append(
            self.build_shape_button.connect_clicked(self._actions.build_shape)
        )
        self.build_shape_button.widget.enabled = False
        self.texture_model_button = self._document.create_button("Texture model")
        self.texture_model_button.widget.stable_id = (
            "multiview-studio.shape-output.texture"
        )
        self._connections.append(
            self.texture_model_button.connect_clicked(self._actions.texture_model)
        )
        self.texture_model_button.widget.enabled = False
        content.add_preferred_child(self.shape_path_label)
        content.add_preferred_child(self.build_shape_button.widget)
        content.add_preferred_child(self.texture_model_button.widget)
        group.set_content(content)
        self.left_content.add_preferred_child(group.widget)

    def _seed_input(
        self,
        parent,
        label: str,
        suffix: str,
        value: int,
        callback: Callable[[int], None],
    ):
        row = self._document.create_hstack("MultiviewStudioSeedRow")
        row.set_layout_spacing(4.0)
        caption = self._document.create_label(label)
        field = self._document.create_text_input(str(value))
        field.widget.stable_id = f"multiview-studio.setting.{suffix}"

        def changed(text: str) -> None:
            if self._syncing or not text.strip():
                return
            try:
                seed = int(text)
            except ValueError:
                return
            callback(seed)

        self._connections.append(field.connect_changed(changed))
        row.add_flex_child(caption, 1.0)
        row.add_fixed_child(field.widget, 112.0)
        parent.add_preferred_child(row)
        return field

    def _slot_item(self, slot) -> CollectionItem:
        key = slot.key
        if key == _FRONT_KEY:
            title = "Front"
        elif key == _BACK_KEY:
            title = "Back"
        else:
            title = f"{key.azimuth:03d}°"
        state = "ready" if slot.populated else "empty"
        lease = self._load_preview_lease(
            f"slot:{key.stable_id}", slot.image_path, (160, 120)
        )
        texture_id = 0
        if lease is not None and not lease.empty:
            texture_id = int(lease.texture.id)
        return CollectionItem(
            key.stable_id,
            title,
            f"{_ELEVATION_LABELS[key.elevation]} · {state}",
            texture_id=texture_id,
            # Empty slots deliberately have no asset-browser glyph.  Their
            # dark rectangular card is the placeholder.
            icon="",
        )

    def _on_slot_selection_changed(self, indices: list[int]) -> None:
        if self._syncing or not indices:
            return
        self._select_slot(indices[-1])

    def _on_slot_activated(self, index: int, _item) -> None:
        self._select_slot(index)

    def _select_slot(self, index: int) -> None:
        if not 0 <= index < len(self._slot_keys):
            return
        self._selected_key = self._slot_keys[index]
        self._apply_selected_view()
        self._request_repaint()

    def _apply_selected_view(self) -> None:
        slot = self._project.slot(self._selected_key)
        if self._selected_key == _FRONT_KEY:
            label = "Front"
        elif self._selected_key == _BACK_KEY:
            label = "Back"
        else:
            label = _ELEVATION_LABELS[self._selected_key.elevation]
        self.selected_title.text = (
            f"{label} · {self._selected_key.azimuth:03d}°"
        )
        self.selected_path.text = self._path_label(slot.image_path)
        self._set_preview(
            "workspace:selected",
            self.selected_image,
            slot.image_path,
            max_size=(1600, 1200),
        )

    def _on_slot_context_menu(self, index: int, x: float, y: float) -> None:
        if not 0 <= index < len(self._slot_keys):
            return
        self._select_slot(index)
        self.slot_grid.select(index)
        key = self._slot_keys[index]
        self._context_key = key
        slot = self._project.slot(key)
        source = _SOURCE_KEYS.get(key)
        commands = [
            CommandData(
                "load",
                "Replace source image..." if source else "Load image...",
                enabled=not self._busy,
            ),
            CommandData(
                "generate",
                "Regenerate view" if slot.populated else "Generate view",
                enabled=(
                    source is None
                    and bool(self._project.front_path)
                    and not self._busy
                ),
            ),
            CommandData(
                "clear",
                "Clear view",
                enabled=source is None and slot.populated and not self._busy,
            ),
        ]
        self.context_model.set_commands(commands)
        self.context_menu.show(Point(x, y), self.root.bounds)
        self._request_repaint()

    def _on_context_activated(self, _index: int, _command_id, command) -> None:
        key = self._context_key
        if key is None:
            return
        action = command.stable_id
        if action == "load":
            source = _SOURCE_KEYS.get(key)
            if source is None:
                self._actions.pick_slot(key)
            else:
                self._actions.pick_source(source)
        elif action == "generate":
            self._actions.generate_view(key)
        elif action == "clear":
            self._actions.clear_slot(key)

    def _spin(
        self,
        parent,
        label: str,
        suffix: str,
        value: int,
        minimum: int,
        maximum: int,
        step: int,
        callback: Callable[[int], None],
    ):
        row = self._document.create_hstack("MultiviewStudioSettingRow")
        row.set_layout_spacing(4.0)
        caption = self._document.create_label(label)
        field = self._document.create_spin_box(float(value))
        field.widget.stable_id = f"multiview-studio.setting.{suffix}"
        field.set_range(float(minimum), float(maximum))
        field.step = float(step)
        field.decimals = 0
        self._connections.append(
            field.connect_changed(
                lambda changed: None
                if self._syncing
                else callback(int(round(changed)))
            )
        )
        row.add_flex_child(caption, 1.0)
        row.add_fixed_child(field.widget, 112.0)
        parent.add_preferred_child(row)
        return field

    def _float_spin(
        self,
        parent,
        label: str,
        suffix: str,
        value: float,
        minimum: float,
        maximum: float,
        step: float,
        callback: Callable[[float], None],
    ):
        row = self._document.create_hstack("MultiviewStudioFloatSpinRow")
        row.set_layout_spacing(4.0)
        caption = self._document.create_label(label)
        field = self._document.create_spin_box(float(value))
        field.widget.stable_id = f"multiview-studio.setting.{suffix}"
        field.set_range(float(minimum), float(maximum))
        field.step = float(step)
        field.decimals = 4
        self._connections.append(
            field.connect_changed(
                lambda changed: None
                if self._syncing
                else callback(float(changed))
            )
        )
        row.add_flex_child(caption, 1.0)
        row.add_fixed_child(field.widget, 112.0)
        parent.add_preferred_child(row)
        return field

    def _checkbox(
        self,
        parent,
        label: str,
        suffix: str,
        callback: Callable[[bool], None],
    ):
        row = self._document.create_hstack("MultiviewStudioCheckboxRow")
        row.set_layout_spacing(4.0)
        checkbox = self._document.create_checkbox(True)
        checkbox.widget.stable_id = f"multiview-studio.setting.{suffix}"
        caption = self._document.create_label(label)
        self._connections.append(
            checkbox.connect_changed(
                lambda checked: None
                if self._syncing
                else callback(bool(checked))
            )
        )
        row.add_preferred_child(checkbox.widget)
        row.add_flex_child(caption, 1.0)
        parent.add_preferred_child(row)
        return checkbox

    def _toolbar_button(self, label: str, callback: Callable[[], None]):
        button = self._document.create_button(label)
        button.widget.stable_id = (
            "multiview-studio.action."
            + label.lower().replace(" ", "-").replace(".", "")
        )
        self._connections.append(button.connect_clicked(callback))
        self.toolbar.add_preferred_child(button.widget)
        return button

    def _set_preview(
        self,
        preview_id: str,
        widget,
        image_path: str,
        *,
        max_size: tuple[int, int] = (256, 192),
    ) -> None:
        lease = self._load_preview_lease(preview_id, image_path, max_size)
        if lease is None or lease.empty:
            widget.clear_texture()
            return
        widget.set_texture(lease.texture, self._preview_sizes[preview_id])

    def _load_preview_lease(
        self,
        preview_id: str,
        image_path: str,
        max_size: tuple[int, int],
    ):
        source_path = Path(image_path) if image_path else None
        if source_path is not None and source_path.is_file():
            stat = source_path.stat()
            signature = (str(source_path), stat.st_mtime_ns, stat.st_size)
        else:
            signature = (image_path, 0, 0)
        if self._preview_signatures.get(preview_id) == signature:
            return self._leases.get(preview_id)
        self._preview_signatures[preview_id] = signature
        lease = self._leases.get(preview_id)
        if source_path is None or not source_path.is_file():
            if lease is not None and not lease.empty:
                lease.clear()
            self._preview_sizes.pop(preview_id, None)
            return None
        with Image.open(source_path) as source:
            preview = source.convert("RGBA")
            preview.thumbnail(max_size)
            # PIL commonly exposes a C-contiguous but read-only array.  In
            # that case ascontiguousarray returns the same read-only storage,
            # while the native upload binding requires writable CPU memory.
            pixels = np.array(
                preview,
                dtype=np.uint8,
                order="C",
                copy=True,
            )
        if lease is None:
            lease = self._texture_lease_factory()
            self._leases[preview_id] = lease
        lease.set_rgba8(pixels, TextureEncoding.SRGB)
        self._preview_sizes[preview_id] = Size(preview.width, preview.height)
        return lease

    @staticmethod
    def _path_label(path: str) -> str:
        if not path:
            return "Empty"
        resolved = Path(path)
        return resolved.name if resolved.is_file() else f"Missing: {resolved.name}"
