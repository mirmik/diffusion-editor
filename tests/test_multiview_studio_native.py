from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from termin.gui_native import (
    ModifierFlag,
    Rect,
    tc_ui_document_create,
    tc_ui_document_destroy,
)

from diffusion_editor.multiview_studio.model import (
    MultiviewProject,
    RefineCube,
    ViewKey,
)
from diffusion_editor.multiview_studio.controller import (
    MultiviewStudioController,
)
from diffusion_editor.multiview_studio.native_app import (
    NativeMultiviewStudioApplication,
    _gltf_point_to_viewport,
    _viewport_point_to_gltf,
)
from diffusion_editor.multiview_studio.native_view import (
    NativeMultiviewStudioView,
)


class _Actions:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class _FakeLease:
    def __init__(self):
        self.texture = object()
        self.empty = True
        self.upload = None

    def set_rgba8(self, pixels, encoding):
        assert pixels.dtype == np.uint8
        assert pixels.flags.c_contiguous
        assert pixels.flags.writeable
        self.upload = (pixels.copy(), encoding)
        self.empty = False

    def clear(self):
        self.empty = True

    def close(self):
        self.empty = True


class _FakeImageWidget:
    def __init__(self):
        self.upload = None

    def set_texture(self, texture, size):
        self.upload = (texture, size)

    def clear_texture(self):
        self.upload = None


def test_native_multiview_view_builds_compact_navigator_and_workspace():
    document = tc_ui_document_create()
    view = NativeMultiviewStudioView(
        document,
        _Actions(),
        request_repaint=lambda: None,
        texture_lease_factory=lambda: None,
    )
    try:
        assert view.root.stable_id == "multiview-studio.root"
        assert view.menu_bar.widget.stable_id == "multiview-studio.menu-bar"
        assert tuple(view.menu_models) == ("file", "generate", "view")
        file_commands = tuple(
            command.data.stable_id for command in view.menu_models["file"].commands
        )
        assert file_commands == (
            "file.new",
            "file.open",
            "file.open_recent",
            "file.separator.save",
            "file.save",
            "file.save_as",
            "file.separator.quit",
            "app.quit",
        )
        assert len(view.slot_widgets) == 24
        project = MultiviewProject()

        view.apply_project(project, None, True)
        document.layout_roots(Rect(0.0, 0.0, 1700.0, 950.0))

        empty_index = view.slot_widgets[ViewKey("low", 45)]["index"]
        assert view.slot_model.item(empty_index).subtitle.endswith("empty")
        assert view.slot_grid.column_count == 8
        assert view.slot_grid.row_count == 3
        assert view.slot_model.item(empty_index).icon == ""
        assert view.slot_grid.widget.stable_id == "multiview-studio.view-grid"
        assert view.workspace_splitter.widget.stable_id == (
            "multiview-studio.workspace.splitter"
        )
        assert tuple(view.model_shading_buttons) == (
            "flat",
            "smooth",
            "wireframe",
        )
        assert tuple(view.refine_cube_controls) == ("x", "y", "z", "side")
        assert all(not row.visible for row in view.refine_cube_value_rows)
        assert not view.pick_refine_cube_button.widget.enabled
        assert not view.confirm_refine_cube_button.widget.enabled
        assert view.refine_cube_status.text == "No refine region"
        assert view.setting_controls[
            "postprocess.fill_holes"
        ].checked
        assert view.setting_controls[
            "postprocess.remesh"
        ].checked
        assert not view.setting_controls["postprocess.final_repair"].checked
        assert view.setting_controls[
            "postprocess.remove_degenerate_faces"
        ].checked
        assert abs(
            view.setting_controls["postprocess.fill_hole_perimeter"].value
            - 0.03
        ) < 1e-6
        assert view.slot_model.item(
            view.slot_widgets[ViewKey("eye", 0)]["index"]
        ).text == "Front"
        assert view.build_shape_button.widget.stable_id == (
            "multiview-studio.shape-output.build"
        )
        assert view.texture_model_button.widget.stable_id == (
            "multiview-studio.shape-output.texture"
        )
        assert view.setting_controls["texture.texture_size"].value == 2048
        assert view.setting_controls["texture.texture_size"].step == 512.0
        assert not view.texture_model_button.widget.enabled
        assert not hasattr(view, "open_shape_button")
        assert not view.build_shape_button.widget.enabled
        assert "0/24 populated" in view.status.text

        with_cube = replace(project, refine_cube=RefineCube(
            center=(1.0, 2.0, 3.0),
            side=0.5,
            geometry_fingerprint="sha256:test",
        ))
        view.apply_project(with_cube, None, True)
        assert all(row.visible for row in view.refine_cube_value_rows)
        assert view.refine_cube_controls["x"].value == 1.0
        assert view.refine_cube_controls["side"].value == 0.5

        unsaved_with_sources = MultiviewProject().with_source(
            "front", "/tmp/front.png"
        ).with_source("back", "/tmp/back.png")
        view.apply_project(unsaved_with_sources, None, True)
        assert view.generate_four_button.widget.enabled
        assert view.generate_missing_button.widget.enabled
        assert view.generate_all_button.widget.enabled
        assert view.build_shape_button.widget.enabled
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_refine_cube_coordinate_conversion_matches_native_glb_preview():
    gltf = (2.0, 3.0, 5.0)

    viewport = _gltf_point_to_viewport(gltf)

    assert viewport == (2.0, -5.0, 3.0)
    assert _viewport_point_to_gltf(viewport) == gltf


def test_recent_project_submenu_routes_open_and_clear(tmp_path: Path):
    class Actions(_Actions):
        def __init__(self):
            self.opened = []
            self.cleared = 0

        def open_recent_project(self, path):
            self.opened.append(path)

        def clear_recent_projects(self):
            self.cleared += 1

    first = tmp_path / "first.mvstudio.json"
    second = tmp_path / "second.mvstudio.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    actions = Actions()
    document = tc_ui_document_create()
    view = NativeMultiviewStudioView(
        document,
        actions,
        request_repaint=lambda: None,
        texture_lease_factory=lambda: None,
    )
    try:
        view.set_recent_projects((first.resolve(), second.resolve()))
        commands = view.recent_model.commands
        assert commands[0].data.label.startswith("1. first.mvstudio.json")
        assert commands[1].data.label.startswith("2. second.mvstudio.json")

        view._on_menu_activated(0, commands[1].id, commands[1].data)
        view._on_menu_activated(0, commands[-1].id, commands[-1].data)

        assert actions.opened == [str(second.resolve())]
        assert actions.cleared == 1
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_file_menu_shortcuts_route_to_studio_actions():
    class Actions(_Actions):
        def __init__(self):
            self.calls = []

        def new_project(self):
            self.calls.append("new")

        def open_project(self):
            self.calls.append("open")

        def save_project(self):
            self.calls.append("save")

        def save_project_as(self):
            self.calls.append("save-as")

        def quit(self):
            self.calls.append("quit")

    actions = Actions()
    document = tc_ui_document_create()
    view = NativeMultiviewStudioView(
        document,
        actions,
        request_repaint=lambda: None,
        texture_lease_factory=lambda: None,
    )
    try:
        ctrl = int(ModifierFlag.Ctrl)
        ctrl_shift = int(ModifierFlag.Ctrl | ModifierFlag.Shift)
        assert view.dispatch_shortcut(ord("n"), ctrl)
        assert view.dispatch_shortcut(ord("o"), ctrl)
        assert view.dispatch_shortcut(ord("s"), ctrl)
        assert view.dispatch_shortcut(ord("s"), ctrl_shift)
        assert view.dispatch_shortcut(ord("q"), ctrl)
        assert actions.calls == ["new", "open", "save", "save-as", "quit"]
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_model_shading_buttons_drive_mounted_viewport():
    document = tc_ui_document_create()
    view = NativeMultiviewStudioView(
        document,
        _Actions(),
        request_repaint=lambda: None,
        texture_lease_factory=lambda: None,
    )
    modes = []

    class Viewport:
        widget = document.create_panel("test-model-viewport").widget

        @staticmethod
        def set_shading_mode(mode):
            modes.append(mode)

    try:
        view.mount_model_viewport(Viewport())
        view._set_model_shading("smooth")
        view._set_model_shading("wireframe")

        assert modes == ["flat", "smooth", "wireframe"]
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_preview_upload_copies_pil_pixels_to_writable_c_array(tmp_path: Path):
    image_path = tmp_path / "preview.png"
    Image.new("RGB", (31, 47), "blue").save(image_path)
    document = tc_ui_document_create()
    leases = []

    def lease_factory():
        lease = _FakeLease()
        leases.append(lease)
        return lease

    view = NativeMultiviewStudioView(
        document,
        _Actions(),
        request_repaint=lambda: None,
        texture_lease_factory=lease_factory,
    )
    widget = _FakeImageWidget()
    try:
        view._set_preview("test", widget, str(image_path))

        assert len(leases) == 1
        assert leases[0].upload is not None
        assert widget.upload is not None
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_first_save_adopts_generated_views_and_shape_from_unsaved_workspace(
    tmp_path: Path,
):
    application = object.__new__(NativeMultiviewStudioApplication)
    application._unsaved_workspace = tempfile.TemporaryDirectory(
        prefix="multiview-studio-test-"
    )
    application.controller = MultiviewStudioController()
    try:
        source = (
            Path(application._unsaved_workspace.name)
            / "views"
            / "mv-low-045.png"
        )
        source.parent.mkdir(parents=True)
        source.write_bytes(b"generated")
        key = ViewKey("low", 45)
        application.controller.set_slot_image(key, source)
        shape = (
            Path(application._unsaved_workspace.name)
            / "shape-runs"
            / "shape-test"
            / "shape.glb"
        )
        shape.parent.mkdir(parents=True)
        shape.write_bytes(b"shape")
        application.controller.set_shape_path(shape)

        application._adopt_unsaved_artifacts(
            tmp_path / "saved" / "project.mvstudio.json"
        )

        adopted = Path(application.controller.project.slot(key).image_path)
        assert adopted == tmp_path / "saved" / "views" / source.name
        assert adopted.read_bytes() == b"generated"
        adopted_shape = Path(application.controller.project.shape_path)
        assert Path(application.controller.project.geometry_path) == adopted_shape
        assert adopted_shape == (
            tmp_path / "saved" / "shape-runs" / "shape-test" / "shape.glb"
        )
        assert adopted_shape.read_bytes() == b"shape"
    finally:
        application._unsaved_workspace.cleanup()


def test_first_save_adopts_geometry_and_texture_runs_separately(tmp_path: Path):
    application = object.__new__(NativeMultiviewStudioApplication)
    application._unsaved_workspace = tempfile.TemporaryDirectory(
        prefix="multiview-studio-texture-test-"
    )
    application.controller = MultiviewStudioController()
    try:
        session = Path(application._unsaved_workspace.name)
        geometry = session / "shape-runs" / "shape-test" / "geometry.glb"
        textured = session / "texture-runs" / "texture-test" / "textured.glb"
        geometry.parent.mkdir(parents=True)
        textured.parent.mkdir(parents=True)
        geometry.write_bytes(b"geometry")
        textured.write_bytes(b"texture")
        application.controller.set_geometry_path(geometry)
        application.controller.set_textured_shape_path(textured)

        application._adopt_unsaved_artifacts(
            tmp_path / "saved" / "project.mvstudio.json"
        )

        project = application.controller.project
        assert Path(project.geometry_path) == (
            tmp_path / "saved" / "shape-runs" / "shape-test" / "geometry.glb"
        )
        assert Path(project.shape_path) == (
            tmp_path
            / "saved"
            / "texture-runs"
            / "texture-test"
            / "textured.glb"
        )
    finally:
        application._unsaved_workspace.cleanup()
