from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from tcbase import MouseButton
from termin.gui_native import (
    ModifierFlag,
    Point,
    PointerEvent,
    PointerEventType,
    Rect,
    Size,
    tc_ui_document_create,
    tc_ui_document_destroy,
)

from diffusion_editor.multiview_studio.model import (
    MultiviewProject,
    RefineCube,
    RefineShapeResult,
    RefineViewPatch,
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
import diffusion_editor.multiview_studio.native_app as native_app_module
from diffusion_editor.multiview_studio.native_view import (
    NativeMultiviewStudioView,
)


class _Actions:
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class _FakeLease:
    def __init__(self):
        self.texture = type("Texture", (), {"id": 1})()
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


def test_error_dialog_is_written_to_log_before_it_is_shown(monkeypatch):
    events = []

    class Box:
        def connect_finished(self, _callback):
            return object()

        def show(self, _rect):
            events.append("shown")

    class Document:
        def create_message_box(self, title, message, _kind):
            events.append(("created", title, message))
            return Box()

    application = object.__new__(NativeMultiviewStudioApplication)
    application.document = Document()
    application.composition = type(
        "Composition", (), {"request_repaint": lambda _self: None}
    )()
    application._message_boxes = []
    application._connections = []
    monkeypatch.setattr(
        native_app_module.log,
        "error",
        lambda message: events.append(("logged", message)),
    )

    application._show_error("Refine failed", "CUDA out of memory")

    assert events == [
        ("logged", "Refine failed: CUDA out of memory"),
        ("created", "Refine failed", "CUDA out of memory"),
        "shown",
    ]


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
        assert view.mesh_panel.widget.stable_id == (
            "multiview-studio.navigator.meshes"
        )
        assert view.mesh_grid.widget.stable_id == "multiview-studio.mesh-grid"
        assert view.mesh_model.item_count == 1
        assert view.mesh_model.item(0).text == "Main"
        assert view.mesh_model.item(0).icon == "cube"
        assert tuple(view.model_shading_buttons) == (
            "flat",
            "smooth",
            "wireframe",
        )
        assert not view.show_mask_button.widget.enabled
        assert not view.paint_mask_button.widget.enabled
        assert not view.stop_refine_button.widget.enabled
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
        assert all(not row.visible for row in view.refine_cube_value_rows)
        view.set_refine_cube_visible(True)
        assert all(row.visible for row in view.refine_cube_value_rows)
        assert view.refine_cube_controls["x"].value == 1.0
        assert view.refine_cube_controls["side"].value == 0.5
        view.set_refine_cube_visible(False)
        assert all(not row.visible for row in view.refine_cube_value_rows)

        confirmed = replace(with_cube.refine_cube, confirmed=True)
        with_regions = replace(
            with_cube,
            refine_cube=confirmed,
            refine_regions=(confirmed, confirmed),
        )
        view.apply_project(with_regions, None, True)
        assert view.mesh_model.item_count == 3
        assert [view.mesh_model.item(index).text for index in range(3)] == [
            "Main",
            "Region 1",
            "Region 2",
        ]

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


def test_selected_refine_region_switches_left_panel_and_marks_view_patch(
    tmp_path: Path,
):
    image = tmp_path / "right.png"
    key = ViewKey("eye", 90)
    region = RefineCube(
        center=(0.0, 0.0, 0.0),
        side=1.0,
        geometry_fingerprint="sha256:test",
        confirmed=True,
    )
    project = replace(
        MultiviewProject().with_slot(key, image_path=str(image)),
        refine_cube=region,
        refine_regions=(region,),
        refine_view_patches=((RefineViewPatch(
            key, (0.2, 0.1, 0.8, 0.9)
        ),),),
    )
    document = tc_ui_document_create()
    view = NativeMultiviewStudioView(
        document,
        _Actions(),
        request_repaint=lambda: None,
        texture_lease_factory=_FakeLease,
    )
    try:
        view.apply_project(project, None, True)
        view.set_selected_mesh(1)
        view._selected_key = key
        view._apply_selected_view()

        assert view.refine_settings_widget.visible
        assert all(not widget.visible for widget in view.main_settings_widgets)
        assert "1 view patches" in view.refine_region_summary.text
        assert view.slot_model.item(
            view.slot_widgets[key]["index"]
        ).subtitle.endswith("patch 60×80%")
        assert view.refine_patch_controls["x"].value == 20.0
        assert view.refine_patch_controls["height"].value == 80.0

        view.set_busy(True)
        assert view.stop_refine_button.widget.enabled
        assert not view.refine_region_button.widget.enabled
        view.set_busy(False)
        assert not view.stop_refine_button.widget.enabled

        view.set_selected_mesh(0)
        assert not view.refine_settings_widget.visible
        assert all(widget.visible for widget in view.main_settings_widgets)
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_refine_patch_buttons_target_selected_region_and_view(tmp_path: Path):
    class Actions(_Actions):
        def __init__(self):
            self.patches = []

        def set_refine_view_patch(self, region_index, key, bounds):
            self.patches.append((region_index, key, bounds))

    image = tmp_path / "front.png"
    key = ViewKey("eye", 0)
    region = RefineCube(
        center=(0.0, 0.0, 0.0),
        side=1.0,
        geometry_fingerprint="sha256:test",
        confirmed=True,
    )
    project = replace(
        MultiviewProject().with_source("front", str(image)),
        refine_cube=region,
        refine_regions=(region,),
    )
    actions = Actions()
    document = tc_ui_document_create()
    view = NativeMultiviewStudioView(
        document,
        actions,
        request_repaint=lambda: None,
        texture_lease_factory=_FakeLease,
    )
    try:
        view.apply_project(project, None, True)
        view.set_selected_mesh(1)
        view._use_full_refine_patch()
        view._remove_refine_patch()

        assert actions.patches == [
            (0, key, (0.0, 0.0, 1.0, 1.0)),
            (0, key, None),
        ]
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_dragging_selected_view_commits_normalized_refine_patch():
    class Actions(_Actions):
        def __init__(self):
            self.patches = []

        def set_refine_view_patch(self, region_index, key, bounds):
            self.patches.append((region_index, key, bounds))

    key = ViewKey("eye", 0)
    region = RefineCube(
        center=(0.0, 0.0, 0.0),
        side=1.0,
        geometry_fingerprint="sha256:test",
        confirmed=True,
    )
    project = replace(
        MultiviewProject().with_source("front", "/missing/front.png"),
        refine_cube=region,
        refine_regions=(region,),
    )
    actions = Actions()
    document = tc_ui_document_create()
    view = NativeMultiviewStudioView(
        document,
        actions,
        request_repaint=lambda: None,
        texture_lease_factory=_FakeLease,
    )
    try:
        view.apply_project(project, None, True)
        view.set_selected_mesh(1)
        document.layout_roots(Rect(0.0, 0.0, 1700.0, 950.0))
        view._preview_sizes["workspace:selected"] = Size(200.0, 100.0)
        down = PointerEvent()
        down.type = PointerEventType.Down
        down.button = MouseButton.LEFT
        view._on_patch_pointer(Point(20.0, 10.0), down)
        assert view._patch_drag is not None

        destination = view.selected_image.image_to_widget(Point(160.0, 80.0))
        up = PointerEvent()
        up.type = PointerEventType.Up
        up.button = MouseButton.LEFT
        up.x = destination.x
        up.y = destination.y
        view._on_captured_patch_pointer(Point(), up)

        assert actions.patches == [
            (0, key, (0.1, 0.1, 0.8, 0.8)),
        ]
    finally:
        view.close()
        tc_ui_document_destroy(document)


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
        texture_lease_factory=_FakeLease,
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


def test_mask_buttons_keep_state_when_shading_changes():
    document = tc_ui_document_create()
    view = NativeMultiviewStudioView(
        document,
        _Actions(),
        request_repaint=lambda: None,
        texture_lease_factory=lambda: None,
    )
    try:
        view.set_model_mask_state(
            available=True, visible=True, painting=False
        )
        view._set_model_shading("wireframe")

        assert view.show_mask_button.widget.enabled
        assert view.paint_mask_button.widget.enabled
        assert view._model_mask_visible
        assert not view._model_mask_painting
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_mask_visibility_and_painting_are_independent_from_model_shading():
    class View:
        states = []
        modes = []
        statuses = []

        def set_model_mask_state(self, **state):
            self.states.append(state)

        def _set_model_shading(self, mode):
            self.modes.append(mode)

        def set_status(self, text):
            self.statuses.append(text)

    class Viewport:
        visibility = []
        painting = []

        def set_refine_mask_visible(self, visible):
            self.visibility.append(visible)

        def set_refine_mask_edit_enabled(self, enabled):
            self.painting.append(enabled)

    application = object.__new__(NativeMultiviewStudioApplication)
    application.view = View()
    application.reconstruction_viewport = Viewport()
    application._displayed_mesh_index = 1
    application._refine_mask_visible = False
    application._refine_mask_painting = False

    application.toggle_refine_mask_painting()
    application.set_model_shading("wireframe")
    application.toggle_refine_mask_painting()

    assert application.view.modes == ["wireframe"]
    assert application.reconstruction_viewport.visibility == [True, True]
    assert application.reconstruction_viewport.painting == [True, False]
    assert application.view.states[-1] == {
        "available": True,
        "visible": True,
        "painting": False,
    }


def test_mesh_grid_single_selection_and_activation_have_distinct_actions():
    class Actions(_Actions):
        def __init__(self):
            self.calls = []

        def select_refine_region(self, index):
            self.calls.append(("cube", index))

        def select_mesh(self, index):
            self.calls.append(("mesh", index))

    actions = Actions()
    document = tc_ui_document_create()
    view = NativeMultiviewStudioView(
        document,
        actions,
        request_repaint=lambda: None,
        texture_lease_factory=lambda: None,
    )
    region = RefineCube(
        center=(0.0, 0.0, 0.0),
        side=1.0,
        geometry_fingerprint="sha256:test",
        confirmed=True,
    )
    try:
        view.apply_project(
            replace(MultiviewProject(), refine_regions=(region,)),
            None,
            False,
        )

        view._on_mesh_selection_changed([1])
        view._on_mesh_activated(1, view.mesh_model.item(1))

        assert actions.calls == [("cube", 1), ("mesh", 1)]
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_refined_mesh_variant_exposes_standalone_texturing(tmp_path: Path):
    class Actions(_Actions):
        def __init__(self):
            self.calls = []

        def select_refine_region(self, index):
            self.calls.append(("cube", index))

        def select_refine_result(self, region_index, result_index):
            self.calls.append(("result", region_index, result_index))

        def texture_refine_result(self, region_index, result_index):
            self.calls.append(("texture", region_index, result_index))

    image = tmp_path / "right.png"
    refined = tmp_path / "refined.glb"
    Image.new("RGB", (8, 8), "white").save(image)
    refined.touch()

    region = RefineCube(
        center=(0.0, 0.0, 0.0),
        side=1.0,
        geometry_fingerprint="sha256:test",
        confirmed=True,
    )
    result = RefineShapeResult(
        geometry_fingerprint="sha256:test",
        request_key="request",
        source_region_path="source.glb",
        refined_region_path=str(refined),
        source_slat_path="source.npz",
        refined_slat_path="refined.npz",
        manifest_path="result.json",
    )
    project = MultiviewProject().with_slot(
        ViewKey("eye", 90), image_path=str(image)
    )
    project = replace(
        project,
        refine_regions=(region,),
        refine_view_patches=((
            RefineViewPatch(ViewKey("eye", 90), (0.1, 0.2, 0.8, 0.9)),
        ),),
        refine_shape_results=((result,),),
    )
    actions = Actions()
    document = tc_ui_document_create()
    view = NativeMultiviewStudioView(
        document,
        actions,
        request_repaint=lambda: None,
        texture_lease_factory=_FakeLease,
    )
    try:
        view.apply_project(project, None, False)

        assert view.mesh_model.item_count == 3
        assert view.mesh_model.item(2).text == "R1 · Refined 1"
        view._on_mesh_selection_changed([2])
        view._on_mesh_activated(2, view.mesh_model.item(2))
        view.set_selected_refine_result(0, 0)
        assert view.texture_refine_button.widget.enabled
        view._texture_refine_result()

        assert actions.calls == [
            ("cube", 1),
            ("result", 0, 0),
            ("texture", 0, 0),
        ]
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_mesh_icons_switch_the_single_model_viewport(tmp_path: Path):
    shape = tmp_path / "shape.glb"
    shape.write_bytes(b"mesh")
    region = RefineCube(
        center=(1.0, 2.0, 3.0),
        side=0.5,
        geometry_fingerprint="sha256:test",
        confirmed=True,
    )
    project = replace(
        MultiviewProject(),
        geometry_path=str(shape),
        shape_path=str(shape),
        refine_cube=region,
        refine_regions=(region,),
    )
    class View:
        selected = []
        cube_visible = []
        statuses = []

        def set_selected_mesh(self, index):
            self.selected.append(index)

        def set_status(self, text):
            self.statuses.append(text)

        def set_refine_cube_visible(self, visible):
            self.cube_visible.append(visible)

    class Viewport:
        calls = []

        def load_glb(self, path, *, fit_camera):
            self.calls.append(("main", path, fit_camera))

        def show_cube_mesh_subset(self, center, side, *, fit_camera):
            self.calls.append(("region", center, side, fit_camera))
            return 3, 1, 1

        def clear_refine_cube(self):
            self.calls.append(("clear-cube",))

        def set_refine_cube(self, center, side, confirmed):
            self.calls.append(("set-cube", center, side, confirmed))

        def cancel_refine_cube_pick(self):
            pass

    application = object.__new__(NativeMultiviewStudioApplication)
    application.controller = MultiviewStudioController(project)
    application.view = View()
    application.reconstruction_viewport = Viewport()
    application._loaded_shape_path = ""
    application._displayed_mesh_signature = None
    application._selected_mesh_index = 0
    application._displayed_mesh_index = 0
    application._refine_cube_before_edit = None
    application._refine_cube_visible = False
    application._refine_cube_visible_before_edit = False
    application._editing_refine_region_index = None
    application._refine_region_before_edit = None
    application._refine_cube_editing = False

    application.select_refine_region(1)
    application.select_mesh(1)
    application.select_mesh(0)

    assert application.view.selected == [1, 1, 0]
    assert application.view.cube_visible == [True, False, False]
    assert [call[0] for call in application.reconstruction_viewport.calls] == [
        "main",
        "set-cube",
        "region",
        "clear-cube",
        "main",
        "clear-cube",
    ]


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


def test_first_save_preserves_nested_refined_region_texture_run(tmp_path: Path):
    application = object.__new__(NativeMultiviewStudioApplication)
    application._unsaved_workspace = tempfile.TemporaryDirectory(
        prefix="multiview-studio-region-texture-test-"
    )
    application.controller = MultiviewStudioController()
    try:
        session = Path(application._unsaved_workspace.name)
        geometry = session / "shape-runs" / "shape-test" / "geometry.glb"
        geometry.parent.mkdir(parents=True)
        geometry.write_bytes(b"geometry")
        application.controller.set_geometry_path(geometry)
        application.controller.set_refine_cube((0.0, 0.0, 0.0), 1.0)
        application.controller.confirm_refine_cube()
        fingerprint = application.controller.project.refine_regions[
            0
        ].geometry_fingerprint
        run = session / "refine-runs" / "region-1-request"
        texture_run = run / "texture-runs" / "texture-request"
        texture_run.mkdir(parents=True)
        paths = {
            "source": run / "source-region.glb",
            "refined": run / "refined-region.glb",
            "source_slat": run / "source-shape-slat.npz",
            "refined_slat": run / "refined-shape-slat.npz",
            "manifest": run / "result.json",
            "textured": texture_run / "textured.glb",
            "texture_shape": texture_run / "encoded-shape-slat.npz",
            "texture_slat": texture_run / "texture-slat.npz",
            "texture_manifest": texture_run / "result.json",
        }
        for path in paths.values():
            path.touch()
        application.controller.add_refine_shape_result(0, RefineShapeResult(
            geometry_fingerprint=fingerprint,
            request_key="request",
            source_region_path=str(paths["source"]),
            refined_region_path=str(paths["refined"]),
            source_slat_path=str(paths["source_slat"]),
            refined_slat_path=str(paths["refined_slat"]),
            manifest_path=str(paths["manifest"]),
            textured_region_path=str(paths["textured"]),
            texture_key="texture-request",
            texture_shape_slat_path=str(paths["texture_shape"]),
            texture_slat_path=str(paths["texture_slat"]),
            texture_manifest_path=str(paths["texture_manifest"]),
        ))

        application._adopt_unsaved_artifacts(
            tmp_path / "saved" / "project.mvstudio.json"
        )

        result = application.controller.project.region_refine_results(0)[0]
        expected_run = tmp_path / "saved" / "refine-runs" / run.name
        assert Path(result.refined_region_path) == expected_run / "refined-region.glb"
        assert Path(result.textured_region_path) == (
            expected_run
            / "texture-runs"
            / "texture-request"
            / "textured.glb"
        )
        assert Path(result.texture_manifest_path).is_file()
    finally:
        application._unsaved_workspace.cleanup()
