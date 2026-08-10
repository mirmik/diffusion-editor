from __future__ import annotations

import pytest

from termin.gui_native import (
    ModifierFlag,
    Rect,
    Size,
    tc_ui_document_create,
    tc_ui_document_destroy,
)

from diffusion_editor.app.native_shell import MENU_COMMANDS, NativeEditorView
from diffusion_editor.generation.types import (
    RECONSTRUCTION_STAGES,
    ReconstructionParameters,
    ReconstructionStage,
    ReconstructionStageStatus,
)


EXPECTED_COMMANDS = (
    "file.new",
    "file.new_from_image",
    "file.open",
    "file.save",
    "file.save_as",
    "file.import",
    "file.export",
    "app.quit",
    "edit.undo",
    "edit.redo",
    "edit.redo.ctrl_y",
    "edit.copy",
    "edit.copy_visible",
    "edit.paste",
    "edit.settings",
    "selection.all",
    "selection.background",
    "selection.clear",
    "selection.invert",
    "layer.new",
    "layer.new_3d_reconstruction",
    "layer.remove",
    "layer.flatten",
    "layer.detect",
    "generation.3d",
    "generation.3d_cancel",
    "view.3d_light_from_camera",
    "view.fit",
    "view.agent_panel",
)


def test_select_menu_contains_background_segmentation_command():
    selection_menu = next(
        command_ids
        for menu_id, _label, command_ids in MENU_COMMANDS
        if menu_id == "selection"
    )

    assert selection_menu == (
        "selection.all",
        "selection.background",
        "selection.clear",
        None,
        "selection.invert",
    )


def _snapshot_by_id(document):
    return {
        item["stable_id"]: item
        for item in document.inspect_snapshot()["widgets"]
        if item["stable_id"]
    }


def test_native_shell_snapshot_has_stable_layout_and_command_inventory():
    document = tc_ui_document_create()
    repaint_requests = []
    titles = []
    view = NativeEditorView(
        document,
        lambda: repaint_requests.append(True),
        titles.append,
        {},
    )
    try:
        document.layout_roots(Rect(0.0, 0.0, 1000.0, 700.0))
        snapshot = _snapshot_by_id(document)

        assert view.command_inventory == EXPECTED_COMMANDS
        assert {
            "diffusion-editor.root",
            "diffusion-editor.menu-bar",
            "diffusion-editor.toolbar",
            "diffusion-editor.toolbar-workspace-edge",
            "diffusion-editor.main-splitter",
            "diffusion-editor.left-panel",
            "diffusion-editor.workspace-splitter",
            "diffusion-editor.canvas-host",
            "diffusion-editor.right-host",
            "diffusion-editor.right-splitter",
            "diffusion-editor.layer-panel",
            "diffusion-editor.agent-panel",
            "diffusion-editor.status",
            "diffusion-editor.status-workspace-edge",
        } <= snapshot.keys()

        root = snapshot["diffusion-editor.root"]["bounds"]
        menu = snapshot["diffusion-editor.menu-bar"]["bounds"]
        toolbar = snapshot["diffusion-editor.toolbar"]["bounds"]
        toolbar_edge = snapshot[
            "diffusion-editor.toolbar-workspace-edge"]["bounds"]
        main = snapshot["diffusion-editor.main-splitter"]["bounds"]
        status_edge = snapshot[
            "diffusion-editor.status-workspace-edge"]["bounds"]
        status = snapshot["diffusion-editor.status"]["bounds"]
        left = snapshot["diffusion-editor.left-panel"]["bounds"]
        canvas = snapshot["diffusion-editor.canvas-host"]["bounds"]
        layers = snapshot["diffusion-editor.layer-panel"]["bounds"]
        agent = snapshot["diffusion-editor.agent-panel"]["bounds"]
        right_host = snapshot["diffusion-editor.right-host"]["bounds"]
        assert (root.width, root.height) == (1000.0, 700.0)
        assert (menu.y, menu.height) == (0.0, 28.0)
        assert (toolbar.y, toolbar.height) == (28.0, 34.0)
        assert (toolbar_edge.y, toolbar_edge.height) == (62.0, 2.0)
        assert (main.y, main.height) == (64.0, 612.0)
        assert (status_edge.y, status_edge.height) == (676.0, 2.0)
        assert (status.y, status.height) == (678.0, 22.0)
        toolbar_background = view.toolbar.widget.resolve_style().background
        status_background = view.status_bar.widget.resolve_style().background
        toolbar_edge_color = view.toolbar_workspace_edge.resolve_style().background
        status_edge_color = view.status_workspace_edge.resolve_style().background
        assert (
            toolbar_background.r,
            toolbar_background.g,
            toolbar_background.b,
        ) == pytest.approx((0.075, 0.080, 0.095))
        assert (
            status_background.r,
            status_background.g,
            status_background.b,
        ) == pytest.approx((0.075, 0.080, 0.095))
        assert (
            toolbar_edge_color.r,
            toolbar_edge_color.g,
            toolbar_edge_color.b,
        ) == pytest.approx((0.32, 0.35, 0.40))
        assert (
            status_edge_color.r,
            status_edge_color.g,
            status_edge_color.b,
        ) == pytest.approx((0.32, 0.35, 0.40))
        assert left.x + left.width <= canvas.x
        assert canvas.x + canvas.width <= layers.x
        assert (
            layers.x,
            layers.y,
            layers.width,
            layers.height,
        ) == (
            right_host.x,
            right_host.y,
            right_host.width,
            right_host.height,
        )
        assert not view.agent_panel_visible
        assert not view.right_splitter.widget.visible
        assert agent.width == 0.0
        assert right_host.width < canvas.width
        assert view.workspace_splitter.split_fraction == pytest.approx(0.72)

        assert view.activate_command("view.agent_panel")
        document.layout_roots(Rect(0.0, 0.0, 1000.0, 700.0))
        snapshot = _snapshot_by_id(document)
        layers = snapshot["diffusion-editor.layer-panel"]["bounds"]
        agent = snapshot["diffusion-editor.agent-panel"]["bounds"]
        assert view.agent_panel_visible
        assert view.right_splitter.widget.visible
        assert view.workspace_splitter.split_fraction == pytest.approx(0.46)
        assert layers.x + layers.width <= agent.x
        assert (left.y, canvas.y, layers.y, agent.y) == (
            main.y, main.y, main.y, main.y)
        assert (left.height, canvas.height, layers.height, agent.height) == (
            main.height, main.height, main.height, main.height)

        view_model = view.menu_models["view"]
        agent_command = next(
            command
            for command in view_model.commands
            if command.data.stable_id == "view.agent_panel"
        )
        assert agent_command.data.checked

        assert view.activate_command("view.agent_panel")
        document.layout_roots(Rect(0.0, 0.0, 1000.0, 700.0))
        snapshot = _snapshot_by_id(document)
        assert not view.agent_panel_visible
        assert not view.right_splitter.widget.visible
        assert view.workspace_splitter.split_fraction == pytest.approx(0.72)
        layers = snapshot["diffusion-editor.layer-panel"]["bounds"]
        right_host = snapshot["diffusion-editor.right-host"]["bounds"]
        assert (
            layers.x,
            layers.y,
            layers.width,
            layers.height,
        ) == (
            right_host.x,
            right_host.y,
            right_host.width,
            right_host.height,
        )
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_native_left_panel_scrolls_preferred_sections_without_overlap():
    document = tc_ui_document_create()
    view = NativeEditorView(
        document,
        lambda: None,
        lambda _title: None,
        {},
    )

    class MountedView:
        def __init__(self, widget):
            self.widget = widget

    controls = document.create_vstack("TestCanvasControls")
    controls.stable_id = "test.canvas-controls"
    controls.preferred_size = Size(240.0, 600.0)
    generation = document.create_vstack("TestGenerationPanels")
    generation.stable_id = "test.generation-panels"
    generation.preferred_size = Size(240.0, 400.0)
    try:
        view.mount_canvas_controls(MountedView(controls))
        view.mount_generation_panels(MountedView(generation), object())
        document.layout_roots(Rect(0.0, 0.0, 1280.0, 800.0))
        snapshot = _snapshot_by_id(document)
        controls_bounds = snapshot["test.canvas-controls"]["bounds"]
        generation_bounds = snapshot["test.generation-panels"]["bounds"]

        assert controls_bounds.y + controls_bounds.height <= generation_bounds.y
        assert view.left_scroll.content_size.height >= 1004.0
        assert snapshot["diffusion-editor.left-panel"]["bounds"].height == 712.0
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_reconstruction_context_reparents_canvas_and_owns_3d_toolbar():
    document = tc_ui_document_create()
    activated = []
    view = NativeEditorView(
        document,
        lambda: None,
        lambda _title: None,
        {
            "generation.3d": lambda: activated.append("generate"),
        },
    )

    class MountedView:
        def __init__(self, widget):
            self.widget = widget

    canvas = document.create_vstack("TestCanvas")
    canvas.stable_id = "test.canvas"
    viewport = document.create_vstack("TestReconstructionViewport")
    viewport.stable_id = "test.reconstruction-viewport"
    try:
        view.mount_canvas(MountedView(canvas))
        view.mount_reconstruction_viewport(MountedView(viewport))
        global_ids = {
            command.data.stable_id
            for command in view.toolbar_model.commands
        }
        context_ids = {
            command.data.stable_id
            for command in view.reconstruction_toolbar_model.commands
        }
        assert "generation.3d" not in global_ids
        assert "view.3d_light_from_camera" not in global_ids
        assert {
            "generation.3d",
            "generation.3d_cancel",
            "view.3d_light_from_camera",
        } <= context_ids
        selected_stages = []
        view.set_reconstruction_stage_handler(selected_stages.append)
        view._activate_reconstruction_stage(ReconstructionStage.HR_COORDINATES)
        assert selected_stages == [ReconstructionStage.HR_COORDINATES]
        changed_parameters = []
        view.set_reconstruction_parameter_handler(
            lambda key, value: changed_parameters.append((key, value))
        )
        view._change_reconstruction_parameter("steps", 18.0)
        assert changed_parameters == [("steps", 18)]
        parameters = ReconstructionParameters(
            seed=123,
            steps=18,
            resolution=1280,
            lr_conditioning_resolution=1024,
            manual_fov_degrees=45.0,
            decimation_target=350_000,
            texture_size=4096,
            low_vram=False,
        )
        view.update_reconstruction_parameters(parameters, busy=True)
        assert view.reconstruction_parameter_controls["seed"].value == 123.0
        assert view.reconstruction_parameter_controls[
            "resolution"
        ].selected_index == 1
        assert view.reconstruction_parameter_controls[
            "lr_conditioning_resolution"
        ].selected_index == 1
        assert all(
            not control.widget.enabled
            for control in view.reconstruction_parameter_controls.values()
        )
        statuses = {
            stage: ReconstructionStageStatus.PENDING
            for stage in RECONSTRUCTION_STAGES
        }
        statuses[ReconstructionStage.SPARSE_OCCUPANCY] = (
            ReconstructionStageStatus.READY
        )
        view.update_reconstruction_stages(
            statuses,
            {ReconstructionStage.SPARSE_OCCUPANCY: (12, 12)},
            ReconstructionStage.HR_COORDINATES,
            ReconstructionStage.SPARSE_OCCUPANCY,
        )
        assert view.reconstruction_stage_checks[
            ReconstructionStage.SPARSE_OCCUPANCY
        ].checked is True
        document.layout_roots(Rect(0.0, 0.0, 1000.0, 700.0))
        snapshot = _snapshot_by_id(document)
        host = snapshot["diffusion-editor.canvas-host"]["bounds"]
        left_panel = snapshot["diffusion-editor.left-panel"]["bounds"]
        canvas_bounds = snapshot["test.canvas"]["bounds"]
        assert (canvas_bounds.x, canvas_bounds.width) == (host.x, host.width)
        assert snapshot[
            "diffusion-editor.reconstruction.panel"
        ]["parent"] is None
        assert view.reconstruction_mode is False

        view.set_reconstruction_context(True, "generating")
        document.layout_roots(Rect(0.0, 0.0, 1000.0, 700.0))
        snapshot = _snapshot_by_id(document)
        canvas_bounds = snapshot["test.canvas"]["bounds"]
        viewport_bounds = snapshot["test.reconstruction-viewport"]["bounds"]
        toolbar = snapshot[
            "diffusion-editor.reconstruction.toolbar"
        ]["bounds"]
        reconstruction_panel = snapshot[
            "diffusion-editor.reconstruction.panel"
        ]["bounds"]
        assert snapshot[
            "diffusion-editor.reconstruction.panel"
        ]["parent"] is not None
        assert snapshot["diffusion-editor.left-panel"]["parent"] is None
        assert canvas_bounds.width < host.width
        assert viewport_bounds.width == pytest.approx(canvas_bounds.width)
        assert (canvas_bounds.y, canvas_bounds.height) == (
            host.y,
            host.height,
        )
        assert (
            reconstruction_panel.x,
            reconstruction_panel.y,
            reconstruction_panel.width,
            reconstruction_panel.height,
        ) == (
            left_panel.x,
            left_panel.y,
            left_panel.width,
            left_panel.height,
        )
        assert toolbar.height >= 3 * 20.0
        for command_id in (
                "generation-3d",
                "generation-3d_cancel",
                "view-3d_light_from_camera"):
            button = snapshot[
                f"diffusion-editor.reconstruction.{command_id}"
            ]["bounds"]
            assert button.x >= reconstruction_panel.x
            assert button.x + button.width <= (
                reconstruction_panel.x + reconstruction_panel.width
            )
        assert len(view.reconstruction_stage_buttons) == len(
            RECONSTRUCTION_STAGES
        )
        assert len(view.reconstruction_stage_checks) == len(
            RECONSTRUCTION_STAGES
        )
        for stage in RECONSTRUCTION_STAGES:
            assert snapshot[
                f"diffusion-editor.reconstruction.stage.{stage.value}"
            ]["parent"] is not None
        for key in (
            "seed",
            "steps",
            "resolution",
            "lr_conditioning_resolution",
            "manual_fov_degrees",
            "decimation_target",
            "texture_size",
            "low_vram",
        ):
            assert snapshot[
                f"diffusion-editor.reconstruction.parameter.{key}"
            ]["parent"] is not None
            assert snapshot[
                f"diffusion-editor.reconstruction.stage-check.{stage.value}"
            ]["parent"] is not None
        assert view.reconstruction_stage_buttons[
            ReconstructionStage.LR_SHAPE_FLOW
        ].widget.enabled is False
        assert view.reconstruction_status.text == "Generating"
        assert view.activate_command("generation.3d")
        assert activated == ["generate"]

        view.set_reconstruction_context(False)
        document.layout_roots(Rect(0.0, 0.0, 1000.0, 700.0))
        snapshot = _snapshot_by_id(document)
        canvas_bounds = snapshot["test.canvas"]["bounds"]
        host = snapshot["diffusion-editor.canvas-host"]["bounds"]
        assert (canvas_bounds.x, canvas_bounds.width) == (host.x, host.width)
        assert snapshot["diffusion-editor.left-panel"]["parent"] is not None
        assert snapshot[
            "diffusion-editor.reconstruction.panel"
        ]["parent"] is None
        assert view.reconstruction_mode is False
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_native_shortcuts_and_command_state_use_app_owned_handlers():
    document = tc_ui_document_create()
    activated = []
    repaint_requests = []
    view = NativeEditorView(
        document,
        lambda: repaint_requests.append(True),
        lambda _title: None,
        {
            "file.save": lambda: activated.append("save"),
            "edit.redo": lambda: activated.append("redo"),
        },
    )
    try:
        ctrl = int(ModifierFlag.Ctrl)
        assert view.menu_bar.dispatch_shortcut(ord("s"), ctrl)
        assert activated == ["save"]

        view.set_command_state("file.save", enabled=False)
        assert not view.menu_bar.dispatch_shortcut(ord("s"), ctrl)
        assert activated == ["save"]

        view.set_command_state("file.save", enabled=True)
        assert view.menu_bar.dispatch_shortcut(ord("s"), ctrl)
        assert view.menu_bar.dispatch_shortcut(ord("y"), ctrl)
        assert activated == ["save", "save", "redo"]

        view.set_command_handler(
            "selection.all",
            lambda: activated.append("select-all"),
        )
        assert view.activate_command("selection.all")
        assert activated[-1] == "select-all"
        assert len(repaint_requests) == 2
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_native_status_and_window_title_are_presentation_state():
    document = tc_ui_document_create()
    repaint_requests = []
    titles = []
    view = NativeEditorView(
        document,
        lambda: repaint_requests.append(True),
        titles.append,
        {},
    )
    try:
        view.set_status("Saving")
        view.set_window_title("picture.png — Diffusion Editor")

        assert view.status_bar.text == "Saving"
        assert view.window_title == "picture.png — Diffusion Editor"
        assert titles == ["picture.png — Diffusion Editor"]
        assert repaint_requests == [True]
    finally:
        view.close()
        tc_ui_document_destroy(document)
