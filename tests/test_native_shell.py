from __future__ import annotations

import pytest

from termin.gui_native import (
    KeyCode,
    ModifierFlag,
    Rect,
    Size,
    tc_ui_document_create,
    tc_ui_document_destroy,
)

from diffusion_editor.app.native_shell import MENU_COMMANDS, NativeEditorView
from diffusion_editor.generation.reconstruction_workspace import (
    build_legacy_workspace,
)
from diffusion_editor.generation.types import (
    RECONSTRUCTION_BACKEND_PARAMETER_KEYS,
    RECONSTRUCTION_STAGES,
    ReconstructionParameters,
    ReconstructionBackend,
    ReconstructionRefineParameters,
    ReconstructionRun,
    ReconstructionRunKind,
    ReconstructionStage,
    ReconstructionStageArtifact,
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
    "edit.clear_selected_pixels",
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
            self.shading_modes = []

        def set_shading_mode(self, mode):
            self.shading_modes.append(mode)

    canvas = document.create_vstack("TestCanvas")
    canvas.stable_id = "test.canvas"
    viewport = document.create_vstack("TestReconstructionViewport")
    viewport.stable_id = "test.reconstruction-viewport"
    mounted_viewport = MountedView(viewport)
    try:
        view.mount_canvas(MountedView(canvas))
        view.mount_reconstruction_viewport(mounted_viewport)
        assert view.reconstruction_workspace_mode is False
        assert view.reconstruction_workspace_open.widget.enabled is True
        assert "lr.refine" in view.reconstruction_workspace_operation_buttons
        assert (
            "local.upscale"
            in view.reconstruction_workspace_operation_buttons
        )
        assert (
            "local.generate_geometry"
            in view.reconstruction_workspace_operation_buttons
        )
        view._select_reconstruction_workspace_operation("local.upscale")
        assert view.reconstruction_workspace_operation_rows[
            "local.upscale"
        ].visible is True
        assert view.reconstruction_workspace_inspector_title.text == (
            "Upscale local conditioning"
        )
        assert "SR/upscaled condition" in (
            view.reconstruction_workspace_inspector_outputs.text
        )
        workspace_statuses = {
            stage: ReconstructionStageStatus.PENDING
            for stage in RECONSTRUCTION_STAGES
        }
        workspace_statuses[ReconstructionStage.SOURCE_IMAGE] = (
            ReconstructionStageStatus.READY
        )
        workspace_statuses[ReconstructionStage.SPARSE_OCCUPANCY] = (
            ReconstructionStageStatus.READY
        )
        workspace = build_legacy_workspace(
            ReconstructionParameters(),
            workspace_statuses,
            {
                ReconstructionStage.SOURCE_IMAGE:
                ReconstructionStageArtifact(
                    ReconstructionStage.SOURCE_IMAGE,
                    "/tmp/source.png",
                    "image",
                ),
                ReconstructionStage.SPARSE_OCCUPANCY:
                ReconstructionStageArtifact(
                    ReconstructionStage.SPARSE_OCCUPANCY,
                    "/tmp/sparse.glb",
                    "mesh",
                ),
            },
            (),
            None,
        )
        workspace_parameters = ReconstructionParameters(
            pixal3d_sparse_seed=101,
            pixal3d_sparse_steps=9,
        )
        view.update_reconstruction_workspace(
            ReconstructionBackend.PIXAL3D, workspace, workspace_parameters
        )
        view._select_reconstruction_workspace_operation("sparse.generate")
        assert view.reconstruction_workspace_parameter_rows[
            "pixal3d_sparse_seed"
        ].visible is True
        assert view.reconstruction_workspace_parameter_rows[
            "pixal3d_lr_seed"
        ].visible is False
        assert view.reconstruction_workspace_parameter_controls[
            "pixal3d_sparse_seed"
        ].value == 101
        assert view.reconstruction_workspace_variant_combo.item_count == 1
        assert view.reconstruction_workspace_variant_combo.item_text(0) == (
            "Current generation · ready"
        )
        assert view.reconstruction_workspace_artifact_combo.item_count == 1
        assert "Sparse occupancy preview" in (
            view.reconstruction_workspace_artifact_combo.item_text(0)
        )
        assert view.reconstruction_workspace_actions[
            "preview"
        ].widget.enabled is True
        workspace_actions = []
        view.set_reconstruction_workspace_handler(
            lambda action, value: workspace_actions.append((action, value))
        )
        view._preview_reconstruction_workspace_artifact()
        assert workspace_actions == [(
            "preview_artifact",
            "legacy:current:sparse.generate:sparse_occupancy",
        )]
        assert view.reconstruction_workspace_actions[
            "generate"
        ].widget.enabled is True
        view._run_reconstruction_workspace_operation()
        assert workspace_actions[-1] == (
            "generate_to_operation", "sparse.generate"
        )
        view._change_reconstruction_workspace_parameter(
            "pixal3d_sparse_seed", 202
        )
        assert workspace_actions[-1] == (
            "set_operation_parameter", ("pixal3d_sparse_seed", 202)
        )
        view._select_reconstruction_workspace_operation("lr.refine")
        assert view.reconstruction_workspace_actions[
            "generate"
        ].widget.enabled is False
        assert view.reconstruction_workspace_mask_panel.visible is True
        assert "not connected yet" in view.reconstruction_workspace_status.text
        workspace_refine_actions = []
        view.set_reconstruction_refine_handler(
            lambda action, value:
            workspace_refine_actions.append((action, value))
        )
        view.reconstruction_workspace_mask_controls["paint"].checked = True
        assert workspace_refine_actions == [("paint", True)]
        view._select_reconstruction_workspace_operation("sparse.generate")
        assert view.reconstruction_workspace_mask_panel.visible is False
        view._select_reconstruction_workspace_operation("hr.refine")
        assert view.reconstruction_workspace_mask_panel.visible is True
        view._set_reconstruction_workspace_mode(True)
        assert view.reconstruction_workspace_mode is True
        view._set_reconstruction_workspace_mode(False)
        assert view.reconstruction_workspace_mode is False
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
        assert view.reconstruction_shading_combo.item_count == 5
        assert [
            view.reconstruction_shading_combo.item_text(index)
            for index in range(view.reconstruction_shading_combo.item_count)
        ] == ["Flat", "Smooth", "Unlit", "Normals", "Wireframe"]
        view.reconstruction_shading_combo.selected_index = 1
        assert mounted_viewport.shading_modes == ["smooth"]
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
        trellis_parameters = ReconstructionParameters(
            backend=ReconstructionBackend.TRELLIS2,
        )
        view.update_reconstruction_parameters(trellis_parameters, busy=False)
        parameter_controls = view.reconstruction_parameter_controls
        assert parameter_controls["backend"].selected_index == 1
        assert parameter_controls["backend"].widget.enabled is True
        assert parameter_controls[
            "lr_conditioning_resolution"
        ].widget.enabled is False
        assert parameter_controls["manual_fov_degrees"].widget.enabled is False
        spar3d_parameters = ReconstructionParameters(
            backend=ReconstructionBackend.SPAR3D,
            spar3d_guidance_scale=4.5,
        )
        view.update_reconstruction_parameters(spar3d_parameters, busy=False)
        assert parameter_controls["backend"].selected_index == 2
        assert parameter_controls["spar3d_guidance_scale"].value == 4.5
        assert parameter_controls["spar3d_guidance_scale"].widget.enabled is True
        assert parameter_controls["steps"].widget.enabled is False
        assert parameter_controls["resolution"].widget.enabled is False
        assert parameter_controls["decimation_target"].widget.enabled is False
        hunyuan_parameters = ReconstructionParameters(
            backend=ReconstructionBackend.HUNYUAN3D21,
            hunyuan3d21_guidance_scale=6.0,
            hunyuan3d21_octree_resolution=256,
            hunyuan3d21_texture_steps=16,
            hunyuan3d21_texture_guidance_scale=4.0,
        )
        view.update_reconstruction_parameters(hunyuan_parameters, busy=False)
        assert parameter_controls["backend"].selected_index == 4
        assert parameter_controls["hunyuan3d21_guidance_scale"].value == 6.0
        assert parameter_controls[
            "hunyuan3d21_octree_resolution"
        ].selected_index == 2
        assert parameter_controls["hunyuan3d21_texture_steps"].value == 16.0
        assert parameter_controls[
            "hunyuan3d21_texture_guidance_scale"
        ].value == 4.0
        assert parameter_controls["resolution"].widget.enabled is False
        assert parameter_controls["texture_size"].widget.enabled is True
        assert parameter_controls["low_vram"].widget.enabled is False
        sam_parameters = ReconstructionParameters(
            backend=ReconstructionBackend.SAM3D_OBJECTS,
            sam3d_sparse_steps=23,
            sam3d_slat_steps=19,
            sam3d_sparse_guidance_scale=6.5,
            sam3d_slat_guidance_scale=1.25,
            sam3d_simplify=0.9,
        )
        view.update_reconstruction_parameters(sam_parameters, busy=False)
        assert parameter_controls["backend"].selected_index == 5
        assert parameter_controls["sam3d_sparse_steps"].value == 23.0
        assert parameter_controls["sam3d_slat_steps"].value == 19.0
        assert parameter_controls["sam3d_sparse_guidance_scale"].value == 6.5
        assert parameter_controls["sam3d_slat_guidance_scale"].value == 1.25
        assert parameter_controls["sam3d_simplify"].value == pytest.approx(0.9)
        assert parameter_controls["sam3d_sparse_steps"].widget.enabled is True
        assert parameter_controls["steps"].widget.enabled is False
        assert parameter_controls["resolution"].widget.enabled is False
        assert parameter_controls["decimation_target"].widget.enabled is False
        assert parameter_controls["texture_size"].widget.enabled is True
        assert parameter_controls["low_vram"].widget.enabled is False
        for backend in ReconstructionBackend:
            view.update_reconstruction_parameters(
                ReconstructionParameters(backend=backend), busy=False
            )
            visible = {
                key for key, widget
                in view.reconstruction_parameter_widgets.items()
                if widget.visible
            }
            assert visible == RECONSTRUCTION_BACKEND_PARAMETER_KEYS[backend]
            assert {
                key for key, control in parameter_controls.items()
                if control.widget.enabled
            } == visible
        refine_actions = []
        view.set_reconstruction_refine_handler(
            lambda action, value: refine_actions.append((action, value))
        )
        view._activate_reconstruction_refine("paint", True)
        assert refine_actions == [("paint", True)]
        runs = (
            ReconstructionRun(
                "base-run", ReconstructionRunKind.BASE,
                "/tmp/base.glb", "/tmp/source.png",
            ),
            ReconstructionRun(
                "refined-run", ReconstructionRunKind.MASKED_REFINE,
                "/tmp/refined.glb", "/tmp/source.png",
                parent_run_id="base-run",
            ),
        )
        refine_parameters = ReconstructionRefineParameters(
            strength=0.6, steps=12, seed=456,
            resize_detail_to_1024=False,
        )
        view.update_reconstruction_refine(
            refine_parameters,
            runs,
            "refined-run",
            mask_ready=True,
            can_refine=True,
            can_texture_refine=True,
            paint_active=True,
            erase_active=True,
            brush_size=72,
            brush_hardness=0.25,
            brush_flow=0.75,
        )
        refine_controls = view.reconstruction_refine_controls
        assert refine_controls["paint"].checked is True
        assert refine_controls["erase"].checked is True
        assert refine_controls["brush_size"].value == 72.0
        assert refine_controls["strength"].value == pytest.approx(0.6)
        assert refine_controls["steps"].value == 12.0
        assert refine_controls["seed"].value == 456.0
        assert refine_controls["resize_detail_to_1024"].checked is False
        assert refine_controls["version"].item_count == 2
        assert refine_controls["version"].item_text(0) == "Base (Pixal3D)"
        assert refine_controls["version"].item_text(1) == "Refined 1"
        assert refine_controls["version"].selected_index == 1
        assert refine_controls["run"].widget.enabled is True
        assert refine_controls["run_texture"].widget.enabled is True
        workspace_mask_controls = (
            view.reconstruction_workspace_mask_controls
        )
        assert workspace_mask_controls["paint"].checked is True
        assert workspace_mask_controls["erase"].checked is True
        assert workspace_mask_controls["brush_size"].value == 72.0
        assert workspace_mask_controls["brush_hardness"].value == 0.25
        assert workspace_mask_controls["brush_flow"].value == 0.75
        assert workspace_mask_controls[
            "resize_detail_to_1024"
        ].checked is False
        assert workspace_mask_controls["strength"].value == pytest.approx(0.6)
        assert workspace_mask_controls["steps"].value == 12.0
        assert workspace_mask_controls["seed"].value == 456.0
        assert workspace_mask_controls["clear"].widget.enabled is True
        assert view.reconstruction_workspace_actions[
            "refine"
        ].widget.enabled is True
        view._run_reconstruction_workspace_refine()
        assert refine_actions[-1] == ("run", None)
        view._select_reconstruction_workspace_operation("texture.refine")
        assert view.reconstruction_workspace_actions[
            "refine"
        ].widget.enabled is True
        view._run_reconstruction_workspace_refine()
        assert refine_actions[-1] == ("run_texture", None)
        assert view.reconstruction_refine_title.visible is True
        assert view.reconstruction_refine_panel.visible is True
        assert view.reconstruction_versions_panel.visible is True
        trellis_run = ReconstructionRun(
            "trellis-run",
            ReconstructionRunKind.BASE,
            "/tmp/trellis.glb",
            "/tmp/source.png",
            backend=ReconstructionBackend.TRELLIS2,
        )
        view.update_reconstruction_refine(
            refine_parameters,
            (trellis_run,),
            trellis_run.run_id,
            mask_ready=True,
            refine_supported=False,
            can_refine=False,
        )
        assert refine_controls["paint"].widget.enabled is False
        assert refine_controls["run"].widget.enabled is False
        assert refine_controls["version"].widget.enabled is True
        assert refine_controls["version"].item_text(0) == "Base (TRELLIS.2)"
        assert view.reconstruction_refine_title.visible is False
        assert view.reconstruction_refine_panel.visible is False
        assert view.reconstruction_versions_panel.visible is True
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
            backend=ReconstructionBackend.SPAR3D,
        )
        assert view.reconstruction_stage_checks[
            ReconstructionStage.SPARSE_OCCUPANCY
        ].checked is True
        assert view.reconstruction_stage_buttons[
            ReconstructionStage.POINT_CLOUD
        ].widget.enabled is True
        assert view.reconstruction_stage_buttons[
            ReconstructionStage.SPARSE_OCCUPANCY
        ].widget.enabled is False
        statuses[ReconstructionStage.TEXTURE_LATENT] = (
            ReconstructionStageStatus.READY
        )
        view.update_reconstruction_stages(
            statuses,
            {},
            ReconstructionStage.FINAL_MESH,
            ReconstructionStage.TEXTURE_LATENT,
            backend=ReconstructionBackend.HUNYUAN3D21,
        )
        assert view.reconstruction_stage_buttons[
            ReconstructionStage.TEXTURE_LATENT
        ].widget.enabled is True
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
            "edit.clear_selected_pixels": lambda: activated.append("clear"),
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
        assert view.menu_bar.dispatch_shortcut(KeyCode.Delete.value, 0)
        assert activated[-1] == "clear"

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
