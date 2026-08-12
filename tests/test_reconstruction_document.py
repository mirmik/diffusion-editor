from __future__ import annotations

import hashlib
import numpy as np
from PIL import Image

from diffusion_editor.app.layer_tree import LayerTreeCoordinator
from diffusion_editor.app.native_root import NativeEditorRoot
from diffusion_editor.document.commands import (
    AddReconstructionLayerCommand,
    PublishReconstructionResultCommand,
    SelectReconstructionRunCommand,
)
from diffusion_editor.document.document_service import DocumentService
from diffusion_editor.document.history import HistoryManager
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.document.reconstruction import (
    ReconstructionLayer,
    ReconstructionStatus,
)
from diffusion_editor.generation.reconstruction_controller import (
    ReconstructionControllerEvent,
)
from diffusion_editor.generation.reconstruction_workspace import (
    PIXAL3D_PIPELINE,
    ReconstructionWorkspace,
    build_legacy_workspace,
)
from diffusion_editor.generation.types import (
    ReconstructionBackend,
    ReconstructionLrVariant,
    ReconstructionParameters,
    ReconstructionRefineParameters,
    ReconstructionRun,
    ReconstructionRunKind,
    ReconstructionResult,
    ReconstructionStage,
    ReconstructionStageArtifact,
    ReconstructionStageEvent,
    ReconstructionStageStatus,
)


def _document():
    stack = LayerStack(tile_size=8)
    image = np.zeros((8, 10, 4), dtype=np.uint8)
    image[:, :, :3] = (20, 40, 60)
    image[:, :, 3] = 255
    stack.init_from_image(image)
    history = HistoryManager(stack.load_state)
    return stack, DocumentService(stack, history, stack.load_state)


def test_reconstruction_node_is_undoable_and_not_composited() -> None:
    stack, document = _document()
    before = stack.composite()

    document.execute(AddReconstructionLayerCommand("Character reconstruction"))

    node = stack.active_layer
    assert isinstance(node, ReconstructionLayer)
    assert node.accepts_pixel_edits is False
    assert node.contributes_to_composite is False
    assert np.array_equal(stack.composite(), before)

    tree = LayerTreeCoordinator(stack, document)
    projected = tree.state.roots[0]
    assert projected.stable_id == node.id
    assert projected.node_type == "reconstruction"
    assert projected.status == "empty"
    assert tree.state.can_attach_tool is False
    assert tree.state.can_flatten is False

    assert document.undo() == "New 3D Reconstruction"
    assert stack.find_layer_by_id(node.id) is None
    assert document.redo() == "New 3D Reconstruction"
    assert stack.find_layer_by_id(node.id) is node
    tree.close()


def test_reconstruction_result_is_bound_to_its_node_and_roundtrips(tmp_path) -> None:
    stack, document = _document()
    document.execute(AddReconstructionLayerCommand("Character reconstruction"))
    node = stack.active_layer
    assert isinstance(node, ReconstructionLayer)
    node.generation_parameters = ReconstructionParameters(
        backend=ReconstructionBackend.TRELLIS2,
        seed=99,
        steps=20,
        resolution=1280,
        lr_conditioning_resolution=1024,
        manual_fov_degrees=50.0,
        decimation_target=300_000,
        texture_size=4096,
        low_vram=False,
    )
    node.refine_parameters = ReconstructionRefineParameters(
        strength=0.55,
        steps=11,
        seed=456,
        local_resolution=1280,
        rescale_t=2.5,
        guidance_strength=6.0,
        resize_detail_to_1024=False,
    )
    node.lr_refine_parameters = ReconstructionRefineParameters(
        strength=0.45, steps=9, seed=457,
    )
    node.texture_refine_parameters = ReconstructionRefineParameters(
        strength=0.65, steps=13, seed=458,
        resize_detail_to_1024=False,
    )
    glb_path = tmp_path / "character.glb"
    glb_path.write_bytes(b"glTF")

    document.execute(PublishReconstructionResultCommand(
        node,
        str(glb_path),
        vertex_count=120,
        triangle_count=40,
        mesh_count=1,
    ))

    assert node.reconstruction_status == ReconstructionStatus.READY
    assert node.glb_path == str(glb_path)
    assert len(node.runs) == 1
    assert node.active_run is node.base_run
    assert node.base_run is not None
    assert node.base_run.kind is ReconstructionRunKind.BASE
    assert (node.vertex_count, node.triangle_count, node.mesh_count) == (
        120, 40, 1)

    snapshot = stack.serialize_state()
    stack.load_state(snapshot)
    restored = stack.find_layer_by_id(node.id)
    assert isinstance(restored, ReconstructionLayer)
    assert restored.reconstruction_status == ReconstructionStatus.READY
    assert restored.glb_path == str(glb_path)
    assert restored.generation_parameters == node.generation_parameters
    assert restored.generation_parameters.backend is ReconstructionBackend.TRELLIS2
    assert restored.refine_parameters == node.refine_parameters
    assert restored.lr_refine_parameters == node.lr_refine_parameters
    assert (
        restored.texture_refine_parameters
        == node.texture_refine_parameters
    )
    assert len(restored.runs) == 1
    assert restored.active_run is restored.base_run
    assert restored.target_stage is ReconstructionStage.FINAL_MESH
    assert restored.selected_preview_stage is ReconstructionStage.FINAL_MESH
    assert restored.stage_statuses[ReconstructionStage.FINAL_MESH] is (
        ReconstructionStageStatus.READY
    )
    assert restored.stage_artifacts[
        ReconstructionStage.FINAL_MESH
    ].path == str(glb_path)


def test_refined_run_is_appended_without_replacing_base(tmp_path) -> None:
    stack, document = _document()
    document.execute(AddReconstructionLayerCommand("Character reconstruction"))
    node = stack.active_layer
    assert isinstance(node, ReconstructionLayer)
    base = tmp_path / "base.glb"
    refined = tmp_path / "refined.glb"
    base_hr = tmp_path / "base-hr.glb"
    refined_hr = tmp_path / "refined-hr.glb"
    base.write_bytes(b"base")
    refined.write_bytes(b"refined")
    base_hr.write_bytes(b"base-hr")
    refined_hr.write_bytes(b"refined-hr")
    node.apply_stage_event(ReconstructionStageEvent(
        ReconstructionStage.HR_SHAPE_LATENT,
        ReconstructionStageStatus.READY,
        8,
        8,
        ReconstructionStageArtifact(
            ReconstructionStage.HR_SHAPE_LATENT,
            str(base_hr),
            "mesh",
        ),
    ))
    document.execute(PublishReconstructionResultCommand(
        node, str(base), 100, 30, 1,
        checkpoint_path=str(tmp_path / "base.npz"),
    ))
    base_run = node.active_run
    assert base_run is not None

    node.apply_stage_event(ReconstructionStageEvent(
        ReconstructionStage.HR_SHAPE_LATENT,
        ReconstructionStageStatus.READY,
        6,
        6,
        ReconstructionStageArtifact(
            ReconstructionStage.HR_SHAPE_LATENT,
            str(refined_hr),
            "mesh",
        ),
    ))
    document.execute(PublishReconstructionResultCommand(
        node, str(refined), 110, 35, 1,
        run_kind=ReconstructionRunKind.MASKED_REFINE,
        parent_run_id=base_run.run_id,
        checkpoint_path=str(tmp_path / "refined.npz"),
    ))

    assert len(node.runs) == 2
    assert node.base_run is base_run
    assert node.active_run is not None
    assert node.active_run.kind is ReconstructionRunKind.MASKED_REFINE
    assert node.active_run.parent_run_id == base_run.run_id
    refined_run = node.active_run
    assert node.glb_path == str(refined)
    assert {
        artifact.stage: artifact.path
        for artifact in node.active_run.stage_artifacts
    }[ReconstructionStage.HR_SHAPE_LATENT] == str(refined_hr)

    node.selected_preview_stage = ReconstructionStage.HR_SHAPE_LATENT
    node.preview_stage_pinned = True

    document.execute(SelectReconstructionRunCommand(node, base_run.run_id))

    assert node.active_run is base_run
    assert node.glb_path == str(base)
    assert node.selected_preview_stage is ReconstructionStage.HR_SHAPE_LATENT
    assert node.stage_artifacts[
        ReconstructionStage.HR_SHAPE_LATENT
    ].path == str(base_hr)
    assert node.stage_progress[ReconstructionStage.HR_SHAPE_LATENT] == (8, 8)
    assert node.stage_artifacts[ReconstructionStage.FINAL_MESH].path == str(base)
    assert document.undo() == "Select 3D Reconstruction Version"
    assert node.glb_path == str(refined)
    assert node.selected_preview_stage is ReconstructionStage.HR_SHAPE_LATENT
    assert node.stage_artifacts[
        ReconstructionStage.HR_SHAPE_LATENT
    ].path == str(refined_hr)

    document.execute(SelectReconstructionRunCommand(node, base_run.run_id))
    loaded = []

    class Viewport:
        def load_glb(self, path):
            loaded.append(path)

    class Application:
        layer_stack = stack
        status = ""

        def set_status(self, status):
            self.status = status

    root = NativeEditorRoot.__new__(NativeEditorRoot)
    application = Application()
    application.document = document
    root.application = application
    root.reconstruction_controller = None
    root.canvas_controls_coordinator = None
    root.view = None
    root._presented_reconstruction_id = None
    root._ensure_reconstruction_viewport = lambda: Viewport()

    root._handle_reconstruction_refine("select_run", refined_run.run_id)

    assert node.active_run is refined_run
    assert node.selected_preview_stage is ReconstructionStage.HR_SHAPE_LATENT
    assert loaded == [str(refined_hr)]


def test_root_refine_entry_point_uses_active_run_checkpoint(tmp_path) -> None:
    stack, document = _document()
    document.execute(AddReconstructionLayerCommand("Character reconstruction"))
    node = stack.active_layer
    assert isinstance(node, ReconstructionLayer)
    checkpoint = tmp_path / "base.npz"
    checkpoint.write_bytes(b"checkpoint")
    texture_checkpoint = tmp_path / "base-texture.npz"
    texture_checkpoint.write_bytes(b"texture-checkpoint")
    source = tmp_path / "source.png"
    Image.new("RGBA", (10, 8), (10, 20, 30, 255)).save(source)
    document.execute(PublishReconstructionResultCommand(
        node, str(tmp_path / "base.glb"), 100, 30, 1,
        source_path=str(source),
        checkpoint_path=str(checkpoint),
        texture_checkpoint_path=str(texture_checkpoint),
    ))
    parent = node.active_run
    assert parent is not None

    class Application:
        layer_stack = stack
        status_text = ""

        def set_status(self, text):
            self.status_text = text

    class Controller:
        is_busy = False
        captured = None
        lr_captured = None
        texture_captured = None

        def start_refine(
            self, image, mask, base_checkpoint_path, *,
            parameters, generation_parameters,
        ):
            self.captured = (
                image, mask, base_checkpoint_path,
                parameters, generation_parameters,
            )
            return ReconstructionControllerEvent(status="Refining")

        def start_lr_refine(
            self, image, mask, session_checkpoint_path, *,
            parameters, generation_parameters,
        ):
            self.lr_captured = (
                image, mask, session_checkpoint_path,
                parameters, generation_parameters,
            )
            return ReconstructionControllerEvent(status="Refining LR")

        def start_texture_refine(
            self, image, mask, shape_path, texture_path, *,
            parameters, generation_parameters,
        ):
            self.texture_captured = (
                image, mask, shape_path, texture_path,
                parameters, generation_parameters,
            )
            return ReconstructionControllerEvent(status="Refining texture")

    controller = Controller()
    root = NativeEditorRoot.__new__(NativeEditorRoot)
    root.application = Application()
    root.reconstruction_controller = controller
    root.canvas_controls_coordinator = None
    root._reconstruction_job_node_id = None
    root._reconstruction_parent_run_id = None

    event = root.start_reconstruction_refine(
        node,
        Image.new("RGB", (16, 16)),
        Image.new("L", (16, 16), 255),
    )

    assert event is not None and event.status == "Refining"
    assert controller.captured[2] == str(checkpoint)
    assert root._reconstruction_job_node_id == node.id
    assert root._reconstruction_parent_run_id == parent.run_id
    assert node.reconstruction_status is ReconstructionStatus.GENERATING

    stack.selection.data[2:6, 3:7] = 1.0
    root._start_selected_reconstruction_refine(node)

    (
        captured_image, captured_mask, _, captured_parameters,
        captured_generation_parameters,
    ) = controller.captured
    assert captured_image.size == (10, 8)
    assert captured_mask.size == (10, 8)
    assert np.asarray(captured_mask)[2:6, 3:7].min() == 255
    assert captured_parameters == node.refine_parameters
    assert captured_generation_parameters == node.generation_parameters

    texture_event = root.start_reconstruction_texture_refine(
        node,
        Image.new("RGB", (10, 8)),
        Image.new("L", (10, 8), 255),
        parameters=node.refine_parameters,
    )

    assert texture_event is not None
    assert texture_event.status == "Refining texture"
    assert controller.texture_captured[2:4] == (
        str(checkpoint), str(texture_checkpoint)
    )

    node.runs = ()
    node.active_run_id = None
    node.intermediate_source_path = str(source)
    node.intermediate_shape_checkpoint_path = str(checkpoint)
    node.intermediate_texture_checkpoint_path = str(texture_checkpoint)
    node.resume_checkpoint_path = str(checkpoint)
    node.resume_stage = ReconstructionStage.LR_SHAPE_LATENT
    node.resume_source_sha256 = "source-hash"
    root._start_selected_reconstruction_refine(node)
    assert controller.captured[2] == str(checkpoint)
    assert root._reconstruction_parent_run_id is None
    root._start_selected_reconstruction_texture_refine(
        node, parameters=node.texture_refine_parameters
    )
    assert controller.texture_captured[2:4] == (
        str(checkpoint), str(texture_checkpoint)
    )
    assert controller.texture_captured[4] == node.texture_refine_parameters
    root._start_selected_reconstruction_lr_refine(node)
    assert controller.lr_captured[2] == str(checkpoint)
    assert controller.lr_captured[3] == node.lr_refine_parameters
    assert root._reconstruction_job_source_sha256 == "source-hash"

    refined_checkpoint = tmp_path / "refined-lr.npz"
    refined_checkpoint.write_bytes(b"refined-lr")
    base_variant = ReconstructionLrVariant(
        "lr-base", "Base LR", str(checkpoint), str(source),
        ReconstructionStageArtifact(
            ReconstructionStage.LR_SHAPE_LATENT,
            str(tmp_path / "base-lr.glb"),
            "mesh",
        ),
    )
    refined_variant = ReconstructionLrVariant(
        "lr-refined-1", "Refined LR 1", str(refined_checkpoint),
        str(source),
        ReconstructionStageArtifact(
            ReconstructionStage.LR_SHAPE_LATENT,
            str(tmp_path / "refined-lr.glb"),
            "mesh",
        ),
        parent_variant_id="lr-base",
    )
    node.lr_variants = (base_variant, refined_variant)
    node.selected_lr_refine_source_id = "lr-base"
    root._start_selected_reconstruction_lr_refine(node)
    assert controller.lr_captured[2] == str(checkpoint)
    assert root._reconstruction_lr_parent_variant_id == "lr-base"

    node.selected_lr_refine_source_id = "lr-refined-1"
    root._start_selected_reconstruction_lr_refine(node)
    assert controller.lr_captured[2] == str(refined_checkpoint)
    assert root._reconstruction_lr_parent_variant_id == "lr-refined-1"


def test_reconstruction_node_tracks_target_progress_and_preview(tmp_path) -> None:
    node = ReconstructionLayer("Staged")
    node.target_stage = ReconstructionStage.HR_COORDINATES
    node.begin_staged_generation()

    assert node.stage_statuses[ReconstructionStage.HR_COORDINATES] is (
        ReconstructionStageStatus.PENDING
    )
    assert node.stage_statuses[ReconstructionStage.HR_SHAPE_FLOW] is (
        ReconstructionStageStatus.SKIPPED
    )

    preview = ReconstructionStageArtifact(
        ReconstructionStage.SPARSE_OCCUPANCY,
        str(tmp_path / "sparse.glb"),
        "mesh",
    )
    node.apply_stage_event(ReconstructionStageEvent(
        ReconstructionStage.SPARSE_OCCUPANCY,
        ReconstructionStageStatus.READY,
        12,
        12,
        preview,
    ))

    assert node.selected_preview_stage is ReconstructionStage.SPARSE_OCCUPANCY
    assert node.stage_progress[ReconstructionStage.SPARSE_OCCUPANCY] == (12, 12)
    assert node.stage_artifacts[ReconstructionStage.SPARSE_OCCUPANCY] == preview

    node.selected_preview_stage = ReconstructionStage.SPARSE_OCCUPANCY
    node.preview_stage_pinned = True
    final_preview = ReconstructionStageArtifact(
        ReconstructionStage.FINAL_MESH,
        str(tmp_path / "final.glb"),
        "mesh",
    )
    node.apply_stage_event(ReconstructionStageEvent(
        ReconstructionStage.FINAL_MESH,
        ReconstructionStageStatus.READY,
        artifact=final_preview,
    ))

    assert node.selected_preview_stage is ReconstructionStage.SPARSE_OCCUPANCY
    assert node.stage_artifacts[ReconstructionStage.FINAL_MESH] == final_preview


def test_spar3d_staged_generation_only_enables_its_real_stages() -> None:
    node = ReconstructionLayer("SPAR3D object")
    node.generation_parameters = ReconstructionParameters(
        backend=ReconstructionBackend.SPAR3D
    )
    node.target_stage = ReconstructionStage.FINAL_MESH

    node.begin_staged_generation()

    assert node.stage_statuses[ReconstructionStage.SOURCE_IMAGE] is (
        ReconstructionStageStatus.PENDING
    )
    assert node.stage_statuses[ReconstructionStage.POINT_CLOUD] is (
        ReconstructionStageStatus.PENDING
    )
    assert node.stage_statuses[ReconstructionStage.SPARSE_OCCUPANCY] is (
        ReconstructionStageStatus.SKIPPED
    )
    assert node.stage_statuses[ReconstructionStage.FINAL_MESH] is (
        ReconstructionStageStatus.PENDING
    )


def test_root_binds_late_worker_result_to_launching_node(tmp_path) -> None:
    stack, document = _document()
    background = stack.active_layer
    document.execute(AddReconstructionLayerCommand("Character reconstruction"))
    node = stack.active_layer
    assert isinstance(node, ReconstructionLayer)
    node.generation_parameters = ReconstructionParameters(
        backend=ReconstructionBackend.TRELLIS2
    )
    glb_path = tmp_path / "character.glb"
    glb_path.write_bytes(b"glTF")

    class Application:
        layer_stack = stack
        status_text = ""

        def set_status(self, text):
            self.status_text = text

    class Controller:
        is_busy = False
        event = None

        def start(self, image, *, parameters, target_stage):
            assert image.size == (10, 8)
            assert parameters.seed == 42
            assert parameters.backend is ReconstructionBackend.TRELLIS2
            assert target_stage.value == "final_mesh"
            return ReconstructionControllerEvent(status="Generating")

        def poll(self):
            event, self.event = self.event, None
            return event

    class Viewport:
        mesh_count = 0

        def load_glb(self, path):
            assert path == str(glb_path)
            self.mesh_count = 1
            return 120, 40, 1

    application = Application()
    application.document = document
    controller = Controller()
    viewport = Viewport()
    root = NativeEditorRoot.__new__(NativeEditorRoot)
    root.application = application
    root.reconstruction_controller = controller
    root.reconstruction_viewport = viewport
    root._reconstruction_job_node_id = None
    root._presented_reconstruction_id = None
    root._ensure_reconstruction_viewport = lambda: viewport
    root._read_reconstruction_glb_stats = lambda _path: (120, 40, 1)

    root._start_reconstruction()
    assert root._reconstruction_job_node_id == node.id
    assert node.reconstruction_status == ReconstructionStatus.GENERATING

    stack.active_layer = background
    controller.event = ReconstructionControllerEvent(
        result=ReconstructionResult(
            str(glb_path),
            "source.png",
            backend=ReconstructionBackend.TRELLIS2,
        )
    )
    root._poll_reconstruction()

    assert node.reconstruction_status == ReconstructionStatus.READY
    assert node.glb_path == str(glb_path)
    assert node.active_run is not None
    assert node.active_run.backend is ReconstructionBackend.TRELLIS2
    assert stack.active_layer is background


def test_workspace_generate_action_routes_to_legacy_stage() -> None:
    stack, _document_service = _document()
    node = ReconstructionLayer("Workspace")
    stack.insert_layer(node)

    class Application:
        layer_stack = stack
        status = ""

        def set_status(self, value):
            self.status = value

    class Controller:
        is_busy = False

    root = NativeEditorRoot.__new__(NativeEditorRoot)
    root.application = Application()
    root.reconstruction_controller = Controller()
    started = []
    root._start_reconstruction = lambda: started.append(node.target_stage)

    root._handle_reconstruction_workspace(
        "generate_to_operation", "lr.generate"
    )

    assert started == [ReconstructionStage.LR_SHAPE_LATENT]
    assert node.target_stage is ReconstructionStage.LR_SHAPE_LATENT


def test_partial_pixal3d_result_keeps_refine_checkpoints(tmp_path) -> None:
    stack, _document_service = _document()
    node = ReconstructionLayer("Workspace")
    stack.insert_layer(node)
    source = tmp_path / "source.png"
    Image.new("RGBA", (10, 8)).save(source)
    shape_checkpoint = tmp_path / "shape.npz"
    shape_checkpoint.write_bytes(b"shape")
    resume_checkpoint = tmp_path / "resume.npz"
    resume_checkpoint.write_bytes(b"resume")
    result = ReconstructionResult(
        str(tmp_path / "not-built-yet.glb"),
        str(source),
        completed_stage=ReconstructionStage.HR_SHAPE_FLOW,
        checkpoint_path=str(shape_checkpoint),
        resume_checkpoint_path=str(resume_checkpoint),
    )

    class Application:
        layer_stack = stack
        status = ""

        def set_status(self, value):
            self.status = value

    class Controller:
        is_busy = False

        def poll(self):
            return ReconstructionControllerEvent(result=result)

    root = NativeEditorRoot.__new__(NativeEditorRoot)
    root.application = Application()
    root.reconstruction_controller = Controller()
    root._reconstruction_job_node_id = node.id
    root._reconstruction_parent_run_id = None
    root._reconstruction_job_source_sha256 = "source-hash"
    root._reconstruction_job_parameters = node.generation_parameters
    root.view = None

    root._poll_reconstruction()

    assert node.intermediate_source_path == str(source)
    assert node.intermediate_shape_checkpoint_path == str(shape_checkpoint)
    assert node.resume_checkpoint_path == str(resume_checkpoint)
    assert node.resume_stage is ReconstructionStage.HR_SHAPE_FLOW
    assert node.reconstruction_status is ReconstructionStatus.READY


def test_lr_results_keep_base_and_refined_variants(tmp_path) -> None:
    stack, _document_service = _document()
    node = ReconstructionLayer("Workspace")
    stack.insert_layer(node)
    source = tmp_path / "source.png"
    Image.new("RGBA", (10, 8)).save(source)
    base_checkpoint = tmp_path / "base-lr.npz"
    base_checkpoint.write_bytes(b"base")
    refined_checkpoint = tmp_path / "refined-lr.npz"
    refined_checkpoint.write_bytes(b"refined")
    generated_preview = tmp_path / "lr-generated.glb"
    generated_preview.write_bytes(b"generated")
    base_preview = ReconstructionStageArtifact(
        ReconstructionStage.LR_SHAPE_LATENT,
        str(tmp_path / "base-lr.glb"),
        "mesh",
    )
    refined_preview = ReconstructionStageArtifact(
        ReconstructionStage.LR_SHAPE_LATENT,
        str(tmp_path / "refined-lr.glb"),
        "mesh",
    )
    results = [
        ReconstructionResult(
            base_preview.path,
            str(source),
            completed_stage=ReconstructionStage.LR_SHAPE_LATENT,
            artifacts=(base_preview,),
            resume_checkpoint_path=str(base_checkpoint),
        ),
        ReconstructionResult(
            refined_preview.path,
            str(source),
            completed_stage=ReconstructionStage.LR_SHAPE_LATENT,
            artifacts=(refined_preview,),
            kind=ReconstructionRunKind.MASKED_REFINE,
            resume_checkpoint_path=str(refined_checkpoint),
            refine_generated_path=str(generated_preview),
        ),
    ]

    class Application:
        layer_stack = stack

        def set_status(self, _value):
            pass

    class Controller:
        is_busy = False

        def poll(self):
            return ReconstructionControllerEvent(result=results.pop(0))

    root = NativeEditorRoot.__new__(NativeEditorRoot)
    root.application = Application()
    root.reconstruction_controller = Controller()
    root.view = None
    root._reconstruction_job_source_sha256 = "source-hash"
    root._reconstruction_job_parameters = node.generation_parameters
    root._reconstruction_parent_run_id = None
    root._reconstruction_lr_parent_variant_id = None
    root._reconstruction_job_node_id = node.id
    root._poll_reconstruction()

    assert [item.variant_id for item in node.lr_variants] == ["lr-base"]
    assert node.selected_lr_refine_source_id == "lr-base"
    assert node.accepted_lr_variant_id == "lr-base"

    root._reconstruction_job_node_id = node.id
    root._reconstruction_lr_parent_variant_id = "lr-base"
    root._poll_reconstruction()

    assert [item.variant_id for item in node.lr_variants] == [
        "lr-base", "lr-refined-1",
    ]
    assert node.lr_variants[1].parent_variant_id == "lr-base"
    assert node.lr_variants[1].refine_generated_path == str(generated_preview)
    assert node.selected_lr_refine_source_id == "lr-base"
    assert node.accepted_lr_variant_id == "lr-refined-1"
    assert node.resume_checkpoint_path == str(refined_checkpoint)


def test_workspace_parameter_action_updates_stage_override() -> None:
    stack, _document_service = _document()
    node = ReconstructionLayer("Workspace")
    stack.insert_layer(node)

    class Application:
        layer_stack = stack

    class Controller:
        is_busy = False

    root = NativeEditorRoot.__new__(NativeEditorRoot)
    root.application = Application()
    root.reconstruction_controller = Controller()
    root._refresh_reconstruction_panel = lambda _node: None

    root._handle_reconstruction_workspace(
        "set_operation_parameter", ("pixal3d_hr_seed", 777)
    )

    assert node.generation_parameters.pixal3d_hr_seed == 777

    root._handle_reconstruction_workspace(
        "reset_operation_parameters", "hr.generate"
    )
    assert node.generation_parameters.pixal3d_hr_seed == -1


def test_reconstruction_start_resumes_pixal3d_prefix(tmp_path) -> None:
    stack, _document_service = _document()
    node = ReconstructionLayer("Workspace")
    stack.insert_layer(node)
    checkpoint = tmp_path / "resume.npz"
    checkpoint.write_bytes(b"checkpoint")
    node.resume_checkpoint_path = str(checkpoint)
    node.resume_stage = ReconstructionStage.SPARSE_OCCUPANCY
    node.resume_parameters = node.generation_parameters
    node.generation_parameters = ReconstructionParameters(
        pixal3d_lr_seed=999
    )
    node.resume_source_sha256 = hashlib.sha256(
        Image.fromarray(stack.composite(), mode="RGBA").tobytes()
    ).hexdigest()
    node.target_stage = ReconstructionStage.LR_SHAPE_LATENT
    node.stage_statuses[ReconstructionStage.SPARSE_OCCUPANCY] = (
        ReconstructionStageStatus.READY
    )

    class Application:
        layer_stack = stack
        def set_status(self, _value):
            pass

    class Controller:
        is_busy = False
        calls = []
        def start(self, image, **kwargs):
            self.calls.append(kwargs)
            return ReconstructionControllerEvent(status="Resuming")

    root = NativeEditorRoot.__new__(NativeEditorRoot)
    root.application = Application()
    root.reconstruction_controller = Controller()
    root._ensure_reconstruction_viewport = lambda: None
    root._refresh_reconstruction_panel = lambda _node: None
    root._set_reconstruction_status = lambda _node, _status: None
    root._reconstruction_job_node_id = None

    root._start_reconstruction()

    assert root.reconstruction_controller.calls[0][
        "resume_checkpoint_path"
    ] == str(checkpoint)
    assert node.stage_statuses[ReconstructionStage.SPARSE_OCCUPANCY] is (
        ReconstructionStageStatus.READY
    )


def test_reconstruction_start_invalidates_resume_after_source_change(
        tmp_path) -> None:
    stack, _document_service = _document()
    node = ReconstructionLayer("Workspace")
    stack.insert_layer(node)
    checkpoint = tmp_path / "resume.npz"
    checkpoint.write_bytes(b"checkpoint")
    node.resume_checkpoint_path = str(checkpoint)
    node.resume_stage = ReconstructionStage.SPARSE_OCCUPANCY
    node.resume_parameters = node.generation_parameters
    node.resume_source_sha256 = "different-source"
    node.target_stage = ReconstructionStage.LR_SHAPE_LATENT

    class Application:
        layer_stack = stack
        def set_status(self, _value):
            pass

    class Controller:
        is_busy = False
        calls = []
        def start(self, image, **kwargs):
            self.calls.append(kwargs)
            return ReconstructionControllerEvent(status="Fresh")

    root = NativeEditorRoot.__new__(NativeEditorRoot)
    root.application = Application()
    root.reconstruction_controller = Controller()
    root._ensure_reconstruction_viewport = lambda: None
    root._refresh_reconstruction_panel = lambda _node: None
    root._set_reconstruction_status = lambda _node, _status: None

    root._start_reconstruction()

    assert "resume_checkpoint_path" not in root.reconstruction_controller.calls[0]
    assert node.stage_statuses[ReconstructionStage.SPARSE_OCCUPANCY] is (
        ReconstructionStageStatus.PENDING
    )


def test_workspace_preview_loads_graph_mesh_without_changing_legacy_stage(
        tmp_path) -> None:
    stack, _document_service = _document()
    node = ReconstructionLayer("Workspace")
    stack.insert_layer(node)
    node.selected_preview_stage = ReconstructionStage.SPARSE_OCCUPANCY
    mesh_path = tmp_path / "hr.glb"
    mesh_path.write_bytes(b"glb")

    workspace = ReconstructionWorkspace(PIXAL3D_PIPELINE)
    operation = workspace.plan_operation(
        "hr.generate",
        model_identity="pixal3d:test",
        worker_protocol="test",
        operation_id="hr-operation",
    )
    artifact = workspace.publish_artifact(
        operation.operation_id,
        "hr_mesh",
        path=str(mesh_path),
        artifact_id="hr-artifact",
    )
    loaded = []

    class Application:
        layer_stack = stack
        status = ""

        def set_status(self, value):
            self.status = value

    class Viewport:
        def load_glb(self, path):
            loaded.append(path)

    root = NativeEditorRoot.__new__(NativeEditorRoot)
    root.application = Application()
    root._active_reconstruction_workspace = workspace
    root._presented_reconstruction_id = None
    root._ensure_reconstruction_viewport = lambda: Viewport()

    root._handle_reconstruction_workspace(
        "preview_artifact", artifact.artifact_id
    )

    assert loaded == [str(mesh_path)]
    assert workspace.selected_artifact_id == artifact.artifact_id
    assert node.selected_preview_stage is ReconstructionStage.SPARSE_OCCUPANCY


def test_refine_generated_artifact_uses_separate_viewport(tmp_path) -> None:
    generated = tmp_path / "refine-generated.glb"
    generated.write_bytes(b"glTF")
    run = ReconstructionRun(
        "refined",
        ReconstructionRunKind.MASKED_REFINE,
        str(tmp_path / "merged.glb"),
        str(tmp_path / "source.png"),
        refine_generated_path=str(generated),
    )
    workspace = build_legacy_workspace(
        ReconstructionParameters(), {}, {}, (run,), "refined"
    )
    operation = workspace.operations_for_spec("hr.refine")[0]

    class View:
        reconstruction_workspace_mode = True
        visibility = []

        def set_reconstruction_refine_view_visible(self, visible):
            self.visibility.append(visible)

    class Viewport:
        loaded = []
        clear_count = 0

        def load_glb(self, path):
            self.loaded.append(path)

        def clear_model(self):
            self.clear_count += 1

    root = NativeEditorRoot.__new__(NativeEditorRoot)
    root.view = View()
    root.reconstruction_refine_viewport = Viewport()
    root._presented_refine_artifact_path = None
    root._ensure_reconstruction_refine_viewport = (
        lambda: root.reconstruction_refine_viewport
    )

    root._sync_reconstruction_refine_output(
        workspace, "hr.refine", operation.operation_id
    )
    root._sync_reconstruction_refine_output(
        workspace, "hr.refine", operation.operation_id
    )

    assert root.view.visibility == [True, True]
    assert root.reconstruction_refine_viewport.loaded == [str(generated)]

    root._sync_reconstruction_refine_output(workspace, "hr.generate")
    assert root.view.visibility[-1] is False
    assert root.reconstruction_refine_viewport.clear_count == 1
