from __future__ import annotations

import numpy as np
from PIL import Image

from diffusion_editor.app.layer_tree import LayerTreeCoordinator
from diffusion_editor.app.native_root import NativeEditorRoot
from diffusion_editor.document.commands import (
    AddReconstructionLayerCommand,
    PublishReconstructionResultCommand,
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
from diffusion_editor.generation.types import (
    ReconstructionParameters,
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
        seed=99,
        steps=20,
        resolution=1280,
        lr_conditioning_resolution=1024,
        manual_fov_degrees=50.0,
        decimation_target=300_000,
        texture_size=4096,
        low_vram=False,
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
    base.write_bytes(b"base")
    refined.write_bytes(b"refined")
    document.execute(PublishReconstructionResultCommand(
        node, str(base), 100, 30, 1,
        checkpoint_path=str(tmp_path / "base.npz"),
    ))
    base_run = node.active_run
    assert base_run is not None

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
    assert node.glb_path == str(refined)


def test_root_refine_entry_point_uses_active_run_checkpoint(tmp_path) -> None:
    stack, document = _document()
    document.execute(AddReconstructionLayerCommand("Character reconstruction"))
    node = stack.active_layer
    assert isinstance(node, ReconstructionLayer)
    checkpoint = tmp_path / "base.npz"
    checkpoint.write_bytes(b"checkpoint")
    document.execute(PublishReconstructionResultCommand(
        node, str(tmp_path / "base.glb"), 100, 30, 1,
        checkpoint_path=str(checkpoint),
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

        def start_refine(self, image, mask, base_checkpoint_path, *, parameters):
            self.captured = (image, mask, base_checkpoint_path, parameters)
            return ReconstructionControllerEvent(status="Refining")

    controller = Controller()
    root = NativeEditorRoot.__new__(NativeEditorRoot)
    root.application = Application()
    root.reconstruction_controller = controller
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


def test_root_binds_late_worker_result_to_launching_node(tmp_path) -> None:
    stack, document = _document()
    background = stack.active_layer
    document.execute(AddReconstructionLayerCommand("Character reconstruction"))
    node = stack.active_layer
    assert isinstance(node, ReconstructionLayer)
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

    root._start_reconstruction()
    assert root._reconstruction_job_node_id == node.id
    assert node.reconstruction_status == ReconstructionStatus.GENERATING

    stack.active_layer = background
    controller.event = ReconstructionControllerEvent(
        result=ReconstructionResult(str(glb_path), "source.png")
    )
    root._poll_reconstruction()

    assert node.reconstruction_status == ReconstructionStatus.READY
    assert node.glb_path == str(glb_path)
    assert stack.active_layer is background
