from __future__ import annotations

import numpy as np

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
from diffusion_editor.generation.types import ReconstructionResult


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
    assert (node.vertex_count, node.triangle_count, node.mesh_count) == (
        120, 40, 1)

    snapshot = stack.serialize_state()
    stack.load_state(snapshot)
    restored = stack.find_layer_by_id(node.id)
    assert isinstance(restored, ReconstructionLayer)
    assert restored.reconstruction_status == ReconstructionStatus.READY
    assert restored.glb_path == str(glb_path)


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

        def start(self, image, *, seed):
            assert image.size == (10, 8)
            assert seed == 42
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
