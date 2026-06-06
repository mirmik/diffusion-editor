"""Tests for DocumentService and command bus integration."""

import numpy as np
from PIL import Image

from diffusion_editor.document.document_service import (
    CallbackCommand, CommandBus, DocumentService,
)
from diffusion_editor.document.commands import (
    AddLayerCommand, SetLayerOpacityCommand, FlattenLayersCommand,
    SnapshotCallbackCommand, ClearLayerMaskCommand, SetLayerPatchRectCommand,
    ClearLayerPatchRectCommand, ReplaceLayerMaskCommand,
    ApplyGeneratedResultCommand, SetLayerSelectionCommand,
    AttachLayerToolCommand, DetachLayerToolCommand,
)
from diffusion_editor.document.layer import Layer
from diffusion_editor.document.tool import DiffusionTool
from diffusion_editor.document.history import HistoryManager
from diffusion_editor.document.layer_stack import LayerStack


def _diff_layer() -> Layer:
    tool = DiffusionTool(
        source_patch=None,
        patch_x=0, patch_y=0, patch_w=8, patch_h=8,
        prompt="", negative_prompt="",
        strength=0.5, guidance_scale=7.0, steps=20, seed=1,
    )
    layer = Layer("Diff", 8, 8)
    layer.tool = tool
    return layer


class _DummyStack:
    def __init__(self):
        self.value = 0

    def serialize_state(self) -> bytes:
        return str(self.value).encode("ascii")

    def load_state(self, snapshot: bytes) -> None:
        self.value = int(snapshot.decode("ascii"))


def test_document_service_snapshot_action_undo_redo():
    stack = _DummyStack()
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)

    service.execute_snapshot_action("inc", lambda: setattr(stack, "value", stack.value + 1))

    assert stack.value == 1
    assert service.undo() == "inc"
    assert stack.value == 0
    assert service.redo() == "inc"
    assert stack.value == 1


def test_document_service_snapshot_action_skips_noop():
    stack = _DummyStack()
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)

    service.execute_snapshot_action("noop", lambda: None)

    assert service.undo() is None


def test_command_bus_execute_registers_history():
    events = []

    def _undo():
        events.append("undo")

    def _redo():
        events.append("redo")

    history = HistoryManager(lambda _snapshot: None)
    bus = CommandBus(history)
    bus.execute(CallbackCommand(
        label="cmd",
        do_fn=lambda: events.append("do"),
        undo_fn=_undo,
        redo_fn=_redo,
    ))

    assert events == ["do"]
    assert history.undo() == "cmd"
    assert events == ["do", "undo"]
    assert history.redo() == "cmd"
    assert events == ["do", "undo", "redo"]


def test_document_service_add_layer_command_undo_redo():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.full((8, 8, 4), 255, dtype=np.uint8))
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)

    service.execute(AddLayerCommand(name="Layer 1"))

    assert len(stack.layers) == 2
    assert service.undo() == "New Layer"
    assert len(stack.layers) == 1
    assert service.redo() == "New Layer"
    assert len(stack.layers) == 2


def test_add_layer_command_can_insert_offset_image_layer():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.zeros((8, 8, 4), dtype=np.uint8))
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)
    image = np.zeros((3, 4, 4), dtype=np.uint8)
    image[:, :, 3] = 255

    service.execute(AddLayerCommand(
        name="Patch",
        image=image,
        x=2,
        y=3,
        label="Paste",
    ))

    layer = stack.active_layer
    assert layer.width == 4
    assert layer.height == 3
    assert layer.x == 2
    assert layer.y == 3
    assert stack.composite()[3, 2, 3] == 255


def test_document_service_set_opacity_command():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.full((8, 8, 4), 255, dtype=np.uint8))
    layer = stack.layers[0]
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)

    service.execute(SetLayerOpacityCommand(layer=layer, opacity=0.5))

    assert layer.opacity == 0.5
    assert service.undo() == "Set Opacity"
    assert stack.layers[0].opacity == 1.0


def test_document_service_flatten_command():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.full((8, 8, 4), 255, dtype=np.uint8))
    stack.add_layer("Top", np.zeros((8, 8, 4), dtype=np.uint8))
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)

    service.execute(FlattenLayersCommand())

    assert len(stack.layers) == 1
    assert stack.layers[0].name == "Background"
    assert service.undo() == "Flatten Layers"
    assert len(stack.layers) == 2


def test_document_service_snapshot_callback_command():
    stack = _DummyStack()
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)

    service.execute(SnapshotCallbackCommand(
        label="inc-cb",
        apply_fn=lambda _layer_stack: setattr(stack, "value", stack.value + 3),
    ))

    assert stack.value == 3
    assert service.undo() == "inc-cb"
    assert stack.value == 0


def test_document_service_clear_layer_mask_command():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.full((8, 8, 4), 255, dtype=np.uint8))
    layer = _diff_layer()
    layer.mask.data[2:4, 2:4] = 1.0
    stack.insert_layer(layer)
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)

    service.execute(ClearLayerMaskCommand(layer=layer, label="Clear Mask"))

    assert not layer.has_mask()
    assert service.undo() == "Clear Mask"
    restored = stack.active_layer
    assert restored is not None
    assert restored.tool is not None
    assert restored.has_mask()


def test_document_service_layer_patch_rect_commands():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.full((8, 8, 4), 255, dtype=np.uint8))
    layer = _diff_layer()
    stack.insert_layer(layer)
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)

    service.execute(SetLayerPatchRectCommand(
        layer=layer,
        rect=(1, 2, 5, 6),
        label="Set Rect",
    ))
    assert layer.patch_rect == (1, 2, 5, 6)

    service.execute(ClearLayerPatchRectCommand(layer=layer, label="Clear Rect"))
    assert layer.patch_rect is None

    assert service.undo() == "Clear Rect"
    assert stack.active_layer is not None
    assert stack.active_layer.patch_rect == (1, 2, 5, 6)


def test_document_service_replace_layer_mask_command():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.full((8, 8, 4), 255, dtype=np.uint8))
    layer = _diff_layer()
    stack.insert_layer(layer)
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0:2, 0:2] = 255

    service.execute(ReplaceLayerMaskCommand(layer=layer, mask=mask))

    assert np.array_equal(layer.mask.data, mask.astype(np.float32) / 255.0)
    assert service.undo() == "Apply Segmentation Mask"


def test_document_service_attach_detach_layer_tool_commands():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.full((8, 8, 4), 255, dtype=np.uint8))
    layer = stack.active_layer
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)
    tool = DiffusionTool(
        source_patch=None,
        patch_x=0, patch_y=0, patch_w=8, patch_h=8,
        prompt="p", negative_prompt="",
        strength=0.5, guidance_scale=7.0, steps=20, seed=1,
    )

    service.execute(AttachLayerToolCommand(layer=layer, tool=tool))

    assert isinstance(layer.tool, DiffusionTool)
    assert service.undo() == "Attach Tool"
    assert stack.active_layer.tool is None
    assert service.redo() == "Attach Tool"
    assert isinstance(stack.active_layer.tool, DiffusionTool)

    service.execute(DetachLayerToolCommand(layer=stack.active_layer))

    assert stack.active_layer.tool is None
    assert service.undo() == "Remove Tool"
    assert isinstance(stack.active_layer.tool, DiffusionTool)


def test_document_service_replace_layer_mask_command_accepts_bool_mask():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.full((8, 8, 4), 255, dtype=np.uint8))
    layer = stack.active_layer
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 2:5] = True

    service.execute(ReplaceLayerMaskCommand(layer=layer, mask=mask))

    assert layer.mask.data.dtype == np.float32
    assert np.all(layer.mask.data[1:3, 2:5] == 1.0)
    assert np.max(layer.mask.data) == 1.0


def test_document_service_set_layer_selection_command_initializes_canvas_shape():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.full((8, 10, 4), 255, dtype=np.uint8))
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)
    selection = np.zeros((8, 10), dtype=np.uint8)
    selection[2:5, 3:7] = 255

    service.execute(SetLayerSelectionCommand(mask=selection))

    assert stack.selection.data.shape == (8, 10)
    assert stack.selection.data.dtype == np.float32
    assert np.all(stack.selection.data[2:5, 3:7] == 1.0)
    assert service.undo() == "Set Selection"
    assert stack.selection.is_empty


def test_document_service_apply_generated_result_command():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.full((8, 8, 4), 255, dtype=np.uint8))
    layer = _diff_layer()
    stack.insert_layer(layer)
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)
    result = Image.fromarray(np.full((8, 8, 3), (200, 10, 10), dtype=np.uint8), "RGB")

    service.execute(ApplyGeneratedResultCommand(
        layer=layer,
        result_image=result,
        label="Apply Result",
    ))

    assert np.any(layer.image[:, :, 3] > 0)
    assert service.undo() == "Apply Result"
    assert np.all(stack.active_layer.image == 0)
