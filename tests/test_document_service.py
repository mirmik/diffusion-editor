"""Tests for DocumentService and command bus integration."""

import numpy as np
from PIL import Image

from diffusion_editor.document_service import (
    CallbackCommand, CommandBus, DocumentService,
)
from diffusion_editor.commands import (
    AddLayerCommand, SetLayerOpacityCommand, FlattenLayersCommand,
    SnapshotCallbackCommand, ClearLayerMaskCommand, SetManualPatchRectCommand,
    ClearManualPatchRectCommand, ReplaceLayerMaskCommand,
    ApplyGeneratedResultCommand,
)
from diffusion_editor.layer import DiffusionLayer
from diffusion_editor.history import HistoryManager
from diffusion_editor.layer_stack import LayerStack


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
    layer = DiffusionLayer(
        name="Diff",
        width=8,
        height=8,
        source_patch=None,
        patch_x=0,
        patch_y=0,
        patch_w=8,
        patch_h=8,
        prompt="",
        negative_prompt="",
        strength=0.5,
        guidance_scale=7.0,
        steps=20,
        seed=1,
    )
    layer.mask[2:4, 2:4] = 255
    stack.insert_layer(layer)
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)

    service.execute(ClearLayerMaskCommand(layer=layer, label="Clear Mask"))

    assert not layer.has_mask()
    assert service.undo() == "Clear Mask"
    restored = stack.active_layer
    assert restored is not None
    assert hasattr(restored, "has_mask")
    assert restored.has_mask()


def test_document_service_manual_patch_rect_commands():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.full((8, 8, 4), 255, dtype=np.uint8))
    layer = DiffusionLayer(
        name="Diff",
        width=8,
        height=8,
        source_patch=None,
        patch_x=0,
        patch_y=0,
        patch_w=8,
        patch_h=8,
        prompt="",
        negative_prompt="",
        strength=0.5,
        guidance_scale=7.0,
        steps=20,
        seed=1,
    )
    stack.insert_layer(layer)
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)

    service.execute(SetManualPatchRectCommand(
        layer=layer,
        rect=(1, 2, 5, 6),
        label="Set Rect",
    ))
    assert layer.manual_patch_rect == (1, 2, 5, 6)

    service.execute(ClearManualPatchRectCommand(layer=layer, label="Clear Rect"))
    assert layer.manual_patch_rect is None

    assert service.undo() == "Clear Rect"
    assert stack.active_layer is not None
    assert stack.active_layer.manual_patch_rect == (1, 2, 5, 6)


def test_document_service_replace_layer_mask_command():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.full((8, 8, 4), 255, dtype=np.uint8))
    layer = DiffusionLayer(
        name="Diff",
        width=8,
        height=8,
        source_patch=None,
        patch_x=0,
        patch_y=0,
        patch_w=8,
        patch_h=8,
        prompt="",
        negative_prompt="",
        strength=0.5,
        guidance_scale=7.0,
        steps=20,
        seed=1,
    )
    stack.insert_layer(layer)
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[0:2, 0:2] = 255

    service.execute(ReplaceLayerMaskCommand(layer=layer, mask=mask))

    assert np.array_equal(layer.mask, mask)
    assert service.undo() == "Apply Segmentation Mask"


def test_document_service_apply_generated_result_command():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.full((8, 8, 4), 255, dtype=np.uint8))
    layer = DiffusionLayer(
        name="Diff",
        width=8,
        height=8,
        source_patch=None,
        patch_x=0,
        patch_y=0,
        patch_w=8,
        patch_h=8,
        prompt="",
        negative_prompt="",
        strength=0.5,
        guidance_scale=7.0,
        steps=20,
        seed=1,
    )
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
