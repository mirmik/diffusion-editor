"""Extra command/history tests for DocumentService."""

from __future__ import annotations

import numpy as np

from diffusion_editor.commands import (
    MoveLayerCommand,
    SetLayerVisibilityCommand,
)
from diffusion_editor.document_service import DocumentService
from diffusion_editor.history import HistoryManager
from diffusion_editor.layer_stack import LayerStack


def _base_stack() -> LayerStack:
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.full((8, 8, 4), 255, dtype=np.uint8))
    stack.add_layer("Top A", np.zeros((8, 8, 4), dtype=np.uint8))
    stack.add_layer("Top B", np.zeros((8, 8, 4), dtype=np.uint8))
    return stack


def _service(stack: LayerStack) -> DocumentService:
    history = HistoryManager(stack.load_state)
    return DocumentService(stack, history, stack.load_state)


def test_set_visibility_command_undo_redo():
    stack = _base_stack()
    layer = stack.layers[0]
    service = _service(stack)

    service.execute(SetLayerVisibilityCommand(layer=layer, visible=False))
    assert layer.visible is False
    assert service.undo() == "Toggle Visibility"
    assert stack.layers[0].visible is True
    assert service.redo() == "Toggle Visibility"
    assert stack.layers[0].visible is False


def test_move_layer_command_undo_redo():
    stack = _base_stack()
    service = _service(stack)
    top_before = stack.layers[0]
    bottom_before = stack.layers[-1]

    # Move top layer to bottom at root.
    service.execute(MoveLayerCommand(layer=top_before, new_parent=None, index=len(stack.layers)))
    assert stack.layers[-1].name == top_before.name
    assert stack.layers[0].name != top_before.name

    assert service.undo() == "Move Layer"
    assert stack.layers[0].name == top_before.name
    assert stack.layers[-1].name == bottom_before.name

    assert service.redo() == "Move Layer"
    assert stack.layers[-1].name == top_before.name


def test_clear_history_resets_undo_redo_stacks():
    stack = _base_stack()
    service = _service(stack)
    layer = stack.layers[0]
    service.execute(SetLayerVisibilityCommand(layer=layer, visible=False))
    assert service.undo() == "Toggle Visibility"
    service.redo()

    service.clear_history()
    assert service.undo() is None
    assert service.redo() is None
