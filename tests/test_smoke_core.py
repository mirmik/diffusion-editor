"""Core smoke tests that should pass on CI without GPU/UI runtime."""

from __future__ import annotations

import numpy as np

from diffusion_editor.commands import AddLayerCommand
from diffusion_editor.document_service import DocumentService
from diffusion_editor.history import HistoryManager
from diffusion_editor.layer_stack import LayerStack


def test_core_startup_smoke():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.full((4, 4, 4), 255, dtype=np.uint8))
    history = HistoryManager(stack.load_state)
    service = DocumentService(stack, history, stack.load_state)

    service.execute(AddLayerCommand(name="Layer 1"))
    assert len(stack.layers) == 2
    assert service.undo() == "New Layer"
    assert len(stack.layers) == 1
