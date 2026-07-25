import numpy as np

from diffusion_editor.app.canvas_status import (
    CanvasStatusCoordinator,
    format_bytes,
)
from diffusion_editor.canvas.brush import BrushToolMode
from diffusion_editor.canvas.editor_canvas_controller import (
    EditorCanvasController,
)
from diffusion_editor.document.document_service import DocumentService
from diffusion_editor.document.history import HistoryManager
from diffusion_editor.document.layer_stack import LayerStack


def _status_coordinator():
    image = np.zeros((24, 32, 4), dtype=np.uint8)
    stack = LayerStack(tile_size=8)
    stack.init_from_image(image)
    history = HistoryManager(lambda _snapshot: None)
    document = DocumentService(stack, history, lambda _snapshot: None)
    canvas = EditorCanvasController(
        stack,
        gpu_compositing=False,
        set_image=lambda _image: None,
        set_overlay=lambda _overlay: None,
    )
    statuses = []
    previous_moves = []
    canvas.on_mouse_moved = lambda x, y: previous_moves.append((x, y))
    coordinator = CanvasStatusCoordinator(
        stack,
        document,
        canvas,
        statuses.append,
    )
    return stack, canvas, coordinator, statuses, previous_moves


def test_canvas_status_formats_live_context_and_preserves_callback_chain():
    stack, canvas, coordinator, statuses, previous_moves = (
        _status_coordinator())
    canvas.brush.set_size(37)
    canvas.set_brush_tool(BrushToolMode.SMUDGE)
    stack.active_layer.name = "Retouch"

    canvas.pointer_move(7, 9)

    assert previous_moves == [(7, 9)]
    assert statuses == [
        "32x24 | (7,9) | Retouch | smudge:37px | Hist:0B Cache:0B"
    ]

    canvas.pointer_move(40, 30)
    assert statuses[-1] == (
        "32x24 | Retouch | smudge:37px | Hist:0B Cache:0B")

    previous = coordinator._previous_mouse_moved
    coordinator.close()
    assert canvas.on_mouse_moved is previous


def test_canvas_status_replaces_operation_text_only_after_pointer_motion():
    _stack, canvas, coordinator, statuses, _previous_moves = (
        _status_coordinator())
    statuses.append("Imported: source.png")

    assert statuses[-1] == "Imported: source.png"
    canvas.pointer_move(1, 2)
    assert statuses[-1].startswith("32x24 | (1,2)")
    coordinator.close()


def test_format_bytes_uses_compact_legacy_units():
    assert format_bytes(512) == "512B"
    assert format_bytes(1536) == "2K"
    assert format_bytes(3 * 1024 * 1024) == "3.0M"
