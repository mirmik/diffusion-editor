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


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _status_coordinator(*, clock=None, update_interval=0.0):
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
        get_status=lambda: statuses[-1] if statuses else "Ready",
        clock=clock or _Clock(),
        update_interval=update_interval,
    )
    return stack, document, canvas, coordinator, statuses, previous_moves


def test_canvas_status_formats_live_context_and_preserves_callback_chain():
    stack, _document, canvas, coordinator, statuses, previous_moves = (
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
    _stack, _document, canvas, coordinator, statuses, _previous_moves = (
        _status_coordinator())
    statuses.append("Imported: source.png")

    assert statuses[-1] == "Imported: source.png"
    canvas.pointer_move(1, 2)
    assert statuses[-1].startswith("32x24 | (1,2)")
    coordinator.close()


def test_canvas_status_coalesces_motion_and_delivers_latest_position():
    clock = _Clock()
    _stack, _document, canvas, coordinator, statuses, previous_moves = (
        _status_coordinator(clock=clock, update_interval=0.04))

    canvas.pointer_move(1, 2)
    clock.advance(0.01)
    canvas.pointer_move(3, 4)
    clock.advance(0.01)
    canvas.pointer_move(5, 6)

    assert previous_moves == [(1, 2), (3, 4), (5, 6)]
    assert len(statuses) == 1
    assert "(1,2)" in statuses[0]
    assert coordinator.flush() is False

    clock.advance(0.02)
    assert coordinator.flush() is True
    assert len(statuses) == 2
    assert "(5,6)" in statuses[-1]
    coordinator.close()


def test_canvas_status_skips_unchanged_text_but_replaces_operation_message():
    clock = _Clock()
    _stack, _document, canvas, coordinator, statuses, _previous_moves = (
        _status_coordinator(clock=clock, update_interval=0.04))

    canvas.pointer_move(1, 2)
    clock.advance(0.04)
    canvas.pointer_move(1, 2)
    assert len(statuses) == 1

    statuses.append("Imported: source.png")
    clock.advance(0.04)
    canvas.pointer_move(1, 2)
    assert len(statuses) == 3
    assert statuses[-1].startswith("32x24 | (1,2)")
    coordinator.close()


def test_canvas_status_memory_counters_are_cached_by_revision(monkeypatch):
    clock = _Clock()
    stack, document, canvas, coordinator, _statuses, _previous_moves = (
        _status_coordinator(clock=clock, update_interval=0.0))
    calls = {"history": 0, "cache": 0}
    history_memory_bytes = document.memory_bytes
    cache_memory_bytes = stack.cache_memory_bytes

    def count_history():
        calls["history"] += 1
        return history_memory_bytes()

    def count_cache():
        calls["cache"] += 1
        return cache_memory_bytes()

    monkeypatch.setattr(document, "memory_bytes", count_history)
    monkeypatch.setattr(stack, "cache_memory_bytes", count_cache)

    canvas.pointer_move(1, 1)
    canvas.pointer_move(2, 2)
    assert calls == {"history": 1, "cache": 1}

    document.push_callbacks(
        "Edit",
        undo_fn=lambda: None,
        redo_fn=lambda: None,
        size_bytes=128,
    )
    canvas.pointer_move(3, 3)
    assert calls == {"history": 2, "cache": 1}

    stack.composite()
    canvas.pointer_move(4, 4)
    assert calls == {"history": 2, "cache": 2}
    coordinator.close()


def test_format_bytes_uses_compact_legacy_units():
    assert format_bytes(512) == "512B"
    assert format_bytes(1536) == "2K"
    assert format_bytes(3 * 1024 * 1024) == "3.0M"
