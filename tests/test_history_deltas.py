import random

import numpy as np
import pytest

from diffusion_editor.document.commands import (
    AddLayerCommand,
    MoveLayerCommand,
    RemoveLayerCommand,
    SetLayerNameCommand,
    SetLayerOpacityCommand,
    SetLayerVisibilityCommand,
)
from diffusion_editor.document.document_service import (
    DocumentService,
    _snapshots_semantically_equal,
)
from diffusion_editor.document.history import HistoryManager
from diffusion_editor.document.layer_stack import LayerStack


def _document(
        width=32,
        height=24,
        *,
        limit=64 * 1024 * 1024,
        max_entries=50):
    stack = LayerStack(tile_size=8)
    stack.init_from_image(np.zeros((height, width, 4), dtype=np.uint8))
    history = HistoryManager(
        stack.load_state,
        max_entries=max_entries,
        max_memory_bytes=limit,
    )
    return stack, history, DocumentService(stack, history, stack.load_state)


def test_metadata_delta_cost_is_independent_of_canvas_pixels():
    stack, history, document = _document(2048, 2048)
    layer = stack.active_layer
    stack.serialize_state = lambda: (_ for _ in ()).throw(
        AssertionError("metadata command serialized the document"))

    document.execute(SetLayerOpacityCommand(layer, 0.5))

    assert history.memory_bytes() <= 16
    assert document.undo() == "Set Opacity"
    assert layer.opacity == 1.0


@pytest.mark.parametrize("canvas_size", [2048, 4096])
def test_multilayer_metadata_benchmark_reports_constant_entry_cost(
        canvas_size):
    stack, history, document = _document(canvas_size, canvas_size)
    overlay = np.zeros((8, 8, 4), dtype=np.uint8)
    document.execute(AddLayerCommand("overlay-a", overlay))
    document.execute(AddLayerCommand("overlay-b", overlay))
    document.clear_history()
    layer = stack.active_layer
    stack.serialize_state = lambda: (_ for _ in ()).throw(
        AssertionError("metadata command serialized the document"))

    document.execute(SetLayerOpacityCommand(layer, 0.5))

    # This assertion is the benchmark report: one opacity delta is 16 bytes
    # for both 2K and 4K multilayer documents.
    assert history.memory_bytes() == 16


def test_continuous_opacity_updates_coalesce_into_one_undo_step():
    stack, history, document = _document()
    layer = stack.active_layer
    for opacity in (0.8, 0.6, 0.4):
        document.execute(SetLayerOpacityCommand(layer, opacity))

    assert layer.opacity == 0.4
    assert document.undo() == "Set Opacity"
    assert layer.opacity == 1.0
    assert not history.can_undo
    assert document.redo() == "Set Opacity"
    assert layer.opacity == 0.4


def test_tree_and_metadata_delta_round_trip_preserves_document_bytes():
    stack, history, document = _document()
    initial = stack.serialize_state()
    document.execute(AddLayerCommand("A"))
    layer_a = stack.active_layer
    document.execute(SetLayerNameCommand(layer_a, "Alpha"))
    document.execute(SetLayerOpacityCommand(layer_a, 0.7))
    document.execute(SetLayerVisibilityCommand(layer_a, False))
    document.execute(AddLayerCommand("B"))
    layer_b = stack.active_layer
    document.execute(MoveLayerCommand(layer_b, layer_a, 0))
    document.execute(RemoveLayerCommand(layer_b))
    final = stack.serialize_state()

    while history.can_undo:
        document.undo()
    assert _snapshots_semantically_equal(initial, stack.serialize_state())
    while history.can_redo:
        document.redo()
    assert _snapshots_semantically_equal(final, stack.serialize_state())


def test_random_metadata_execute_undo_redo_round_trip_preserves_state():
    rng = random.Random(910)
    stack, history, document = _document(96, 64, max_entries=100)
    for name in ("A", "B", "C"):
        document.execute(AddLayerCommand(name))
    document.clear_history()
    layers = list(stack.layers)
    initial = stack.serialize_state()

    for step in range(80):
        layer = rng.choice(layers)
        operation = rng.randrange(3)
        if operation == 0:
            document.execute(SetLayerOpacityCommand(layer, rng.random()))
        elif operation == 1:
            document.execute(SetLayerVisibilityCommand(layer, bool(step % 2)))
        else:
            document.execute(SetLayerNameCommand(layer, f"layer-{step}"))
    final = stack.serialize_state()

    while history.can_undo:
        document.undo()
    assert _snapshots_semantically_equal(initial, stack.serialize_state())
    while history.can_redo:
        document.redo()
    assert _snapshots_semantically_equal(final, stack.serialize_state())


def test_history_admission_never_retains_an_oversized_entry():
    history = HistoryManager(lambda _snapshot: None, max_memory_bytes=8)
    history.push_callbacks(
        "too large", lambda: None, lambda: None, size_bytes=9)
    assert history.memory_bytes() == 0
    assert not history.can_undo

    history.push_callbacks(
        "small", lambda: None, lambda: None, size_bytes=8)
    assert history.memory_bytes() == 8
    assert history.can_undo
