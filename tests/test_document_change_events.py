import logging

import numpy as np

from diffusion_editor.document.change_event import DocumentChangeKind
from diffusion_editor.document.layer_stack import LayerStack


def _stack() -> LayerStack:
    stack = LayerStack(tile_size=8)
    stack.init_from_image(np.zeros((6, 8, 4), dtype=np.uint8))
    return stack


def test_subscriptions_detach_in_any_order_and_unsubscribe_is_idempotent():
    stack = _stack()
    first = []
    second = []
    first_handle = stack.subscribe(first.append)
    second_handle = stack.subscribe(second.append)

    stack.set_layer_name(stack.active_layer, "One")
    first_handle.unsubscribe()
    first_handle.unsubscribe()
    stack.set_layer_name(stack.active_layer, "Two")
    second_handle.close()

    assert [event.revision for event in first] == [2]
    assert [event.revision for event in second] == [2, 3]
    assert first[0].kind == DocumentChangeKind.METADATA


def test_failing_subscriber_does_not_suppress_remaining_subscribers(caplog):
    stack = _stack()
    received = []

    def fail(_event):
        raise RuntimeError("subscriber exploded")

    stack.subscribe(fail)
    stack.subscribe(received.append)
    with caplog.at_level(logging.ERROR):
        stack.set_layer_name(stack.active_layer, "Still committed")

    assert len(received) == 1
    assert received[0].kind == DocumentChangeKind.METADATA
    assert "subscriber exploded" in caplog.text


def test_pixel_event_is_immutable_revisioned_and_clips_local_dirty_rect():
    stack = _stack()
    layer = stack.active_layer
    events = []
    stack.subscribe(events.append)

    event = stack.publish_change(
        DocumentChangeKind.PIXELS,
        layers=(layer,),
        dirty_rect=(-4, 2, 20, 10),
    )

    assert event is events[0]
    assert event.revision == 2
    assert event.layer_ids == (layer.id,)
    assert event.dirty_rect == (0, 2, 8, 6)
    try:
        event.revision = 99
    except AttributeError:
        pass
    else:
        raise AssertionError("document change events must be immutable")


def test_each_semantic_mutation_advances_exactly_one_revision():
    stack = _stack()
    initial = stack.revision
    stack.set_layer_name(stack.active_layer, "Renamed")
    assert stack.revision == initial + 1
    stack.set_layer_name(stack.active_layer, "Renamed")
    assert stack.revision == initial + 1
    stack.active_layer = stack.active_layer
    assert stack.revision == initial + 1
