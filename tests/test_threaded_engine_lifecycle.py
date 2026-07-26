from __future__ import annotations

import threading
import time

from PIL import Image

from diffusion_editor.engines.lama_engine import LamaEngine
from diffusion_editor.engines.threaded_lifecycle import (
    EngineTaskQueue,
    SingleWorkerEventQueue,
)
from diffusion_editor.generation.types import LamaRequest, LamaResult


class _FakeLamaClient:
    def __init__(self, inference):
        self._inference = inference
        self.shutdown_calls = 0

    def inpaint(self, image, mask, _cancel):
        return self._inference(image, mask)

    def shutdown(self, _timeout=1.0):
        self.shutdown_calls += 1


def _poll(queue: EngineTaskQueue, timeout: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        event = queue.poll_event()
        if event is not None:
            return event
        time.sleep(0.001)
    raise AssertionError("timed out waiting for engine event")


def test_concurrent_submit_accepts_exactly_one_task():
    queue = EngineTaskQueue()
    started = threading.Event()
    release = threading.Event()
    barrier = threading.Barrier(17)
    accepted: list[bool] = []
    accepted_lock = threading.Lock()

    def operation(_cancel):
        started.set()
        release.wait(1.0)
        return "done"

    def submit():
        barrier.wait()
        result = queue.submit("inference", operation, name="race-test")
        with accepted_lock:
            accepted.append(result)

    contenders = [threading.Thread(target=submit) for _ in range(16)]
    for contender in contenders:
        contender.start()
    barrier.wait()
    for contender in contenders:
        contender.join()

    assert sum(accepted) == 1
    assert started.wait(1.0)
    release.set()
    assert _poll(queue).result == "done"


def test_completion_and_exception_are_immutable_single_delivery_events():
    queue = EngineTaskQueue()

    assert queue.submit(
        "load",
        lambda _cancel: "model",
        meta="path",
        name="success-test",
    )
    event = _poll(queue)
    assert (event.task_type, event.result, event.error, event.meta) == (
        "load",
        "model",
        None,
        "path",
    )
    assert queue.poll_event() is None

    def fail(_cancel):
        raise ValueError("broken model")

    assert queue.submit("load", fail, name="failure-test")
    event = _poll(queue)
    assert event.result is None
    assert event.error == "broken model"
    assert queue.poll_event() is None


def test_terminal_event_is_visible_only_after_worker_slot_is_released():
    queue = EngineTaskQueue()

    assert queue.submit("load", lambda _cancel: "model", name="load-test")
    event = _poll(queue)

    assert event.result == "model"
    assert not queue.is_busy
    assert queue.submit(
        "inference",
        lambda _cancel: "image",
        name="inference-test",
    )
    assert _poll(queue).result == "image"


def test_back_to_back_terminal_events_cannot_be_inverted():
    queue = SingleWorkerEventQueue[str]()
    first_put_started = threading.Event()
    release_first_put = threading.Event()
    second_operation_started = threading.Event()
    original_put = queue._events.put

    def controlled_put(queued_event):
        event, is_terminal = queued_event
        if is_terminal and event == "first":
            first_put_started.set()
            assert release_first_put.wait(1.0)
        original_put(queued_event)

    queue._events.put = controlled_put

    assert queue.submit(
        lambda _cancel: "first",
        name="first-terminal-order",
    )
    assert first_put_started.wait(1.0)

    accepted: list[bool] = []

    def submit_second():
        def operation(_cancel):
            second_operation_started.set()
            return "second"

        accepted.append(queue.submit(
            operation,
            name="second-terminal-order",
        ))

    contender = threading.Thread(target=submit_second)
    contender.start()

    # The first terminal Queue.put is deliberately paused. The lifecycle lock
    # must keep a second submit from starting until that event is enqueued.
    assert not second_operation_started.wait(0.05)
    release_first_put.set()
    contender.join(1.0)
    assert not contender.is_alive()
    assert accepted == [False]
    assert not second_operation_started.is_set()
    assert queue.is_busy
    assert queue.poll_event() == "first"
    assert not queue.is_busy

    assert queue.submit(
        lambda _cancel: "second",
        name="second-terminal-order-after-poll",
    )
    deadline = time.monotonic() + 1.0
    second = None
    while second is None and time.monotonic() < deadline:
        second = queue.poll_event()
        time.sleep(0.001)
    assert second == "second"
    assert queue.drain_events() == ()


def test_cancel_is_cooperative_and_publishes_no_result():
    queue = EngineTaskQueue()
    started = threading.Event()

    def operation(cancel):
        started.set()
        assert cancel.wait(1.0)
        return "must be discarded"

    assert queue.submit("inference", operation, name="cancel-test")
    assert started.wait(1.0)
    assert queue.cancel()
    event = _poll(queue)
    assert event.result is None
    assert event.error == "Operation cancelled"
    assert not queue.cancel()


def test_shutdown_timeout_does_not_unload_live_engine_state():
    started = threading.Event()
    release = threading.Event()

    def inference(_image, _mask):
        started.set()
        release.wait(1.0)
        return Image.new("RGB", (1, 1))

    client = _FakeLamaClient(inference)
    engine = LamaEngine(client=client)
    request = LamaRequest(
        image=Image.new("RGB", (1, 1)),
        mask_image=Image.new("L", (1, 1)),
    )
    assert engine.submit_request(request)
    assert started.wait(1.0)

    engine.shutdown(timeout=0.001)
    assert client.shutdown_calls == 0
    assert engine.is_busy

    release.set()
    deadline = time.monotonic() + 1.0
    while engine.is_busy and time.monotonic() < deadline:
        time.sleep(0.001)
    engine.shutdown()
    assert client.shutdown_calls == 1


def test_lama_result_is_published_once_through_the_queue():
    expected = Image.new("RGB", (2, 2), "red")
    engine = LamaEngine(
        client=_FakeLamaClient(lambda _image, _mask: expected)
    )
    request = LamaRequest(
        image=Image.new("RGB", (2, 2)),
        mask_image=Image.new("L", (2, 2)),
    )

    assert engine.submit_request(request)
    deadline = time.monotonic() + 1.0
    event = None
    while event is None and time.monotonic() < deadline:
        event = engine.poll_event()
        time.sleep(0.001)

    assert event is not None
    assert isinstance(event.result, LamaResult)
    assert event.result.image is expected
    assert engine.poll_event() is None


def test_parallel_python_stress_has_no_lost_or_duplicated_events():
    queue = EngineTaskQueue()
    seen: list[int] = []

    for value in range(500):
        while not queue.submit(
            "inference",
            lambda _cancel, item=value: item,
            name=f"stress-{value}",
        ):
            time.sleep(0)
        seen.append(_poll(queue).result)

    assert seen == list(range(500))
