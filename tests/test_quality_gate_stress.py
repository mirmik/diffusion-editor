from __future__ import annotations

import sys
import threading
import time

import numpy as np
from PIL import Image
import pytest

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.engines.threaded_lifecycle import EngineTaskQueue
from diffusion_editor.workers.lama_process import LamaProcessClient
from diffusion_editor.workers.ml_process import MlProcessClient
from diffusion_editor.workers.segmentation_process import (
    SegmentationProcessClient,
)


class _Settings:
    def __init__(self):
        self.values = {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


def _poll(queue: EngineTaskQueue, timeout: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        event = queue.poll_event()
        if event is not None:
            return event
        time.sleep(0.0005)
    raise AssertionError("timed out waiting for stress event")


def test_repeated_start_open_submit_cancel_close_is_bounded():
    started = time.monotonic()
    for iteration in range(100):
        application = EditorApplication(
            settings=_Settings(),
            engines=EngineSet.create_default(),
        )
        pixels = np.full((8, 8, 4), iteration % 256, dtype=np.uint8)
        application.layer_stack.init_from_image(pixels)

        queue = EngineTaskQueue()
        operation_started = threading.Event()

        def operation(cancel):
            operation_started.set()
            cancel.wait(0.5)
            return iteration

        assert queue.submit(
            "inference",
            operation,
            name=f"quality-cancel-{iteration}",
        )
        assert operation_started.wait(0.5)
        assert queue.cancel()
        event = _poll(queue)
        assert event.error == "Operation cancelled"
        assert queue.shutdown(0.5)

        application.close()
        application.close()
        assert application.closed
        assert len(application.layer_stack.layers) == 1
    assert time.monotonic() - started < 20.0


@pytest.mark.parametrize("iteration", range(5))
def test_lama_worker_repeated_crash_restart(iteration):
    del iteration
    client = LamaProcessClient(
        python=sys.executable,
        backend="crash",
        startup_timeout=2.0,
        request_timeout=2.0,
    )
    image = Image.new("RGB", (4, 4), "red")
    mask = Image.new("L", (4, 4), 255)
    with pytest.raises(RuntimeError, match="exited with code 37"):
        client.inpaint(image, mask, threading.Event())
    client._backend = "identity"
    try:
        assert client.inpaint(
            image, mask, threading.Event()
        ).getpixel((0, 0)) == (255, 0, 0)
    finally:
        client.shutdown()


@pytest.mark.parametrize("iteration", range(5))
def test_segmentation_worker_repeated_crash_restart(iteration):
    del iteration
    client = SegmentationProcessClient(
        python=sys.executable,
        backend="crash",
        startup_timeout=2.0,
        request_timeout=2.0,
    )
    image = np.full((4, 4, 4), 255, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="exited with code 38"):
        client.segment(image, threading.Event())
    client._backend = "threshold"
    try:
        assert client.segment(image, threading.Event()).shape == (4, 4)
    finally:
        client.shutdown()


@pytest.mark.parametrize("iteration", range(5))
def test_ml_worker_repeated_crash_restart(iteration):
    del iteration
    client = MlProcessClient(
        python=sys.executable,
        backend="crash",
        startup_timeout=2.0,
        request_timeout=2.0,
    )
    with pytest.raises(RuntimeError, match="exited with code 39"):
        client.request("gpu_available", {}, threading.Event())
    client._backend = "fake"
    try:
        assert client.request(
            "gpu_available", {}, threading.Event()
        ) == {"available": False}
    finally:
        client.shutdown()
