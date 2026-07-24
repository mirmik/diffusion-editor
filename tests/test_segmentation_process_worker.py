from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pytest

from diffusion_editor.engines.segmentation_engine import SegmentationEngine
from diffusion_editor.generation.types import (
    SegmentationRequest,
    SegmentationResult,
)
from diffusion_editor.workers.segmentation_process import (
    SegmentationProcessClient,
)
from diffusion_editor.workers.segmentation_protocol import (
    SegmentationProtocolError,
)


def _image() -> np.ndarray:
    image = np.full((4, 6, 4), 255, dtype=np.uint8)
    image[:, :3, :3] = 0
    return image


def _client(backend: str, timeout: float = 2.0) -> SegmentationProcessClient:
    return SegmentationProcessClient(
        python=sys.executable,
        backend=backend,
        startup_timeout=timeout,
        request_timeout=timeout,
    )


def test_threshold_worker_round_trip_and_progress_keep_main_gil_disabled():
    client = _client("threshold")
    progress: list[str] = []
    before = sys._is_gil_enabled()
    try:
        mask = client.segment(
            _image(),
            threading.Event(),
            on_progress=progress.append,
        )
        assert mask.shape == (4, 6)
        assert np.all(mask[:, :3] == 255)
        assert np.all(mask[:, 3:] == 0)
        assert progress == ["Running segmentation"]
        assert client.is_running
        assert sys._is_gil_enabled() is before is False
        assert "rembg" not in sys.modules
        assert "cv2" not in sys.modules
    finally:
        client.shutdown()
    assert not client.is_running


def test_cancel_terminates_hung_segmentation_worker():
    client = _client("hang", timeout=5.0)
    cancel = threading.Event()
    timer = threading.Timer(0.1, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            client.segment(_image(), cancel)
    finally:
        timer.cancel()
        client.shutdown()
    assert time.monotonic() - started < 2.0
    assert not client.is_running


@pytest.mark.parametrize(
    ("backend", "error"),
    [
        ("crash", "exited with code 38"),
        ("malformed", "malformed JSON"),
    ],
)
def test_segmentation_crash_and_malformed_response_restart(
    backend,
    error,
):
    client = _client(backend)
    with pytest.raises(
        (RuntimeError, SegmentationProtocolError),
        match=error,
    ):
        client.segment(_image(), threading.Event())
    assert not client.is_running

    client._backend = "threshold"
    try:
        assert client.segment(
            _image(),
            threading.Event(),
        ).shape == (4, 6)
    finally:
        client.shutdown()


def test_segmentation_timeout_stops_worker():
    client = _client("hang", timeout=0.15)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        client.segment(_image(), threading.Event())
    assert time.monotonic() - started < 2.0
    assert not client.is_running


def test_segmentation_engine_inverts_worker_mask_once():
    class FakeClient:
        is_running = False

        def segment(self, _image, _cancel, on_progress=None):
            if on_progress is not None:
                on_progress("fake")
            return np.array([[0, 255]], dtype=np.uint8)

        def shutdown(self, _timeout=1.0):
            pass

    engine = SegmentationEngine(client=FakeClient())
    assert engine.submit_request(
        SegmentationRequest(image=_image(), invert=True)
    )
    deadline = time.monotonic() + 1.0
    event = None
    while event is None and time.monotonic() < deadline:
        event = engine.poll_event()
        time.sleep(0.001)

    assert event is not None
    assert isinstance(event.result, SegmentationResult)
    assert np.array_equal(
        event.result.mask,
        np.array([[255, 0]], dtype=np.uint8),
    )
    assert engine.poll_event() is None


def test_segmentation_engine_shutdown_terminates_hung_subprocess():
    client = _client("hang", timeout=5.0)
    engine = SegmentationEngine(client=client)
    assert engine.submit_request(
        SegmentationRequest(image=_image(), invert=False)
    )

    deadline = time.monotonic() + 1.0
    while not client.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.is_running

    engine.shutdown(timeout=1.0)

    assert not engine.is_busy
    assert not client.is_running
