from __future__ import annotations

import sys
import threading
import time

from PIL import Image
import pytest

from diffusion_editor.engines.lama_engine import LamaEngine
from diffusion_editor.generation.types import LamaRequest
from diffusion_editor.workers.lama_process import LamaProcessClient
from diffusion_editor.workers.lama_protocol import LamaProtocolError


def _image(color: str = "red") -> Image.Image:
    return Image.new("RGB", (3, 2), color)


def _mask() -> Image.Image:
    return Image.new("L", (3, 2), 255)


def _client(backend: str, timeout: float = 2.0) -> LamaProcessClient:
    return LamaProcessClient(
        python=sys.executable,
        backend=backend,
        startup_timeout=timeout,
        request_timeout=timeout,
    )


def test_identity_worker_round_trip_keeps_main_process_gil_disabled():
    client = _client("identity")
    before = sys._is_gil_enabled()
    try:
        result = client.inpaint(_image(), _mask(), threading.Event())
        assert result.size == (3, 2)
        assert result.getpixel((0, 0)) == (255, 0, 0)
        assert client.is_running
        assert sys._is_gil_enabled() is before is False
        assert "simple_lama_inpainting" not in sys.modules
        assert "cv2" not in sys.modules
    finally:
        client.shutdown()
    assert not client.is_running


def test_cancel_terminates_hung_worker_with_a_bounded_error():
    client = _client("hang", timeout=5.0)
    cancel = threading.Event()
    timer = threading.Timer(0.1, cancel.set)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            client.inpaint(_image(), _mask(), cancel)
    finally:
        timer.cancel()
        client.shutdown()
    assert time.monotonic() - started < 2.0
    assert not client.is_running


@pytest.mark.parametrize(
    ("backend", "error"),
    [
        ("crash", "exited with code 37"),
        ("malformed", "malformed JSON"),
    ],
)
def test_crash_and_malformed_response_are_bounded_and_restartable(
    backend,
    error,
):
    client = _client(backend)
    with pytest.raises((RuntimeError, LamaProtocolError), match=error):
        client.inpaint(_image(), _mask(), threading.Event())
    assert not client.is_running

    client._backend = "identity"
    try:
        assert client.inpaint(
            _image("blue"),
            _mask(),
            threading.Event(),
        ).getpixel((0, 0)) == (0, 0, 255)
    finally:
        client.shutdown()


def test_request_timeout_stops_worker():
    client = _client("hang", timeout=0.15)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="timed out"):
        client.inpaint(_image(), _mask(), threading.Event())
    assert time.monotonic() - started < 2.0
    assert not client.is_running


def test_engine_shutdown_terminates_a_hung_subprocess():
    client = _client("hang", timeout=5.0)
    engine = LamaEngine(client=client)
    assert engine.submit_request(LamaRequest(image=_image(), mask_image=_mask()))

    deadline = time.monotonic() + 1.0
    while not client.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.is_running

    engine.shutdown(timeout=1.0)

    assert not engine.is_busy
    assert not client.is_running
