from __future__ import annotations

import sys
import threading
import time

from PIL import Image

from diffusion_editor.engines.reconstruction_engine import ReconstructionEngine
from diffusion_editor.generation.reconstruction_controller import (
    ReconstructionController,
)
from diffusion_editor.workers.pixal3d_process import Pixal3DProcessClient


_FAKE_INFERENCE = r"""
import argparse
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('--image')
parser.add_argument('--output')
parser.add_argument('--model_path')
parser.add_argument('--resolution')
parser.add_argument('--steps')
parser.add_argument('--seed')
parser.add_argument('--decimation-target')
parser.add_argument('--texture-size')
parser.add_argument('--low_vram', action='store_true')
args = parser.parse_args()
if Path(args.model_path, 'sleep').exists():
    time.sleep(30)
Path(args.output).write_bytes(b'glTF-fake')
"""


def _client(tmp_path, *, sleeping: bool = False) -> Pixal3DProcessClient:
    root = tmp_path / "pixal3d"
    root.mkdir()
    (root / "inference.py").write_text(_FAKE_INFERENCE)
    model = tmp_path / "model"
    model.mkdir()
    if sleeping:
        (model / "sleep").touch()
    return Pixal3DProcessClient(
        python=sys.executable,
        root=root,
        model_path=model,
        resolution=1024,
        steps=1,
    )


def test_pixal3d_client_publishes_temporary_glb_and_cleans_it(tmp_path):
    client = _client(tmp_path)
    output, source = client.generate(
        Image.new("RGBA", (8, 6), (10, 20, 30, 255)),
        42,
        threading.Event(),
    )

    assert output.read_bytes() == b"glTF-fake"
    assert source.is_file()
    artifact_root = output.parent
    client.shutdown()
    assert not artifact_root.exists()


def test_pixal3d_client_terminates_cancelled_process(tmp_path):
    client = _client(tmp_path, sleeping=True)
    cancel = threading.Event()
    errors = []

    thread = threading.Thread(
        target=lambda: _capture_error(
            errors,
            lambda: client.generate(Image.new("RGB", (2, 2)), 1, cancel),
        )
    )
    thread.start()
    time.sleep(0.2)
    cancel.set()
    thread.join(timeout=5.0)

    assert not thread.is_alive()
    assert errors and "cancelled" in str(errors[0]).lower()
    client.shutdown()


def test_reconstruction_controller_reports_worker_result(tmp_path):
    engine = ReconstructionEngine(_client(tmp_path))
    controller = ReconstructionController(engine)

    started = controller.start(Image.new("RGBA", (4, 4)), seed=9)
    assert started.status and "Pixal3D" in started.status
    event = None
    deadline = time.monotonic() + 5.0
    while event is None and time.monotonic() < deadline:
        event = controller.poll()
        time.sleep(0.01)

    assert event is not None and event.result is not None
    assert event.result.glb_path.endswith("model.glb")
    engine.shutdown()


def _capture_error(errors, callback) -> None:
    try:
        callback()
    except Exception as exc:
        errors.append(exc)
