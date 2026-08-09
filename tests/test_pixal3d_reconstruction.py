from __future__ import annotations

import json
import math
import sys
import threading
import time
from pathlib import Path

from PIL import Image
import pytest

from diffusion_editor.engines.reconstruction_engine import ReconstructionEngine
from diffusion_editor.generation.reconstruction_controller import (
    ReconstructionController,
)
from diffusion_editor.workers.pixal3d_process import Pixal3DProcessClient
from diffusion_editor.generation.types import (
    ReconstructionParameters,
    ReconstructionStage,
    ReconstructionStageStatus,
)


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
parser.add_argument('--manual-fov')
parser.add_argument('--low_vram', action='store_true')
args = parser.parse_args()
if Path(args.model_path, 'sleep').exists():
    time.sleep(30)
Path(args.output).write_bytes(b'glTF-fake')
"""


_FAKE_STAGED_RUNNER = r"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--pixal3d-root')
parser.add_argument('--image')
parser.add_argument('--output')
parser.add_argument('--events')
parser.add_argument('--model_path')
parser.add_argument('--target-stage')
parser.add_argument('--resolution')
parser.add_argument('--steps')
parser.add_argument('--seed')
parser.add_argument('--decimation-target')
parser.add_argument('--texture-size')
parser.add_argument('--manual-fov')
parser.add_argument('--low_vram', action='store_true')
args = parser.parse_args()
Path(args.output).with_name('parameters.json').write_text(json.dumps(vars(args)))
preview = Path(args.output).with_name('sparse.glb')
preview.write_bytes(b'glTF-preview')
with Path(args.events).open('w') as stream:
    stream.write(json.dumps({
        'stage': 'sparse_occupancy',
        'status': 'running',
        'progress': 1,
        'total': 2,
    }) + '\n')
    stream.write(json.dumps({
        'stage': 'sparse_occupancy',
        'status': 'ready',
        'progress': 2,
        'total': 2,
        'artifact_path': str(preview),
        'preview_kind': 'mesh',
    }) + '\n')
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
        runner_path=root / "inference.py",
        staged=False,
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


def test_staged_client_stops_at_target_and_streams_preview(tmp_path):
    root = tmp_path / "pixal3d"
    root.mkdir()
    runner = tmp_path / "staged.py"
    runner.write_text(_FAKE_STAGED_RUNNER)
    model = tmp_path / "model"
    model.mkdir()
    client = Pixal3DProcessClient(
        python=sys.executable,
        root=root,
        model_path=model,
        runner_path=runner,
        staged=True,
        steps=2,
    )
    events = []

    output, source = client.generate(
        Image.new("RGB", (4, 3)),
        7,
        threading.Event(),
        target_stage=ReconstructionStage.SPARSE_OCCUPANCY,
        on_event=events.append,
    )

    assert source.is_file()
    assert not output.exists()
    assert [event.stage for event in events] == [
        ReconstructionStage.SOURCE_IMAGE,
        ReconstructionStage.SPARSE_OCCUPANCY,
        ReconstructionStage.SPARSE_OCCUPANCY,
    ]
    assert events[-1].status is ReconstructionStageStatus.READY
    assert events[-1].artifact is not None
    assert Path(events[-1].artifact.path).read_bytes() == b"glTF-preview"
    assert {item.stage for item in client.artifacts} == {
        ReconstructionStage.SOURCE_IMAGE,
        ReconstructionStage.SPARSE_OCCUPANCY,
    }
    client.shutdown()


def test_staged_client_passes_generation_parameter_snapshot(tmp_path):
    root = tmp_path / "pixal3d"
    root.mkdir()
    runner = tmp_path / "staged.py"
    runner.write_text(_FAKE_STAGED_RUNNER)
    model = tmp_path / "model"
    model.mkdir()
    client = Pixal3DProcessClient(
        python=sys.executable,
        root=root,
        model_path=model,
        runner_path=runner,
        staged=True,
    )
    parameters = ReconstructionParameters(
        seed=123,
        steps=7,
        resolution=1280,
        manual_fov_degrees=45.0,
        decimation_target=350_000,
        texture_size=4096,
        low_vram=False,
    )

    output, _source = client.generate(
        Image.new("RGB", (4, 3)),
        parameters.seed,
        threading.Event(),
        parameters=parameters,
        target_stage=ReconstructionStage.SPARSE_OCCUPANCY,
    )
    captured = json.loads(output.with_name("parameters.json").read_text())

    assert captured["seed"] == "123"
    assert captured["steps"] == "7"
    assert captured["resolution"] == "1280"
    assert captured["decimation_target"] == "350000"
    assert captured["texture_size"] == "4096"
    assert float(captured["manual_fov"]) == pytest.approx(math.pi / 4)
    assert captured["low_vram"] is False
    client.shutdown()


def test_reconstruction_controller_reports_worker_result(tmp_path):
    engine = ReconstructionEngine(_client(tmp_path))
    controller = ReconstructionController(engine)

    started = controller.start(Image.new("RGBA", (4, 4)), seed=9)
    assert started.status and "Pixal3D" in started.status
    event = None
    stage_events = []
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        polled = controller.poll()
        if polled is not None and polled.stage_event is not None:
            stage_events.append(polled.stage_event)
        elif polled is not None:
            event = polled
            break
        time.sleep(0.01)

    assert event is not None and event.result is not None
    assert event.result.glb_path.endswith("model.glb")
    assert stage_events[0].stage.value == "source_image"
    engine.shutdown()


def _capture_error(errors, callback) -> None:
    try:
        callback()
    except Exception as exc:
        errors.append(exc)
