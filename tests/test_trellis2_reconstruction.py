from __future__ import annotations

import json
import sys
import threading

from PIL import Image

from diffusion_editor.engines.reconstruction_engine import ReconstructionEngine
from diffusion_editor.generation.types import (
    ReconstructionBackend,
    ReconstructionParameters,
    ReconstructionRequest,
    ReconstructionStage,
)
from diffusion_editor.workers.trellis2_process import Trellis2ProcessClient


_FAKE_RUNNER = r"""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--trellis2-root')
parser.add_argument('--image')
parser.add_argument('--output')
parser.add_argument('--events')
parser.add_argument('--model-path')
parser.add_argument('--target-stage')
parser.add_argument('--resolution')
parser.add_argument('--steps')
parser.add_argument('--seed')
parser.add_argument('--decimation-target')
parser.add_argument('--texture-size')
parser.add_argument('--low-vram', action='store_true')
args = parser.parse_args()
output = Path(args.output)
captured = {
    **vars(args),
    'attention_backend': os.environ.get('ATTN_BACKEND'),
    'sparse_attention_backend': os.environ.get('SPARSE_ATTN_BACKEND'),
}
output.with_name('parameters.json').write_text(json.dumps(captured))
output.with_name('preprocessed.png').write_bytes(b'png')
preview = output.with_name('sparse.glb')
preview.write_bytes(b'sparse')
if args.target_stage == 'final_mesh':
    output.write_bytes(b'glTF-trellis2')
with Path(args.events).open('w') as stream:
    stream.write(json.dumps({
        'stage': 'sparse_occupancy',
        'status': 'ready',
        'artifact_path': str(preview),
        'preview_kind': 'mesh',
    }) + '\n')
    if args.target_stage == 'final_mesh':
        stream.write(json.dumps({
            'stage': 'final_mesh',
            'status': 'ready',
            'artifact_path': str(output),
            'preview_kind': 'mesh',
        }) + '\n')
"""

_FAILING_RUNNER = r"""
import sys

print('synthetic TRELLIS failure')
raise SystemExit(7)
"""


def test_trellis2_client_uses_its_runtime_and_common_stage_protocol(tmp_path):
    root = tmp_path / "trellis2"
    model = root / "models" / "TRELLIS.2-4B"
    model.mkdir(parents=True)
    runner = tmp_path / "runner.py"
    runner.write_text(_FAKE_RUNNER)
    client = Trellis2ProcessClient(
        python=sys.executable,
        root=root,
        model_path=model,
        runner_path=runner,
    )
    events = []
    parameters = ReconstructionParameters(
        backend=ReconstructionBackend.TRELLIS2,
        seed=77,
        steps=9,
        resolution=1280,
        decimation_target=310_000,
        texture_size=4096,
        low_vram=False,
    )

    output, source = client.generate(
        Image.new("RGBA", (12, 10), "white"),
        parameters.seed,
        threading.Event(),
        parameters=parameters,
        on_event=events.append,
    )

    captured = json.loads(output.with_name("parameters.json").read_text())
    assert output.read_bytes() == b"glTF-trellis2"
    assert source.is_file()
    assert captured["trellis2_root"] == str(root)
    assert captured["model_path"] == str(model)
    assert captured["resolution"] == "1280"
    assert captured["steps"] == "9"
    assert captured["seed"] == "77"
    assert captured["decimation_target"] == "310000"
    assert captured["texture_size"] == "4096"
    assert captured["low_vram"] is False
    assert captured["attention_backend"] == "sdpa"
    assert captured["sparse_attention_backend"] == "xformers"
    assert [event.stage for event in events] == [
        ReconstructionStage.SOURCE_IMAGE,
        ReconstructionStage.SPARSE_OCCUPANCY,
        ReconstructionStage.FINAL_MESH,
    ]
    assert client.checkpoint_path is None
    assert client.texture_checkpoint_path is None
    assert client.conditioning_path is not None
    client.shutdown()


def test_reconstruction_engine_routes_trellis_backend() -> None:
    class Client:
        def __init__(self, name):
            self.name = name
            self.calls = []
            self.artifacts = ()
            self.conditioning_path = None
            self.checkpoint_path = None
            self.texture_checkpoint_path = None

        def generate(self, image, seed, cancel, **kwargs):
            self.calls.append((image.copy(), seed, cancel, kwargs))
            return "/tmp/model.glb", "/tmp/source.png"

        def shutdown(self, _timeout):
            pass

    pixal = Client("pixal")
    trellis = Client("trellis")
    engine = ReconstructionEngine(pixal, trellis_client=trellis)
    request = ReconstructionRequest(
        Image.new("RGB", (4, 4)),
        ReconstructionParameters(backend=ReconstructionBackend.TRELLIS2),
    )

    result = engine._run(request, threading.Event(), lambda _event: None)

    assert not pixal.calls
    assert len(trellis.calls) == 1
    assert result.backend is ReconstructionBackend.TRELLIS2
    assert result.checkpoint_path is None
    engine.shutdown()


def test_trellis2_client_reports_subprocess_log_tail(tmp_path) -> None:
    root = tmp_path / "trellis2"
    model = root / "models" / "TRELLIS.2-4B"
    model.mkdir(parents=True)
    runner = tmp_path / "failing_runner.py"
    runner.write_text(_FAILING_RUNNER)
    client = Trellis2ProcessClient(
        python=sys.executable,
        root=root,
        model_path=model,
        runner_path=runner,
    )

    try:
        client.generate(
            Image.new("RGBA", (12, 10), "white"),
            42,
            threading.Event(),
        )
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("failing TRELLIS.2 runner must raise RuntimeError")

    assert "TRELLIS.2 exited with code 7" in message
    assert "synthetic TRELLIS failure" in message
    client.shutdown()
