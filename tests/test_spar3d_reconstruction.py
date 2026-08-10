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
from diffusion_editor.workers.spar3d_process import Spar3DProcessClient


_FAKE_RUNNER = r"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--spar3d-root')
parser.add_argument('--image')
parser.add_argument('--output')
parser.add_argument('--events')
parser.add_argument('--model-path')
parser.add_argument('--target-stage')
parser.add_argument('--seed')
parser.add_argument('--guidance-scale')
parser.add_argument('--texture-size')
parser.add_argument('--low-vram', action='store_true')
args = parser.parse_args()
output = Path(args.output)
output.with_name('parameters.json').write_text(json.dumps(vars(args)))
output.with_name('preprocessed.png').write_bytes(b'png')
points = output.with_name('points.ply')
points.write_bytes(b'ply')
with Path(args.events).open('w') as stream:
    stream.write(json.dumps({
        'stage': 'point_cloud',
        'status': 'ready',
        'artifact_path': str(points),
        'preview_kind': 'points',
    }) + '\n')
    if args.target_stage == 'final_mesh':
        output.write_bytes(b'glTF-spar3d')
        stream.write(json.dumps({
            'stage': 'final_mesh',
            'status': 'ready',
            'artifact_path': str(output),
            'preview_kind': 'mesh',
        }) + '\n')
"""


def _client(tmp_path) -> Spar3DProcessClient:
    root = tmp_path / "spar3d"
    model = root / "models" / "stable-point-aware-3d"
    model.mkdir(parents=True)
    runner = tmp_path / "runner.py"
    runner.write_text(_FAKE_RUNNER)
    return Spar3DProcessClient(
        python=sys.executable,
        root=root,
        model_path=model,
        runner_path=runner,
    )


def test_spar3d_client_publishes_point_cloud_then_final_mesh(tmp_path) -> None:
    client = _client(tmp_path)
    events = []
    parameters = ReconstructionParameters(
        backend=ReconstructionBackend.SPAR3D,
        seed=77,
        texture_size=4096,
        low_vram=False,
        spar3d_guidance_scale=4.25,
    )

    output, source = client.generate(
        Image.new("RGBA", (12, 10), "white"),
        parameters.seed,
        threading.Event(),
        parameters=parameters,
        on_event=events.append,
    )

    captured = json.loads(output.with_name("parameters.json").read_text())
    assert output.read_bytes() == b"glTF-spar3d"
    assert source.is_file()
    assert captured["seed"] == "77"
    assert captured["guidance_scale"] == "4.25"
    assert captured["texture_size"] == "4096"
    assert captured["low_vram"] is False
    assert [event.stage for event in events] == [
        ReconstructionStage.SOURCE_IMAGE,
        ReconstructionStage.POINT_CLOUD,
        ReconstructionStage.FINAL_MESH,
    ]
    assert events[1].artifact is not None
    assert events[1].artifact.preview_kind == "points"
    assert events[1].artifact.path.endswith("points.ply")
    assert client.conditioning_path is not None
    client.shutdown()


def test_spar3d_client_can_stop_after_editable_point_cloud(tmp_path) -> None:
    client = _client(tmp_path)
    events = []

    output, _source = client.generate(
        Image.new("RGBA", (8, 8), "white"),
        42,
        threading.Event(),
        target_stage=ReconstructionStage.POINT_CLOUD,
        on_event=events.append,
    )

    assert not output.exists()
    assert [event.stage for event in events] == [
        ReconstructionStage.SOURCE_IMAGE,
        ReconstructionStage.POINT_CLOUD,
    ]
    assert client.artifacts[-1].preview_kind == "points"
    client.shutdown()


def test_reconstruction_engine_routes_spar3d_backend() -> None:
    class Client:
        def __init__(self):
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

    pixal = Client()
    trellis = Client()
    spar3d = Client()
    engine = ReconstructionEngine(
        pixal,
        trellis_client=trellis,
        spar3d_client=spar3d,
    )
    request = ReconstructionRequest(
        Image.new("RGB", (4, 4)),
        ReconstructionParameters(backend=ReconstructionBackend.SPAR3D),
    )

    result = engine._run(request, threading.Event(), lambda _event: None)

    assert not pixal.calls
    assert not trellis.calls
    assert len(spar3d.calls) == 1
    assert result.backend is ReconstructionBackend.SPAR3D
    engine.shutdown()
