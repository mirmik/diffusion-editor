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
from diffusion_editor.workers.hi3dgen_process import Hi3DGenProcessClient


_FAKE_RUNNER = r"""
import argparse
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--hi3dgen-root')
parser.add_argument('--stable-normal-root')
parser.add_argument('--image')
parser.add_argument('--output')
parser.add_argument('--events')
parser.add_argument('--model-path')
parser.add_argument('--target-stage')
parser.add_argument('--sparse-steps')
parser.add_argument('--slat-steps')
parser.add_argument('--guidance-scale')
parser.add_argument('--normal-resolution')
parser.add_argument('--seed')
parser.add_argument('--decimation-target')
args = parser.parse_args()
output = Path(args.output)
captured = {
    **vars(args),
    'spconv_algo': os.environ.get('SPCONV_ALGO'),
    'xformers_disabled': os.environ.get('XFORMERS_DISABLED'),
}
output.with_name('parameters.json').write_text(json.dumps(captured))
output.with_name('preprocessed.png').write_bytes(b'png')
normal = output.with_name('normal.png')
normal.write_bytes(b'normal')
sparse = output.with_name('sparse.glb')
sparse.write_bytes(b'sparse')
shape = output.with_name('shape.glb')
shape.write_bytes(b'shape')
with Path(args.events).open('w') as stream:
    stream.write(json.dumps({
        'stage': 'normal_map',
        'status': 'ready',
        'artifact_path': str(normal),
        'preview_kind': 'image',
    }) + '\n')
    if args.target_stage != 'normal_map':
        stream.write(json.dumps({
            'stage': 'sparse_occupancy',
            'status': 'ready',
            'artifact_path': str(sparse),
            'preview_kind': 'mesh',
        }) + '\n')
    if args.target_stage in ('hr_shape_latent', 'final_mesh'):
        stream.write(json.dumps({
            'stage': 'hr_shape_latent',
            'status': 'ready',
            'artifact_path': str(shape),
            'preview_kind': 'mesh',
        }) + '\n')
    if args.target_stage == 'final_mesh':
        output.write_bytes(b'glTF-hi3dgen')
        stream.write(json.dumps({
            'stage': 'final_mesh',
            'status': 'ready',
            'artifact_path': str(output),
            'preview_kind': 'mesh',
        }) + '\n')
"""


def _client(tmp_path) -> Hi3DGenProcessClient:
    root = tmp_path / "Stable3DGen"
    model = root / "weights" / "trellis-normal-v0-1"
    model.mkdir(parents=True)
    (root / "weights" / "yoso-normal-v1-8-1").mkdir()
    (root / "weights" / "BiRefNet").mkdir()
    stable_normal = tmp_path / "StableNormal"
    stable_normal.mkdir()
    runner = tmp_path / "runner.py"
    runner.write_text(_FAKE_RUNNER)
    return Hi3DGenProcessClient(
        python=sys.executable,
        root=root,
        model_path=model,
        stable_normal_root=stable_normal,
        runner_path=runner,
    )


def test_hi3dgen_client_publishes_stages_and_parameters(tmp_path) -> None:
    client = _client(tmp_path)
    events = []
    parameters = ReconstructionParameters(
        backend=ReconstructionBackend.HI3DGEN,
        seed=77,
        steps=18,
        decimation_target=310_000,
        hi3dgen_slat_steps=7,
        hi3dgen_guidance_scale=4.25,
        hi3dgen_normal_resolution=1024,
    )

    output, source = client.generate(
        Image.new("RGBA", (12, 10), "white"),
        parameters.seed,
        threading.Event(),
        parameters=parameters,
        on_event=events.append,
    )

    captured = json.loads(output.with_name("parameters.json").read_text())
    assert output.read_bytes() == b"glTF-hi3dgen"
    assert source.is_file()
    assert captured["sparse_steps"] == "18"
    assert captured["slat_steps"] == "7"
    assert captured["guidance_scale"] == "4.25"
    assert captured["normal_resolution"] == "1024"
    assert captured["seed"] == "77"
    assert captured["decimation_target"] == "310000"
    assert captured["spconv_algo"] == "native"
    assert captured["xformers_disabled"] == "1"
    assert [event.stage for event in events] == [
        ReconstructionStage.SOURCE_IMAGE,
        ReconstructionStage.NORMAL_MAP,
        ReconstructionStage.SPARSE_OCCUPANCY,
        ReconstructionStage.HR_SHAPE_LATENT,
        ReconstructionStage.FINAL_MESH,
    ]
    assert events[1].artifact is not None
    assert events[1].artifact.preview_kind == "image"
    assert client.conditioning_path is not None
    client.shutdown()


def test_hi3dgen_client_can_stop_after_normal_map(tmp_path) -> None:
    client = _client(tmp_path)
    events = []

    output, _source = client.generate(
        Image.new("RGBA", (8, 8), "white"),
        42,
        threading.Event(),
        target_stage=ReconstructionStage.NORMAL_MAP,
        on_event=events.append,
    )

    assert not output.exists()
    assert [event.stage for event in events] == [
        ReconstructionStage.SOURCE_IMAGE,
        ReconstructionStage.NORMAL_MAP,
    ]
    client.shutdown()


def test_reconstruction_engine_routes_hi3dgen_backend() -> None:
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
    hi3dgen = Client()
    engine = ReconstructionEngine(
        pixal,
        trellis_client=trellis,
        spar3d_client=spar3d,
        hi3dgen_client=hi3dgen,
    )
    request = ReconstructionRequest(
        Image.new("RGB", (4, 4)),
        ReconstructionParameters(backend=ReconstructionBackend.HI3DGEN),
    )

    result = engine._run(request, threading.Event(), lambda _event: None)

    assert not pixal.calls
    assert not trellis.calls
    assert not spar3d.calls
    assert len(hi3dgen.calls) == 1
    assert result.backend is ReconstructionBackend.HI3DGEN
    engine.shutdown()
