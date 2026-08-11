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
from diffusion_editor.workers.sam3d_objects_process import (
    Sam3DObjectsProcessClient,
)


_FAKE_RUNNER = r"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
for name in ('sam3d-root', 'config', 'image', 'output', 'events',
             'target-stage', 'seed', 'sparse-steps', 'slat-steps',
             'sparse-guidance', 'slat-guidance', 'simplify', 'texture-size'):
    parser.add_argument('--' + name)
args = parser.parse_args()
output = Path(args.output)
output.with_name('parameters.json').write_text(json.dumps(vars(args)))
output.with_name('conditioning.png').write_bytes(b'png')
output.with_name('structured-latent.pt').write_bytes(b'latent')
output.with_name('gaussian-splat.ply').write_bytes(b'gaussian')
points = output.with_name('pointmap.ply'); points.write_bytes(b'points')
sparse = output.with_name('sparse-occupancy.ply'); sparse.write_bytes(b'sparse')
raw = output.with_name('raw-mesh.glb'); raw.write_bytes(b'raw')
gaussian = output.with_name('gaussian-preview.ply'); gaussian.write_bytes(b'gs')
stages = [
    ('point_cloud', points, 'points'),
    ('sparse_occupancy', sparse, 'points'),
    ('hr_shape_flow', None, None),
    ('hr_shape_latent', raw, 'mesh'),
    ('texture_latent', gaussian, 'points'),
    ('final_mesh', output, 'mesh'),
]
target_index = [stage for stage, _, _ in stages].index(args.target_stage)
with Path(args.events).open('w') as stream:
    for index, (stage, artifact, kind) in enumerate(stages):
        if index > target_index:
            break
        if stage == 'final_mesh':
            output.write_bytes(b'glTF-sam')
        payload = {'stage': stage, 'status': 'ready'}
        if artifact is not None:
            payload.update(artifact_path=str(artifact), preview_kind=kind)
        stream.write(json.dumps(payload) + '\n')
"""


def _client(tmp_path) -> Sam3DObjectsProcessClient:
    root = tmp_path / "sam3d-objects"
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "pipeline.yaml").write_text("_target_: fake")
    runner = tmp_path / "runner.py"
    runner.write_text(_FAKE_RUNNER)
    return Sam3DObjectsProcessClient(
        python=sys.executable,
        root=root,
        checkpoints=checkpoints,
        runner_path=runner,
    )


def test_sam3d_objects_client_publishes_stages_and_parameters(tmp_path) -> None:
    client = _client(tmp_path)
    events = []
    parameters = ReconstructionParameters(
        backend=ReconstructionBackend.SAM3D_OBJECTS,
        seed=77,
        texture_size=4096,
        sam3d_sparse_steps=21,
        sam3d_slat_steps=17,
        sam3d_sparse_guidance_scale=6.5,
        sam3d_slat_guidance_scale=1.25,
        sam3d_simplify=0.9,
    )
    output, source = client.generate(
        Image.new("RGBA", (12, 10), "white"),
        parameters.seed,
        threading.Event(),
        parameters=parameters,
        on_event=events.append,
    )
    captured = json.loads(output.with_name("parameters.json").read_text())
    assert output.read_bytes() == b"glTF-sam"
    assert source.is_file()
    assert captured["sparse_steps"] == "21"
    assert captured["slat_steps"] == "17"
    assert captured["sparse_guidance"] == "6.5"
    assert captured["slat_guidance"] == "1.25"
    assert captured["simplify"] == "0.9"
    assert captured["texture_size"] == "4096"
    assert [event.stage for event in events] == [
        ReconstructionStage.SOURCE_IMAGE,
        ReconstructionStage.POINT_CLOUD,
        ReconstructionStage.SPARSE_OCCUPANCY,
        ReconstructionStage.HR_SHAPE_FLOW,
        ReconstructionStage.HR_SHAPE_LATENT,
        ReconstructionStage.TEXTURE_LATENT,
        ReconstructionStage.FINAL_MESH,
    ]
    assert client.conditioning_path is not None
    assert client.checkpoint_path is not None
    assert client.texture_checkpoint_path is not None
    client.shutdown()


def test_sam3d_objects_client_can_stop_after_sparse_occupancy(tmp_path) -> None:
    client = _client(tmp_path)
    events = []
    output, _ = client.generate(
        Image.new("RGBA", (8, 8), "white"),
        42,
        threading.Event(),
        target_stage=ReconstructionStage.SPARSE_OCCUPANCY,
        on_event=events.append,
    )
    assert not output.exists()
    assert [event.stage for event in events] == [
        ReconstructionStage.SOURCE_IMAGE,
        ReconstructionStage.POINT_CLOUD,
        ReconstructionStage.SPARSE_OCCUPANCY,
    ]
    client.shutdown()


def test_sam3d_objects_parameters_roundtrip() -> None:
    parameters = ReconstructionParameters(
        backend=ReconstructionBackend.SAM3D_OBJECTS,
        sam3d_sparse_steps=20,
        sam3d_slat_steps=18,
        sam3d_sparse_guidance_scale=6.0,
        sam3d_slat_guidance_scale=1.5,
        sam3d_simplify=0.87,
    )
    assert ReconstructionParameters.from_dict(parameters.to_dict()) == parameters


def test_reconstruction_engine_routes_sam3d_objects_backend() -> None:
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

    clients = [Client() for _ in range(6)]
    engine = ReconstructionEngine(
        clients[0],
        trellis_client=clients[1],
        spar3d_client=clients[2],
        hi3dgen_client=clients[3],
        hunyuan3d21_client=clients[4],
        sam3d_objects_client=clients[5],
    )
    request = ReconstructionRequest(
        Image.new("RGB", (4, 4)),
        ReconstructionParameters(backend=ReconstructionBackend.SAM3D_OBJECTS),
    )
    result = engine._run(request, threading.Event(), lambda _event: None)
    assert all(not client.calls for client in clients[:5])
    assert len(clients[5].calls) == 1
    assert result.backend is ReconstructionBackend.SAM3D_OBJECTS
    engine.shutdown()
