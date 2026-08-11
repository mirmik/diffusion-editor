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
from diffusion_editor.workers.hunyuan3d21_process import Hunyuan3D21ProcessClient


_FAKE_RUNNER = r"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
for name in ('comfy-root', 'node-root', 'image', 'output', 'events', 'dit',
             'vae', 'latent-output', 'target-stage', 'shape-steps',
             'shape-guidance', 'octree-resolution', 'texture-steps',
             'texture-guidance', 'texture-size', 'seed', 'decimation-target',
             'gltf-transform'):
    parser.add_argument('--' + name)
args = parser.parse_args()
output = Path(args.output)
output.with_name('parameters.json').write_text(json.dumps(vars(args)))
output.with_name('preprocessed.png').write_bytes(b'png')
Path(args.latent_output).write_bytes(b'latent')
shape = output.with_name('raw-mesh.glb')
shape.write_bytes(b'shape')
texture = output.with_name('textured-preview.glb')
texture.write_bytes(b'texture')
with Path(args.events).open('w') as stream:
    stream.write(json.dumps({'stage': 'hr_shape_flow', 'status': 'ready'}) + '\n')
    if args.target_stage != 'hr_shape_flow':
        stream.write(json.dumps({
            'stage': 'hr_shape_latent', 'status': 'ready',
            'artifact_path': str(shape), 'preview_kind': 'mesh',
        }) + '\n')
    if args.target_stage in ('texture_flow', 'texture_latent', 'final_mesh'):
        stream.write(json.dumps({'stage': 'texture_flow', 'status': 'ready'}) + '\n')
    if args.target_stage in ('texture_latent', 'final_mesh'):
        stream.write(json.dumps({
            'stage': 'texture_latent', 'status': 'ready',
            'artifact_path': str(texture), 'preview_kind': 'mesh',
        }) + '\n')
    if args.target_stage == 'final_mesh':
        output.write_bytes(b'glTF-hunyuan')
        stream.write(json.dumps({
            'stage': 'final_mesh', 'status': 'ready',
            'artifact_path': str(output), 'preview_kind': 'mesh',
        }) + '\n')
"""


def _client(tmp_path) -> Hunyuan3D21ProcessClient:
    root = tmp_path / "ComfyUI"
    node = root / "custom_nodes" / "ComfyUI-Hunyuan3d-2-1"
    node.mkdir(parents=True)
    dit = root / "models" / "diffusion_models" / "dit.ckpt"
    vae = root / "models" / "vae" / "vae.ckpt"
    dit.parent.mkdir(parents=True)
    vae.parent.mkdir(parents=True)
    dit.write_bytes(b"dit")
    vae.write_bytes(b"vae")
    runner = tmp_path / "runner.py"
    runner.write_text(_FAKE_RUNNER)
    gltf_transform = tmp_path / "gltf-transform"
    gltf_transform.write_text("")
    return Hunyuan3D21ProcessClient(
        python=sys.executable,
        root=root,
        node_root=node,
        dit_path=dit,
        vae_path=vae,
        runner_path=runner,
        gltf_transform=gltf_transform,
    )


def test_hunyuan3d21_client_publishes_stages_and_parameters(tmp_path) -> None:
    client = _client(tmp_path)
    events = []
    parameters = ReconstructionParameters(
        backend=ReconstructionBackend.HUNYUAN3D21,
        seed=77,
        steps=24,
        decimation_target=310_000,
        texture_size=4096,
        hunyuan3d21_guidance_scale=6.25,
        hunyuan3d21_octree_resolution=512,
        hunyuan3d21_texture_steps=14,
        hunyuan3d21_texture_guidance_scale=4.5,
    )
    output, source = client.generate(
        Image.new("RGBA", (12, 10), "white"),
        parameters.seed,
        threading.Event(),
        parameters=parameters,
        on_event=events.append,
    )
    captured = json.loads(output.with_name("parameters.json").read_text())
    assert output.read_bytes() == b"glTF-hunyuan"
    assert source.is_file()
    assert captured["shape_steps"] == "24"
    assert captured["shape_guidance"] == "6.25"
    assert captured["octree_resolution"] == "512"
    assert captured["texture_steps"] == "14"
    assert captured["texture_guidance"] == "4.5"
    assert captured["texture_size"] == "4096"
    assert captured["decimation_target"] == "310000"
    assert [event.stage for event in events] == [
        ReconstructionStage.SOURCE_IMAGE,
        ReconstructionStage.HR_SHAPE_FLOW,
        ReconstructionStage.HR_SHAPE_LATENT,
        ReconstructionStage.TEXTURE_FLOW,
        ReconstructionStage.TEXTURE_LATENT,
        ReconstructionStage.FINAL_MESH,
    ]
    assert client.checkpoint_path is not None
    assert client.conditioning_path is not None
    client.shutdown()


def test_hunyuan3d21_client_can_stop_after_raw_shape(tmp_path) -> None:
    client = _client(tmp_path)
    events = []
    output, _source = client.generate(
        Image.new("RGBA", (8, 8), "white"),
        42,
        threading.Event(),
        target_stage=ReconstructionStage.HR_SHAPE_LATENT,
        on_event=events.append,
    )
    assert not output.exists()
    assert [event.stage for event in events] == [
        ReconstructionStage.SOURCE_IMAGE,
        ReconstructionStage.HR_SHAPE_FLOW,
        ReconstructionStage.HR_SHAPE_LATENT,
    ]
    client.shutdown()


def test_hunyuan3d21_parameters_roundtrip() -> None:
    parameters = ReconstructionParameters(
        backend=ReconstructionBackend.HUNYUAN3D21,
        hunyuan3d21_guidance_scale=6.5,
        hunyuan3d21_octree_resolution=256,
        hunyuan3d21_texture_steps=18,
        hunyuan3d21_texture_guidance_scale=4.25,
    )
    assert ReconstructionParameters.from_dict(parameters.to_dict()) == parameters


def test_reconstruction_engine_routes_hunyuan3d21_backend() -> None:
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

    clients = [Client() for _ in range(5)]
    engine = ReconstructionEngine(
        clients[0],
        trellis_client=clients[1],
        spar3d_client=clients[2],
        hi3dgen_client=clients[3],
        hunyuan3d21_client=clients[4],
    )
    request = ReconstructionRequest(
        Image.new("RGB", (4, 4)),
        ReconstructionParameters(backend=ReconstructionBackend.HUNYUAN3D21),
    )
    result = engine._run(request, threading.Event(), lambda _event: None)
    assert all(not client.calls for client in clients[:4])
    assert len(clients[4].calls) == 1
    assert result.backend is ReconstructionBackend.HUNYUAN3D21
    engine.shutdown()
