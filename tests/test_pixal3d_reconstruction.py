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
from diffusion_editor.workers.pixal3d_staged_runner import Pixal3DRuntime
from diffusion_editor.generation.types import (
    ReconstructionParameters,
    ReconstructionRefineParameters,
    ReconstructionStage,
    ReconstructionStageStatus,
    pixal3d_resume_parameters_compatible,
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
parser.add_argument('--sparse-seed')
parser.add_argument('--sparse-steps')
parser.add_argument('--lr-seed')
parser.add_argument('--lr-steps')
parser.add_argument('--hr-seed')
parser.add_argument('--hr-steps')
parser.add_argument('--texture-seed')
parser.add_argument('--texture-steps')
parser.add_argument('--decimation-target')
parser.add_argument('--texture-size')
parser.add_argument('--manual-fov')
parser.add_argument('--low_vram', action='store_true')
args = parser.parse_args()
if Path(args.model_path, 'sleep').exists():
    time.sleep(30)
Path(args.output).write_bytes(b'glTF-fake')
"""


def test_pixal3d_resume_compatibility_only_freezes_completed_phases():
    previous = ReconstructionParameters(
        pixal3d_sparse_seed=10,
        pixal3d_lr_seed=20,
    )
    changed_lr = ReconstructionParameters(
        pixal3d_sparse_seed=10,
        pixal3d_lr_seed=21,
    )

    assert pixal3d_resume_parameters_compatible(
        previous, changed_lr, ReconstructionStage.SPARSE_OCCUPANCY
    )
    assert not pixal3d_resume_parameters_compatible(
        previous, changed_lr, ReconstructionStage.LR_SHAPE_LATENT
    )


def test_persistent_runtime_caches_preprocessed_source_by_content(tmp_path):
    class Pipeline:
        calls = 0
        def preprocess_image(self, image):
            self.calls += 1
            return image.convert("RGB")

    runtime = Pixal3DRuntime.__new__(Pixal3DRuntime)
    runtime.pipeline = Pipeline()
    runtime._source_sha256 = None
    runtime._prepared_image = None
    runtime._auto_camera = None
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (3, 2), "red").save(first)
    second.write_bytes(first.read_bytes())

    first_image, first_hash = runtime.prepare_image(first)
    second_image, second_hash = runtime.prepare_image(second)

    assert runtime.pipeline.calls == 1
    assert first_hash == second_hash
    assert first_image is not second_image


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
parser.add_argument('--lr-conditioning-resolution')
parser.add_argument('--steps')
parser.add_argument('--seed')
parser.add_argument('--sparse-seed')
parser.add_argument('--sparse-steps')
parser.add_argument('--lr-seed')
parser.add_argument('--lr-steps')
parser.add_argument('--hr-seed')
parser.add_argument('--hr-steps')
parser.add_argument('--texture-seed')
parser.add_argument('--texture-steps')
parser.add_argument('--decimation-target')
parser.add_argument('--texture-size')
parser.add_argument('--manual-fov')
parser.add_argument('--checkpoint')
parser.add_argument('--texture-checkpoint')
parser.add_argument('--session-checkpoint')
parser.add_argument('--resume-checkpoint')
parser.add_argument('--lr-refine-checkpoint')
parser.add_argument('--refine-checkpoint')
parser.add_argument('--texture-refine-checkpoint')
parser.add_argument('--refine-mask')
parser.add_argument('--refine-strength')
parser.add_argument('--refine-steps')
parser.add_argument('--refine-seed')
parser.add_argument('--refine-rescale-t')
parser.add_argument('--refine-guidance')
parser.add_argument('--resize-refine-detail', action='store_true')
parser.add_argument('--low_vram', action='store_true')
args = parser.parse_args()
Path(args.output).with_name('parameters.json').write_text(json.dumps(vars(args)))
if args.session_checkpoint:
    Path(args.session_checkpoint).write_bytes(b'npz-resume')
if args.lr_refine_checkpoint:
    Path(args.output).write_bytes(b'glTF-lr-refined')
    Path(args.output).with_name('lr-refine-generated.glb').write_bytes(
        b'glTF-lr-generated')
    with Path(args.events).open('w') as stream:
        stream.write(json.dumps({
            'stage': 'lr_shape_flow',
            'status': 'ready',
            'progress': int(args.refine_steps),
            'total': int(args.refine_steps),
        }) + '\n')
        stream.write(json.dumps({
            'stage': 'lr_shape_latent',
            'status': 'ready',
            'artifact_path': args.output,
            'preview_kind': 'mesh',
        }) + '\n')
    raise SystemExit(0)
if args.texture_refine_checkpoint:
    Path(args.output).write_bytes(b'glTF-texture-refined')
    Path(args.texture_checkpoint).write_bytes(b'npz-texture-refined')
    with Path(args.events).open('w') as stream:
        stream.write(json.dumps({
            'stage': 'texture_flow',
            'status': 'ready',
            'progress': int(args.refine_steps),
            'total': int(args.refine_steps),
        }) + '\n')
        stream.write(json.dumps({
            'stage': 'final_mesh',
            'status': 'ready',
            'artifact_path': args.output,
            'preview_kind': 'mesh',
        }) + '\n')
    raise SystemExit(0)
if args.refine_checkpoint:
    Path(args.output).write_bytes(b'glTF-refined')
    Path(args.output).with_name('hr-refine-generated.glb').write_bytes(
        b'glTF-hr-generated')
    Path(args.checkpoint).write_bytes(b'npz-refined')
    Path(args.texture_checkpoint).write_bytes(b'npz-texture')
    with Path(args.events).open('w') as stream:
        stream.write(json.dumps({
            'stage': 'hr_shape_flow',
            'status': 'ready',
            'progress': int(args.refine_steps),
            'total': int(args.refine_steps),
        }) + '\n')
        stream.write(json.dumps({
            'stage': 'final_mesh',
            'status': 'ready',
            'artifact_path': args.output,
            'preview_kind': 'mesh',
        }) + '\n')
    raise SystemExit(0)
preview = Path(args.output).with_name('sparse.glb')
preview.write_bytes(b'glTF-preview')
reported_stage = 'lr_shape_flow' if args.resume_checkpoint else 'sparse_occupancy'
with Path(args.events).open('w') as stream:
    stream.write(json.dumps({
        'stage': reported_stage,
        'status': 'running',
        'progress': 1,
        'total': 2,
    }) + '\n')
    stream.write(json.dumps({
        'stage': reported_stage,
        'status': 'ready',
        'progress': 2,
        'total': 2,
        'artifact_path': str(preview),
        'preview_kind': 'mesh',
    }) + '\n')
"""


_FAKE_PERSISTENT_RUNNER = r"""
import argparse
import json
from pathlib import Path
import sys

if '--server' not in sys.argv:
    raise SystemExit(2)
startup_file = Path(sys.argv[sys.argv.index('--pixal3d-root') + 1]) / 'starts'
count = int(startup_file.read_text()) if startup_file.exists() else 0
startup_file.write_text(str(count + 1))
wire = sys.stdout.buffer
wire.write(b'{"protocol":1,"type":"ready"}\n')
wire.flush()
for line in sys.stdin.buffer:
    request = json.loads(line)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output')
    parser.add_argument('--events')
    parser.add_argument('--session-checkpoint')
    parser.add_argument('--resume-checkpoint')
    parser.add_argument('--target-stage')
    args, _unknown = parser.parse_known_args(request['arguments'])
    stage = 'lr_shape_latent' if args.resume_checkpoint else 'sparse_occupancy'
    Path(args.session_checkpoint).write_bytes(b'persistent-checkpoint')
    with Path(args.events).open('w') as stream:
        stream.write(json.dumps({
            'stage': stage,
            'status': 'ready',
        }) + '\n')
    wire.write((json.dumps({
        'protocol': 1,
        'type': 'result',
        'request_id': request['request_id'],
    }) + '\n').encode())
    wire.flush()
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
        lr_conditioning_resolution=1024,
        manual_fov_degrees=45.0,
        decimation_target=350_000,
        texture_size=4096,
        low_vram=False,
        pixal3d_sparse_seed=11,
        pixal3d_sparse_steps=3,
        pixal3d_lr_seed=22,
        pixal3d_lr_steps=4,
        pixal3d_hr_seed=33,
        pixal3d_hr_steps=5,
        pixal3d_texture_seed=44,
        pixal3d_texture_steps=6,
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
    assert captured["lr_conditioning_resolution"] == "1024"
    assert captured["decimation_target"] == "350000"
    assert captured["texture_size"] == "4096"
    assert captured["checkpoint"].endswith("shape-checkpoint.npz")
    assert captured["sparse_seed"] == "11"
    assert captured["sparse_steps"] == "3"
    assert captured["lr_seed"] == "22"
    assert captured["lr_steps"] == "4"
    assert captured["hr_seed"] == "33"
    assert captured["hr_steps"] == "5"
    assert captured["texture_seed"] == "44"
    assert captured["texture_steps"] == "6"
    assert float(captured["manual_fov"]) == pytest.approx(math.pi / 4)
    assert captured["low_vram"] is False
    client.shutdown()


def test_staged_client_passes_previous_session_checkpoint_for_resume(tmp_path):
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

    client.generate(
        Image.new("RGB", (4, 3)), 7, threading.Event(),
        target_stage=ReconstructionStage.SPARSE_OCCUPANCY,
    )
    previous = client.resume_checkpoint_path
    assert previous is not None and previous.read_bytes() == b"npz-resume"

    resumed_events = []
    output, _source = client.generate(
        Image.new("RGB", (4, 3)), 7, threading.Event(),
        target_stage=ReconstructionStage.LR_SHAPE_LATENT,
        resume_checkpoint_path=previous,
        on_event=resumed_events.append,
    )
    captured = json.loads(output.with_name("parameters.json").read_text())
    assert captured["resume_checkpoint"] == str(previous)
    assert client.resume_checkpoint_path is not None
    assert client.resume_checkpoint_path != previous
    assert ReconstructionStage.SPARSE_OCCUPANCY not in {
        event.stage for event in resumed_events
    }
    assert ReconstructionStage.LR_SHAPE_FLOW in {
        event.stage for event in resumed_events
    }
    client.shutdown()


def test_persistent_client_reuses_one_worker_for_resume(tmp_path):
    root = tmp_path / "pixal3d"
    root.mkdir()
    runner = tmp_path / "persistent.py"
    runner.write_text(_FAKE_PERSISTENT_RUNNER)
    model = tmp_path / "model"
    model.mkdir()
    client = Pixal3DProcessClient(
        python=sys.executable,
        root=root,
        model_path=model,
        runner_path=runner,
        staged=True,
        persistent=True,
    )

    first_events = []
    client.generate(
        Image.new("RGB", (4, 3)), 7, threading.Event(),
        target_stage=ReconstructionStage.SPARSE_OCCUPANCY,
        on_event=first_events.append,
    )
    checkpoint = client.resume_checkpoint_path
    second_events = []
    client.generate(
        Image.new("RGB", (4, 3)), 7, threading.Event(),
        target_stage=ReconstructionStage.LR_SHAPE_LATENT,
        resume_checkpoint_path=checkpoint,
        on_event=second_events.append,
    )

    assert (root / "starts").read_text() == "1"
    assert [event.stage for event in first_events] == [
        ReconstructionStage.SOURCE_IMAGE,
        ReconstructionStage.SPARSE_OCCUPANCY,
    ]
    assert [event.stage for event in second_events] == [
        ReconstructionStage.SOURCE_IMAGE,
        ReconstructionStage.LR_SHAPE_LATENT,
    ]
    assert client._worker_process is not None
    assert client._worker_process.poll() is None
    client.shutdown()
    assert client._worker_process is None


def test_staged_client_runs_masked_refine_as_separate_artifact(tmp_path):
    root = tmp_path / "pixal3d"
    root.mkdir()
    runner = tmp_path / "staged.py"
    runner.write_text(_FAKE_STAGED_RUNNER)
    model = tmp_path / "model"
    model.mkdir()
    base_checkpoint = tmp_path / "base.npz"
    base_checkpoint.write_bytes(b"npz-base")
    client = Pixal3DProcessClient(
        python=sys.executable,
        root=root,
        model_path=model,
        runner_path=runner,
        staged=True,
    )
    events = []

    output, condition = client.refine(
        Image.new("RGB", (16, 16), "white"),
        Image.new("L", (16, 16), 255),
        base_checkpoint,
        threading.Event(),
        parameters=ReconstructionRefineParameters(
            strength=0.6, steps=8, seed=321
        ),
        on_event=events.append,
    )

    captured = json.loads(output.with_name("parameters.json").read_text())
    assert output.read_bytes() == b"glTF-refined"
    assert condition.is_file()
    assert client.checkpoint_path is not None
    assert client.checkpoint_path.read_bytes() == b"npz-refined"
    assert client.texture_checkpoint_path is not None
    assert client.texture_checkpoint_path.read_bytes() == b"npz-texture"
    assert client.refine_generated_path is not None
    assert client.refine_generated_path.read_bytes() == b"glTF-hr-generated"
    assert captured["refine_checkpoint"] == str(base_checkpoint)
    assert captured["refine_strength"] == "0.6"
    assert captured["refine_steps"] == "8"
    assert captured["refine_seed"] == "321"
    assert captured["resize_refine_detail"] is True
    assert [event.stage for event in events] == [
        ReconstructionStage.HR_SHAPE_FLOW,
        ReconstructionStage.FINAL_MESH,
    ]
    client.shutdown()


def test_staged_client_runs_masked_lr_refine_to_resume_checkpoint(tmp_path):
    root = tmp_path / "pixal3d"
    root.mkdir()
    runner = tmp_path / "staged.py"
    runner.write_text(_FAKE_STAGED_RUNNER)
    model = tmp_path / "model"
    model.mkdir()
    base_checkpoint = tmp_path / "lr-session.npz"
    base_checkpoint.write_bytes(b"npz-lr")
    client = Pixal3DProcessClient(
        python=sys.executable,
        root=root,
        model_path=model,
        runner_path=runner,
        staged=True,
    )
    events = []

    output, source = client.refine_lr(
        Image.new("RGB", (16, 16), "white"),
        Image.new("L", (16, 16), 255),
        base_checkpoint,
        threading.Event(),
        parameters=ReconstructionRefineParameters(
            strength=0.55, steps=7, seed=1234
        ),
        generation_parameters=ReconstructionParameters(
            lr_conditioning_resolution=1024,
            resolution=1536,
        ),
        on_event=events.append,
    )

    captured = json.loads(output.with_name("parameters.json").read_text())
    assert output.read_bytes() == b"glTF-lr-refined"
    assert source.is_file()
    assert client.resume_checkpoint_path is not None
    assert client.resume_checkpoint_path.read_bytes() == b"npz-resume"
    assert client.refine_generated_path is not None
    assert client.refine_generated_path.read_bytes() == b"glTF-lr-generated"
    assert captured["lr_refine_checkpoint"] == str(base_checkpoint)
    assert captured["lr_conditioning_resolution"] == "1024"
    assert captured["resolution"] == "1536"
    assert captured["refine_strength"] == "0.55"
    assert [event.stage for event in events] == [
        ReconstructionStage.LR_SHAPE_FLOW,
        ReconstructionStage.LR_SHAPE_LATENT,
    ]
    client.shutdown()


def test_staged_client_runs_masked_texture_refine(tmp_path):
    root = tmp_path / "pixal3d"
    root.mkdir()
    runner = tmp_path / "staged.py"
    runner.write_text(_FAKE_STAGED_RUNNER)
    model = tmp_path / "model"
    model.mkdir()
    shape_checkpoint = tmp_path / "shape.npz"
    texture_checkpoint = tmp_path / "texture.npz"
    shape_checkpoint.write_bytes(b"npz-shape")
    texture_checkpoint.write_bytes(b"npz-texture")
    client = Pixal3DProcessClient(
        python=sys.executable,
        root=root,
        model_path=model,
        runner_path=runner,
        staged=True,
    )
    events = []

    output, source = client.refine_texture(
        Image.new("RGB", (16, 16), "white"),
        Image.new("L", (16, 16), 255),
        shape_checkpoint,
        texture_checkpoint,
        threading.Event(),
        parameters=ReconstructionRefineParameters(
            strength=0.5, steps=6, seed=99,
            resize_detail_to_1024=False,
        ),
        on_event=events.append,
    )

    captured = json.loads(output.with_name("parameters.json").read_text())
    assert output.read_bytes() == b"glTF-texture-refined"
    assert source.is_file()
    assert client.checkpoint_path == shape_checkpoint
    assert client.texture_checkpoint_path is not None
    assert client.texture_checkpoint_path.read_bytes() == b"npz-texture-refined"
    assert captured["texture_refine_checkpoint"] == str(texture_checkpoint)
    assert captured["refine_strength"] == "0.5"
    assert captured["resize_refine_detail"] is False
    assert [event.stage for event in events] == [
        ReconstructionStage.TEXTURE_FLOW,
        ReconstructionStage.FINAL_MESH,
    ]
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
    assert event.result.resume_checkpoint_path is None
    assert stage_events[0].stage.value == "source_image"
    engine.shutdown()


def test_reconstruction_controller_snapshots_refine_inputs() -> None:
    class Engine:
        is_busy = False
        request = None
        job_id = None

        def submit_refine_request(self, request, *, job_id=None):
            self.request = request
            self.job_id = job_id
            return True

    engine = Engine()
    controller = ReconstructionController(engine)
    condition = Image.new("RGB", (8, 8), "white")
    mask = Image.new("L", (8, 8), 255)

    event = controller.start_refine(
        condition,
        mask,
        "/tmp/base.npz",
        parameters=ReconstructionRefineParameters(strength=0.6, steps=8),
    )

    assert event.status and "Refining" in event.status
    assert engine.job_id.startswith("reconstruction_refine_")
    assert engine.request.base_checkpoint_path == "/tmp/base.npz"
    assert engine.request.parameters.strength == 0.6
    assert engine.request.conditioning_image is not condition
    assert engine.request.mask_image is not mask


def test_reconstruction_controller_snapshots_lr_refine_inputs() -> None:
    class Engine:
        is_busy = False
        request = None
        job_id = None

        def submit_lr_refine_request(self, request, *, job_id=None):
            self.request = request
            self.job_id = job_id
            return True

    engine = Engine()
    controller = ReconstructionController(engine)
    condition = Image.new("RGB", (8, 8), "white")
    mask = Image.new("L", (8, 8), 255)

    event = controller.start_lr_refine(
        condition,
        mask,
        "/tmp/lr-session.npz",
        parameters=ReconstructionRefineParameters(strength=0.4, steps=9),
    )

    assert event.status and "LR" in event.status
    assert engine.job_id.startswith("reconstruction_lr_refine_")
    assert engine.request.session_checkpoint_path == "/tmp/lr-session.npz"
    assert engine.request.parameters.steps == 9
    assert engine.request.conditioning_image is not condition
    assert engine.request.mask_image is not mask


def test_reconstruction_controller_snapshots_texture_refine_inputs() -> None:
    class Engine:
        is_busy = False
        request = None
        job_id = None

        def submit_texture_refine_request(self, request, *, job_id=None):
            self.request = request
            self.job_id = job_id
            return True

    engine = Engine()
    controller = ReconstructionController(engine)
    condition = Image.new("RGB", (8, 8), "white")
    mask = Image.new("L", (8, 8), 255)

    event = controller.start_texture_refine(
        condition,
        mask,
        "/tmp/shape.npz",
        "/tmp/texture.npz",
        parameters=ReconstructionRefineParameters(strength=0.45, steps=7),
    )

    assert event.status and "texture" in event.status.lower()
    assert engine.job_id.startswith("reconstruction_texture_refine_")
    assert engine.request.shape_checkpoint_path == "/tmp/shape.npz"
    assert engine.request.texture_checkpoint_path == "/tmp/texture.npz"
    assert engine.request.conditioning_image is not condition
    assert engine.request.mask_image is not mask


def _capture_error(errors, callback) -> None:
    try:
        callback()
    except Exception as exc:
        errors.append(exc)
