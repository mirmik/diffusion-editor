from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import threading

import numpy as np
import pytest

from diffusion_editor.multiview_studio.model import (
    MultiviewProject,
    MeshPostprocessSettings,
    RefineCube,
    RefineShapeResult,
    RefineViewPatch,
    TrellisShapeSettings,
    TrellisTextureSettings,
    ViewKey,
)
from diffusion_editor.multiview_studio.trellis_generation import (
    TrellisShapeGenerator,
    _postprocess_payload,
    schedule_images,
)
from diffusion_editor.multiview_studio.trellis_mesh_postprocess import (
    _exact_degenerate_faces,
    _topology_counts,
    postprocess_key,
)
from diffusion_editor.multiview_studio.trellis_shape_runner import (
    _gltf_y_up_transform,
)
from diffusion_editor.multiview_studio.trellis_texture_generation import (
    TrellisTextureGenerator,
    schedule_texture_images,
)
from diffusion_editor.multiview_studio.trellis_texture_runner import (
    _write_texture_result,
)
from diffusion_editor.multiview_studio.controller import MultiviewStudioController
from diffusion_editor.multiview_studio.trellis_refine_generation import (
    TrellisRegionRefineGenerator,
)
from diffusion_editor.multiview_studio.trellis_refine_runner import (
    DECODER_ACTIVATION_CHUNK_BYTES,
    ENCODER_MLP_CHUNK_BYTES,
    REFINE_PIPELINE_MODEL_NAMES,
    _decode_mesh_cache,
)


def test_shape_schedule_warms_up_front_then_cycles_all_populated(tmp_path: Path):
    front = tmp_path / "front.png"
    right = tmp_path / "right.png"
    left = tmp_path / "left.png"
    front.touch()
    right.touch()
    left.touch()
    project = MultiviewProject().with_source("front", str(front))
    project = project.with_slot(ViewKey("eye", 90), image_path=str(right))
    project = project.with_slot(ViewKey("eye", 270), image_path=str(left))
    project = replace(
        project,
        trellis=TrellisShapeSettings(total_steps=7, warmup_steps=3),
    )

    schedule = schedule_images(project)

    assert [key for key, _path in schedule] == [
        ViewKey("eye", 0),
        ViewKey("eye", 0),
        ViewKey("eye", 0),
        ViewKey("eye", 0),
        ViewKey("eye", 90),
        ViewKey("eye", 270),
        ViewKey("eye", 0),
    ]
    assert [path for _key, path in schedule[:3]] == [front, front, front]


def test_shape_export_conversion_is_standard_gltf_y_up():
    transform = _gltf_y_up_transform()

    assert (transform @ [2.0, 3.0, 5.0, 1.0]).tolist() == [
        2.0,
        5.0,
        -3.0,
        1.0,
    ]


def test_refine_worker_loads_only_models_used_by_shape_refine():
    assert REFINE_PIPELINE_MODEL_NAMES == (
        "shape_slat_flow_model_1024",
        "shape_slat_decoder",
    )
    assert ENCODER_MLP_CHUNK_BYTES == 256 * 1024 * 1024
    assert DECODER_ACTIVATION_CHUNK_BYTES == 256 * 1024 * 1024


def test_refine_worker_reuses_decoded_mesh_cache_without_decoder(tmp_path: Path):
    cache = tmp_path / "decoded-region-raw.npz"
    np.savez(
        cache,
        vertices=np.asarray(((1.0, 2.0, 3.0),), dtype=np.float64),
        faces=np.asarray(((0, 0, 0),), dtype=np.int64),
    )

    class DecoderMustNotRun:
        def decode_shape_slat(self, *_args):
            raise AssertionError("decoded mesh cache was ignored")

    vertices, faces = _decode_mesh_cache(
        DecoderMustNotRun(), object(), 1536, cache, np, object()
    )

    assert vertices.dtype == np.float32
    assert faces.dtype == np.int32
    assert vertices.tolist() == [[1.0, 2.0, 3.0]]
    assert faces.tolist() == [[0, 0, 0]]


def test_postprocess_key_changes_only_with_postprocess_configuration():
    project = MultiviewProject()
    first = _postprocess_payload(project)
    changed = replace(
        project,
        trellis=replace(
            project.trellis,
            postprocess=replace(project.trellis.postprocess, remesh=False),
        ),
    )

    assert postprocess_key(first, 1024) == postprocess_key(first, 1024)
    assert postprocess_key(first, 1024) != postprocess_key(
        _postprocess_payload(changed), 1024
    )


def test_generation_cache_survives_postprocess_tuning_but_not_seed_change(
    tmp_path: Path,
):
    front = tmp_path / "front.png"
    front.write_bytes(b"front")
    model_path = tmp_path / "model"
    model_path.mkdir()
    python = tmp_path / "python"
    python.touch()
    generator = TrellisShapeGenerator(
        python=python,
        trellis_root=tmp_path,
        model_path=model_path,
    )
    project = MultiviewProject().with_source("front", str(front))
    run = tmp_path / "shape-runs" / "shape-test"
    run.mkdir(parents=True)
    shape = run / "shape.glb"
    cache = run / "decoded-mesh-z-up.npz"
    shape.touch()
    cache.touch()
    project = replace(project, shape_path=str(shape))
    (run / "result.json").write_text(
        json.dumps(
            {
                "generation_key": generator._generation_key(project),
                "shape_cache": str(cache),
                "actual_resolution": 1024,
            }
        ),
        encoding="utf-8",
    )

    tuned = replace(
        project,
        trellis=replace(
            project.trellis,
            decimation_target=1_000_000,
            postprocess=MeshPostprocessSettings(remesh=False),
        ),
    )
    changed_seed = replace(
        tuned,
        trellis=replace(tuned.trellis, seed=tuned.trellis.seed + 1),
    )

    assert generator.has_reusable_cache(project)
    assert generator.has_reusable_cache(tuned)
    assert not generator.has_reusable_cache(changed_seed)

    relocated_front = tmp_path / "saved" / "front.png"
    relocated_front.parent.mkdir()
    shutil.copy2(front, relocated_front)
    relocated = tuned.with_source("front", str(relocated_front))
    assert generator._generation_key(relocated) == generator._generation_key(tuned)


def test_topology_counts_boundary_and_multi_face_edges():
    import numpy as np

    faces = np.asarray(
        [
            [0, 1, 2],
            [1, 0, 3],
            [0, 1, 4],
        ],
        dtype=np.int64,
    )

    assert _topology_counts(faces) == {
        "boundary_edges": 6,
        "multi_face_nonmanifold_edges": 1,
    }


def test_exact_zero_area_faces_are_identified_without_area_epsilon():
    import numpy as np

    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    faces = np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int64)

    assert _exact_degenerate_faces(vertices, faces).tolist() == [False, True]


def test_region_refine_request_is_deterministic_and_reuses_result(
    tmp_path: Path,
):
    python = tmp_path / "python"
    model = tmp_path / "model"
    image = tmp_path / "right.png"
    geometry = tmp_path / "geometry.glb"
    python.touch()
    model.mkdir()
    image.write_bytes(b"image")
    geometry.write_bytes(b"mesh")
    controller = MultiviewStudioController()
    controller.set_geometry_path(geometry)
    controller.set_slot_image(ViewKey("eye", 90), image)
    controller.set_refine_cube((0.0, 0.0, 0.0), 1.0)
    controller.confirm_refine_cube()
    controller.set_refine_mask_weights(0, (((0, 1.0),),))
    controller.set_refine_view_patch(
        0, ViewKey("eye", 90), (0.1, 0.2, 0.8, 0.9)
    )
    controller.set_refine_shape_setting(0, "steps", 3)
    generator = TrellisRegionRefineGenerator(
        python=python, trellis_root=tmp_path, model_path=model
    )
    calls = []

    def fake_worker(command, *, result_path, **_kwargs):
        calls.append(command)
        request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        output = result_path.parent
        artifacts = {
            "source_region": output / "source-region.glb",
            "shape": output / "refined-region.glb",
            "source_slat": output / "source-shape-slat.npz",
            "refined_slat": output / "refined-shape-slat.npz",
        }
        for path in artifacts.values():
            path.touch()
        result_path.write_text(json.dumps({
            "request_key": request["request_key"],
            "geometry_fingerprint": request["geometry_fingerprint"],
            **{key: str(path) for key, path in artifacts.items()},
        }), encoding="utf-8")

    generator._run_worker = fake_worker
    arrays = (
        np.asarray(((-0.4, 0.0, 0.0), (0.4, 0.0, 0.0), (0.0, 0.4, 0.0))),
        np.asarray(((0, 1, 2),), dtype=np.uint32),
        np.asarray((1.0, 0.0, 0.0), dtype=np.float32),
    )
    manifest = tmp_path / "project.mvstudio.json"

    first = generator.generate(
        controller.project, 0, manifest, arrays, threading.Event()
    )
    second = generator.generate(
        controller.project, 0, manifest, arrays, threading.Event()
    )
    request = json.loads(Path(calls[0][-1]).read_text(encoding="utf-8"))

    assert first == second
    assert len(calls) == 1
    assert request["schedule"] == ["eye-090", "eye-090", "eye-090"]
    assert request["patches"][0]["bounds"] == [0.1, 0.2, 0.8, 0.9]


def test_worker_failure_is_streamed_and_persisted(
    tmp_path: Path, monkeypatch
):
    worker = tmp_path / "failing-worker.py"
    worker.write_text(
        "import sys\n"
        "print('[load] encoder', flush=True)\n"
        "print('torch.OutOfMemoryError: CUDA out of memory', flush=True)\n"
        "raise SystemExit(7)\n",
        encoding="utf-8",
    )
    generator = TrellisShapeGenerator(
        python=Path(sys.executable),
        trellis_root=tmp_path,
        model_path=tmp_path,
    )
    logged = []
    monkeypatch.setattr(
        "diffusion_editor.multiview_studio.trellis_generation.log.info",
        logged.append,
    )
    errors = []
    monkeypatch.setattr(
        "diffusion_editor.multiview_studio.trellis_generation.log.error",
        errors.append,
    )
    progress = []
    result_path = tmp_path / "run" / "result.json"

    with pytest.raises(RuntimeError, match="Full worker log") as failure:
        generator._run_worker(
            [sys.executable, str(worker)],
            result_path=result_path,
            cancel=threading.Event(),
            on_progress=progress.append,
            operation="test refine",
        )

    worker_log = result_path.parent / "worker.log"
    assert worker_log.read_text(encoding="utf-8").splitlines() == [
        "[load] encoder",
        "torch.OutOfMemoryError: CUDA out of memory",
    ]
    assert progress == [
        "[load] encoder",
        "torch.OutOfMemoryError: CUDA out of memory",
    ]
    assert any("CUDA out of memory" in message for message in logged)
    assert errors and str(worker_log) in errors[-1]
    assert str(worker_log) in str(failure.value)


def test_texture_schedule_uses_its_own_steps_and_every_view(tmp_path: Path):
    front = tmp_path / "front.png"
    right = tmp_path / "right.png"
    front.touch()
    right.touch()
    project = MultiviewProject().with_source("front", str(front))
    project = project.with_slot(ViewKey("eye", 90), image_path=str(right))
    project = replace(
        project,
        texture=TrellisTextureSettings(total_steps=5, warmup_steps=2),
    )

    assert [key for key, _path in schedule_texture_images(project)] == [
        ViewKey("eye", 0),
        ViewKey("eye", 0),
        ViewKey("eye", 0),
        ViewKey("eye", 90),
        ViewKey("eye", 0),
    ]


def test_texture_fingerprint_uses_geometry_not_displayed_material(tmp_path: Path):
    front = tmp_path / "front.png"
    geometry = tmp_path / "geometry.glb"
    first_texture = tmp_path / "first-texture.glb"
    second_texture = tmp_path / "second-texture.glb"
    front.write_bytes(b"front")
    geometry.write_bytes(b"geometry-a")
    first_texture.write_bytes(b"material-a")
    second_texture.write_bytes(b"material-b")
    model = tmp_path / "model"
    model.mkdir()
    generator = TrellisTextureGenerator(
        python=tmp_path / "python",
        trellis_root=tmp_path,
        model_path=model,
    )
    project = MultiviewProject().with_source("front", str(front))
    project = replace(
        project,
        geometry_path=str(geometry),
        shape_path=str(first_texture),
    )
    first_key = generator.texture_key(project)

    displayed_changed = replace(project, shape_path=str(second_texture))
    displayed_changed_key = generator.texture_key(displayed_changed)
    geometry.write_bytes(b"geometry-b")
    geometry_changed_key = generator.texture_key(project)

    assert displayed_changed_key == first_key
    assert geometry_changed_key != first_key


def test_texture_generator_returns_matching_cached_result(tmp_path: Path):
    front = tmp_path / "front.png"
    geometry = tmp_path / "geometry.glb"
    python = tmp_path / "python"
    model = tmp_path / "model"
    front.write_bytes(b"front")
    geometry.write_bytes(b"geometry")
    python.touch()
    model.mkdir()
    project = MultiviewProject().with_source("front", str(front))
    project = replace(project, geometry_path=str(geometry), shape_path=str(geometry))
    generator = TrellisTextureGenerator(
        python=python,
        trellis_root=tmp_path,
        model_path=model,
    )
    key = generator.texture_key(project)
    run = tmp_path / "texture-runs" / f"texture-{key[:16]}"
    run.mkdir(parents=True)
    textured = run / "textured.glb"
    textured.write_bytes(b"textured")
    (run / "result.json").write_text(
        json.dumps({"texture_key": key, "shape": str(textured)}),
        encoding="utf-8",
    )

    progress = []
    result = generator.generate(
        project,
        tmp_path / "project.mvstudio.json",
        threading.Event(),
        progress.append,
    )

    assert result == textured
    assert progress == [f"[texture-cache] {textured.name}"]


def test_refined_region_texture_uses_only_manual_patches_and_updates_result(
    tmp_path: Path,
):
    python = tmp_path / "python"
    model = tmp_path / "model"
    image = tmp_path / "right.png"
    geometry = tmp_path / "refined.glb"
    refine_run = tmp_path / "refine-runs" / "region-1-request"
    manifest = refine_run / "result.json"
    python.touch()
    model.mkdir()
    image.write_bytes(b"image")
    refine_run.mkdir(parents=True)
    geometry.write_bytes(b"geometry")
    manifest.write_text("{}", encoding="utf-8")
    region = RefineCube(
        center=(0.0, 0.0, 0.0),
        side=1.0,
        geometry_fingerprint="sha256:test",
        confirmed=True,
    )
    source = RefineShapeResult(
        geometry_fingerprint="sha256:test",
        request_key="shape-request",
        source_region_path=str(tmp_path / "source.glb"),
        refined_region_path=str(geometry),
        source_slat_path=str(tmp_path / "source.npz"),
        refined_slat_path=str(tmp_path / "refined.npz"),
        manifest_path=str(manifest),
    )
    project = MultiviewProject().with_slot(
        ViewKey("eye", 90), image_path=str(image)
    )
    project = replace(
        project,
        texture=TrellisTextureSettings(total_steps=3, warmup_steps=2),
        refine_regions=(region,),
        refine_view_patches=((
            RefineViewPatch(ViewKey("eye", 90), (0.1, 0.2, 0.8, 0.9)),
        ),),
        refine_shape_results=((source,),),
    )
    generator = TrellisTextureGenerator(
        python=python,
        trellis_root=tmp_path,
        model_path=model,
    )
    requests = []

    def fake_worker(command, *, result_path, **_kwargs):
        request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        requests.append(request)
        output = result_path.parent
        textured = output / "textured.glb"
        shape_slat = output / "encoded-shape-slat.npz"
        texture_slat = output / "texture-slat.npz"
        for path in (textured, shape_slat, texture_slat):
            path.touch()
        result_path.write_text(json.dumps({
            "texture_key": request["texture_key"],
            "shape": str(textured),
            "shape_slat": str(shape_slat),
            "texture_slat": str(texture_slat),
        }), encoding="utf-8")

    generator._run_worker = fake_worker
    result = generator.generate_refined_region(
        project,
        0,
        0,
        tmp_path / "project.mvstudio.json",
        threading.Event(),
    )

    assert requests[0]["input_mesh"] == str(geometry)
    assert requests[0]["schedule"] == ["eye-090"] * 3
    assert requests[0]["warmup_steps"] == 0
    assert requests[0]["views"] == [{
        "id": "eye-090",
        "image": str(image),
        "bounds": [0.1, 0.2, 0.8, 0.9],
    }]
    assert result.request_key == source.request_key
    assert Path(result.textured_region_path).name == "textured.glb"
    assert result.texture_key == requests[0]["texture_key"]


def test_texture_result_writer_accepts_list_bounds(tmp_path: Path):
    class Mesh:
        vertices = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        faces = ((0, 1, 1),)
        bounds = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]

    request = {
        "texture_key": "texture-key",
        "resolution": 1024,
        "texture_size": 2048,
        "schedule": ["eye-000"],
    }
    shape = tmp_path / "textured.glb"
    shape.touch()

    _write_texture_result(
        request,
        tmp_path,
        tmp_path / "refined.glb",
        Mesh(),
        0,
        Mesh(),
        shape,
        np,
    )

    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["source_bounds"] == Mesh.bounds
    assert result["output_bounds"] == Mesh.bounds
