from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import threading
import types

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
    _alternating_sample,
    _gltf_y_up_transform,
)
from diffusion_editor.multiview_studio.trellis_texture_generation import (
    TrellisTextureGenerator,
    schedule_texture_images,
)
from diffusion_editor.multiview_studio.trellis_texture_runner import (
    _texture_model_names,
    _write_texture_result,
)
from diffusion_editor.multiview_studio.controller import MultiviewStudioController
from diffusion_editor.multiview_studio.trellis_refine_generation import (
    TrellisRegionRefineGenerator,
    _refine_view_schedule,
    _region_postprocess_payload,
)
from diffusion_editor.multiview_studio.trellis_refine_runner import (
    DECODER_ACTIVATION_CHUNK_BYTES,
    ENCODER_MLP_CHUNK_BYTES,
    REFINE_PIPELINE_MODEL_NAMES,
    SPARSE_STRUCTURE_ENCODER,
    _decode_mesh_cache,
    _postprocess_refined_mesh,
    _region_stage_strengths,
    _restore_protected,
    _reuse_decoded_shape_cache,
    _save_sparse,
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
        "sparse_structure_flow_model",
        "sparse_structure_decoder",
        "shape_slat_flow_model_512",
        "shape_slat_flow_model_1024",
        "shape_slat_decoder",
    )
    assert SPARSE_STRUCTURE_ENCODER.endswith("ss_enc_conv3d_16l8_fp16")
    assert ENCODER_MLP_CHUNK_BYTES == 256 * 1024 * 1024
    assert DECODER_ACTIVATION_CHUNK_BYTES == 256 * 1024 * 1024


def test_region_strength_changes_only_coarse_structure():
    assert _region_stage_strengths(0.35) == {
        "occupancy": 0.35,
        "shape_512": 1.0,
        "shape_high_resolution": 1.0,
    }
    assert _region_stage_strengths(0.0)["shape_512"] == 1.0
    with pytest.raises(ValueError, match="structure strength"):
        _region_stage_strengths(1.01)


def test_refine_schedule_warms_up_front_then_cycles_manual_patches():
    patches = (
        RefineViewPatch(ViewKey("eye", 90), (0.0, 0.0, 1.0, 1.0)),
        RefineViewPatch(ViewKey("eye", 0), (0.0, 0.0, 1.0, 1.0)),
        RefineViewPatch(ViewKey("eye", 180), (0.0, 0.0, 1.0, 1.0)),
    )

    assert _refine_view_schedule(patches, 8, 4) == (
        "eye-000",
        "eye-000",
        "eye-000",
        "eye-000",
        "eye-090",
        "eye-000",
        "eye-180",
        "eye-090",
    )


def test_sparse_cache_writes_c_contiguous_coordinates(tmp_path: Path):
    class ArrayTensor:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def float(self):
            return self

        def numpy(self):
            return self.value

    coords = np.asfortranarray(
        np.asarray(((0, 1, 2, 3), (0, 4, 5, 6)), dtype=np.int32)
    )
    feats = np.asfortranarray(np.ones((2, 32), dtype=np.float32))
    path = tmp_path / "slat.npz"

    _save_sparse(
        path,
        types.SimpleNamespace(
            coords=ArrayTensor(coords), feats=ArrayTensor(feats)
        ),
    )

    with np.load(path, allow_pickle=False) as saved:
        assert saved["coords"].flags.c_contiguous
        assert saved["feats"].flags.c_contiguous


def test_latent_inpaint_restores_protected_tokens_only():
    source = np.asarray(((1.0, 2.0), (3.0, 4.0)), dtype=np.float32)
    candidate = np.asarray(((9.0, 9.0), (8.0, 8.0)), dtype=np.float32)
    protected = np.asarray(((1.0,), (0.0,)), dtype=np.float32)

    restored = _restore_protected(source, candidate, protected)

    assert restored.tolist() == [[1.0, 2.0], [8.0, 8.0]]


def test_strength_one_empty_protection_is_identical_to_main_noise_start():
    class Sampler:
        sigma_min = 1e-5

        def sample_once(self, _model, sample, *_args, **_kwargs):
            return types.SimpleNamespace(pred_x_prev=sample + 0.25)

    sampler = Sampler()
    noise = np.asarray(((-2.0, 3.0), (4.0, -5.0)), dtype=np.float32)
    source = np.asarray(((100.0, 200.0), (300.0, 400.0)), dtype=np.float32)
    conditions = {"eye-000": {"cond": object(), "neg_cond": object()}}
    schedule = ("eye-000", "eye-000")
    kwargs = {
        "stage": "test",
        "rescale_t": 5.0,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.7,
        "guidance_interval": (0.6, 1.0),
    }

    main = _alternating_sample(
        sampler, object(), noise, conditions, schedule, **kwargs
    )
    refine = _alternating_sample(
        sampler,
        object(),
        noise,
        conditions,
        schedule,
        source=source,
        protected_mask=np.zeros((2, 1), dtype=np.float32),
        strength=1.0,
        **kwargs,
    )

    assert np.array_equal(refine, main)
    assert not np.array_equal(refine, source)


def test_shared_sampler_restores_protected_source_after_every_step():
    observed = []

    class Sampler:
        sigma_min = 1e-5

        def sample_once(self, _model, sample, *_args, **_kwargs):
            observed.append(sample.copy())
            return types.SimpleNamespace(pred_x_prev=sample + 10.0)

    source = np.asarray(((1.0, 2.0), (3.0, 4.0)), dtype=np.float32)
    noise = np.full_like(source, 50.0)
    protected = np.asarray(((1.0,), (0.0,)), dtype=np.float32)
    conditions = {"eye-000": {"cond": object(), "neg_cond": object()}}

    result = _alternating_sample(
        Sampler(),
        object(),
        noise,
        conditions,
        ("eye-000", "eye-000"),
        stage="test",
        rescale_t=3.0,
        guidance_strength=1.0,
        guidance_rescale=0.5,
        guidance_interval=(0.6, 1.0),
        source=source,
        protected_mask=protected,
        strength=1.0,
    )

    assert all(row[0].tolist() == source[0].tolist() for row in observed)
    assert result[0].tolist() == source[0].tolist()
    assert result[1].tolist() != source[1].tolist()


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


def test_refine_worker_delegates_to_shared_main_postprocess(
    tmp_path: Path, monkeypatch
):
    called = {}
    shared = types.ModuleType("trellis_mesh_postprocess")

    def fake_postprocess(cache, output, settings, resolution, *, progress):
        called.update(
            cache=cache,
            output=output,
            settings=settings,
            resolution=resolution,
            progress=progress,
        )
        return output / "shape.glb", {"status": "shared"}

    shared.run_mesh_postprocess = fake_postprocess
    monkeypatch.setitem(sys.modules, "trellis_mesh_postprocess", shared)
    cache = tmp_path / "decoded.npz"
    settings = {"cleanup": True}

    result = _postprocess_refined_mesh(cache, tmp_path, settings, 1536)

    assert result == (tmp_path / "shape.glb", {"status": "shared"})
    assert called == {
        "cache": cache,
        "output": tmp_path,
        "settings": settings,
        "resolution": 1536,
        "progress": called["progress"],
    }
    assert callable(called["progress"])


def test_region_shape_cache_handoff_uses_raw_decode_and_migrates_legacy_stage(
    tmp_path: Path,
):
    raw = tmp_path / "decoded-region-raw.npz"
    raw.touch()
    prepared = tmp_path / "decoded-region-postprocess-input.npz"
    prepared.touch()
    stage = tmp_path / "shape-stage-result.json"
    stage.write_text(json.dumps({
        "shape_request_key": "shape-key",
        "shape": str(prepared),
        "postprocess_cache": str(prepared),
        "pre_simplification": {"faces": 250_000},
    }), encoding="utf-8")

    assert _reuse_decoded_shape_cache(stage, raw, "shape-key")

    migrated = json.loads(stage.read_text(encoding="utf-8"))
    assert migrated["shape"] == str(raw.resolve())
    assert migrated["postprocess_cache"] == str(raw.resolve())
    assert "pre_simplification" not in migrated


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
    front = tmp_path / "front.png"
    image = tmp_path / "right.png"
    geometry = tmp_path / "geometry.glb"
    python.touch()
    model.mkdir()
    front.write_bytes(b"front image")
    image.write_bytes(b"image")
    geometry.write_bytes(b"mesh")
    controller = MultiviewStudioController()
    controller.set_source("front", front)
    controller.set_geometry_path(geometry)
    controller.set_slot_image(ViewKey("eye", 90), image)
    controller.set_refine_cube((0.0, 0.0, 0.0), 1.0)
    controller.confirm_refine_cube()
    controller.set_refine_mask_weights(0, (((0, 1.0),),))
    controller.set_refine_view_patch(
        0, ViewKey("eye", 90), (0.1, 0.2, 0.8, 0.9)
    )
    controller.set_refine_view_patch(
        0, ViewKey("eye", 0), (0.2, 0.1, 0.9, 0.8)
    )
    controller.set_refine_shape_setting(0, "warmup_steps", 3)
    controller.set_refine_shape_setting(0, "steps", 7)
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
    assert len(calls) == 2
    assert Path(calls[0][1]).name == "trellis_refine_runner.py"
    assert Path(calls[1][1]).name == "trellis_refine_postprocess_runner.py"
    assert request["schedule"] == [
        "eye-000",
        "eye-000",
        "eye-000",
        "eye-000",
        "eye-090",
        "eye-000",
        "eye-090",
    ]
    assert request["warmup_steps"] == 3
    right_patch = next(
        patch for patch in request["patches"] if patch["id"] == "eye-090"
    )
    assert right_patch["bounds"] == [0.1, 0.2, 0.8, 0.9]
    assert request["postprocess"] == _region_postprocess_payload(
        controller.project, controller.project.region_refine_settings(0)
    )
    assert request["postprocess"]["decimation_target"] == 250_000
    assert request["postprocess_resolution"] == 1024
    assert request["shape_request_key"] == generator.shape_refine_key(
        controller.project, 0, arrays
    )

    changed_main_postprocess = replace(
        controller.project,
        trellis=replace(
            controller.project.trellis,
            postprocess=replace(
                controller.project.trellis.postprocess, remesh=False
            ),
        ),
    )
    assert generator.refine_key(
        changed_main_postprocess, 0, arrays
    ) == generator.refine_key(controller.project, 0, arrays)
    assert generator.shape_refine_key(
        changed_main_postprocess, 0, arrays
    ) == generator.shape_refine_key(controller.project, 0, arrays)

    changed_warmup = replace(
        controller.project,
        refine_shape_settings=(replace(
            controller.project.region_refine_settings(0), warmup_steps=2
        ),),
    )
    assert generator.shape_refine_key(
        changed_warmup, 0, arrays
    ) != generator.shape_refine_key(controller.project, 0, arrays)

    region_settings = controller.project.region_refine_settings(0)
    changed_region_postprocess = replace(
        controller.project,
        refine_shape_settings=(replace(
            region_settings,
            postprocess=replace(region_settings.postprocess, remesh=False),
        ),),
    )
    assert generator.refine_key(
        changed_region_postprocess, 0, arrays
    ) != generator.refine_key(controller.project, 0, arrays)
    assert generator.shape_refine_key(
        changed_region_postprocess, 0, arrays
    ) == generator.shape_refine_key(controller.project, 0, arrays)


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


def test_texture_resume_loads_only_models_needed_after_each_cache_stage():
    assert _texture_model_names(
        1024,
        shape_cached=False,
        guides_cached=False,
        texture_cached=False,
    ) == (
        "tex_slat_decoder",
        "shape_slat_encoder",
        "tex_slat_flow_model_1024",
    )
    assert _texture_model_names(
        1024,
        shape_cached=True,
        guides_cached=True,
        texture_cached=False,
    ) == (
        "tex_slat_decoder",
        "tex_slat_flow_model_1024",
    )
    assert _texture_model_names(
        1024,
        shape_cached=True,
        guides_cached=True,
        texture_cached=True,
    ) == ("tex_slat_decoder",)
    assert _texture_model_names(
        1024,
        shape_cached=True,
        guides_cached=False,
        texture_cached=True,
    ) == ("tex_slat_decoder", "shape_slat_encoder")
