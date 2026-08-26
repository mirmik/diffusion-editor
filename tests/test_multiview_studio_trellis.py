from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import threading

from diffusion_editor.multiview_studio.model import (
    MultiviewProject,
    MeshPostprocessSettings,
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
