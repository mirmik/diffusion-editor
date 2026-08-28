from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from diffusion_editor.multiview_studio.controller import (
    MultiviewStudioController,
)
from diffusion_editor.multiview_studio.model import (
    MeshPostprocessSettings,
    MultiviewProject,
    RefineCube,
    RefineFaceMask,
    RefineShapeResult,
    RefineShapeSettings,
    RefineViewPatch,
    TrellisShapeSettings,
    TrellisTextureSettings,
    ViewKey,
    view_schedule,
)


def test_project_has_canonical_24_view_grid_and_maps_sources():
    project = MultiviewProject()

    assert len(project.slots) == 24
    assert project.slots[0].key == ViewKey("low", 0)
    assert project.slots[-1].key == ViewKey("elevated", 315)

    project = project.with_source("front", "/tmp/front.png")
    project = project.with_source("back", "/tmp/back.png")

    assert project.slot(ViewKey("eye", 0)).image_path == "/tmp/front.png"
    assert project.slot(ViewKey("eye", 180)).image_path == "/tmp/back.png"
    assert project.validate_shape_request() == ()


def test_refine_cfg_defaults_to_one_in_new_and_legacy_settings():
    assert RefineShapeSettings().cfg == 1.0
    assert RefineShapeSettings.from_dict({}).cfg == 1.0
    assert RefineShapeSettings().warmup_steps == 15
    assert RefineShapeSettings.from_dict({}).warmup_steps == 15


def test_every_populated_slot_participates_in_schedule():
    project = MultiviewProject().with_source("front", "/tmp/front.png")
    project = project.with_slot(ViewKey("eye", 90), image_path="/tmp/right.png")

    assert project.slot(ViewKey("eye", 0)).populated
    assert [slot.key for slot in project.populated_slots()] == [
        ViewKey("eye", 0),
        ViewKey("eye", 90),
    ]
    assert view_schedule(project.slots, 4, 2) == (
        ViewKey("eye", 0),
        ViewKey("eye", 0),
        ViewKey("eye", 0),
        ViewKey("eye", 90),
    )


def test_project_roundtrip_uses_relative_paths_inside_project(tmp_path: Path):
    front = tmp_path / "front.png"
    back = tmp_path / "back.png"
    front.touch()
    back.touch()
    manifest = tmp_path / "vaan.mvstudio.json"
    project = MultiviewProject().with_source("front", str(front))
    project = project.with_source("back", str(back))
    project = MultiviewProject(
        front_path=project.front_path,
        back_path=project.back_path,
        qwen_seed=17,
        slots=project.slots,
        trellis=TrellisShapeSettings(
            seed=23,
            total_steps=34,
            warmup_steps=10,
            resolution=1536,
            decimation_target=250_000,
            postprocess=MeshPostprocessSettings(
                fill_holes=False,
                fill_hole_perimeter=0.0125,
                remesh=True,
                simplify=False,
                cleanup=True,
                final_repair=False,
                remove_isolated_double_faces=True,
            ),
        ),
    )

    project.save(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    restored = MultiviewProject.load(manifest)

    assert payload["sources"] == {"front": "front.png", "back": "back.png"}
    assert payload["trellis"]["postprocess"]["fill_hole_perimeter"] == 0.0125
    assert restored == project


def test_legacy_exclusion_flag_is_ignored(tmp_path: Path):
    front = tmp_path / "front.png"
    front.touch()
    manifest = tmp_path / "legacy.mvstudio.json"
    MultiviewProject().with_source("front", str(front)).save(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    front_view = next(
        view
        for view in payload["views"]
        if view["elevation"] == "eye" and view["azimuth"] == 0
    )
    front_view["include_in_trellis"] = False
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    restored = MultiviewProject.load(manifest)

    assert restored.populated_slots() == (restored.slot(ViewKey("eye", 0)),)


def test_legacy_shape_path_becomes_texture_geometry_source(tmp_path: Path):
    shape = tmp_path / "legacy.glb"
    shape.touch()
    manifest = tmp_path / "legacy-shape.mvstudio.json"
    project = replace(MultiviewProject(), shape_path=str(shape))
    project.save(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("geometry")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    restored = MultiviewProject.load(manifest)

    assert restored.geometry_path == str(shape)
    assert restored.shape_path == str(shape)


def test_legacy_meshlib_repair_is_disabled_when_zero_face_stage_is_absent(
    tmp_path: Path,
):
    manifest = tmp_path / "legacy-postprocess.mvstudio.json"
    MultiviewProject().save(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    postprocess = payload["trellis"]["postprocess"]
    postprocess["final_repair"] = True
    postprocess.pop("remove_degenerate_faces")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    restored = MultiviewProject.load(manifest)

    assert not restored.trellis.postprocess.final_repair
    assert restored.trellis.postprocess.remove_degenerate_faces


def test_controller_tracks_dirty_state_and_persists(tmp_path: Path):
    events = []
    controller = MultiviewStudioController()
    controller.connect(lambda project, path, dirty: events.append((path, dirty)))

    controller.set_slot_image(ViewKey("low", 45), tmp_path / "low-45.png")
    assert controller.dirty

    path = controller.save(tmp_path / "project.mvstudio.json")
    assert path.is_file()
    assert not controller.dirty
    assert events[-1] == (path, False)


def test_controller_updates_fixed_mesh_postprocess_settings():
    controller = MultiviewStudioController()

    controller.set_mesh_postprocess("remesh", False)
    controller.set_mesh_postprocess("fill_hole_perimeter", 0.0075)

    assert not controller.project.trellis.postprocess.remesh
    assert controller.project.trellis.postprocess.fill_hole_perimeter == 0.0075
    assert controller.dirty


def test_texture_settings_and_geometry_source_roundtrip(tmp_path: Path):
    front = tmp_path / "front.png"
    geometry = tmp_path / "geometry.glb"
    textured = tmp_path / "textured.glb"
    front.touch()
    geometry.touch()
    textured.touch()
    manifest = tmp_path / "texture.mvstudio.json"
    project = MultiviewProject().with_source("front", str(front))
    project = replace(
        project,
        texture=TrellisTextureSettings(
            seed=99,
            total_steps=7,
            warmup_steps=6,
            resolution=512,
            texture_size=1536,
        ),
        geometry_path=str(geometry),
        shape_path=str(textured),
    )

    project.save(manifest)
    restored = MultiviewProject.load(manifest)

    assert restored == project
    assert restored.geometry_path == str(geometry)
    assert restored.shape_path == str(textured)


def test_texture_atlas_size_supports_1536():
    settings = TrellisTextureSettings(texture_size=1536)

    assert settings.texture_size == 1536


def test_refine_cube_roundtrip_is_bound_to_geometry(tmp_path: Path):
    geometry = tmp_path / "geometry.glb"
    geometry.write_bytes(b"mesh-v1")
    manifest = tmp_path / "refine.mvstudio.json"
    controller = MultiviewStudioController()
    controller.set_geometry_path(geometry)
    controller.set_refine_cube((1.0, 2.0, 3.0), 0.75)
    controller.confirm_refine_cube()

    controller.save(manifest)
    restored = MultiviewProject.load(manifest)

    assert restored.refine_cube is not None
    assert restored.refine_cube.center == (1.0, 2.0, 3.0)
    assert restored.refine_cube.side == 0.75
    assert restored.refine_cube.confirmed
    assert restored.refine_regions == (restored.refine_cube,)


def test_refine_face_mask_roundtrip_and_region_update_preserve_source_faces(
    tmp_path: Path,
):
    geometry = tmp_path / "geometry.glb"
    geometry.write_bytes(b"mesh-v1")
    manifest = tmp_path / "mask.mvstudio.json"
    controller = MultiviewStudioController()
    controller.set_geometry_path(geometry)
    controller.set_refine_cube((0.0, 0.0, 0.0), 1.0)
    controller.confirm_refine_cube()
    controller.set_refine_mask(0, ((9, 2, 2), (), (17,)))

    controller.select_refine_region(0)
    controller.update_refine_cube((0.25, 0.0, 0.0), 1.25)
    controller.confirm_refine_cube(0)
    controller.save(manifest)
    restored = MultiviewProject.load(manifest)

    assert restored.refine_masks == (
        RefineFaceMask(
            restored.refine_regions[0].geometry_fingerprint,
            ((2, 9), (), (17,)),
        ),
    )


def test_refine_vertex_weights_roundtrip_as_sparse_mask(tmp_path: Path):
    geometry = tmp_path / "geometry.glb"
    geometry.write_bytes(b"mesh-v1")
    manifest = tmp_path / "weighted-mask.mvstudio.json"
    controller = MultiviewStudioController()
    controller.set_geometry_path(geometry)
    controller.set_refine_cube((0.0, 0.0, 0.0), 1.0)
    controller.confirm_refine_cube()
    controller.set_refine_mask_weights(
        0,
        (((9, 0.25), (2, 1.0), (9, 0.75)), (), ((17, 0.5),)),
    )

    controller.save(manifest)
    payload = json.loads(manifest.read_text())
    restored = MultiviewProject.load(manifest)

    assert payload["refine"]["masks"][0]["kind"] == (
        "weighted-source-vertices"
    )
    assert restored.refine_masks[0].mesh_faces == ()
    assert restored.refine_masks[0].mesh_vertex_weights == (
        ((2, 1.0), (9, 0.75)),
        (),
        ((17, 0.5),),
    )


def test_refine_view_patches_roundtrip_and_follow_region_edits(tmp_path: Path):
    geometry = tmp_path / "geometry.glb"
    geometry.write_bytes(b"mesh-v1")
    front = tmp_path / "front.png"
    right = tmp_path / "right.png"
    front.touch()
    right.touch()
    manifest = tmp_path / "view-patches.mvstudio.json"
    controller = MultiviewStudioController()
    controller.set_geometry_path(geometry)
    controller.set_source("front", front)
    controller.set_slot_image(ViewKey("eye", 90), right)
    controller.set_refine_cube((0.0, 0.0, 0.0), 1.0)
    controller.confirm_refine_cube()

    controller.set_refine_view_patch(
        0, ViewKey("eye", 90), (0.2, 0.1, 0.8, 0.9)
    )
    controller.set_refine_view_patch(
        0, ViewKey("eye", 0), (0.1, 0.2, 0.6, 0.7)
    )
    controller.select_refine_region(0)
    controller.update_refine_cube((0.25, 0.0, 0.0), 1.25)
    controller.confirm_refine_cube(0)
    controller.save(manifest)
    restored = MultiviewProject.load(manifest)

    assert restored.view_patches(0) == (
        RefineViewPatch(ViewKey("eye", 0), (0.1, 0.2, 0.6, 0.7)),
        RefineViewPatch(ViewKey("eye", 90), (0.2, 0.1, 0.8, 0.9)),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["refine"]["view_patches"][0][0]["width"] == 0.5


def test_refine_view_patch_requires_populated_view_and_clears_with_slot(
    tmp_path: Path,
):
    geometry = tmp_path / "geometry.glb"
    geometry.write_bytes(b"mesh")
    view_path = tmp_path / "right.png"
    view_path.touch()
    key = ViewKey("eye", 90)
    controller = MultiviewStudioController()
    controller.set_geometry_path(geometry)
    controller.set_refine_cube((0.0, 0.0, 0.0), 1.0)
    controller.confirm_refine_cube()

    with pytest.raises(ValueError, match="populated view"):
        controller.set_refine_view_patch(0, key, (0.0, 0.0, 1.0, 1.0))

    controller.set_slot_image(key, view_path)
    controller.set_refine_view_patch(0, key, (0.0, 0.0, 1.0, 1.0))
    controller.clear_slot(key)

    assert controller.project.view_patches(0) == ()


def test_project_persists_up_to_eight_confirmed_refine_regions(tmp_path: Path):
    geometry = tmp_path / "geometry.glb"
    geometry.write_bytes(b"mesh-v1")
    manifest = tmp_path / "regions.mvstudio.json"
    controller = MultiviewStudioController()
    controller.set_geometry_path(geometry)

    for index in range(8):
        controller.set_refine_cube((float(index), 0.0, 0.0), 1.0)
        controller.confirm_refine_cube()

    controller.save(manifest)
    restored = MultiviewProject.load(manifest)

    assert len(restored.refine_regions) == 8
    assert [region.center[0] for region in restored.refine_regions] == list(
        map(float, range(8))
    )

    controller.set_refine_cube((9.0, 0.0, 0.0), 1.0)
    with pytest.raises(ValueError, match="at most 8"):
        controller.confirm_refine_cube()


def test_confirming_selected_region_updates_it_without_appending(tmp_path: Path):
    geometry = tmp_path / "geometry.glb"
    geometry.write_bytes(b"mesh-v1")
    controller = MultiviewStudioController()
    controller.set_geometry_path(geometry)
    controller.set_refine_cube((0.0, 0.0, 0.0), 1.0)
    controller.confirm_refine_cube()
    controller.set_refine_cube((2.0, 0.0, 0.0), 1.0)
    controller.confirm_refine_cube()

    controller.select_refine_region(0)
    controller.update_refine_cube((1.0, 2.0, 3.0), 0.5)
    controller.confirm_refine_cube(0)

    assert len(controller.project.refine_regions) == 2
    assert controller.project.refine_regions[0].center == (1.0, 2.0, 3.0)
    assert controller.project.refine_regions[0].side == 0.5
    assert controller.project.refine_regions[0].confirmed
    assert controller.project.refine_regions[1].center == (2.0, 0.0, 0.0)


def test_loading_drops_refine_cube_when_geometry_content_changed(tmp_path: Path):
    geometry = tmp_path / "geometry.glb"
    geometry.write_bytes(b"mesh-v1")
    manifest = tmp_path / "refine.mvstudio.json"
    controller = MultiviewStudioController()
    controller.set_geometry_path(geometry)
    controller.set_refine_cube((0.0, 0.0, 0.0), 1.0, confirmed=True)
    controller.save(manifest)

    geometry.write_bytes(b"mesh-v2")
    restored = MultiviewProject.load(manifest)

    assert restored.refine_cube is None
    assert restored.refine_regions == ()


def test_interactive_refine_cube_update_keeps_geometry_binding(tmp_path: Path):
    geometry = tmp_path / "geometry.glb"
    geometry.write_bytes(b"mesh")
    controller = MultiviewStudioController()
    controller.set_geometry_path(geometry)
    controller.set_refine_cube((0.0, 0.0, 0.0), 1.0, confirmed=True)
    fingerprint = controller.project.refine_cube.geometry_fingerprint

    controller.update_refine_cube((1.0, 2.0, 3.0), 2.5)

    assert controller.project.refine_cube == RefineCube(
        center=(1.0, 2.0, 3.0),
        side=2.5,
        geometry_fingerprint=fingerprint,
        confirmed=False,
    )


def test_numeric_refine_cube_edit_returns_confirmed_cube_to_draft(tmp_path: Path):
    geometry = tmp_path / "geometry.glb"
    geometry.write_bytes(b"mesh")
    controller = MultiviewStudioController()
    controller.set_geometry_path(geometry)
    controller.set_refine_cube((0.0, 0.0, 0.0), 1.0, confirmed=True)

    controller.set_refine_cube_value("y", 2.5)

    assert controller.project.refine_cube.center == (0.0, 2.5, 0.0)
    assert not controller.project.refine_cube.confirmed


def test_refine_settings_and_standalone_results_roundtrip(tmp_path: Path):
    geometry = tmp_path / "geometry.glb"
    geometry.write_bytes(b"mesh")
    run = tmp_path / "refine-runs" / "region-1"
    run.mkdir(parents=True)
    paths = {
        name: run / name
        for name in (
            "source-region.glb",
            "refined-region.glb",
            "source-shape-slat.npz",
            "refined-shape-slat.npz",
            "result.json",
            "texture-runs/texture-1/textured.glb",
            "texture-runs/texture-1/encoded-shape-slat.npz",
            "texture-runs/texture-1/texture-slat.npz",
            "texture-runs/texture-1/result.json",
        )
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    controller = MultiviewStudioController()
    controller.set_geometry_path(geometry)
    controller.set_refine_cube((0.0, 0.0, 0.0), 1.0)
    controller.confirm_refine_cube()
    controller.set_refine_shape_setting(0, "steps", 17)
    controller.set_refine_shape_setting(0, "strength", 0.65)
    controller.set_refine_postprocess(0, "remesh", False)
    controller.set_refine_postprocess(0, "fill_hole_perimeter", 0.0125)
    fingerprint = controller.project.refine_regions[0].geometry_fingerprint
    controller.add_refine_shape_result(0, RefineShapeResult(
        geometry_fingerprint=fingerprint,
        request_key="request-1",
        source_region_path=str(paths["source-region.glb"]),
        refined_region_path=str(paths["refined-region.glb"]),
        source_slat_path=str(paths["source-shape-slat.npz"]),
        refined_slat_path=str(paths["refined-shape-slat.npz"]),
        manifest_path=str(paths["result.json"]),
        textured_region_path=str(
            paths["texture-runs/texture-1/textured.glb"]
        ),
        texture_key="texture-1",
        texture_shape_slat_path=str(
            paths["texture-runs/texture-1/encoded-shape-slat.npz"]
        ),
        texture_slat_path=str(
            paths["texture-runs/texture-1/texture-slat.npz"]
        ),
        texture_manifest_path=str(
            paths["texture-runs/texture-1/result.json"]
        ),
    ))
    manifest = tmp_path / "project.mvstudio.json"

    controller.save(manifest)
    restored = MultiviewProject.load(manifest)

    assert restored.region_refine_settings(0) == RefineShapeSettings(
        steps=17,
        strength=0.65,
        postprocess=MeshPostprocessSettings(
            remesh=False,
            fill_hole_perimeter=0.0125,
        ),
    )
    assert restored.region_refine_results(0)[0].request_key == "request-1"
    assert restored.region_refine_results(0)[0].texture_key == "texture-1"
    assert Path(
        restored.region_refine_results(0)[0].textured_region_path
    ).name == "textured.glb"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["refine"]["shape_results"][0][0]["refined_region"] == (
        "refine-runs/region-1/refined-region.glb"
    )
    assert payload["refine"]["shape_results"][0][0]["textured_region"] == (
        "refine-runs/region-1/texture-runs/texture-1/textured.glb"
    )
    assert not payload["refine"]["shape_settings"][0]["postprocess"][
        "remesh"
    ]


def test_new_refine_region_copies_main_postprocess_then_changes_independently(
    tmp_path: Path,
):
    geometry = tmp_path / "geometry.glb"
    geometry.write_bytes(b"mesh")
    controller = MultiviewStudioController()
    controller.set_geometry_path(geometry)
    controller.set_mesh_postprocess("remesh", False)
    controller.set_refine_cube((0.0, 0.0, 0.0), 1.0)
    controller.confirm_refine_cube()

    assert not controller.project.region_refine_settings(0).postprocess.remesh

    controller.set_refine_postprocess(0, "remesh", True)

    assert controller.project.region_refine_settings(0).postprocess.remesh
    assert not controller.project.trellis.postprocess.remesh


def test_refine_request_allows_empty_protection_mask(
    tmp_path: Path,
):
    geometry = tmp_path / "geometry.glb"
    view = tmp_path / "view.png"
    geometry.write_bytes(b"mesh")
    view.touch()
    controller = MultiviewStudioController()
    controller.set_geometry_path(geometry)
    controller.set_slot_image(ViewKey("eye", 90), view)
    controller.set_refine_cube((0.0, 0.0, 0.0), 1.0)
    controller.confirm_refine_cube()
    controller.set_refine_mask_weights(0, ((), ()))
    controller.set_refine_view_patch(
        0, ViewKey("eye", 90), (0.0, 0.0, 1.0, 1.0)
    )
    controller.set_refine_shape_setting(0, "warmup_steps", 0)

    assert controller.project.validate_refine_request(0) == ()


def test_textured_result_does_not_replace_geometry_source(tmp_path: Path):
    controller = MultiviewStudioController()
    geometry = tmp_path / "geometry.glb"
    textured = tmp_path / "textured.glb"

    controller.set_geometry_path(geometry)
    controller.set_textured_shape_path(textured)

    assert controller.project.geometry_path == str(geometry.resolve())
    assert controller.project.shape_path == str(textured.resolve())


def test_texture_validation_covers_every_populated_view():
    project = MultiviewProject().with_source("front", "/tmp/front.png")
    project = project.with_slot(
        ViewKey("eye", 90), image_path="/tmp/right.png"
    )
    project = replace(
        project,
        geometry_path="/tmp/model.glb",
        texture=TrellisTextureSettings(total_steps=2, warmup_steps=1),
    )

    assert project.validate_texture_request() == (
        "Texture post-warmup steps must cover every populated view once",
    )


def test_trellis_settings_reject_warmup_longer_than_schedule():
    with pytest.raises(ValueError, match="warmup"):
        TrellisShapeSettings(total_steps=10, warmup_steps=11)


def test_shape_validation_requires_one_post_warmup_step_per_view():
    project = MultiviewProject().with_source("front", "/tmp/front.png")
    project = project.with_slot(
        ViewKey("eye", 90), image_path="/tmp/right.png"
    )
    project = replace(
        project,
        trellis=TrellisShapeSettings(total_steps=2, warmup_steps=1),
    )

    assert project.validate_shape_request() == (
        "Post-warmup steps must cover every populated view at least once",
    )
