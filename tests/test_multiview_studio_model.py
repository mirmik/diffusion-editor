from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from diffusion_editor.multiview_studio.controller import (
    MultiviewStudioController,
)
from diffusion_editor.multiview_studio.model import (
    MultiviewProject,
    TrellisShapeSettings,
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


def test_slot_exclusion_is_independent_from_image_and_schedule_cycles():
    project = MultiviewProject().with_source("front", "/tmp/front.png")
    project = project.with_slot(ViewKey("eye", 90), image_path="/tmp/right.png")
    project = project.with_slot(ViewKey("eye", 0), include_in_trellis=False)

    assert project.slot(ViewKey("eye", 0)).populated
    assert not project.slot(ViewKey("eye", 0)).include_in_trellis
    assert [slot.key for slot in project.included_slots()] == [
        ViewKey("eye", 90)
    ]
    assert view_schedule(project.slots, 4, 2) == (
        ViewKey("eye", 0),
        ViewKey("eye", 0),
        ViewKey("eye", 90),
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
    project = project.with_slot(
        ViewKey("elevated", 315), include_in_trellis=False
    )
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
        ),
    )

    project.save(manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    restored = MultiviewProject.load(manifest)

    assert payload["sources"] == {"front": "front.png", "back": "back.png"}
    assert restored == project


def test_controller_tracks_dirty_state_and_persists(tmp_path: Path):
    events = []
    controller = MultiviewStudioController()
    controller.connect(lambda project, path, dirty: events.append((path, dirty)))

    controller.set_slot_included(ViewKey("low", 45), False)
    assert controller.dirty

    path = controller.save(tmp_path / "project.mvstudio.json")
    assert path.is_file()
    assert not controller.dirty
    assert events[-1] == (path, False)


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
        "Post-warmup steps must cover every included view at least once",
    )
