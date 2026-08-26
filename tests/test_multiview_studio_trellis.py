from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from diffusion_editor.multiview_studio.model import (
    MultiviewProject,
    TrellisShapeSettings,
    ViewKey,
)
from diffusion_editor.multiview_studio.trellis_generation import schedule_images
from diffusion_editor.multiview_studio.trellis_shape_runner import (
    _gltf_y_up_transform,
)


def test_shape_schedule_warms_up_front_then_cycles_only_included(tmp_path: Path):
    front = tmp_path / "front.png"
    right = tmp_path / "right.png"
    left = tmp_path / "left.png"
    front.touch()
    right.touch()
    left.touch()
    project = MultiviewProject().with_source("front", str(front))
    project = project.with_slot(ViewKey("eye", 90), image_path=str(right))
    project = project.with_slot(ViewKey("eye", 270), image_path=str(left))
    for slot in project.slots:
        desired = slot.key in {
            ViewKey("eye", 90),
            ViewKey("eye", 270),
        }
        project = project.with_slot(slot.key, include_in_trellis=desired)
    project = replace(
        project,
        trellis=TrellisShapeSettings(total_steps=7, warmup_steps=3),
    )

    schedule = schedule_images(project)

    assert [key for key, _path in schedule] == [
        ViewKey("eye", 0),
        ViewKey("eye", 0),
        ViewKey("eye", 0),
        ViewKey("eye", 90),
        ViewKey("eye", 270),
        ViewKey("eye", 90),
        ViewKey("eye", 270),
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
