from __future__ import annotations

from diffusion_editor.workers.pixal3d_staged_runner import (
    _merge_occupied_cells,
    _preview_transform,
)


def test_merge_occupied_cells_preserves_block_as_one_cuboid() -> None:
    cells = [
        (x, y, z)
        for x in range(2)
        for y in range(2)
        for z in range(2)
    ]

    assert _merge_occupied_cells(cells) == [((0, 0, 0), (2, 2, 2))]


def test_merge_occupied_cells_keeps_disconnected_cell_separate() -> None:
    cells = [(0, 0, 0), (1, 0, 0), (4, 3, 2)]

    assert _merge_occupied_cells(cells) == [
        ((0, 0, 0), (2, 1, 1)),
        ((4, 3, 2), (5, 4, 3)),
    ]


def test_preview_transform_preserves_y_up_and_flips_view_azimuth() -> None:
    transform = _preview_transform()

    assert tuple(transform @ (0.0, 1.0, 0.0, 1.0)) == (0.0, 1.0, 0.0, 1.0)
    assert tuple(transform @ (1.0, 0.0, 2.0, 1.0)) == (-1.0, 0.0, -2.0, 1.0)
