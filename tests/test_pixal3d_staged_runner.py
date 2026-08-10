from __future__ import annotations

import numpy as np

from diffusion_editor.workers.pixal3d_staged_runner import (
    _save_shape_checkpoint,
    _lr_conditioner,
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


def test_lr_conditioner_uses_1024_model_with_lr_projection_grid() -> None:
    class Pipeline:
        image_cond_model_shape_512 = object()
        image_cond_model_shape_1024 = object()

    pipeline = Pipeline()

    assert _lr_conditioner(pipeline, 512) == (
        pipeline.image_cond_model_shape_512,
        None,
    )
    assert _lr_conditioner(pipeline, 1024) == (
        pipeline.image_cond_model_shape_1024,
        32,
    )


def test_shape_checkpoint_is_pickle_free_and_records_compatibility(tmp_path) -> None:
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def detach(self):
            return self

        def to(self, _device):
            return self

        def numpy(self):
            return self.values

    class Latent:
        coords = Tensor([[0, 1, 2, 3]])
        feats = Tensor([[0.25, -0.5]])

    source = tmp_path / "preprocessed.png"
    source.write_bytes(b"source")
    checkpoint = tmp_path / "shape.npz"

    _save_shape_checkpoint(
        checkpoint,
        Latent(),
        1536,
        {"camera_angle_x": 0.8, "distance": 2.0, "mesh_scale": 1.0},
        source,
        tmp_path / "model",
    )

    with np.load(checkpoint, allow_pickle=False) as saved:
        assert saved["protocol_version"].tolist() == [1]
        assert saved["resolution"].tolist() == [1536]
        assert saved["coords"].tolist() == [[0, 1, 2, 3]]
        assert saved["normalized_feats"].tolist() == [[0.25, -0.5]]
        assert len(str(saved["source_sha256"].item())) == 64
