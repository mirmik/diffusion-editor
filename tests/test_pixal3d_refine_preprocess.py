from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from diffusion_editor.workers.pixal3d_staged_runner import (
    _detail_crop_box,
    _preprocess_refine_pair,
    _remap_projected_pixels_to_crop,
    _texture_detail_crop,
)


class _Pipeline:
    low_vram = False


def test_refine_preprocessing_applies_the_same_subject_crop_to_mask() -> None:
    source = np.zeros((80, 120, 4), dtype=np.uint8)
    source[20:60, 40:80, :3] = (200, 120, 40)
    source[20:60, 40:80, 3] = 255
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[30:50, 50:70] = 255

    condition, cropped_mask = _preprocess_refine_pair(
        _Pipeline(),
        Image.fromarray(source, "RGBA"),
        Image.fromarray(mask, "L"),
    )

    assert condition.size == cropped_mask.size
    assert condition.size[0] == condition.size[1]
    condition_alpha_region = np.asarray(condition).sum(axis=2) > 0
    mask_region = np.asarray(cropped_mask) > 0
    assert mask_region.any()
    assert np.all(condition_alpha_region[mask_region])


def test_refine_preprocessing_rejects_a_mask_from_another_canvas() -> None:
    with pytest.raises(ValueError, match="must match source image dimensions"):
        _preprocess_refine_pair(
            _Pipeline(),
            Image.new("RGBA", (32, 32), (255, 255, 255, 128)),
            Image.new("L", (31, 32), 255),
        )


def test_texture_detail_crop_magnifies_the_masked_region() -> None:
    image = Image.new("RGB", (128, 128), "black")
    mask = Image.new("L", (128, 128), 0)
    mask.paste(255, (50, 20, 70, 40))

    detail = _texture_detail_crop(image, mask)

    assert detail.width == detail.height
    assert 20 < detail.width < image.width


def test_detail_crop_box_is_square_and_contains_the_mask() -> None:
    mask = Image.new("L", (128, 128), 0)
    mask.paste(255, (70, 15, 90, 45))

    x0, y0, x1, y1 = _detail_crop_box(mask)

    assert x1 - x0 == y1 - y0
    assert x0 <= 70 < 90 <= x1
    assert y0 <= 15 < 45 <= y1


def test_projected_pixels_are_remapped_into_resized_detail_crop() -> None:
    points = np.array([[[256.0, 384.0], [512.0, 640.0]]], dtype=np.float32)

    mapped = _remap_projected_pixels_to_crop(
        points,
        crop_box=(32, 48, 96, 112),
        full_size=(128, 128),
        projection_resolution=1024,
    )

    np.testing.assert_allclose(mapped[0, 0], (0.0, 0.0))
    np.testing.assert_allclose(mapped[0, 1], (512.0, 512.0))
    np.testing.assert_allclose(points[0, 0], (256.0, 384.0))
