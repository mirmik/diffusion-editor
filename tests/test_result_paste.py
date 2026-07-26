"""Regression tests for clipping generated patches into layer-local images."""

import numpy as np
import pytest
from PIL import Image

from diffusion_editor.document.result_paste import paste_result


def _coordinate_image(width: int, height: int) -> tuple[Image.Image, np.ndarray]:
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :]
    pixels[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None]
    return Image.fromarray(pixels, "RGB"), pixels


def test_paste_result_clips_left_and_top_with_matching_source_crop():
    target = np.zeros((3, 3, 4), dtype=np.uint8)
    result, pixels = _coordinate_image(4, 4)

    changed = paste_result(target, result, -1, -1, 4, 4)

    expected = np.zeros_like(target)
    expected[:, :, :3] = pixels[1:4, 1:4]
    expected[:, :, 3] = 255
    assert changed is True
    assert np.array_equal(target, expected)


def test_paste_result_clips_right_and_bottom():
    target = np.zeros((3, 3, 4), dtype=np.uint8)
    result, pixels = _coordinate_image(4, 4)

    changed = paste_result(target, result, 2, 2, 4, 4)

    expected = np.zeros_like(target)
    expected[2, 2, :3] = pixels[0, 0]
    expected[2, 2, 3] = 255
    assert changed is True
    assert np.array_equal(target, expected)


def test_paste_result_outside_layer_is_a_noop():
    target = np.full((3, 3, 4), 11, dtype=np.uint8)
    before = target.copy()
    result, _pixels = _coordinate_image(4, 4)

    changed = paste_result(target, result, -5, 0, 4, 4)

    assert changed is False
    assert np.array_equal(target, before)


def test_paste_result_uses_layer_coordinates_for_mask_after_clipping():
    target = np.zeros((3, 3, 4), dtype=np.uint8)
    result, _pixels = _coordinate_image(4, 4)
    mask = np.zeros((3, 3), dtype=np.float32)
    mask[0, 0] = 0.25
    mask[2, 2] = 1.0

    paste_result(target, result, -1, -1, 4, 4, mask=mask)

    assert target[0, 0, 3] == 64
    assert target[2, 2, 3] == 255


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"patch_w": 0}, "patch_w"),
        ({"patch_h": 0}, "patch_h"),
        ({"mask": np.zeros((2, 2), dtype=np.uint8)}, "mask shape"),
    ],
)
def test_paste_result_validation_does_not_mutate_target(kwargs, message):
    target = np.full((3, 3, 4), 19, dtype=np.uint8)
    before = target.copy()
    result = Image.new("RGB", (3, 3), "red")
    arguments = {
        "paste_x": 0,
        "paste_y": 0,
        "patch_w": 3,
        "patch_h": 3,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        paste_result(target, result, **arguments)

    assert np.array_equal(target, before)
