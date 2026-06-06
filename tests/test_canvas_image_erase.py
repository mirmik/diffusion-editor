import numpy as np

from diffusion_editor.canvas.brush import Brush
from diffusion_editor.canvas.canvas_image_erase import (
    erase_alpha_region,
    erase_dab,
    erase_line,
)


def test_erase_dab_reduces_alpha_inside_brush_stamp():
    image = np.zeros((9, 9, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    brush = Brush()
    brush.set_size(3)
    brush.set_hardness(1.0)
    brush.set_color(255, 255, 255, 255)

    dirty = erase_dab(image, brush, 4, 4)

    assert dirty == (3, 3, 6, 6)
    assert image[4, 4, 3] == 0
    assert image[0, 0, 3] == 255


def test_erase_dab_clips_to_image_bounds():
    image = np.zeros((5, 5, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    brush = Brush()
    brush.set_size(5)
    brush.set_hardness(1.0)

    dirty = erase_dab(image, brush, 0, 0)

    assert dirty == (0, 0, 3, 3)
    assert image[0, 0, 3] == 0


def test_erase_line_interpolates_between_points():
    image = np.zeros((12, 12, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    brush = Brush()
    brush.set_size(3)
    brush.set_hardness(1.0)

    dirty = erase_line(image, brush, 2, 6, 9, 6)

    assert dirty == (0, 4, 11, 8)
    assert image[6, 2, 3] == 0
    assert image[6, 6, 3] == 0
    assert image[6, 9, 3] == 0


def test_erase_alpha_region_accepts_uint8_mask():
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[:, :, 3] = 200
    erase = np.array([[0, 128], [255, 64]], dtype=np.uint8)

    erase_alpha_region(image, (1, 1, 3, 3), erase)

    assert image[1, 1, 3] == 200
    assert image[1, 2, 3] == 99
    assert image[2, 1, 3] == 0
    assert image[2, 2, 3] == 149
