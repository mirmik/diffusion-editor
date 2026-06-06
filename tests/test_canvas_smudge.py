import numpy as np

from diffusion_editor.canvas.brush import Brush
from diffusion_editor.canvas.canvas_smudge import SmudgeStrokeBuffer, brush_rect_slices


def test_brush_rect_slices_clips_to_image_bounds():
    slices = brush_rect_slices((8, 8), (5, 5), 0, 0)

    assert slices == (2, 2, 5, 5, 0, 0, 3, 3)


def test_smudge_begin_captures_source_pixels():
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    image[3, 3] = (255, 0, 0, 255)
    brush = Brush()
    brush.set_size(3)
    stroke = SmudgeStrokeBuffer()

    stroke.begin(image, brush, 3, 3)

    assert stroke.buffer is not None
    assert stroke.buffer[1, 1].tolist() == [255.0, 0.0, 0.0, 255.0]


def test_smudge_dab_moves_carried_color_into_destination():
    image = np.zeros((12, 12, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    image[4:7, 2:5, :3] = (255, 0, 0)
    brush = Brush()
    brush.set_size(3)
    brush.set_hardness(1.0)
    brush.set_flow(1.0)
    brush.set_color(255, 255, 255, 255)
    stroke = SmudgeStrokeBuffer()
    stroke.begin(image, brush, 3, 5)

    dirty = stroke.dab(image, brush, 8, 5)

    assert dirty == (7, 4, 10, 7)
    assert image[5, 8, 0] > 0


def test_smudge_line_interpolates_between_distant_points():
    image = np.zeros((40, 40, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    image[17:23, 5:11, :3] = (255, 0, 0)
    brush = Brush()
    brush.set_size(6)
    brush.set_hardness(1.0)
    brush.set_flow(1.0)
    brush.set_color(255, 255, 255, 255)
    stroke = SmudgeStrokeBuffer()
    stroke.begin(image, brush, 8, 20)

    dirty = stroke.line(image, brush, 8, 20, 32, 20)

    assert dirty[0] <= 7
    assert dirty[2] >= 35
    assert image[20, 20, 0] > 0
    assert image[20, 32, 0] > 0


def test_smudge_clear_resets_buffer():
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    brush = Brush()
    stroke = SmudgeStrokeBuffer()
    stroke.begin(image, brush, 2, 2)

    stroke.clear()

    assert stroke.buffer is None
