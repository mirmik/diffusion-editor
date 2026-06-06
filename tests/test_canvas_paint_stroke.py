import numpy as np

from diffusion_editor.canvas.canvas_paint_stroke import (
    PaintStrokeBuffer,
    blend_paint_region,
)


def test_blend_paint_region_applies_source_over_color():
    base = np.zeros((4, 4, 4), dtype=np.uint8)
    base[:, :, 3] = 255
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[2, 1] = 255

    out = blend_paint_region(base, mask, (255, 0, 0, 255), (1, 2, 2, 3))

    assert out.shape == (1, 1, 4)
    assert out[0, 0].tolist() == [255, 0, 0, 255]


def test_paint_stroke_buffer_applies_dirty_region_from_base_image():
    layer_image = np.zeros((6, 6, 4), dtype=np.uint8)
    layer_image[:, :, 3] = 255
    stroke = PaintStrokeBuffer()
    assert stroke.begin(layer_image, (255, 0, 0, 255)) is True
    stroke.mask[2, 2] = 255
    layer_image[2, 2] = (0, 0, 255, 255)

    dirty = stroke.apply_region(layer_image, (2, 2, 3, 3))

    assert dirty == (2, 2, 3, 3)
    assert stroke.live_dirty_rect == dirty
    assert layer_image[2, 2].tolist() == [255, 0, 0, 255]


def test_paint_stroke_buffer_tracks_union_dirty_rect():
    layer_image = np.zeros((6, 6, 4), dtype=np.uint8)
    stroke = PaintStrokeBuffer()
    stroke.begin(layer_image, (255, 255, 255, 255))
    stroke.mask[1, 1] = 255
    stroke.mask[4, 4] = 255

    stroke.apply_region(layer_image, (1, 1, 2, 2))
    stroke.apply_region(layer_image, (4, 4, 5, 5))

    assert stroke.live_dirty_rect == (1, 1, 5, 5)


def test_paint_stroke_buffer_clear_resets_runtime_state():
    layer_image = np.zeros((2, 2, 4), dtype=np.uint8)
    stroke = PaintStrokeBuffer()
    stroke.begin(layer_image, (255, 255, 255, 255))
    stroke.mask[0, 0] = 255
    stroke.apply_region(layer_image, (0, 0, 1, 1))

    stroke.clear()

    assert stroke.mask is None
    assert stroke.live_dirty_rect is None
