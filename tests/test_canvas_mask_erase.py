import numpy as np

from diffusion_editor.canvas.canvas_mask_erase import MaskEraseStrokeBuffer


def test_mask_erase_begin_allocates_float_mask():
    stroke = MaskEraseStrokeBuffer()

    assert stroke.begin(4, 5) is True

    assert stroke.mask.shape == (4, 5)
    assert stroke.mask.dtype == np.float32
    assert stroke.dirty_rect is None


def test_mask_erase_accumulates_dirty_rects():
    stroke = MaskEraseStrokeBuffer()
    stroke.begin(8, 8)

    stroke.add_dirty((2, 3, 4, 5))
    stroke.add_dirty((1, 4, 6, 7))

    assert stroke.dirty_rect == (1, 3, 6, 7)


def test_mask_erase_preview_region_caps_layer_mask_by_inverse_erase():
    layer_mask = np.ones((5, 5), dtype=np.float32)
    stroke = MaskEraseStrokeBuffer()
    stroke.begin(5, 5)
    stroke.mask[2, 2] = 0.75

    preview = stroke.preview_region(layer_mask, (2, 2, 3, 3))

    assert preview.shape == (1, 1)
    assert preview[0, 0] == 0.25


def test_mask_erase_apply_to_layer_mask_returns_dirty_rect():
    layer_mask = np.ones((5, 5), dtype=np.float32)
    stroke = MaskEraseStrokeBuffer()
    stroke.begin(5, 5)
    stroke.mask[1:3, 2:4] = 0.5
    stroke.add_dirty((2, 1, 4, 3))

    dirty = stroke.apply_to_layer_mask(layer_mask)

    assert dirty == (2, 1, 4, 3)
    assert np.all(layer_mask[1:3, 2:4] == 0.5)
    assert layer_mask[0, 0] == 1.0


def test_mask_erase_clear_resets_runtime_state():
    stroke = MaskEraseStrokeBuffer()
    stroke.begin(4, 4)
    stroke.add_dirty((0, 0, 1, 1))

    stroke.clear()

    assert stroke.mask is None
    assert stroke.dirty_rect is None
