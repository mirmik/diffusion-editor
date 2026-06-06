import numpy as np

from diffusion_editor.canvas.soft_mask_stroke import SoftMaskBrush, apply_dab, apply_line


def test_apply_dab_paints_float_mask_with_flow():
    mask = np.zeros((9, 9), dtype=np.float32)
    brush = SoftMaskBrush(size=4, hardness=1.0, flow=0.5)

    dirty, stamp = apply_dab(mask, 4, 4, brush)

    assert dirty == (2, 2, 6, 6)
    assert stamp.shape == (4, 4)
    assert mask.max() == 0.5
    assert mask[4, 4] == 0.5


def test_apply_dab_clips_to_mask_bounds():
    mask = np.zeros((5, 5), dtype=np.float32)
    brush = SoftMaskBrush(size=4, hardness=1.0, flow=1.0)

    dirty, stamp = apply_dab(mask, 0, 0, brush)

    assert dirty == (0, 0, 2, 2)
    assert stamp.shape == (2, 2)


def test_apply_dab_erases_by_capping_to_inverse_stamp():
    mask = np.ones((7, 7), dtype=np.float32)
    brush = SoftMaskBrush(size=4, hardness=1.0, flow=0.25)

    apply_dab(mask, 3, 3, brush, erase=True)

    assert mask[3, 3] == 0.75


def test_apply_line_paints_segment_bounds():
    mask = np.zeros((10, 10), dtype=np.float32)
    brush = SoftMaskBrush(size=4, hardness=1.0, flow=1.0)

    dirty, stamp = apply_line(mask, 2, 2, 7, 2, brush)

    assert dirty == (0, 0, 10, 5)
    assert stamp.shape == (5, 10)
    assert mask[2, 2] == 1.0
    assert mask[2, 7] == 1.0


def test_apply_line_degenerates_to_dab_for_tiny_segment():
    line_mask = np.zeros((7, 7), dtype=np.float32)
    dab_mask = np.zeros((7, 7), dtype=np.float32)
    brush = SoftMaskBrush(size=4, hardness=0.5, flow=0.75)

    line_dirty, line_stamp = apply_line(line_mask, 3, 3, 3, 3, brush)
    dab_dirty, dab_stamp = apply_dab(dab_mask, 3, 3, brush)

    assert line_dirty == dab_dirty
    assert np.array_equal(line_stamp, dab_stamp)
    assert np.array_equal(line_mask, dab_mask)


def test_apply_line_returns_empty_when_outside_mask():
    mask = np.zeros((5, 5), dtype=np.float32)
    brush = SoftMaskBrush(size=2, hardness=1.0, flow=1.0)

    assert apply_line(mask, 20, 20, 25, 25, brush) == (None, None)
