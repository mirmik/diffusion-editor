import numpy as np

from diffusion_editor.canvas.canvas_mask_paint import CanvasMaskPainter


def test_mask_painter_dab_paints_mask():
    mask = np.zeros((9, 9), dtype=np.float32)
    painter = CanvasMaskPainter()
    painter.set_brush(5, 1.0, 1.0)

    dirty, _stamp = painter.dab(mask, 4, 4)

    assert dirty == (2, 2, 7, 7)
    assert mask[4, 4] == 1.0


def test_mask_painter_line_paints_between_points():
    mask = np.zeros((12, 12), dtype=np.float32)
    painter = CanvasMaskPainter()
    painter.set_brush(3, 1.0, 1.0)

    dirty, _stamp = painter.line(mask, 2, 6, 9, 6)

    assert dirty[0] <= 1
    assert dirty[2] >= 10
    assert mask[6, 5] > 0.0


def test_mask_painter_eraser_reduces_mask():
    mask = np.ones((9, 9), dtype=np.float32)
    painter = CanvasMaskPainter()
    painter.set_brush(5, 1.0, 1.0)
    painter.set_eraser(True)

    dirty, _stamp = painter.dab(mask, 4, 4)

    assert dirty == (2, 2, 7, 7)
    assert mask[4, 4] == 0.0
    assert mask[0, 0] == 1.0


def test_mask_painter_explicit_erase_overrides_eraser_state():
    mask = np.zeros((9, 9), dtype=np.float32)
    painter = CanvasMaskPainter()
    painter.set_brush(5, 1.0, 1.0)
    painter.set_eraser(True)

    painter.dab(mask, 4, 4, erase=False)

    assert mask[4, 4] == 1.0


def test_mask_painter_clamps_flow():
    mask = np.zeros((5, 5), dtype=np.float32)
    painter = CanvasMaskPainter()
    painter.set_brush(3, 1.0, 2.0)

    painter.dab(mask, 2, 2)

    assert painter.flow == 1.0
    assert mask[2, 2] == 1.0
