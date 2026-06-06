import numpy as np

from diffusion_editor.canvas.canvas_selection_paint import CanvasSelectionPainter


def test_selection_painter_dab_paints_selection():
    selection = np.zeros((9, 9), dtype=np.float32)
    painter = CanvasSelectionPainter()
    painter.set_brush(5, 1.0, 1.0)

    dirty, _stamp = painter.dab(selection, 4, 4)

    assert dirty == (2, 2, 7, 7)
    assert selection[4, 4] == 1.0


def test_selection_painter_line_paints_between_points():
    selection = np.zeros((12, 12), dtype=np.float32)
    painter = CanvasSelectionPainter()
    painter.set_brush(3, 1.0, 1.0)

    dirty, _stamp = painter.line(selection, 2, 6, 9, 6)

    assert dirty[0] <= 1
    assert dirty[2] >= 10
    assert selection[6, 5] > 0.0


def test_selection_painter_eraser_reduces_selection():
    selection = np.ones((9, 9), dtype=np.float32)
    painter = CanvasSelectionPainter()
    painter.set_brush(5, 1.0, 1.0)
    painter.set_eraser(True)

    dirty, _stamp = painter.dab(selection, 4, 4)

    assert dirty == (2, 2, 7, 7)
    assert selection[4, 4] == 0.0
    assert selection[0, 0] == 1.0


def test_selection_painter_ignores_empty_selection_array():
    selection = np.zeros((0, 0), dtype=np.float32)
    painter = CanvasSelectionPainter()

    dirty, stamp = painter.dab(selection, 0, 0)

    assert dirty is None
    assert stamp is None


def test_selection_painter_clamps_flow():
    selection = np.zeros((5, 5), dtype=np.float32)
    painter = CanvasSelectionPainter()
    painter.set_brush(3, 1.0, 3.0)

    painter.dab(selection, 2, 2)

    assert painter.flow == 1.0
    assert selection[2, 2] == 1.0
