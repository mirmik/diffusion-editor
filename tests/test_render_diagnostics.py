import numpy as np

from diffusion_editor.canvas.render_diagnostics import (
    rgba_signature,
    source_contribution_error,
)


def _source():
    image = np.zeros((8, 12, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    image[:, :, 0] = np.arange(12, dtype=np.uint8)[None, :] * 19
    image[:, :, 1] = np.arange(8, dtype=np.uint8)[:, None] * 31
    return image


def test_source_contribution_accepts_exact_and_bounded_rounding():
    source = _source()
    rounded = source.copy()
    rounded[3, 4, 0] += 2

    assert source_contribution_error(source, source.copy()) is None
    assert source_contribution_error(source, rounded) is None


def test_source_contribution_rejects_forced_zero_with_diagnostics():
    source = _source()
    blank = np.zeros_like(source)

    error = source_contribution_error(source, blank)

    assert error is not None
    assert "pixel mismatch" in error
    assert "composite=" in error
    assert "range=0..0" in rgba_signature(blank)


def test_source_contribution_rejects_missing_or_wrong_shape():
    source = _source()

    assert "no composite" in source_contribution_error(source, None)
    assert "shape mismatch" in source_contribution_error(
        source, np.zeros((4, 4, 4), dtype=np.uint8))
