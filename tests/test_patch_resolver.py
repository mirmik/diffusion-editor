import numpy as np

from diffusion_editor.document.layer import Layer
from diffusion_editor.generation.patch_resolver import (
    extract_layer_mask_patch,
    resolve_source_patch,
)


def _rgba(width, height, color):
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:] = color
    return arr


def test_extract_layer_mask_patch_uses_layer_local_mask_with_canvas_rect():
    layer = Layer("Layer", 4, 4, _rgba(4, 4, (0, 0, 0, 0)), x=3, y=2)
    layer.mask.data[1:3, 1:4] = 1.0

    mask = extract_layer_mask_patch(layer, (4, 3, 7, 5))
    pixels = np.array(mask, dtype=np.uint8)

    assert mask.size == (3, 2)
    assert np.all(pixels == 255)


def test_extract_layer_mask_patch_keeps_canvas_area_outside_layer_empty():
    layer = Layer("Layer", 2, 2, _rgba(2, 2, (0, 0, 0, 0)), x=2, y=2)
    layer.mask.data[:, :] = 1.0

    mask = extract_layer_mask_patch(layer, (1, 1, 4, 4))

    pixels = np.array(mask, dtype=np.uint8)
    assert pixels.shape == (3, 3)
    assert pixels[0, 0] == 0
    assert pixels[1, 1] == 255
    assert pixels[2, 2] == 255


def test_resolve_source_patch_uses_existing_rect_as_fallback():
    composite = _rgba(16, 16, (10, 20, 30, 255))
    layer = Layer("Layer", 16, 16, _rgba(16, 16, (0, 0, 0, 0)))

    patch = resolve_source_patch(
        layer,
        composite,
        fallback_canvas_rect=(2, 3, 10, 11),
    )

    assert patch.source == "existing"
    assert patch.canvas_rect == (2, 3, 10, 11)
    assert patch.image.size == (8, 8)
