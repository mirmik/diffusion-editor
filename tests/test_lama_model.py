from __future__ import annotations

from PIL import Image

from diffusion_editor.workers.lama_model import _padded_array


def test_lama_input_arrays_are_normalized_and_padded_to_model_stride():
    image = Image.new("RGB", (11, 9), (255, 128, 0))
    mask = Image.new("L", image.size, 255)

    image_array = _padded_array(image, mode="RGB")
    mask_array = _padded_array(mask, mode="L")

    assert image_array.shape == (3, 16, 16)
    assert mask_array.shape == (1, 16, 16)
    assert image_array.dtype.name == "float32"
    assert mask_array.dtype.name == "float32"
    assert image_array.min() == 0
    assert image_array.max() == 1
    assert mask_array.min() == mask_array.max() == 1
