from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
import sys

from PIL import Image
import pytest

from diffusion_editor.generation.image_edit_profiles import (
    SENSENOVA_U15_PROFILE_ID,
    image_edit_profile,
)
from diffusion_editor.workers import sensenova_image_edit


@pytest.mark.parametrize(
    ("reference", "expected_sizes"),
    [
        (None, [(64, 32)]),
        (Image.new("RGBA", (32, 64), "blue"), [(64, 32), (32, 64)]),
    ],
)
def test_sensenova_adapter_preserves_ordered_single_or_multi_image_input(
        monkeypatch, reference, expected_sizes):
    captured: dict[str, object] = {}

    class FakeModel:
        def it2i_generate(self, tokenizer, prompt, images, **kwargs):
            captured["tokenizer"] = tokenizer
            captured["prompt"] = prompt
            captured["images"] = images
            captured["kwargs"] = kwargs
            return Image.new("RGB", (64, 32), "green")

    model = FakeModel()
    pipeline = sensenova_image_edit.SenseNovaImageEditPipeline.__new__(
        sensenova_image_edit.SenseNovaImageEditPipeline)
    pipeline.tokenizer = object()
    pipeline._smart_resize = lambda *, height, width, **_kwargs: (
        height, width)
    pipeline._offload_context = lambda: nullcontext(model)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=nullcontext),
    )
    monkeypatch.setattr(
        sensenova_image_edit, "_tensor_to_image", lambda output: output)
    parameters = image_edit_profile(SENSENOVA_U15_PROFILE_ID).defaults()
    parameters.update({
        "prompt": "combine image 1 and image 2",
        "width": 64,
        "height": 32,
    })

    result = pipeline.edit(
        Image.new("RGBA", (64, 32), "red"),
        parameters,
        123,
        reference_image=reference,
    )

    assert result.size == (64, 32)
    images = captured["images"]
    assert [image.size for image in images] == expected_sizes
    assert all(image.mode == "RGB" for image in images)
    assert images[0].getpixel((0, 0)) == (255, 0, 0)
    if reference is not None:
        assert images[1].getpixel((0, 0)) == (0, 0, 255)
    assert captured["kwargs"]["seed"] == 123
