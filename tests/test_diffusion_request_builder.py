import numpy as np
import pytest
from PIL import Image

from diffusion_editor.generation.diffusion_request_builder import DiffusionRequestBuilder
from diffusion_editor.document.layer import Layer
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.document.tool import DiffusionTool


def _rgba(width, height, color):
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:] = color
    return arr


def _tool(mode="inpaint"):
    return DiffusionTool(
        source_patch=None,
        patch_x=0, patch_y=0, patch_w=4, patch_h=4,
        prompt="p", negative_prompt="np",
        strength=0.5, guidance_scale=7.0, steps=20, seed=1,
        mode=mode,
    )


def test_build_diffusion_request_uses_layer_patch_in_canvas_coordinates():
    stack = LayerStack()
    stack.init_from_image(_rgba(8, 8, (0, 0, 0, 255)))
    layer = Layer("Diff", 4, 4, _rgba(4, 4, (0, 0, 0, 0)), x=2, y=1)
    layer.patch_rect = (1, 1, 3, 4)
    layer.mask.data[1:4, 1:3] = 1.0
    layer.tool = _tool()
    stack.insert_layer(layer)
    composite = _rgba(8, 8, (10, 20, 30, 255))

    result = DiffusionRequestBuilder(stack).build(layer, composite)

    assert result.error is None
    request = result.request
    assert request.image.size == (2, 3)
    assert request.mask_image.size == (2, 3)
    assert request.width == 2
    assert request.height == 3
    assert layer.tool.patch_x == 3
    assert layer.tool.patch_y == 2
    assert layer.tool.patch_w == 2
    assert layer.tool.patch_h == 3


def test_build_diffusion_request_attaches_ip_adapter_reference_image():
    stack = LayerStack()
    stack.init_from_image(_rgba(6, 6, (0, 0, 0, 255)))
    ref = Layer("Reference", 2, 2, _rgba(2, 2, (50, 60, 70, 255)))
    stack.insert_layer(ref)
    layer = Layer("Diff", 6, 6, _rgba(6, 6, (0, 0, 0, 0)))
    layer.tool = _tool(mode="txt2img")
    layer.tool.ip_adapter_layer_id = ref.id
    layer.tool.ip_adapter_layer_name_hint = ref.name
    stack.insert_layer(layer)

    result = DiffusionRequestBuilder(stack).build(layer, None)

    assert result.error is None
    request = result.request
    assert request.ip_adapter_image is not None
    assert request.ip_adapter_image.size == (2, 2)
    assert request.ip_adapter_image.getpixel((0, 0)) == (50, 60, 70)


@pytest.mark.parametrize("resize", [False, True])
@pytest.mark.parametrize("cached_source", [False, True])
def test_build_diffusion_request_without_mask_uses_full_canvas(resize, cached_source):
    stack = LayerStack()
    stack.init_from_image(_rgba(16, 8, (0, 0, 0, 255)))
    layer = Layer("Diff", 4, 4, _rgba(4, 4, (0, 0, 0, 0)), x=2, y=1)
    layer.tool = _tool(mode="inpaint")
    layer.tool.resize_to_model_resolution = resize
    if cached_source:
        layer.tool.source_patch = Image.new("RGB", (4, 4), (90, 80, 70))
    stack.insert_layer(layer)

    result = DiffusionRequestBuilder(stack).build(
        layer,
        _rgba(16, 8, (10, 20, 30, 255)),
    )

    assert result.error is None
    request = result.request
    expected_size = (1024, 512) if resize else (16, 8)
    assert request.image.size == request.mask_image.size == expected_size
    assert (request.width, request.height) == expected_size
    assert request.image.getpixel((0, 0)) == (10, 20, 30)
    assert request.mask_image.mode == "L"
    assert request.mask_image.getextrema() == (255, 255)
    assert (layer.tool.patch_x, layer.tool.patch_y, layer.tool.patch_w, layer.tool.patch_h) == (0, 0, 16, 8)
    assert not layer.has_mask()


def test_unmasked_inpaint_preserves_explicit_processing_area():
    stack = LayerStack()
    stack.init_from_image(_rgba(16, 8, (0, 0, 0, 255)))
    layer = Layer("Diff", 8, 8, _rgba(8, 8, (0, 0, 0, 0)), x=2, y=0)
    layer.tool = _tool()
    layer.patch_rect = (1, 2, 5, 6)
    stack.insert_layer(layer)
    result = DiffusionRequestBuilder(stack).build(layer, _rgba(16, 8, (10, 20, 30, 255)))
    assert result.error is None
    assert result.request.image.size == result.request.mask_image.size == (4, 4)
    assert result.request.mask_image.getextrema() == (255, 255)
    assert (layer.tool.patch_x, layer.tool.patch_y) == (3, 2)
    assert not layer.has_mask()
