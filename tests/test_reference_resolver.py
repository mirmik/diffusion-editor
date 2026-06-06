import numpy as np

from diffusion_editor.document.layer import Layer
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.generation.reference_resolver import (
    layer_reference_image,
    resolve_ip_adapter_reference,
)
from diffusion_editor.document.tool import DiffusionTool


def _rgba(width, height, color):
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:] = color
    return arr


def _tool():
    return DiffusionTool(
        source_patch=None,
        patch_x=0, patch_y=0, patch_w=4, patch_h=4,
        prompt="", negative_prompt="",
        strength=0.5, guidance_scale=7.0, steps=20, seed=1,
    )


def test_reference_layer_patch_wins_over_alpha_bbox():
    image = _rgba(5, 5, (10, 20, 30, 255))
    layer = Layer("Reference", 5, 5, image)
    layer.patch_rect = (1, 1, 3, 4)

    ref = layer_reference_image(layer)

    assert ref.source == "patch"
    assert ref.local_rect == (1, 1, 3, 4)
    assert ref.image.size == (2, 3)


def test_reference_layer_uses_alpha_bbox_without_patch():
    image = _rgba(5, 5, (0, 0, 0, 0))
    image[2, 3] = (10, 20, 30, 255)
    layer = Layer("Reference", 5, 5, image)

    ref = layer_reference_image(layer)

    assert ref.source == "alpha_bbox"
    assert ref.local_rect == (3, 2, 4, 3)
    assert ref.image.size == (1, 1)
    assert ref.image.getpixel((0, 0)) == (10, 20, 30)


def test_reference_layer_uses_full_layer_when_alpha_is_empty():
    layer = Layer("Reference", 2, 3, _rgba(2, 3, (0, 0, 0, 0)))

    ref = layer_reference_image(layer)

    assert ref.source == "full_layer"
    assert ref.local_rect == (0, 0, 2, 3)
    assert ref.image.size == (2, 3)
    assert ref.image.getpixel((0, 0)) == (255, 255, 255)


def test_resolve_ip_adapter_reference_reports_missing_layer():
    stack = LayerStack()
    stack.init_from_image(_rgba(4, 4, (0, 0, 0, 255)))
    tool = _tool()
    tool.ip_adapter_layer_id = "missing"
    tool.ip_adapter_layer_name_hint = "Old Reference"

    result = resolve_ip_adapter_reference(tool, stack)

    assert result.reference is None
    assert result.error is not None
    assert result.error.message == "IP-Adapter reference missing: Old Reference"
