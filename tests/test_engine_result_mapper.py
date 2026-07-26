"""Tests for mapping engine outputs to document commands."""

import numpy as np
from PIL import Image

from diffusion_editor.document.commands import (
    ApplyGeneratedResultCommand,
    ReplaceLayerMaskCommand,
    SetLayerSelectionCommand,
)
from diffusion_editor.generation.result_mapper import (
    map_segmentation_result, map_lama_result,
    map_instruct_result, map_diffusion_result, map_grounding_result,
)
from diffusion_editor.grounding.types import GroundingDetection, GroundingResult
from diffusion_editor.document.layer import Layer
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.document.tool import DiffusionTool, LamaTool, InstructTool


def _diff_layer() -> Layer:
    tool = DiffusionTool(
        source_patch=None,
        patch_x=0, patch_y=0, patch_w=8, patch_h=8,
        prompt="", negative_prompt="",
        strength=0.5, guidance_scale=7.0, steps=20, seed=1,
    )
    layer = Layer("Diff", 8, 8)
    layer.tool = tool
    return layer


def _lama_layer() -> Layer:
    tool = LamaTool(
        source_patch=None,
        patch_x=0, patch_y=0, patch_w=8, patch_h=8,
    )
    layer = Layer("LaMa", 8, 8)
    layer.tool = tool
    return layer


def _instruct_layer() -> Layer:
    tool = InstructTool(
        source_patch=None,
        patch_x=0, patch_y=0, patch_w=8, patch_h=8,
        instruction="",
        image_guidance_scale=1.5, guidance_scale=7.0, steps=20, seed=1,
    )
    layer = Layer("Instruct", 8, 8)
    layer.tool = tool
    return layer


def test_map_segmentation_result_for_supported_layer():
    layer = _diff_layer()
    mask = np.zeros((8, 8), dtype=np.uint8)
    cmd, status = map_segmentation_result(layer, mask)
    assert isinstance(cmd, ReplaceLayerMaskCommand)
    assert status == "Background mask applied"


def test_map_segmentation_result_for_any_layer():
    """Any layer type (including plain Layer) accepts segmentation results."""
    layer = Layer("Paint", 8, 8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    cmd, status = map_segmentation_result(layer, mask)
    assert isinstance(cmd, ReplaceLayerMaskCommand)
    assert status == "Background mask applied"


def test_map_lama_result():
    layer = _lama_layer()
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), "RGB")
    cmd, status = map_lama_result(layer, img)
    assert isinstance(cmd, ApplyGeneratedResultCommand)
    assert status == "Objects removed (LaMa)"


def test_map_instruct_result():
    layer = _instruct_layer()
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), "RGB")
    cmd, status = map_instruct_result(layer, img, used_seed=123)
    assert isinstance(cmd, ApplyGeneratedResultCommand)
    assert status == "Instruction applied (seed=123)"


def test_map_diffusion_result():
    layer = _diff_layer()
    img = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), "RGB")
    cmd, status = map_diffusion_result(layer, img, used_seed=321)
    assert isinstance(cmd, ApplyGeneratedResultCommand)
    assert status == "Regenerated (seed=321)"


def test_map_grounding_result_uses_masks_when_present():
    layer = Layer("Paint", 3, 2, x=4, y=1)
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 3:5] = True
    result = GroundingResult(
        canvas_width=8,
        canvas_height=8,
        detections=(
            GroundingDetection(
                label="cup",
                x0=0,
                y0=0,
                x1=8,
                y1=8,
                score=0.9,
                mask=mask,
            ),
        ),
    )

    cmd, status = map_grounding_result(layer, result)

    assert isinstance(cmd, SetLayerSelectionCommand)
    assert status == "Grounding: 1 hit(s): cup (90%)"
    assert cmd.mask.shape == (8, 8)
    assert cmd.mask[2:4, 3:5].all()
    assert cmd.mask.sum() == 4


def test_map_grounding_result_falls_back_to_box():
    layer = Layer("Paint", 8, 8)
    result = GroundingResult(
        canvas_width=8,
        canvas_height=8,
        detections=(
            GroundingDetection(
                label="face",
                x0=1,
                y0=2,
                x1=4,
                y1=6,
                score=0.75,
            ),
        ),
    )

    cmd, status = map_grounding_result(layer, result)

    assert isinstance(cmd, SetLayerSelectionCommand)
    assert status == "Grounding: 1 hit(s): face (75%)"
    assert cmd.mask[2:6, 1:4].all()
    assert cmd.mask.sum() == 12


def test_map_grounding_result_keeps_canvas_coordinates_for_offset_layer():
    layer = Layer("Local", 4, 3, x=7, y=-1)
    result = GroundingResult(
        canvas_width=12,
        canvas_height=10,
        detections=(
            GroundingDetection(
                label="object",
                x0=8,
                y0=-2,
                x1=14,
                y1=2,
                score=0.8,
            ),
        ),
    )

    cmd, status = map_grounding_result(layer, result)

    assert isinstance(cmd, SetLayerSelectionCommand)
    assert status == "Grounding: 1 hit(s): object (80%)"
    assert cmd.mask.shape == (10, 12)
    assert cmd.mask[0:2, 8:12].all()
    assert cmd.mask.sum() == 8

    stack = LayerStack()
    stack.init_from_image(np.zeros((10, 12, 4), dtype=np.uint8))
    cmd.apply(stack)
    np.testing.assert_array_equal(stack.selection.data, cmd.mask)


def test_map_grounding_result_rejects_non_canvas_mask_without_command():
    layer = Layer("Local", 4, 3, x=2, y=1)
    result = GroundingResult(
        canvas_width=12,
        canvas_height=10,
        detections=(
            GroundingDetection(
                label="object",
                x0=2,
                y0=1,
                x1=6,
                y1=4,
                score=0.8,
                mask=np.ones((3, 4), dtype=bool),
            ),
        ),
    )

    cmd, status = map_grounding_result(layer, result)

    assert cmd is None
    assert status == (
        "Grounding error: worker mask shape (3, 4) "
        "does not match canvas (10, 12)"
    )
