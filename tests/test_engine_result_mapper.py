"""Tests for mapping engine outputs to document commands."""

import numpy as np
from PIL import Image

from diffusion_editor.commands import ReplaceLayerMaskCommand, ApplyGeneratedResultCommand
from diffusion_editor.engine_result_mapper import (
    map_segmentation_result, map_lama_result,
    map_instruct_result, map_diffusion_result,
)
from diffusion_editor.layer import Layer, DiffusionLayer, LamaLayer, InstructLayer


def _diff_layer() -> DiffusionLayer:
    return DiffusionLayer(
        name="Diff",
        width=8,
        height=8,
        source_patch=None,
        patch_x=0,
        patch_y=0,
        patch_w=8,
        patch_h=8,
        prompt="",
        negative_prompt="",
        strength=0.5,
        guidance_scale=7.0,
        steps=20,
        seed=1,
    )


def _lama_layer() -> LamaLayer:
    return LamaLayer(
        name="LaMa",
        width=8,
        height=8,
        source_patch=None,
        patch_x=0,
        patch_y=0,
        patch_w=8,
        patch_h=8,
    )


def _instruct_layer() -> InstructLayer:
    return InstructLayer(
        name="Instruct",
        width=8,
        height=8,
        source_patch=None,
        patch_x=0,
        patch_y=0,
        patch_w=8,
        patch_h=8,
        instruction="",
        image_guidance_scale=1.5,
        guidance_scale=7.0,
        steps=20,
        seed=1,
    )


def test_map_segmentation_result_for_supported_layer():
    layer = _diff_layer()
    mask = np.zeros((8, 8), dtype=np.uint8)
    cmd, status = map_segmentation_result(layer, mask)
    assert isinstance(cmd, ReplaceLayerMaskCommand)
    assert status == "Background mask applied"


def test_map_segmentation_result_for_unsupported_layer():
    layer = Layer("Paint", 8, 8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    cmd, status = map_segmentation_result(layer, mask)
    assert cmd is None
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
