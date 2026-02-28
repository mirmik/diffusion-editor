"""Mapping engine outputs to document commands and user-facing statuses."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from .layer import Layer, DiffusionLayer, LamaLayer, InstructLayer
from .commands import ReplaceLayerMaskCommand, ApplyGeneratedResultCommand


def map_segmentation_result(layer: Layer | None, seg_mask: np.ndarray
                            ) -> Tuple[ReplaceLayerMaskCommand | None, str]:
    """Build command/status for segmentation output."""
    if isinstance(layer, (DiffusionLayer, LamaLayer)):
        return ReplaceLayerMaskCommand(layer=layer, mask=seg_mask), "Background mask applied"
    return None, "Background mask applied"


def map_lama_result(layer: Layer | None, result_image: Image.Image
                    ) -> Tuple[ApplyGeneratedResultCommand | None, str]:
    """Build command/status for LaMa output."""
    if isinstance(layer, LamaLayer):
        return (
            ApplyGeneratedResultCommand(
                layer=layer,
                result_image=result_image,
                label="Apply LaMa Result",
            ),
            "Objects removed (LaMa)",
        )
    return None, "Objects removed (LaMa)"


def map_instruct_result(layer: Layer | None, result_image: Image.Image, used_seed: int
                        ) -> Tuple[ApplyGeneratedResultCommand | None, str]:
    """Build command/status for InstructPix2Pix output."""
    if isinstance(layer, InstructLayer):
        return (
            ApplyGeneratedResultCommand(
                layer=layer,
                result_image=result_image,
                label="Apply Instruct Result",
            ),
            f"Instruction applied (seed={used_seed})",
        )
    return None, f"Instruction applied (seed={used_seed})"


def map_diffusion_result(layer: Layer | None, result_image: Image.Image, used_seed: int
                         ) -> Tuple[ApplyGeneratedResultCommand | None, str]:
    """Build command/status for diffusion output."""
    if isinstance(layer, DiffusionLayer):
        return (
            ApplyGeneratedResultCommand(
                layer=layer,
                result_image=result_image,
                label="Apply Diffusion Result",
            ),
            f"Regenerated (seed={used_seed})",
        )
    return None, f"Regenerated (seed={used_seed})"

