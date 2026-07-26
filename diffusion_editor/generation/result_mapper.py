"""Mapping engine outputs to document commands and user-facing statuses."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from ..grounding.types import GroundingResult
from ..document.layer import Layer
from ..document.tool import DiffusionTool, LamaTool, InstructTool
from ..document.commands import (
    ApplyGeneratedResultCommand,
    ReplaceLayerMaskCommand,
    SetLayerSelectionCommand,
)


def map_segmentation_result(layer: Layer | None, seg_mask: np.ndarray
                            ) -> Tuple[ReplaceLayerMaskCommand | None, str]:
    """Build command/status for segmentation output."""
    if layer is not None:
        return ReplaceLayerMaskCommand(layer=layer, mask=seg_mask), "Background mask applied"
    return None, "Background mask applied"


def map_lama_result(layer: Layer | None, result_image: Image.Image
                    ) -> Tuple[ApplyGeneratedResultCommand | None, str]:
    """Build command/status for LaMa output."""
    if layer is not None and isinstance(layer.tool, LamaTool):
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
    if layer is not None and isinstance(layer.tool, InstructTool):
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
    if layer is not None and isinstance(layer.tool, DiffusionTool):
        return (
            ApplyGeneratedResultCommand(
                layer=layer,
                result_image=result_image,
                label="Apply Diffusion Result",
            ),
            f"Regenerated (seed={used_seed})",
        )
    return None, f"Regenerated (seed={used_seed})"


def map_grounding_result(layer: Layer | None, result: GroundingResult
                         ) -> Tuple[SetLayerSelectionCommand | None, str]:
    """Build a canvas-space selection from Grounding DINO/SAM output."""
    count = len(result.detections)
    if count == 0:
        return None, "Grounding: nothing found"

    found = ", ".join(
        f"{item.label} ({item.score:.0%})" for item in result.detections
    )
    status = f"Grounding: {count} hit(s): {found}"
    if layer is None:
        return None, status

    width = int(result.canvas_width)
    height = int(result.canvas_height)
    if width <= 0 or height <= 0:
        return None, (
            "Grounding error: invalid worker canvas size "
            f"{width}x{height}"
        )

    combined_selection = np.zeros((height, width), dtype=np.float32)
    for item in result.detections:
        if item.mask is not None:
            mask = np.asarray(item.mask)
            if mask.shape != (height, width):
                return None, (
                    "Grounding error: worker mask shape "
                    f"{mask.shape} does not match canvas "
                    f"{(height, width)}"
                )
            try:
                mask = mask.astype(np.float32, copy=False)
            except (TypeError, ValueError):
                return None, "Grounding error: worker mask is not numeric"
            if not np.isfinite(mask).all():
                return None, "Grounding error: worker mask is not finite"
            if mask.any():
                mask = np.clip(mask, 0.0, 1.0)
                combined_selection = np.maximum(combined_selection, mask)
                continue

        try:
            x0 = max(0, int(item.x0))
            y0 = max(0, int(item.y0))
            x1 = min(width, int(item.x1))
            y1 = min(height, int(item.y1))
        except (TypeError, ValueError, OverflowError):
            return None, "Grounding error: worker box coordinates are invalid"
        if x0 < x1 and y0 < y1:
            combined_selection[y0:y1, x0:x1] = 1.0

    if not combined_selection.any():
        return None, status
    return (
        SetLayerSelectionCommand(
            mask=np.ascontiguousarray(combined_selection),
            label="Detect Objects",
        ),
        status,
    )
