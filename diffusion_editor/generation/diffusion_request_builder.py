"""Build DiffusionEngine requests from editor layer/tool state."""

from __future__ import annotations

from PIL import Image

from .types import (
    DiffusionRequest,
    DiffusionRequestBuildResult,
    GenerationError,
)
from ..document.layer import Layer
from ..document.layer_stack import LayerStack
from .patch_resolver import (
    apply_patch_source_to_tool,
    extract_layer_mask_patch,
    resolve_source_patch,
)
from .reference_resolver import resolve_ip_adapter_reference
from ..document.tool import DiffusionTool

MODEL_RESOLUTION = 1024


def _resize_to_model_resolution(
        image: Image.Image | None,
        mask: Image.Image | None,
        width: int,
        height: int) -> tuple[Image.Image | None, Image.Image | None, int, int]:
    if image is None:
        return image, mask, width, height
    longest = max(width, height)
    if longest == MODEL_RESOLUTION:
        return image, mask, width, height

    scale = MODEL_RESOLUTION / longest
    resized_w = max(8, round(width * scale / 8) * 8)
    resized_h = max(8, round(height * scale / 8) * 8)
    resized_image = image.resize((resized_w, resized_h), Image.LANCZOS)
    resized_mask = None
    if mask is not None:
        resized_mask = mask.resize((resized_w, resized_h), Image.NEAREST)
    return resized_image, resized_mask, resized_w, resized_h


class DiffusionRequestBuilder:
    def __init__(self, layer_stack: LayerStack):
        self._layer_stack = layer_stack

    def build(
            self,
            layer: Layer,
            composite_below: np.ndarray | None) -> DiffusionRequestBuildResult:
        tool = layer.tool
        if not isinstance(tool, DiffusionTool):
            return DiffusionRequestBuildResult(error=GenerationError(
                message="Active layer has no diffusion tool",
                log_message=f"Cannot build diffusion request for layer {layer.name}",
            ))

        if tool.mode == "inpaint" and not layer.has_mask():
            return DiffusionRequestBuildResult(error=GenerationError(
                message="Inpaint requires a mask",
                log_message=f"Inpaint requested without mask on layer {layer.name}",
            ))

        if tool.mode != "txt2img":
            if composite_below is None:
                return DiffusionRequestBuildResult(error=GenerationError(
                    message="No source composite for generation",
                    log_message=(
                        f"Cannot build source patch for layer {layer.name}: "
                        "composite_below is None"
                    ),
                ))
            patch = resolve_source_patch(layer, composite_below)
            if isinstance(patch, GenerationError):
                return DiffusionRequestBuildResult(error=patch)
            if patch is not None:
                apply_patch_source_to_tool(tool, patch)
            if tool.source_patch is None:
                return DiffusionRequestBuildResult(error=GenerationError(
                    message="No source patch for generation",
                    log_message=(
                        f"Cannot build diffusion request for layer {layer.name}: "
                        "source patch is missing"
                    ),
                ))

        mask_image = None
        if tool.mode == "inpaint":
            mask_image = extract_layer_mask_patch(
                layer,
                (
                    tool.patch_x,
                    tool.patch_y,
                    tool.patch_x + tool.patch_w,
                    tool.patch_y + tool.patch_h,
                ),
            )

        reference_result = resolve_ip_adapter_reference(tool, self._layer_stack)
        if reference_result.error is not None:
            return DiffusionRequestBuildResult(error=reference_result.error)
        ip_adapter_image = None
        if reference_result.reference is not None:
            ip_adapter_image = reference_result.reference.image

        submit_image = tool.source_patch
        submit_mask = mask_image
        submit_w = tool.patch_w
        submit_h = tool.patch_h
        if tool.resize_to_model_resolution:
            submit_image, submit_mask, submit_w, submit_h = (
                _resize_to_model_resolution(
                    submit_image,
                    submit_mask,
                    submit_w,
                    submit_h,
                )
            )

        return DiffusionRequestBuildResult(request=DiffusionRequest(
            image=submit_image,
            mask_image=submit_mask,
            prompt=tool.prompt,
            negative_prompt=tool.negative_prompt,
            strength=tool.strength,
            steps=tool.steps,
            guidance_scale=tool.guidance_scale,
            seed=tool.seed,
            mode=tool.mode,
            masked_content=tool.masked_content,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_scale=tool.ip_adapter_scale,
            width=submit_w,
            height=submit_h,
        ))
