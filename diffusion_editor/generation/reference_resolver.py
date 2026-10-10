"""Reference image resolution for IP-Adapter and future reference inputs."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .image_edit_profiles import image_edit_profile
from .text_to_image_profiles import text_to_image_profile
from .types import (
    GenerationError,
    ReferenceImage,
    ReferenceResolveResult,
    Rect,
)
from ..document.layer import Layer
from ..document.layer_stack import LayerStack
from ..document.tool import DiffusionTool, InstructTool, TextToImageTool


def _clip_local_rect(layer: Layer, rect: Rect) -> Rect:
    x0, y0, x1, y1 = rect
    x0 = max(0, min(int(x0), layer.width))
    x1 = max(0, min(int(x1), layer.width))
    y0 = max(0, min(int(y0), layer.height))
    y1 = max(0, min(int(y1), layer.height))
    return x0, y0, x1, y1


def _alpha_bbox(layer: Layer) -> Rect | None:
    alpha = layer.image[:, :, 3]
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def reference_rect_for_layer(layer: Layer) -> tuple[Rect, str]:
    if layer.patch_rect is not None:
        return layer.patch_rect, "patch"
    bbox = _alpha_bbox(layer)
    if bbox is not None:
        return bbox, "alpha_bbox"
    return (0, 0, layer.width, layer.height), "full_layer"


def layer_reference_image(
        layer: Layer,
        *,
        purpose: str = "IP-Adapter",
        preserve_alpha: bool = False,
) -> ReferenceImage | GenerationError:
    raw_rect, source = reference_rect_for_layer(layer)
    rect = _clip_local_rect(layer, raw_rect)
    x0, y0, x1, y1 = rect
    if x1 <= x0 or y1 <= y0:
        return GenerationError(
            message=f"{purpose} reference is empty: {layer.name}",
            log_message=(
                f"{purpose} reference crop is empty for layer {layer.name}: "
                f"{raw_rect}"
            ),
        )

    rgba = layer.image[y0:y1, x0:x1].astype(np.float32)
    alpha = rgba[:, :, 3:4] / 255.0
    rgb = rgba[:, :, :3] * alpha + 255.0 * (1.0 - alpha)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return ReferenceImage(
        image=(Image.fromarray(rgba.astype(np.uint8), "RGBA")
               if preserve_alpha else Image.fromarray(rgb, "RGB")),
        layer_id=layer.id,
        layer_name=layer.name,
        local_rect=rect,
        source=source,
    )


def resolve_ip_adapter_reference(
        tool: DiffusionTool,
        layer_stack: LayerStack) -> ReferenceResolveResult:
    if tool.ip_adapter_layer_id is None:
        return ReferenceResolveResult()

    layer = layer_stack.find_layer_by_id(tool.ip_adapter_layer_id)
    if layer is None:
        hint = tool.ip_adapter_layer_name_hint or tool.ip_adapter_layer_id
        return ReferenceResolveResult(error=GenerationError(
            message=f"IP-Adapter reference missing: {hint}",
            log_message=f"IP-Adapter reference layer not found: {hint}",
        ))

    resolved = layer_reference_image(layer)
    if isinstance(resolved, GenerationError):
        return ReferenceResolveResult(error=resolved)
    return ReferenceResolveResult(reference=resolved)


def resolve_image_edit_reference(
        tool: InstructTool | TextToImageTool,
        layer_stack: LayerStack | None,
) -> Image.Image | GenerationError | None:
    """Resolve an optional AI Edit/Text to Image reference from a layer or file."""
    is_generation = isinstance(tool, TextToImageTool)
    purpose = "Text to Image" if is_generation else "AI Edit"
    profile = (text_to_image_profile if is_generation else image_edit_profile)(tool.model_profile_id)
    preserve_alpha = profile.supports_alpha
    if tool.reference_layer_id is not None:
        layer = (layer_stack.find_layer_by_id(tool.reference_layer_id)
                 if layer_stack is not None else None)
        if layer is None:
            hint = tool.reference_layer_name_hint or tool.reference_layer_id
            return GenerationError(
                message=f"{purpose} reference missing: {hint}",
                log_message=f"{purpose} reference layer not found: {hint}",
            )
        resolved = layer_reference_image(
            layer, purpose=purpose, preserve_alpha=preserve_alpha)
        if isinstance(resolved, GenerationError):
            return resolved
        return resolved.image
    if tool.reference_image is not None:
        return tool.reference_image.copy().convert("RGBA" if preserve_alpha else "RGB")
    return None


def resolve_text_to_image_references(tool: TextToImageTool, layer_stack: LayerStack | None):
    profile = text_to_image_profile(tool.model_profile_id)
    if not profile.supports_reference_image:
        return ()
    if len(tool.references) > profile.max_reference_images:
        return GenerationError(message=f"{profile.title} accepts at most {profile.max_reference_images} references")
    images = []
    for index, ref in enumerate(tool.references, 1):
        if ref.layer_id is not None:
            layer = layer_stack.find_layer_by_id(ref.layer_id) if layer_stack is not None else None
            if layer is None:
                return GenerationError(message=f"Text to Image reference <image{index}> missing or unavailable: {ref.layer_name_hint or ref.layer_id}")
            resolved = layer_reference_image(layer, purpose=f"Reference <image{index}>", preserve_alpha=profile.supports_alpha)
            if isinstance(resolved, GenerationError):
                return resolved
            image = resolved.image
        elif ref.image is not None:
            image = ref.image.copy().convert("RGBA" if profile.supports_alpha else "RGB")
        else:
            return GenerationError(message=f"Text to Image reference <image{index}> is empty")
        images.append(image)
    return tuple(images)
