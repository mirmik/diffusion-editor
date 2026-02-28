"""Command definitions for document mutations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from PIL import Image, ImageFilter

from .layer import Layer, DiffusionLayer, LamaLayer, InstructLayer
from .layer_stack import LayerStack
from .diffusion_brush import paste_result


class SnapshotCommand(Protocol):
    """Command interface executed against LayerStack with snapshot undo/redo."""

    @property
    def label(self) -> str:
        ...

    def apply(self, layer_stack: LayerStack) -> None:
        ...


@dataclass(frozen=True)
class AddLayerCommand:
    name: str
    image: np.ndarray | None = None
    label: str = "New Layer"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.add_layer(self.name, self.image)


@dataclass(frozen=True)
class InsertLayerCommand:
    layer: Layer
    label: str

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.insert_layer(self.layer)


@dataclass(frozen=True)
class RemoveLayerCommand:
    layer: Layer
    label: str = "Remove Layer"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.remove_layer(self.layer)


@dataclass(frozen=True)
class MoveLayerCommand:
    layer: Layer
    new_parent: Layer | None
    index: int
    label: str = "Move Layer"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.move_layer(self.layer, self.new_parent, self.index)


@dataclass(frozen=True)
class SetLayerVisibilityCommand:
    layer: Layer
    visible: bool
    label: str = "Toggle Visibility"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.set_visibility(self.layer, self.visible)


@dataclass(frozen=True)
class SetLayerOpacityCommand:
    layer: Layer
    opacity: float
    label: str = "Set Opacity"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.set_opacity(self.layer, self.opacity)


@dataclass(frozen=True)
class FlattenLayersCommand:
    label: str = "Flatten Layers"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.flatten()


@dataclass(frozen=True)
class SnapshotCallbackCommand:
    """Command adapter for an arbitrary snapshot-based callback."""

    label: str
    apply_fn: Callable[[LayerStack], None]

    def apply(self, layer_stack: LayerStack) -> None:
        self.apply_fn(layer_stack)


@dataclass(frozen=True)
class ClearLayerMaskCommand:
    layer: DiffusionLayer | LamaLayer | InstructLayer
    label: str

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.clear_mask()
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class SetDiffusionIpAdapterRectCommand:
    layer: DiffusionLayer
    rect: tuple[int, int, int, int]
    label: str = "Set IP-Adapter Rect"

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.ip_adapter_rect = self.rect
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class ClearDiffusionIpAdapterRectCommand:
    layer: DiffusionLayer
    label: str = "Clear IP-Adapter Rect"

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.ip_adapter_rect = None
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class SetManualPatchRectCommand:
    layer: DiffusionLayer | InstructLayer
    rect: tuple[int, int, int, int]
    label: str

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.manual_patch_rect = self.rect
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class ClearManualPatchRectCommand:
    layer: DiffusionLayer | InstructLayer
    label: str

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.manual_patch_rect = None
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class ReplaceLayerMaskCommand:
    layer: DiffusionLayer | LamaLayer | InstructLayer
    mask: np.ndarray
    label: str = "Apply Segmentation Mask"

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.mask = self.mask.copy()
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class ApplyGeneratedResultCommand:
    """Apply a generated patch result into a layer image and invalidate caches."""

    layer: DiffusionLayer | LamaLayer | InstructLayer
    result_image: Image.Image
    label: str

    def apply(self, layer_stack: LayerStack) -> None:
        layer = self.layer
        layer.image[:] = 0
        if layer.has_mask():
            mask_pil = Image.fromarray(layer.mask, "L")
            mask_pil = mask_pil.filter(ImageFilter.MaxFilter(7))
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=4))
            mask_arg = np.array(mask_pil, dtype=np.uint8)
        else:
            mask_arg = None
        paste_result(
            layer.image,
            self.result_image,
            layer.patch_x,
            layer.patch_y,
            layer.patch_w,
            layer.patch_h,
            mask=mask_arg,
        )
        layer_stack.mark_layer_dirty(layer)
        if layer_stack.on_changed:
            layer_stack.on_changed()
