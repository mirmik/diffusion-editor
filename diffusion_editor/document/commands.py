"""Command definitions for document mutations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from PIL import Image, ImageFilter

from .layer import Layer
from .tool import Tool, DiffusionTool
from .layer_stack import LayerStack
from .result_paste import paste_result
from .mask import Selection, coerce_mask_data


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
    x: int = 0
    y: int = 0
    label: str = "New Layer"

    def apply(self, layer_stack: LayerStack) -> None:
        if self.image is None:
            layer_stack.add_layer(self.name)
        else:
            layer_stack.insert_image_layer(self.name, self.image, self.x, self.y)


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
class SetLayerNameCommand:
    layer: Layer
    name: str
    label: str = "Rename Layer"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.set_layer_name(self.layer, self.name)


@dataclass(frozen=True)
class AttachLayerToolCommand:
    layer: Layer
    tool: Tool
    label: str = "Attach Tool"

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.tool = self.tool
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class DetachLayerToolCommand:
    layer: Layer
    label: str = "Remove Tool"

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.tool = None
        if layer_stack.on_changed:
            layer_stack.on_changed()


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
class DrawRectCommand:
    layer: Layer
    x: int
    y: int
    width: int
    height: int
    color: tuple[int, int, int, int] = (255, 0, 0, 255)
    thickness: int = 2
    label: str = "Draw Rectangle"

    def apply(self, layer_stack: LayerStack) -> None:
        lx = self.x - self.layer.x
        ly = self.y - self.layer.y
        x0 = max(0, lx)
        y0 = max(0, ly)
        x1 = min(self.layer.width, lx + self.width)
        y1 = min(self.layer.height, ly + self.height)
        if x0 >= x1 or y0 >= y1:
            return
        t = max(1, self.thickness)
        image = self.layer.image
        color = np.array(self.color, dtype=np.uint8)

        # Top edge
        te_bottom = min(y0 + t, y1)
        image[y0:te_bottom, x0:x1] = color
        # Bottom edge
        be_top = max(y1 - t, y0)
        image[be_top:y1, x0:x1] = color
        # Left edge (between top and bottom strips already drawn)
        le_right = min(x0 + t, x1)
        image[y0:y1, x0:le_right] = color
        # Right edge
        re_left = max(x1 - t, x0)
        image[y0:y1, re_left:x1] = color

        layer_stack.mark_layer_dirty(
            self.layer, self.layer.local_rect_to_canvas((x0, y0, x1, y1)))
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class DrawGridCommand:
    layer: Layer
    sections_x: int
    sections_y: int
    color: tuple[int, int, int, int] = (255, 0, 0, 128)
    thickness: int = 1
    label: str = "Draw Grid"

    def apply(self, layer_stack: LayerStack) -> None:
        t = max(1, self.thickness)
        image = self.layer.image
        w, h = self.layer.width, self.layer.height
        color = np.array(self.color, dtype=np.uint8)

        # Vertical lines
        for i in range(self.sections_x + 1):
            x = int(i * w / self.sections_x)
            x0 = max(0, min(x - t // 2, w))
            x1 = max(0, min(x + t // 2 + (t % 2), w))
            if x0 < x1:
                image[0:h, x0:x1] = color

        # Horizontal lines
        for i in range(self.sections_y + 1):
            y = int(i * h / self.sections_y)
            y0 = max(0, min(y - t // 2, h))
            y1 = max(0, min(y + t // 2 + (t % 2), h))
            if y0 < y1:
                image[y0:y1, 0:w] = color

        layer_stack.mark_layer_dirty(self.layer)
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class FillMaskCommand:
    """Fill a binary mask region on a layer with semi-transparent color
    and draw a visible boundary outline."""

    layer: Layer
    mask: np.ndarray  # bool HxW, True = fill
    color: tuple[int, int, int, int] = (255, 0, 0, 128)
    outline_color: tuple[int, int, int, int] | None = None
    label: str = "Fill Mask"

    def apply(self, layer_stack: LayerStack) -> None:
        image = self.layer.image
        if self.mask.shape == image.shape[:2]:
            m = self.mask
            dst_x0 = 0
            dst_y0 = 0
        else:
            cx0 = max(0, self.layer.x)
            cy0 = max(0, self.layer.y)
            cx1 = min(self.mask.shape[1], self.layer.x + self.layer.width)
            cy1 = min(self.mask.shape[0], self.layer.y + self.layer.height)
            if cx1 <= cx0 or cy1 <= cy0:
                return
            dst_x0 = cx0 - self.layer.x
            dst_y0 = cy0 - self.layer.y
            m = self.mask[cy0:cy1, cx0:cx1]
            image = image[dst_y0:dst_y0 + m.shape[0],
                          dst_x0:dst_x0 + m.shape[1]]
        if not m.any():
            return

        color = np.array(self.color, dtype=np.uint8)
        alpha = self.color[3] / 255.0
        blended = image[m].astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
        image[m] = blended.astype(np.uint8)

        if self.outline_color is not None and m.shape[0] > 2 and m.shape[1] > 2:
            # 8-connected morphological boundary: dilated & ~mask
            d = np.zeros_like(m)
            d[:-1] |= m[1:]
            d[1:] |= m[:-1]
            d[:, :-1] |= m[:, 1:]
            d[:, 1:] |= m[:, :-1]
            d[:-1, :-1] |= m[1:, 1:]
            d[:-1, 1:] |= m[1:, :-1]
            d[1:, :-1] |= m[:-1, 1:]
            d[1:, 1:] |= m[:-1, :-1]
            boundary = d & ~m
            if boundary.any():
                oc = np.array(self.outline_color, dtype=np.uint8)
                image[boundary] = oc

        dirty = (dst_x0, dst_y0, dst_x0 + m.shape[1], dst_y0 + m.shape[0])
        layer_stack.mark_layer_dirty(
            self.layer, self.layer.local_rect_to_canvas(dirty))
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class ClearLayerMaskCommand:
    layer: Layer
    label: str

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.clear_mask()
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class SetIpAdapterReferenceLayerCommand:
    layer: Layer
    reference_layer_id: str
    reference_layer_name_hint: str
    label: str = "Set IP-Adapter Reference Layer"

    def apply(self, layer_stack: LayerStack) -> None:
        tool = self.layer.tool
        if isinstance(tool, DiffusionTool):
            tool.ip_adapter_layer_id = self.reference_layer_id
            tool.ip_adapter_layer_name_hint = self.reference_layer_name_hint
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class ClearIpAdapterReferenceLayerCommand:
    layer: Layer
    label: str = "Clear IP-Adapter Reference Layer"

    def apply(self, layer_stack: LayerStack) -> None:
        tool = self.layer.tool
        if isinstance(tool, DiffusionTool):
            tool.ip_adapter_layer_id = None
            tool.ip_adapter_layer_name_hint = ""
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class SetLayerPatchRectCommand:
    layer: Layer
    rect: tuple[int, int, int, int]
    label: str

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.patch_rect = self.rect
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class ClearLayerPatchRectCommand:
    layer: Layer
    label: str

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.patch_rect = None
        if layer_stack.on_changed:
            layer_stack.on_changed()


SetManualPatchRectCommand = SetLayerPatchRectCommand
ClearManualPatchRectCommand = ClearLayerPatchRectCommand


@dataclass(frozen=True)
class ReplaceLayerMaskCommand:
    layer: Layer
    mask: np.ndarray
    label: str = "Apply Segmentation Mask"

    def apply(self, layer_stack: LayerStack) -> None:
        data = coerce_mask_data(self.mask)
        if data.shape == self.layer.mask.data.shape:
            self.layer.mask.data[:] = data
        elif data.shape == (layer_stack.height, layer_stack.width):
            self.layer.mask.clear()
            cx0 = max(0, self.layer.x)
            cy0 = max(0, self.layer.y)
            cx1 = min(layer_stack.width, self.layer.x + self.layer.width)
            cy1 = min(layer_stack.height, self.layer.y + self.layer.height)
            if cx1 > cx0 and cy1 > cy0:
                lx0 = cx0 - self.layer.x
                ly0 = cy0 - self.layer.y
                lx1 = lx0 + (cx1 - cx0)
                ly1 = ly0 + (cy1 - cy0)
                self.layer.mask.data[ly0:ly1, lx0:lx1] = data[cy0:cy1, cx0:cx1]
        else:
            raise ValueError(
                f"mask shape {data.shape} does not match layer mask "
                f"shape {self.layer.mask.data.shape}")
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class SetLayerSelectionCommand:
    """Replace the document-level selection with a mask array (e.g. SAM output)."""

    mask: np.ndarray
    label: str = "Set Selection"

    def apply(self, layer_stack: LayerStack) -> None:
        data = coerce_mask_data(self.mask)
        expected_shape = (layer_stack.height, layer_stack.width)
        if expected_shape != data.shape:
            raise ValueError(
                f"selection shape {data.shape} does not match canvas "
                f"shape {expected_shape}")
        if layer_stack.selection.data.shape != expected_shape:
            layer_stack.selection = Selection(height=layer_stack.height,
                                              width=layer_stack.width)
        layer_stack.selection.data[:] = data
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class ClearSelectionCommand:
    """Clear the document-level selection."""

    label: str = "Clear Selection"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.selection.clear()
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class InvertSelectionCommand:
    """Invert the document-level selection."""

    label: str = "Invert Selection"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.selection.data[:] = 1.0 - layer_stack.selection.data
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class SelectAllCommand:
    """Select the entire canvas."""

    label: str = "Select All"

    def apply(self, layer_stack: LayerStack) -> None:
        h, w = layer_stack.height, layer_stack.width
        if h == 0 or w == 0:
            return
        if layer_stack.selection.data.shape != (h, w):
            layer_stack.selection = Selection(height=h, width=w)
        layer_stack.selection.data[:] = 1.0
        if layer_stack.on_changed:
            layer_stack.on_changed()


@dataclass(frozen=True)
class ApplyGeneratedResultCommand:
    """Apply a generated patch result into a layer image and invalidate caches."""

    layer: Layer
    result_image: Image.Image
    label: str

    def apply(self, layer_stack: LayerStack) -> None:
        layer = self.layer
        tool = layer.tool
        if tool is None:
            return
        layer.image[:] = 0
        if layer.has_mask():
            mask_pil = Image.fromarray(layer.mask.to_uint8(), "L")
            mask_pil = mask_pil.filter(ImageFilter.MaxFilter(7))
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=4))
            mask_arg = np.array(mask_pil, dtype=np.uint8)
        else:
            mask_arg = None
        paste_result(
            layer.image,
            self.result_image,
            tool.patch_x - layer.x,
            tool.patch_y - layer.y,
            tool.patch_w,
            tool.patch_h,
            mask=mask_arg,
        )
        layer_stack.mark_layer_dirty(layer)
        if layer_stack.on_changed:
            layer_stack.on_changed()
