"""Command definitions for document mutations."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from numbers import Integral
from typing import Callable, Protocol

import numpy as np
from PIL import Image, ImageFilter

from .layer import Layer
from .reconstruction import ReconstructionLayer, ReconstructionStatus
from ..generation.types import ReconstructionRun, ReconstructionRunKind
from .change_event import DocumentChangeKind
from .tool import DiffusionTool, InstructTool, Tool
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
class CommandDelta:
    undo_fn: Callable[[], None]
    redo_fn: Callable[[], None]
    size_bytes: int = 0
    coalesce_key: object | None = None


def _location(layer_stack: LayerStack, layer: Layer) -> tuple[Layer | None, int]:
    parent = layer.parent
    siblings = parent.children if parent is not None else layer_stack.layers
    return parent, siblings.index(layer)


def _restore_active(layer_stack: LayerStack, layer_id: str | None) -> None:
    layer_stack.active_layer = layer_stack.find_layer_by_id(layer_id or "")


def _set_tool(layer_stack: LayerStack, layer: Layer, tool: Tool | None) -> None:
    layer.tool = tool
    layer_stack.publish_change(DocumentChangeKind.METADATA, layers=(layer,))


def _attribute_delta(
        layer_stack: LayerStack,
        layer: Layer,
        target: object,
        values: dict[str, object],
        *,
        coalesce_key: object | None = None) -> CommandDelta | None:
    old = {name: getattr(target, name) for name in values}
    if old == values:
        return None

    def assign(next_values: dict[str, object]) -> None:
        for name, value in next_values.items():
            setattr(target, name, value)
        layer_stack.publish_change(
            DocumentChangeKind.METADATA, layers=(layer,))

    assign(values)
    size = sum(
        len(repr(value).encode("utf-8"))
        for value in (*old.values(), *values.values())
    )
    return CommandDelta(
        lambda: assign(old),
        lambda: assign(values),
        size,
        coalesce_key,
    )


def _changed_rect(
        before: np.ndarray,
        after: np.ndarray) -> tuple[int, int, int, int] | None:
    changed = before != after
    if changed.ndim > 2:
        changed = np.any(changed, axis=tuple(range(2, changed.ndim)))
    ys, xs = np.nonzero(changed)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _array_delta_after_apply(
        layer_stack: LayerStack,
        target: np.ndarray,
        before: np.ndarray,
        *,
        layer: Layer | None = None,
        pixels: bool = False) -> CommandDelta | None:
    rect = _changed_rect(before, target)
    if rect is None:
        return None
    x0, y0, x1, y1 = rect
    before_patch = before[y0:y1, x0:x1].copy()
    after_patch = target[y0:y1, x0:x1].copy()

    def assign(patch: np.ndarray) -> None:
        target[y0:y1, x0:x1] = patch
        if pixels and layer is not None:
            layer_stack.mark_layer_dirty(
                layer, layer.local_rect_to_canvas(rect))
            layer_stack.publish_change(
                DocumentChangeKind.PIXELS,
                layers=(layer,),
                dirty_rect=rect,
            )
        else:
            layer_stack.publish_change(
                DocumentChangeKind.METADATA,
                layers=(layer,) if layer is not None else (),
            )

    return CommandDelta(
        lambda: assign(before_patch),
        lambda: assign(after_patch),
        before_patch.nbytes + after_patch.nbytes,
    )


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

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        if layer_stack.width == 0 or layer_stack.height == 0:
            return None
        old_active_id = (
            layer_stack.active_layer.id
            if layer_stack.active_layer is not None else None)
        self.apply(layer_stack)
        layer = layer_stack.active_layer
        if layer is None:
            return None
        parent, index = _location(layer_stack, layer)

        def undo() -> None:
            layer_stack.remove_layer(layer)
            _restore_active(layer_stack, old_active_id)

        def redo() -> None:
            layer_stack.insert_layer(layer)
            layer_stack.move_layer(layer, parent, index)

        return CommandDelta(undo, redo, 128)


@dataclass(frozen=True)
class AddReconstructionLayerCommand:
    name: str
    label: str = "New 3D Reconstruction"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.insert_layer(ReconstructionLayer(self.name))

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        if layer_stack.width == 0 or layer_stack.height == 0:
            return None
        old_active_id = (
            layer_stack.active_layer.id
            if layer_stack.active_layer is not None else None
        )
        layer = ReconstructionLayer(self.name)
        layer_stack.insert_layer(layer)
        parent, index = _location(layer_stack, layer)

        def undo() -> None:
            layer_stack.remove_layer(layer)
            _restore_active(layer_stack, old_active_id)

        def redo() -> None:
            layer_stack.insert_layer(layer)
            layer_stack.move_layer(layer, parent, index)

        return CommandDelta(undo, redo, 256)


@dataclass(frozen=True)
class PublishReconstructionResultCommand:
    layer: ReconstructionLayer
    glb_path: str
    vertex_count: int
    triangle_count: int
    mesh_count: int
    source_path: str = ""
    conditioning_path: str | None = None
    checkpoint_path: str | None = None
    run_kind: ReconstructionRunKind = ReconstructionRunKind.BASE
    parent_run_id: str | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    label: str = "Publish 3D Reconstruction"

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        if layer_stack.find_layer_by_id(self.layer.id) is not self.layer:
            raise ValueError("reconstruction node does not belong to the layer stack")
        run = ReconstructionRun(
            run_id=self.run_id,
            kind=self.run_kind,
            glb_path=str(self.glb_path),
            source_path=str(self.source_path),
            conditioning_path=self.conditioning_path,
            checkpoint_path=self.checkpoint_path,
            parent_run_id=self.parent_run_id,
            vertex_count=int(self.vertex_count),
            triangle_count=int(self.triangle_count),
            mesh_count=int(self.mesh_count),
        )
        runs = (
            (run,)
            if self.run_kind is ReconstructionRunKind.BASE
            else (*self.layer.runs, run)
        )
        return _attribute_delta(
            layer_stack,
            self.layer,
            self.layer,
            {
                "reconstruction_status": ReconstructionStatus.READY,
                "glb_path": str(self.glb_path),
                "vertex_count": int(self.vertex_count),
                "triangle_count": int(self.triangle_count),
                "mesh_count": int(self.mesh_count),
                "runs": runs,
                "active_run_id": run.run_id,
            },
        )


@dataclass(frozen=True)
class InsertLayerCommand:
    layer: Layer
    label: str

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.insert_layer(self.layer)

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        if layer_stack.width == 0 or layer_stack.height == 0:
            return None
        old_active_id = (
            layer_stack.active_layer.id
            if layer_stack.active_layer is not None else None)
        self.apply(layer_stack)
        parent, index = _location(layer_stack, self.layer)

        def undo() -> None:
            layer_stack.remove_layer(self.layer)
            _restore_active(layer_stack, old_active_id)

        def redo() -> None:
            layer_stack.insert_layer(self.layer)
            layer_stack.move_layer(self.layer, parent, index)

        return CommandDelta(undo, redo, 128)


@dataclass(frozen=True)
class RemoveLayerCommand:
    layer: Layer
    label: str = "Remove Layer"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.remove_layer(self.layer)

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta:
        parent, index = _location(layer_stack, self.layer)
        old_active_id = (
            layer_stack.active_layer.id
            if layer_stack.active_layer is not None else None)
        self.apply(layer_stack)

        def undo() -> None:
            layer_stack.insert_layer(self.layer)
            layer_stack.move_layer(self.layer, parent, index)
            _restore_active(layer_stack, old_active_id)

        return CommandDelta(
            undo,
            lambda: layer_stack.remove_layer(self.layer),
            128,
        )


@dataclass(frozen=True)
class MoveLayerCommand:
    layer: Layer
    new_parent: Layer | None
    index: int
    label: str = "Move Layer"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.move_layer(self.layer, self.new_parent, self.index)

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta:
        old_parent, old_index = _location(layer_stack, self.layer)
        old_active_id = (
            layer_stack.active_layer.id
            if layer_stack.active_layer is not None else None)
        self.apply(layer_stack)
        new_parent, new_index = _location(layer_stack, self.layer)

        def undo() -> None:
            layer_stack.move_layer(self.layer, old_parent, old_index)
            _restore_active(layer_stack, old_active_id)

        return CommandDelta(
            undo,
            lambda: layer_stack.move_layer(
                self.layer, new_parent, new_index),
            96,
        )


@dataclass(frozen=True)
class SetLayerVisibilityCommand:
    layer: Layer
    visible: bool
    label: str = "Toggle Visibility"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.set_visibility(self.layer, self.visible)

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        old = self.layer.visible
        if old == self.visible:
            return None
        self.apply(layer_stack)
        return CommandDelta(
            lambda: layer_stack.set_visibility(self.layer, old),
            lambda: layer_stack.set_visibility(self.layer, self.visible),
            16,
        )


@dataclass(frozen=True)
class SetLayerSoloCommand:
    layer: Layer | None
    label: str = "Set Solo Layer"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.set_solo_layer(self.layer)

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        old = layer_stack.solo_layer()
        if old is self.layer:
            return None
        self.apply(layer_stack)
        return CommandDelta(
            lambda: layer_stack.set_solo_layer(old),
            lambda: layer_stack.set_solo_layer(self.layer),
            16,
        )


@dataclass(frozen=True)
class SetLayerOpacityCommand:
    layer: Layer
    opacity: float
    label: str = "Set Opacity"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.set_opacity(self.layer, self.opacity)

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        old = self.layer.opacity
        if old == self.opacity:
            return None
        self.apply(layer_stack)
        return CommandDelta(
            lambda: layer_stack.set_opacity(self.layer, old),
            lambda: layer_stack.set_opacity(self.layer, self.opacity),
            16,
            ("opacity", self.layer.id),
        )


@dataclass(frozen=True)
class SetLayerNameCommand:
    layer: Layer
    name: str
    label: str = "Rename Layer"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.set_layer_name(self.layer, self.name)

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        old = self.layer.name
        if old == self.name:
            return None
        self.apply(layer_stack)
        size = len(old.encode("utf-8")) + len(self.name.encode("utf-8"))
        return CommandDelta(
            lambda: layer_stack.set_layer_name(self.layer, old),
            lambda: layer_stack.set_layer_name(self.layer, self.name),
            size,
        )


@dataclass(frozen=True)
class AttachLayerToolCommand:
    layer: Layer
    tool: Tool
    label: str = "Attach Tool"

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.tool = self.tool
        layer_stack.publish_change(
            DocumentChangeKind.METADATA, layers=(self.layer,))

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta:
        old = self.layer.tool
        self.apply(layer_stack)
        return CommandDelta(
            lambda: _set_tool(layer_stack, self.layer, old),
            lambda: _set_tool(layer_stack, self.layer, self.tool),
            256,
        )


@dataclass(frozen=True)
class DetachLayerToolCommand:
    layer: Layer
    label: str = "Remove Tool"

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.tool = None
        layer_stack.publish_change(
            DocumentChangeKind.METADATA, layers=(self.layer,))

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta:
        old = self.layer.tool
        self.apply(layer_stack)
        return CommandDelta(
            lambda: _set_tool(layer_stack, self.layer, old),
            lambda: _set_tool(layer_stack, self.layer, None),
            256,
        )


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
        layer_stack.publish_change(
            DocumentChangeKind.PIXELS,
            layers=(self.layer,),
            dirty_rect=(x0, y0, x1, y1),
        )

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        before = self.layer.image.copy()
        self.apply(layer_stack)
        return _array_delta_after_apply(
            layer_stack, self.layer.image, before,
            layer=self.layer, pixels=True)


@dataclass(frozen=True)
class DrawGridCommand:
    layer: Layer
    sections_x: int
    sections_y: int
    color: tuple[int, int, int, int] = (255, 0, 0, 128)
    thickness: int = 1
    label: str = "Draw Grid"

    def __post_init__(self) -> None:
        if isinstance(self.sections_x, bool) or not isinstance(
                self.sections_x, Integral):
            raise ValueError("sections_x must be an integer >= 1")
        if isinstance(self.sections_y, bool) or not isinstance(
                self.sections_y, Integral):
            raise ValueError("sections_y must be an integer >= 1")
        if self.sections_x < 1:
            raise ValueError("sections_x must be >= 1")
        if self.sections_y < 1:
            raise ValueError("sections_y must be >= 1")

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
        layer_stack.publish_change(
            DocumentChangeKind.PIXELS, layers=(self.layer,))

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        before = self.layer.image.copy()
        self.apply(layer_stack)
        return _array_delta_after_apply(
            layer_stack, self.layer.image, before,
            layer=self.layer, pixels=True)


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
        layer_stack.publish_change(
            DocumentChangeKind.PIXELS,
            layers=(self.layer,),
            dirty_rect=dirty,
        )

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        before = self.layer.image.copy()
        self.apply(layer_stack)
        return _array_delta_after_apply(
            layer_stack, self.layer.image, before,
            layer=self.layer, pixels=True)


@dataclass(frozen=True)
class ClearLayerMaskCommand:
    layer: Layer
    label: str

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.clear_mask()
        layer_stack.publish_change(
            DocumentChangeKind.METADATA, layers=(self.layer,))

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        before = self.layer.mask.data.copy()
        self.apply(layer_stack)
        return _array_delta_after_apply(
            layer_stack, self.layer.mask.data, before, layer=self.layer)


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
        layer_stack.publish_change(
            DocumentChangeKind.METADATA, layers=(self.layer,))

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        tool = self.layer.tool
        if not isinstance(tool, DiffusionTool):
            return None
        return _attribute_delta(
            layer_stack,
            self.layer,
            tool,
            {
                "ip_adapter_layer_id": self.reference_layer_id,
                "ip_adapter_layer_name_hint": self.reference_layer_name_hint,
            },
        )


@dataclass(frozen=True)
class ClearIpAdapterReferenceLayerCommand:
    layer: Layer
    label: str = "Clear IP-Adapter Reference Layer"

    def apply(self, layer_stack: LayerStack) -> None:
        tool = self.layer.tool
        if isinstance(tool, DiffusionTool):
            tool.ip_adapter_layer_id = None
            tool.ip_adapter_layer_name_hint = ""
        layer_stack.publish_change(
            DocumentChangeKind.METADATA, layers=(self.layer,))

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        tool = self.layer.tool
        if not isinstance(tool, DiffusionTool):
            return None
        return _attribute_delta(
            layer_stack,
            self.layer,
            tool,
            {"ip_adapter_layer_id": None, "ip_adapter_layer_name_hint": ""},
        )


@dataclass(frozen=True)
class UpdateDiffusionToolCommand:
    layer: Layer
    prompt: str
    negative_prompt: str
    strength: float
    guidance_scale: float
    steps: int
    seed: int
    mode: str
    masked_content: str
    ip_adapter_scale: float
    resize_to_model_resolution: bool
    model_path: str
    prediction_type: str
    label: str = "Update Diffusion Settings"

    def apply(self, layer_stack: LayerStack) -> None:
        tool = self.layer.tool
        if not isinstance(tool, DiffusionTool):
            return
        tool.prompt = self.prompt
        tool.negative_prompt = self.negative_prompt
        tool.strength = self.strength
        tool.guidance_scale = self.guidance_scale
        tool.steps = self.steps
        tool.seed = self.seed
        tool.mode = self.mode
        tool.masked_content = self.masked_content
        tool.ip_adapter_scale = self.ip_adapter_scale
        tool.resize_to_model_resolution = self.resize_to_model_resolution
        tool.model_path = self.model_path
        tool.prediction_type = self.prediction_type
        layer_stack.publish_change(
            DocumentChangeKind.METADATA, layers=(self.layer,))

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        tool = self.layer.tool
        if not isinstance(tool, DiffusionTool):
            return None
        return _attribute_delta(
            layer_stack,
            self.layer,
            tool,
            {
                "prompt": self.prompt,
                "negative_prompt": self.negative_prompt,
                "strength": self.strength,
                "guidance_scale": self.guidance_scale,
                "steps": self.steps,
                "seed": self.seed,
                "mode": self.mode,
                "masked_content": self.masked_content,
                "ip_adapter_scale": self.ip_adapter_scale,
                "resize_to_model_resolution": self.resize_to_model_resolution,
                "model_path": self.model_path,
                "prediction_type": self.prediction_type,
            },
            coalesce_key=("diffusion-tool", self.layer.id),
        )


@dataclass(frozen=True)
class UpdateInstructToolCommand:
    layer: Layer
    instruction: str
    image_guidance_scale: float
    guidance_scale: float
    steps: int
    seed: int
    label: str = "Update Instruct Settings"

    def apply(self, layer_stack: LayerStack) -> None:
        tool = self.layer.tool
        if not isinstance(tool, InstructTool):
            return
        tool.instruction = self.instruction
        tool.image_guidance_scale = self.image_guidance_scale
        tool.guidance_scale = self.guidance_scale
        tool.steps = self.steps
        tool.seed = self.seed
        layer_stack.publish_change(
            DocumentChangeKind.METADATA, layers=(self.layer,))

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        tool = self.layer.tool
        if not isinstance(tool, InstructTool):
            return None
        return _attribute_delta(
            layer_stack,
            self.layer,
            tool,
            {
                "instruction": self.instruction,
                "image_guidance_scale": self.image_guidance_scale,
                "guidance_scale": self.guidance_scale,
                "steps": self.steps,
                "seed": self.seed,
            },
            coalesce_key=("instruct-tool", self.layer.id),
        )


@dataclass(frozen=True)
class SetLayerPatchRectCommand:
    layer: Layer
    rect: tuple[int, int, int, int]
    label: str

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.patch_rect = self.rect
        layer_stack.publish_change(
            DocumentChangeKind.METADATA, layers=(self.layer,))

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        return _attribute_delta(
            layer_stack, self.layer, self.layer,
            {"patch_rect": self.rect},
        )


@dataclass(frozen=True)
class ClearLayerPatchRectCommand:
    layer: Layer
    label: str

    def apply(self, layer_stack: LayerStack) -> None:
        self.layer.patch_rect = None
        layer_stack.publish_change(
            DocumentChangeKind.METADATA, layers=(self.layer,))

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        return _attribute_delta(
            layer_stack, self.layer, self.layer,
            {"patch_rect": None},
        )


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
        layer_stack.publish_change(
            DocumentChangeKind.METADATA, layers=(self.layer,))

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        before = self.layer.mask.data.copy()
        self.apply(layer_stack)
        return _array_delta_after_apply(
            layer_stack, self.layer.mask.data, before, layer=self.layer)


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
        layer_stack.publish_change(DocumentChangeKind.METADATA)

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        before = layer_stack.selection.data.copy()
        self.apply(layer_stack)
        return _array_delta_after_apply(
            layer_stack, layer_stack.selection.data, before)


@dataclass(frozen=True)
class ClearSelectionCommand:
    """Clear the document-level selection."""

    label: str = "Clear Selection"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.selection.clear()
        layer_stack.publish_change(DocumentChangeKind.METADATA)

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        before = layer_stack.selection.data.copy()
        self.apply(layer_stack)
        return _array_delta_after_apply(
            layer_stack, layer_stack.selection.data, before)


@dataclass(frozen=True)
class InvertSelectionCommand:
    """Invert the document-level selection."""

    label: str = "Invert Selection"

    def apply(self, layer_stack: LayerStack) -> None:
        layer_stack.selection.data[:] = 1.0 - layer_stack.selection.data
        layer_stack.publish_change(DocumentChangeKind.METADATA)

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        before = layer_stack.selection.data.copy()
        self.apply(layer_stack)
        return _array_delta_after_apply(
            layer_stack, layer_stack.selection.data, before)


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
        layer_stack.publish_change(DocumentChangeKind.METADATA)

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        before = layer_stack.selection.data.copy()
        self.apply(layer_stack)
        return _array_delta_after_apply(
            layer_stack, layer_stack.selection.data, before)


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
        if layer.has_mask():
            mask_pil = Image.fromarray(layer.mask.to_uint8(), "L")
            mask_pil = mask_pil.filter(ImageFilter.MaxFilter(7))
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=4))
            mask_arg = np.array(mask_pil, dtype=np.uint8)
        else:
            mask_arg = None
        replacement = np.zeros_like(layer.image)
        intersects = paste_result(
            replacement,
            self.result_image,
            tool.patch_x - layer.x,
            tool.patch_y - layer.y,
            tool.patch_w,
            tool.patch_h,
            mask=mask_arg,
        )
        if not intersects:
            return
        layer.image[:] = replacement
        layer_stack.mark_layer_dirty(layer)
        layer_stack.publish_change(
            DocumentChangeKind.PIXELS, layers=(layer,))

    def apply_with_history(self, layer_stack: LayerStack) -> CommandDelta | None:
        before = self.layer.image.copy()
        self.apply(layer_stack)
        return _array_delta_after_apply(
            layer_stack, self.layer.image, before,
            layer=self.layer, pixels=True)
