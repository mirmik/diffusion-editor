"""Undoable SVG mutations. Render/validate before changing the document."""
from __future__ import annotations
from dataclasses import dataclass

from .change_event import DocumentChangeKind
from .commands import CommandDelta, _location, _restore_active
from .layer import Layer
from .svg_layer import SvgLayer, render_svg


@dataclass(frozen=True)
class AddSvgLayerCommand:
    name: str
    source: str
    width: int
    height: int
    label: str = "Add SVG Layer"

    def apply_with_history(self, stack):
        if stack.width <= 0 or stack.height <= 0:
            raise ValueError("Create a document before adding an SVG layer")
        layer = SvgLayer(self.name, self.source, self.width, self.height)
        old_active = stack.active_layer.id if stack.active_layer else None
        stack.insert_layer(layer)
        parent, index = _location(stack, layer)

        layer_id = layer.id
        parent_id = parent.id if parent is not None else None
        retained = layer

        def undo():
            nonlocal retained
            retained = stack.find_layer_by_id(layer_id)
            if retained is None:
                raise RuntimeError("Cannot undo SVG insertion: layer is detached")
            stack.remove_layer(retained)
            _restore_active(stack, old_active)

        def redo():
            target_parent = stack.find_layer_by_id(parent_id) if parent_id else None
            if parent_id and target_parent is None:
                raise RuntimeError("Cannot redo SVG insertion: parent is detached")
            stack.insert_layer(retained)
            stack.move_layer(retained, target_parent, index)

        return CommandDelta(undo, redo, layer.svg_content.size_bytes)


@dataclass(frozen=True)
class UpdateSvgLayerCommand:
    layer: SvgLayer
    source: str
    width: int | None = None
    height: int | None = None
    label: str = "Update SVG Layer"

    def apply_with_history(self, stack):
        if stack.find_layer_by_id(self.layer.id) is not self.layer:
            raise ValueError("SVG layer does not belong to this document")
        width = self.layer.width if self.width is None else self.width
        height = self.layer.height if self.height is None else self.height
        before = self.layer.svg_content
        if before.source == self.source and (width, height) == (self.layer.width, self.layer.height):
            return None
        after = render_svg(self.source, width, height)

        layer_id = self.layer.id

        def assign(content):
            layer = stack.find_layer_by_id(layer_id)
            if not isinstance(layer, SvgLayer):
                raise RuntimeError("Cannot update SVG: layer is detached or rasterized")
            resized = layer.image.shape != content.pixels.shape
            layer._install_svg_content(content)
            stack.mark_layer_dirty(layer)
            stack.publish_change(
                DocumentChangeKind.STRUCTURE if resized else DocumentChangeKind.PIXELS,
                layers=(layer,))

        assign(after)
        return CommandDelta(lambda: assign(before), lambda: assign(after),
                            before.size_bytes + after.size_bytes)


@dataclass(frozen=True)
class RasterizeSvgLayerCommand:
    layer: SvgLayer
    label: str = "Rasterize SVG Layer"

    def apply_with_history(self, stack):
        if not isinstance(self.layer, SvgLayer):
            raise ValueError("Select an SVG layer to rasterize")
        if stack.find_layer_by_id(self.layer.id) is not self.layer:
            raise ValueError("SVG layer does not belong to this document")
        source = self.layer
        raster = Layer(source.name, source.width, source.height, source.image.copy(),
                       layer_id=source.id, x=source.x, y=source.y)
        raster.visible, raster.opacity = source.visible, source.opacity
        layer_id = source.id

        def current_layer():
            current = stack.find_layer_by_id(layer_id)
            if current is None:
                raise RuntimeError("Cannot rasterize SVG: layer is detached")
            return current

        def undo():
            nonlocal raster
            raster = current_layer()
            stack.replace_layer(raster, source)

        def redo():
            nonlocal source
            source = current_layer()
            stack.replace_layer(source, raster)

        redo()
        return CommandDelta(undo, redo,
                            source.svg_content.size_bytes + raster.image.nbytes)


@dataclass(frozen=True)
class MoveSvgLayerCommand:
    layer: SvgLayer
    x: int
    y: int
    label: str = "Move SVG Layer"

    def apply_with_history(self, stack):
        if stack.find_layer_by_id(self.layer.id) is not self.layer:
            raise ValueError("SVG layer does not belong to this document")
        if type(self.x) is not int or type(self.y) is not int:
            raise ValueError("Layer offsets must be integer canvas pixels")
        before = self.layer.x, self.layer.y
        after = self.x, self.y
        if before == after:
            return None
        layer_id = self.layer.id

        def assign(offset):
            layer = stack.find_layer_by_id(layer_id)
            if layer is None:
                raise RuntimeError("Cannot move SVG: layer is detached")
            stack.set_layer_offset(layer, *offset)

        assign(after)
        return CommandDelta(lambda: assign(before), lambda: assign(after), 64)
