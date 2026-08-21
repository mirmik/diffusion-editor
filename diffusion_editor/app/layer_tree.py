"""Toolkit-neutral layer-tree projection and intents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

from ..document.commands import (
    AddLayerCommand,
    AttachLayerToolCommand,
    DetachLayerToolCommand,
    FlattenLayersCommand,
    MoveLayerCommand,
    RemoveLayerCommand,
    SetLayerNameCommand,
    SetLayerOpacityCommand,
    SetLayerSoloCommand,
    SetLayerVisibilityCommand,
)
from ..document.document_service import DocumentService
from ..document.change_event import DocumentChangeEvent
from ..document.layer import Layer
from ..document.layer_stack import LayerStack
from ..document.tool import Tool


class LayerTreeAction(str, Enum):
    SELECT = "select"
    ADD = "add"
    REMOVE = "remove"
    FLATTEN = "flatten"
    VISIBILITY = "visibility"
    SOLO = "solo"
    OPACITY = "opacity"
    RENAME = "rename"
    MOVE = "move"
    ATTACH_TOOL = "attach_tool"
    DETACH_TOOL = "detach_tool"


class LayerDropPosition(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    INSIDE = "inside"
    ROOT = "root"


@dataclass(frozen=True)
class LayerTreeIntent:
    action: LayerTreeAction
    layer_id: str | None = None
    value: object = None
    target_id: str | None = None
    position: LayerDropPosition | None = None


@dataclass(frozen=True)
class LayerTreeNodeState:
    stable_id: str
    name: str
    visible: bool
    solo: bool
    tool_type: str | None
    children: tuple["LayerTreeNodeState", ...]
    node_type: str = "raster"
    status: str | None = None


@dataclass(frozen=True)
class LayerTreeState:
    roots: tuple[LayerTreeNodeState, ...]
    active_id: str | None
    opacity: float
    can_add: bool
    can_remove: bool
    can_flatten: bool
    can_rename: bool
    can_attach_tool: bool
    can_detach_tool: bool


class LayerTreePresentation(Protocol):
    def apply_layer_tree_state(self, state: LayerTreeState) -> None: ...


ToolFactory = Callable[[Layer, str], Tool | None]


class LayerTreeCoordinator:
    """Projects LayerStack and executes explicit layer intents."""

    def __init__(
            self,
            layer_stack: LayerStack,
            document: DocumentService,
            *,
            tool_factory: ToolFactory | None = None,
            before_remove_layer: Callable[[Layer], None] | None = None,
            before_detach_tool: Callable[[Layer], None] | None = None,
            set_status: Callable[[str], None] | None = None) -> None:
        self._layer_stack = layer_stack
        self._document = document
        self._tool_factory = tool_factory
        self._before_remove_layer = before_remove_layer
        self._before_detach_tool = before_detach_tool
        self._set_status = set_status or (lambda _text: None)
        self._view: LayerTreePresentation | None = None
        self._closed = False
        self._stack_subscription = layer_stack.subscribe(self._on_stack_changed)
        self._state = self._build_state()

    @property
    def state(self) -> LayerTreeState:
        return self._state

    def bind_view(self, view: LayerTreePresentation) -> None:
        self._require_open()
        self._view = view
        view.apply_layer_tree_state(self._state)

    def handle_intent(self, intent: LayerTreeIntent) -> None:
        self._require_open()
        layer = self._layer(intent.layer_id)
        action = intent.action
        if action == LayerTreeAction.SELECT:
            if layer is not None:
                self._layer_stack.active_layer = layer
            else:
                self.refresh()
        elif action == LayerTreeAction.ADD:
            if self._layer_stack.width > 0 and self._layer_stack.height > 0:
                self._document.execute(AddLayerCommand(
                    name=self._layer_stack.next_name("Layer")))
        elif action == LayerTreeAction.REMOVE:
            if self._can_remove(layer):
                if self._before_remove_layer is not None:
                    self._before_remove_layer(layer)
                self._document.execute(RemoveLayerCommand(layer=layer))
            else:
                self.refresh()
        elif action == LayerTreeAction.FLATTEN:
            layers = self._layer_stack.all_layers()
            if (
                    len(layers) > 1
                    and all(
                        layer.contributes_to_composite for layer in layers
                    )):
                self._document.execute(FlattenLayersCommand())
            else:
                self.refresh()
        elif action == LayerTreeAction.VISIBILITY:
            if layer is not None and layer.contributes_to_composite:
                self._document.execute(SetLayerVisibilityCommand(
                    layer=layer,
                    visible=bool(intent.value),
                ))
            else:
                self.refresh()
        elif action == LayerTreeAction.SOLO:
            if layer is not None and layer.contributes_to_composite:
                next_layer = (
                    None
                    if self._layer_stack.solo_layer_id == layer.id
                    else layer
                )
                self._document.execute(SetLayerSoloCommand(layer=next_layer))
                self._set_status(
                    "Solo off"
                    if next_layer is None
                    else f"Solo: {layer.name}"
                )
            else:
                self.refresh()
        elif action == LayerTreeAction.OPACITY:
            if layer is not None and layer.contributes_to_composite:
                opacity = max(0.0, min(float(intent.value), 1.0))
                self._document.execute(SetLayerOpacityCommand(
                    layer=layer,
                    opacity=opacity,
                ))
            else:
                self.refresh()
        elif action == LayerTreeAction.RENAME:
            if layer is not None:
                name = str(intent.value).strip()
                if name and name != layer.name:
                    self._document.execute(SetLayerNameCommand(
                        layer=layer,
                        name=name,
                    ))
            else:
                self.refresh()
        elif action == LayerTreeAction.MOVE:
            self._move(layer, intent)
        elif action == LayerTreeAction.ATTACH_TOOL:
            self._attach_tool(layer, str(intent.value))
        elif action == LayerTreeAction.DETACH_TOOL:
            self._detach_tool(layer)
        else:
            raise ValueError(f"unsupported layer tree action: {action}")

    def refresh(self) -> None:
        if self._closed:
            return
        self._state = self._build_state()
        if self._view is not None:
            self._view.apply_layer_tree_state(self._state)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._view = None
        self._stack_subscription.unsubscribe()

    def _on_stack_changed(self, _event: DocumentChangeEvent) -> None:
        self.refresh()

    def _build_state(self) -> LayerTreeState:
        active = self._layer_stack.active_layer
        all_layers = self._layer_stack.all_layers()
        return LayerTreeState(
            roots=tuple(self._node(layer) for layer in self._layer_stack.layers),
            active_id=active.id if active is not None else None,
            opacity=active.opacity if active is not None else 1.0,
            can_add=(
                self._layer_stack.width > 0
                and self._layer_stack.height > 0
            ),
            can_remove=self._can_remove(active),
            can_flatten=(
                len(all_layers) > 1
                and all(layer.contributes_to_composite for layer in all_layers)
            ),
            can_rename=active is not None,
            can_attach_tool=(
                active is not None
                and active.accepts_pixel_edits
                and active.tool is None
            ),
            can_detach_tool=active is not None and active.tool is not None,
        )

    def _node(self, layer: Layer) -> LayerTreeNodeState:
        tool_type = (
            str(layer.tool.tool_type)
            if layer.tool is not None else None
        )
        return LayerTreeNodeState(
            stable_id=layer.id,
            name=layer.name,
            visible=layer.visible,
            solo=layer.id == self._layer_stack.solo_layer_id,
            node_type=layer.node_type,
            status=(
                layer.reconstruction_status.value
                if layer.node_type == "reconstruction"
                else None
            ),
            tool_type=tool_type,
            children=tuple(self._node(child) for child in layer.children),
        )

    def _layer(self, stable_id: str | None) -> Layer | None:
        if stable_id is None:
            return None
        return self._layer_stack.find_layer_by_id(stable_id)

    def _can_remove(self, layer: Layer | None) -> bool:
        if layer is None:
            return False
        removed_count = 1 + len(layer.all_descendants())
        return len(self._layer_stack.all_layers()) > removed_count

    def _move(
            self,
            layer: Layer | None,
            intent: LayerTreeIntent) -> None:
        if layer is None or intent.position is None:
            self.refresh()
            return
        target = self._layer(intent.target_id)
        if target is layer or (
                target is not None
                and target in layer.all_descendants()):
            self.refresh()
            return

        position = LayerDropPosition(intent.position)
        if position == LayerDropPosition.INSIDE and target is not None:
            parent, index = target, 0
        elif position in (
                LayerDropPosition.BEFORE, LayerDropPosition.AFTER
        ) and target is not None:
            parent = target.parent
            siblings = (
                parent.children if parent is not None
                else self._layer_stack.layers
            )
            remaining = [candidate for candidate in siblings if candidate is not layer]
            if target not in remaining:
                self.refresh()
                return
            index = remaining.index(target)
            if position == LayerDropPosition.AFTER:
                index += 1
        elif position == LayerDropPosition.ROOT:
            parent = None
            remaining = [
                candidate for candidate in self._layer_stack.layers
                if candidate is not layer
            ]
            index = len(remaining)
        else:
            self.refresh()
            return
        if (
                (layer.node_type == "reconstruction" and parent is not None)
                or (
                    parent is not None
                    and parent.node_type == "reconstruction"
                )):
            self.refresh()
            return
        self._document.execute(MoveLayerCommand(
            layer=layer,
            new_parent=parent,
            index=index,
        ))

    def _attach_tool(self, layer: Layer | None, tool_type: str) -> None:
        if (
                layer is None
                or not layer.accepts_pixel_edits
                or layer.tool is not None
                or self._tool_factory is None):
            self.refresh()
            return
        tool = self._tool_factory(layer, tool_type)
        if tool is None:
            self.refresh()
            return
        label = {
            "diffusion": "Attach Diffusion Tool",
            "lama": "Attach LaMa Tool",
            "instruct": "Attach AI Edit Tool",
        }.get(tool_type, "Attach Tool")
        self._document.execute(AttachLayerToolCommand(
            layer=layer,
            tool=tool,
            label=label,
        ))

    def _detach_tool(self, layer: Layer | None) -> None:
        if layer is None or layer.tool is None:
            self.refresh()
            return
        if self._before_detach_tool is not None:
            self._before_detach_tool(layer)
        self._document.execute(DetachLayerToolCommand(layer=layer))

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("layer tree coordinator is closed")
