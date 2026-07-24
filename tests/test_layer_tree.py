from __future__ import annotations

import numpy as np

from diffusion_editor.app.layer_tree import (
    LayerDropPosition,
    LayerTreeAction,
    LayerTreeCoordinator,
    LayerTreeIntent,
)
from diffusion_editor.document.document_service import DocumentService
from diffusion_editor.document.history import HistoryManager
from diffusion_editor.document.layer import Layer
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.document.tool import LamaTool


def _stack_and_service():
    stack = LayerStack()
    stack.init_from_image(np.zeros((12, 16, 4), dtype=np.uint8))
    history = HistoryManager(stack.load_state)
    return stack, DocumentService(stack, history, stack.load_state)


def _intent(action, layer=None, **kwargs):
    return LayerTreeIntent(
        action,
        layer_id=layer.id if layer is not None else None,
        **kwargs,
    )


def test_layer_tree_projects_nested_stable_ids_and_restores_callback():
    stack, service = _stack_and_service()
    background = stack.active_layer
    stack.add_layer("Group")
    group = stack.active_layer
    child = Layer("Child", stack.width, stack.height)
    group.add_child(child)
    stack.active_layer = child
    calls = []
    previous = lambda: calls.append("changed")
    stack.on_changed = previous

    coordinator = LayerTreeCoordinator(stack, service)
    state = coordinator.state

    assert [node.stable_id for node in state.roots] == [
        group.id, background.id]
    assert state.roots[0].children[0].stable_id == child.id
    assert state.active_id == child.id
    assert stack.on_changed is not previous

    stack.set_layer_name(child, "Renamed externally")
    assert calls == ["changed"]
    assert coordinator.state.roots[0].children[0].name == "Renamed externally"

    coordinator.close()
    assert stack.on_changed is previous


def test_layer_tree_executes_edit_intents_with_undoable_solo():
    stack, service = _stack_and_service()
    first = stack.active_layer
    stack.add_layer("Second")
    second = stack.active_layer
    statuses = []
    coordinator = LayerTreeCoordinator(
        stack, service, set_status=statuses.append)

    coordinator.handle_intent(_intent(
        LayerTreeAction.VISIBILITY, first, value=False))
    coordinator.handle_intent(_intent(
        LayerTreeAction.OPACITY, first, value=1.5))
    coordinator.handle_intent(_intent(
        LayerTreeAction.RENAME, first, value="  Base  "))
    coordinator.handle_intent(_intent(LayerTreeAction.SOLO, first))

    assert not first.visible
    assert first.opacity == 1.0
    assert first.name == "Base"
    assert stack.solo_layer_id == first.id
    assert statuses[-1] == "Solo: Base"

    assert service.undo() == "Set Solo Layer"
    assert stack.solo_layer_id is None
    assert service.redo() == "Set Solo Layer"
    assert stack.solo_layer_id == first.id

    coordinator.handle_intent(_intent(LayerTreeAction.SELECT, second))
    assert stack.active_layer.id == second.id
    coordinator.handle_intent(_intent(LayerTreeAction.REMOVE, second))
    assert stack.find_layer_by_id(second.id) is None
    coordinator.handle_intent(LayerTreeIntent(LayerTreeAction.ADD))
    assert len(stack.all_layers()) == 2
    coordinator.handle_intent(LayerTreeIntent(LayerTreeAction.FLATTEN))
    assert len(stack.all_layers()) == 1


def test_layer_tree_moves_across_tree_and_rejects_cycles_and_stale_ids():
    stack, service = _stack_and_service()
    background = stack.active_layer
    stack.add_layer("Group")
    group = stack.active_layer
    child = Layer("Child", stack.width, stack.height)
    group.add_child(child)
    stack.active_layer = child
    coordinator = LayerTreeCoordinator(stack, service)

    coordinator.handle_intent(_intent(
        LayerTreeAction.MOVE,
        background,
        target_id=group.id,
        position=LayerDropPosition.INSIDE,
    ))
    assert background.parent is group
    assert group.children[0] is background

    coordinator.handle_intent(_intent(
        LayerTreeAction.MOVE,
        group,
        target_id=child.id,
        position=LayerDropPosition.INSIDE,
    ))
    assert group.parent is None

    coordinator.handle_intent(LayerTreeIntent(
        LayerTreeAction.MOVE,
        layer_id="stale",
        target_id=group.id,
        position=LayerDropPosition.AFTER,
    ))
    assert stack.layers == [group]

    coordinator.handle_intent(_intent(
        LayerTreeAction.MOVE,
        child,
        position=LayerDropPosition.ROOT,
    ))
    assert child.parent is None
    assert stack.layers == [group, child]


def test_layer_tree_attach_detach_uses_injected_tool_boundary():
    stack, service = _stack_and_service()
    layer = stack.active_layer
    detached = []

    def make_tool(target, tool_type):
        assert target is layer
        assert tool_type == "lama"
        return LamaTool(None, 0, 0, 16, 12)

    coordinator = LayerTreeCoordinator(
        stack,
        service,
        tool_factory=make_tool,
        before_detach_tool=detached.append,
    )
    coordinator.handle_intent(_intent(
        LayerTreeAction.ATTACH_TOOL, layer, value="lama"))
    assert layer.tool.tool_type == "lama"
    assert coordinator.state.can_detach_tool

    coordinator.handle_intent(_intent(LayerTreeAction.DETACH_TOOL, layer))
    assert detached == [layer]
    assert layer.tool is None
    assert coordinator.state.can_attach_tool


def test_layer_tree_ignores_invalid_mutation_targets():
    stack, service = _stack_and_service()
    coordinator = LayerTreeCoordinator(stack, service)
    before = stack.serialize_state()

    coordinator.handle_intent(LayerTreeIntent(
        LayerTreeAction.RENAME, layer_id="missing", value="Nope"))
    coordinator.handle_intent(LayerTreeIntent(
        LayerTreeAction.OPACITY, layer_id="missing", value=0.2))
    coordinator.handle_intent(LayerTreeIntent(
        LayerTreeAction.REMOVE, layer_id="missing"))

    assert stack.serialize_state() == before


def test_layer_tree_does_not_remove_the_only_root_subtree():
    stack, service = _stack_and_service()
    root = stack.active_layer
    child = Layer("Child", stack.width, stack.height)
    root.add_child(child)
    stack.active_layer = root
    coordinator = LayerTreeCoordinator(stack, service)

    assert not coordinator.state.can_remove
    coordinator.handle_intent(_intent(LayerTreeAction.REMOVE, root))

    assert stack.layers == [root]
    assert root.children == [child]
