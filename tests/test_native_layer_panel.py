from __future__ import annotations

from termin.gui_native import (
    KeyCode,
    KeyEvent,
    KeyEventType,
    Rect,
    TreeDropPosition,
    tc_ui_document_create,
    tc_ui_document_destroy,
)

from diffusion_editor.app.layer_tree import (
    LayerDropPosition,
    LayerTreeAction,
    LayerTreeNodeState,
    LayerTreeState,
)
from diffusion_editor.app.native_layer_panel import NativeLayerPanel


def _node(
        stable_id,
        name,
        *,
        children=(),
        visible=True,
        solo=False,
        tool_type=None):
    return LayerTreeNodeState(
        stable_id, name, visible, solo, tool_type, tuple(children))


def _state(*roots, active_id="child", opacity=0.6):
    return LayerTreeState(
        roots=tuple(roots),
        active_id=active_id,
        opacity=opacity,
        can_add=True,
        can_remove=True,
        can_flatten=True,
        can_rename=True,
        can_attach_tool=True,
        can_detach_tool=False,
    )


def _panel(state):
    document = tc_ui_document_create()
    intents = []
    panel = NativeLayerPanel(
        document,
        state,
        intents.append,
        viewport_rect=lambda: Rect(0.0, 0.0, 640.0, 480.0),
    )
    assert document.add_root(panel.widget.handle)
    return document, panel, intents


def test_native_layer_panel_sync_is_feedback_free_and_incremental():
    state = _state(_node(
        "group", "Group", children=(_node("child", "Child"),)))
    document, panel, intents = _panel(state)
    child_node = panel._stable_to_node["child"]
    group_node = panel._stable_to_node["group"]
    panel.expansion.set_expanded(group_node, True)

    panel.apply_layer_tree_state(_state(
        _node(
            "group",
            "Renamed group",
            children=(_node("child", "Updated", visible=False),),
        ),
        active_id="group",
        opacity=0.25,
    ))

    assert intents == []
    assert panel._stable_to_node["child"] == child_node
    assert panel.tree.selected_node == group_node
    assert panel.expansion.expanded(group_node)
    assert panel.opacity.value == 0.25
    assert panel.model.node(child_node).item.text == "[ ] Updated"

    panel.close()
    tc_ui_document_destroy(document)


def test_native_layer_panel_restores_expansion_and_selection_after_rebuild():
    state = _state(_node(
        "group", "Group", children=(_node("child", "Child"),)))
    document, panel, intents = _panel(state)
    old_group = panel._stable_to_node["group"]
    panel.expansion.set_expanded(old_group, True)

    panel.apply_layer_tree_state(_state(
        _node(
            "group",
            "Group",
            children=(
                _node("child", "Child"),
                _node("new", "New"),
            ),
        ),
        active_id="child",
    ))

    group = panel._stable_to_node["group"]
    assert panel.expansion.expanded(group)
    assert panel.tree.selected_node == panel._stable_to_node["child"]
    assert intents == []

    panel.close()
    tc_ui_document_destroy(document)


def test_native_layer_panel_emits_selection_controls_and_drop_intents():
    state = _state(
        _node("first", "First"),
        _node("child", "Child"),
    )
    document, panel, intents = _panel(state)

    panel.tree.select(panel._stable_to_node["first"])
    panel.visible.checked = False
    panel.solo.checked = True
    panel.opacity.value = 0.4
    panel._on_drop_requested(
        panel._stable_to_node["first"],
        panel._stable_to_node["child"],
        TreeDropPosition.After,
    )

    assert [intent.action for intent in intents] == [
        LayerTreeAction.SELECT,
        LayerTreeAction.VISIBILITY,
        LayerTreeAction.SOLO,
        LayerTreeAction.OPACITY,
        LayerTreeAction.MOVE,
    ]
    assert intents[-1].layer_id == "first"
    assert intents[-1].target_id == "child"
    assert intents[-1].position == LayerDropPosition.AFTER

    panel.close()
    tc_ui_document_destroy(document)


def test_native_layer_panel_rename_and_confirmed_delete_flows():
    state = _state(
        _node("first", "First"),
        _node("child", "Child"),
    )
    document, panel, intents = _panel(state)
    enter = KeyEvent()
    enter.type = KeyEventType.Down
    enter.key = KeyCode.Enter
    escape = KeyEvent()
    escape.type = KeyEventType.Down
    escape.key = KeyCode.Escape

    panel._show_rename("child")
    assert panel.rename_dialog.open
    panel.rename_dialog.value = "  Renamed  "
    document.dispatch_key_event(enter)
    assert intents[-1].action == LayerTreeAction.RENAME
    assert intents[-1].layer_id == "child"
    assert intents[-1].value == "  Renamed  "

    count = len(intents)
    assert document.set_focus(panel.tree.handle)
    delete = KeyEvent()
    delete.type = KeyEventType.Down
    delete.key = KeyCode.Delete
    document.dispatch_key_event(delete)
    assert panel._dialogs[-1].open
    document.dispatch_key_event(escape)
    assert len(intents) == count

    document.dispatch_key_event(delete)
    assert panel._dialogs[-1].open
    document.dispatch_key_event(enter)
    assert intents[-1].action == LayerTreeAction.REMOVE
    assert intents[-1].layer_id == "child"

    panel.close()
    tc_ui_document_destroy(document)


def test_native_layer_panel_context_commands_follow_tool_state():
    state = _state(_node("child", "Child"))
    document, panel, _intents = _panel(state)
    native_node = panel._stable_to_node["child"]

    panel._on_context_menu_requested(native_node, 10.0, 10.0)
    enabled = {
        command.data.stable_id: command.data.enabled
        for command in panel.context_model.commands
    }
    assert enabled["rename"]
    assert enabled["attach.diffusion"]
    assert enabled["attach.lama"]
    assert enabled["attach.instruct"]
    assert not enabled["detach"]
    assert enabled["delete"]

    panel.apply_layer_tree_state(_state(
        _node("child", "Child", tool_type="lama")))
    panel._on_context_menu_requested(native_node, 10.0, 10.0)
    enabled = {
        command.data.stable_id: command.data.enabled
        for command in panel.context_model.commands
    }
    assert not enabled["attach.diffusion"]
    assert not enabled["attach.lama"]
    assert not enabled["attach.instruct"]
    assert enabled["detach"]

    panel.close()
    tc_ui_document_destroy(document)
