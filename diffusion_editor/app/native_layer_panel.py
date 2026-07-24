"""termin-gui-native projection of the layer tree."""

from __future__ import annotations

from typing import Callable

from termin.gui_native import (
    CollectionItem,
    CommandData,
    CommandKind,
    CommandModel,
    MessageBoxKind,
    Point,
    Rect,
    TcDocument,
    TreeDropPosition,
    TreeExpansionModel,
    TreeModel,
)

from .layer_tree import (
    LayerDropPosition,
    LayerTreeAction,
    LayerTreeIntent,
    LayerTreeNodeState,
    LayerTreeState,
)


class NativeLayerPanel:
    """Native tree/model view with explicit layer intents."""

    def __init__(
            self,
            document: TcDocument,
            state: LayerTreeState,
            on_intent: Callable[[LayerTreeIntent], None],
            viewport_rect: Callable[[], Rect]) -> None:
        self._document = document
        self._on_intent = on_intent
        self._viewport_rect = viewport_rect
        self._state = state
        self._closed = False
        self._syncing = False
        self._connections: list[object] = []
        self._stable_to_node: dict[str, int] = {}
        self._node_to_stable: dict[int, str] = {}
        self._structure: tuple[tuple[str, str | None, int], ...] = ()
        self._context_layer_id: str | None = None
        self._rename_layer_id: str | None = None
        self._dialogs: list[object] = []
        self._dialog_connections: list[object] = []

        self.widget = document.create_vstack("NativeLayerPanel")
        self.widget.stable_id = "diffusion-editor.layer-panel-content"
        self.widget.set_layout_spacing(4.0)

        self.model = TreeModel()
        self.expansion = TreeExpansionModel()
        self.tree = document.create_tree_widget(self.model, self.expansion)
        self.tree.widget.stable_id = "diffusion-editor.layer-tree"
        self.tree.set_row_height(26.0)
        self.tree.set_row_spacing(1.0)
        self.tree.set_indent_size(18.0)
        self.tree.draggable = True
        self._connections.extend((
            self.tree.connect_selection_changed(self._on_selection_changed),
            self.tree.connect_context_menu_requested(
                self._on_context_menu_requested),
            self.tree.connect_delete_requested(self._on_delete_requested),
            self.tree.connect_drop_requested(self._on_drop_requested),
        ))
        self.widget.add_flex_child(self.tree.widget, 1.0)

        buttons = document.create_hstack("NativeLayerButtons")
        buttons.set_layout_spacing(4.0)
        self.add_button = document.create_button("+")
        self.add_button.widget.stable_id = "diffusion-editor.layer.add"
        self.remove_button = document.create_button("-")
        self.remove_button.widget.stable_id = "diffusion-editor.layer.remove"
        self.flatten_button = document.create_button("Flatten")
        self.flatten_button.widget.stable_id = "diffusion-editor.layer.flatten"
        self._connections.extend((
            self.add_button.connect_clicked(
                lambda: self._emit(LayerTreeAction.ADD)),
            self.remove_button.connect_clicked(self._request_active_delete),
            self.flatten_button.connect_clicked(
                lambda: self._emit(LayerTreeAction.FLATTEN)),
        ))
        buttons.add_fixed_child(self.add_button.widget, 30.0)
        buttons.add_fixed_child(self.remove_button.widget, 30.0)
        buttons.add_flex_child(self.flatten_button.widget, 1.0)
        self.widget.add_preferred_child(buttons)

        active_row = document.create_hstack("NativeLayerActiveFlags")
        active_row.set_layout_spacing(4.0)
        self.visible = document.create_checkbox(True)
        self.visible.widget.stable_id = "diffusion-editor.layer.visible"
        self.solo = document.create_checkbox(False)
        self.solo.widget.stable_id = "diffusion-editor.layer.solo"
        self._connections.extend((
            self.visible.connect_changed(self._on_visibility_changed),
            self.solo.connect_changed(self._on_solo_changed),
        ))
        self._add_checkbox_label(
            active_row, self.visible, "Visible", "visible")
        self._add_checkbox_label(active_row, self.solo, "Solo", "solo")
        self.widget.add_preferred_child(active_row)

        self.opacity = document.create_slider_edit(state.opacity)
        self.opacity.widget.stable_id = "diffusion-editor.layer.opacity"
        self.opacity.label = "Opacity"
        self.opacity.set_range(0.0, 1.0)
        self.opacity.set_step(0.01)
        self.opacity.set_decimals(2)
        self._connections.append(
            self.opacity.connect_changed(self._on_opacity_changed))
        self.widget.add_preferred_child(self.opacity.widget)

        self.context_model = CommandModel()
        self.context_menu = document.create_menu(self.context_model)
        self.context_menu.widget.stable_id = (
            "diffusion-editor.layer.context-menu")
        self._connections.append(
            self.context_menu.connect_activated(self._on_context_activated))

        self.rename_dialog = document.create_input_dialog(
            "Rename Layer",
            "Layer name",
            "",
        )
        self.rename_dialog.widget.stable_id = (
            "diffusion-editor.layer.rename-dialog")
        self._connections.append(
            self.rename_dialog.connect_value_finished(
                self._on_rename_finished))

        self.apply_layer_tree_state(state)

    def apply_layer_tree_state(self, state: LayerTreeState) -> None:
        if self._closed:
            return
        self._syncing = True
        try:
            next_structure = self._state_structure(state)
            if next_structure == self._structure:
                for node in self._walk_nodes(state.roots):
                    native_node = self._stable_to_node.get(node.stable_id)
                    if native_node is not None and self.model.contains(native_node):
                        self.model.update(native_node, self._item(node))
            else:
                self._rebuild_model(state)
                self._structure = next_structure

            selected = self._stable_to_node.get(state.active_id or "")
            if selected is not None and self.model.contains(selected):
                self.tree.select(selected)
            else:
                self.tree.clear_selection()

            active = self._find_state_node(state.roots, state.active_id)
            self.visible.checked = active.visible if active is not None else False
            self.solo.checked = active.solo if active is not None else False
            self.opacity.value = state.opacity
            has_active = active is not None
            self.visible.widget.enabled = has_active
            self.solo.widget.enabled = has_active
            self.opacity.widget.enabled = has_active
            self.add_button.widget.enabled = state.can_add
            self.remove_button.widget.enabled = state.can_remove
            self.flatten_button.widget.enabled = state.can_flatten
            self._state = state
        finally:
            self._syncing = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._on_intent = lambda _intent: None
        self._connections.clear()

    def _rebuild_model(self, state: LayerTreeState) -> None:
        expanded_ids = {
            stable_id
            for stable_id, node in self._stable_to_node.items()
            if self.model.contains(node) and self.expansion.expanded(node)
        }
        self.model.clear()
        self.expansion.clear()
        self._stable_to_node.clear()
        self._node_to_stable.clear()

        def append(
                node_state: LayerTreeNodeState,
                parent: int | None = None) -> int:
            if parent is None:
                native_node = self.model.append_root(self._item(node_state))
            else:
                native_node = self.model.append_child(
                    parent, self._item(node_state))
            self._stable_to_node[node_state.stable_id] = native_node
            self._node_to_stable[native_node] = node_state.stable_id
            for child in node_state.children:
                append(child, native_node)
            if node_state.children and (
                    not expanded_ids
                    or node_state.stable_id in expanded_ids):
                self.expansion.set_expanded(native_node, True)
            return native_node

        for root in state.roots:
            append(root)
        self.expansion.reconcile(self.model)

    @staticmethod
    def _item(node: LayerTreeNodeState) -> CollectionItem:
        flags = "[V]" if node.visible else "[ ]"
        if node.solo:
            flags += "[S]"
        subtitle = (
            f"{node.tool_type.capitalize()} tool"
            if node.tool_type else ""
        )
        return CollectionItem(
            node.stable_id,
            f"{flags} {node.name}",
            subtitle,
        )

    def _add_checkbox_label(
            self, parent, checkbox, label: str, suffix: str) -> None:
        cell = self._document.create_hstack("NativeLayerFlagCell")
        cell.set_layout_spacing(2.0)
        text = self._document.create_label(label, "NativeLayerFlagLabel")
        text.stable_id = f"diffusion-editor.layer.{suffix}.label"
        cell.add_preferred_child(checkbox.widget)
        cell.add_flex_child(text, 1.0)
        parent.add_flex_child(cell, 1.0)

    def _on_selection_changed(self, native_node: int) -> None:
        if self._syncing or self._closed:
            return
        stable_id = self._node_to_stable.get(native_node)
        if stable_id is not None:
            self._emit(LayerTreeAction.SELECT, layer_id=stable_id)

    def _on_visibility_changed(self, visible: bool) -> None:
        if not self._syncing:
            self._emit(
                LayerTreeAction.VISIBILITY,
                layer_id=self._state.active_id,
                value=visible,
            )

    def _on_solo_changed(self, _solo: bool) -> None:
        if not self._syncing:
            self._emit(
                LayerTreeAction.SOLO,
                layer_id=self._state.active_id,
            )

    def _on_opacity_changed(self, opacity: float) -> None:
        if not self._syncing:
            self._emit(
                LayerTreeAction.OPACITY,
                layer_id=self._state.active_id,
                value=opacity,
            )

    def _on_context_menu_requested(
            self, native_node: int, x: float, y: float) -> None:
        if self._closed:
            return
        stable_id = self._node_to_stable.get(native_node)
        if stable_id is None:
            return
        self.tree.select(native_node)
        self._context_layer_id = stable_id
        node = self._find_state_node(self._state.roots, stable_id)
        has_tool = node is not None and node.tool_type is not None
        commands = [
            CommandData("rename", "Rename", enabled=node is not None),
            CommandData("separator.0", kind=CommandKind.Separator),
            CommandData(
                "attach.diffusion",
                "Attach Diffusion Tool",
                enabled=node is not None and not has_tool,
            ),
            CommandData(
                "attach.lama",
                "Attach LaMa Tool",
                enabled=node is not None and not has_tool,
            ),
            CommandData(
                "attach.instruct",
                "Attach Instruct Tool",
                enabled=node is not None and not has_tool,
            ),
            CommandData(
                "detach",
                "Remove Tool",
                enabled=has_tool,
            ),
            CommandData("separator.1", kind=CommandKind.Separator),
            CommandData(
                "delete",
                "Delete Layer",
                enabled=self._state.can_remove,
            ),
        ]
        self.context_model.set_commands(commands)
        self.context_menu.show(Point(x, y), self._safe_viewport())

    def _on_context_activated(
            self, _index: int, _command_id, command) -> None:
        layer_id = self._context_layer_id
        action = command.stable_id
        if action == "rename":
            self._show_rename(layer_id)
        elif action.startswith("attach."):
            self._emit(
                LayerTreeAction.ATTACH_TOOL,
                layer_id=layer_id,
                value=action.split(".", 1)[1],
            )
        elif action == "detach":
            self._emit(LayerTreeAction.DETACH_TOOL, layer_id=layer_id)
        elif action == "delete":
            self._request_delete(layer_id)

    def _on_delete_requested(self, native_node: int, _item) -> None:
        self._request_delete(self._node_to_stable.get(native_node))

    def _request_active_delete(self) -> None:
        self._request_delete(self._state.active_id)

    def _request_delete(self, layer_id: str | None) -> None:
        node = self._find_state_node(self._state.roots, layer_id)
        if node is None or not self._state.can_remove:
            return
        dialog = self._document.create_message_box(
            "Delete Layer",
            f'Delete layer "{node.name}"?',
            MessageBoxKind.Question,
        )
        dialog.widget.stable_id = "diffusion-editor.layer.delete-dialog"
        self._dialog_connections.append(dialog.connect_finished(
            lambda result, target=layer_id: self._on_delete_finished(
                target, result.action_id)))
        self._dialogs.append(dialog)
        dialog.show(self._safe_viewport())

    def _on_delete_finished(self, layer_id: str, action_id: str) -> None:
        if action_id == "yes":
            self._emit(LayerTreeAction.REMOVE, layer_id=layer_id)

    def _show_rename(self, layer_id: str | None) -> None:
        node = self._find_state_node(self._state.roots, layer_id)
        if node is None or self.rename_dialog.open:
            return
        self._rename_layer_id = layer_id
        self.rename_dialog.value = node.name
        self.rename_dialog.show(self._safe_viewport())

    def _on_rename_finished(self, value: str | None) -> None:
        layer_id, self._rename_layer_id = self._rename_layer_id, None
        if value is not None and layer_id is not None:
            self._emit(
                LayerTreeAction.RENAME,
                layer_id=layer_id,
                value=value,
            )

    def _on_drop_requested(
            self,
            dragged: int,
            target: int,
            position: TreeDropPosition) -> None:
        positions = {
            TreeDropPosition.Before: LayerDropPosition.BEFORE,
            TreeDropPosition.After: LayerDropPosition.AFTER,
            TreeDropPosition.Inside: LayerDropPosition.INSIDE,
            TreeDropPosition.Root: LayerDropPosition.ROOT,
        }
        mapped = positions.get(position)
        dragged_id = self._node_to_stable.get(dragged)
        if mapped is None or dragged_id is None:
            return
        self._on_intent(LayerTreeIntent(
            action=LayerTreeAction.MOVE,
            layer_id=dragged_id,
            target_id=self._node_to_stable.get(target),
            position=mapped,
        ))

    def _emit(
            self,
            action: LayerTreeAction,
            *,
            layer_id: str | None = None,
            value=None) -> None:
        if not self._closed:
            self._on_intent(LayerTreeIntent(
                action=action,
                layer_id=layer_id,
                value=value,
            ))

    def _safe_viewport(self) -> Rect:
        viewport = self._viewport_rect()
        if viewport.width <= 0 or viewport.height <= 0:
            return Rect(0.0, 0.0, 640.0, 480.0)
        return viewport

    @classmethod
    def _walk_nodes(
            cls,
            roots: tuple[LayerTreeNodeState, ...]):
        for node in roots:
            yield node
            yield from cls._walk_nodes(node.children)

    @classmethod
    def _state_structure(
            cls,
            state: LayerTreeState) -> tuple[tuple[str, str | None, int], ...]:
        result = []

        def walk(nodes, parent_id):
            for index, node in enumerate(nodes):
                result.append((node.stable_id, parent_id, index))
                walk(node.children, node.stable_id)

        walk(state.roots, None)
        return tuple(result)

    @classmethod
    def _find_state_node(
            cls,
            roots: tuple[LayerTreeNodeState, ...],
            stable_id: str | None) -> LayerTreeNodeState | None:
        if stable_id is None:
            return None
        for node in roots:
            if node.stable_id == stable_id:
                return node
            found = cls._find_state_node(node.children, stable_id)
            if found is not None:
                return found
        return None
