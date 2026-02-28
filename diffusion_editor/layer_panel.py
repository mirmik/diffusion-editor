"""LayerPanel — tcgui panel with tree of layers + management buttons."""

from __future__ import annotations

from tcgui.widgets.vstack import VStack
from tcgui.widgets.hstack import HStack
from tcgui.widgets.button import Button
from tcgui.widgets.label import Label
from tcgui.widgets.tree import TreeNode, TreeWidget
from tcgui.widgets.checkbox import Checkbox
from tcgui.widgets.slider_edit import SliderEdit
from tcgui.widgets.menu import Menu, MenuItem
from tcgui.widgets.dialog import Dialog
from tcgui.widgets.text_input import TextInput
from tcgui.widgets.message_box import MessageBox, Buttons
from tcgui.widgets.units import px, pct

from .layer_stack import LayerStack
from .layer import Layer, DiffusionLayer, LamaLayer, InstructLayer


class LayerPanel(VStack):
    def __init__(self, layer_stack: LayerStack):
        super().__init__()
        self._layer_stack = layer_stack
        self._id_to_layer: dict[int, Layer] = {}
        self._layer_to_node: dict[int, TreeNode] = {}
        self._node_to_layer: dict[int, Layer] = {}
        self._updating = False
        self.spacing = 6
        self.preferred_width = px(220)

        # Callbacks
        self.on_create_diffusion: callable = None
        self.on_create_lama: callable = None
        self.on_create_instruct: callable = None
        self.on_add_layer: callable = None
        self.on_remove_layer: callable = None
        self.on_flatten_layers: callable = None
        self.on_move_layer: callable = None  # (layer, new_parent, index)
        self.on_toggle_visibility: callable = None  # (layer, visible)
        self.on_opacity_changed: callable = None  # (layer, opacity)

        # Tree widget (stretch to fill remaining space)
        self._tree = TreeWidget()
        self._tree.preferred_width = pct(100)
        self._tree.stretch = True
        self._tree.row_height = 24
        self._tree.row_spacing = 1
        self._tree.draggable = True
        self._tree.on_select = self._on_tree_select
        self._tree.on_drop = self._on_tree_drop
        self._tree.context_menu = self._build_context_menu()
        self.add_child(self._tree)

        # Buttons row: + - Flatten
        btn_row = HStack()
        btn_row.spacing = 4

        add_btn = Button()
        add_btn.text = "+"
        add_btn.preferred_width = px(30)
        add_btn.on_click = self._on_add
        btn_row.add_child(add_btn)

        rm_btn = Button()
        rm_btn.text = "-"
        rm_btn.preferred_width = px(30)
        rm_btn.on_click = self._on_remove
        btn_row.add_child(rm_btn)

        flatten_btn = Button()
        flatten_btn.text = "Flatten"
        flatten_btn.preferred_width = px(60)
        flatten_btn.on_click = self._on_flatten
        btn_row.add_child(flatten_btn)

        self.add_child(btn_row)

        # Opacity slider
        self._opacity_slider = SliderEdit()
        self._opacity_slider.label = "Opacity"
        self._opacity_slider.min_value = 0.0
        self._opacity_slider.max_value = 1.0
        self._opacity_slider.value = 1.0
        self._opacity_slider.step = 0.01
        self._opacity_slider.preferred_width = pct(100)
        self._opacity_slider.on_changed = self._on_opacity_changed
        self.add_child(self._opacity_slider)

        # Create special layers
        diff_btn = Button()
        diff_btn.text = "Diffusion"
        diff_btn.preferred_width = pct(100)
        diff_btn.on_click = lambda: (self.on_create_diffusion and self.on_create_diffusion())
        self.add_child(diff_btn)

        lama_btn = Button()
        lama_btn.text = "LaMa"
        lama_btn.preferred_width = pct(100)
        lama_btn.on_click = lambda: (self.on_create_lama and self.on_create_lama())
        self.add_child(lama_btn)

        instruct_btn = Button()
        instruct_btn.text = "Instruct"
        instruct_btn.preferred_width = pct(100)
        instruct_btn.on_click = lambda: (self.on_create_instruct and self.on_create_instruct())
        self.add_child(instruct_btn)

    # ------------------------------------------------------------------
    # Sync from stack
    # ------------------------------------------------------------------

    def sync_from_stack(self):
        self._updating = True
        self._id_to_layer.clear()
        self._layer_to_node.clear()
        self._node_to_layer.clear()

        # Remove all roots
        self._tree.root_nodes.clear()
        self._tree._dirty = True

        for layer in self._layer_stack.layers:
            node = self._create_node(layer)
            self._tree.add_root(node)

        # Select active layer
        active = self._layer_stack.active_layer
        if active is not None:
            node = self._layer_to_node.get(id(active))
            if node is not None:
                self._tree._select_node(node)

        self._sync_opacity_slider()
        self._updating = False

    def _create_node(self, layer: Layer) -> TreeNode:
        if isinstance(layer, DiffusionLayer):
            prefix = "[D] "
        elif isinstance(layer, LamaLayer):
            prefix = "[L] "
        elif isinstance(layer, InstructLayer):
            prefix = "[I] "
        else:
            prefix = ""

        # HStack with visibility checkbox + name label
        row = HStack()
        row.spacing = 4

        vis_cb = Checkbox()
        vis_cb.checked = layer.visible
        vis_cb.text = ""
        vis_cb.preferred_width = px(16)

        def _make_vis_handler(ly):
            def handler(checked):
                if not self._updating:
                    if self.on_toggle_visibility:
                        self.on_toggle_visibility(ly, checked)
                    else:
                        self._layer_stack.set_visibility(ly, checked)
            return handler
        vis_cb.on_changed = _make_vis_handler(layer)
        row.add_child(vis_cb)

        name_lbl = Label()
        name_lbl.text = prefix + layer.name
        name_lbl.font_size = 13
        name_lbl.color = (0.9, 0.9, 0.9, 1.0)
        row.add_child(name_lbl)

        node = TreeNode(content=row)
        node.expanded = True

        self._id_to_layer[id(layer)] = layer
        self._layer_to_node[id(layer)] = node
        self._node_to_layer[id(node)] = layer

        for child in layer.children:
            child_node = self._create_node(child)
            node.add_node(child_node)

        return node

    def _layer_from_node(self, node: TreeNode) -> Layer | None:
        return self._node_to_layer.get(id(node))

    def _build_context_menu(self) -> Menu:
        panel = self
        tree = self._tree

        class _LayerContextMenu(Menu):
            def show(self, ui, x: float, y: float):
                node = tree._node_at_y(y)
                if node is not None:
                    tree._select_node(node)
                    panel._on_tree_select(node)
                super().show(ui, x, y)

        menu = _LayerContextMenu()
        menu.items = [
            MenuItem("Rename", on_click=self._on_rename),
            MenuItem("Delete", on_click=self._on_remove),
        ]
        return menu

    # ------------------------------------------------------------------
    # Tree callbacks
    # ------------------------------------------------------------------

    def _on_opacity_changed(self, value: float):
        if self._updating:
            return
        layer = self._layer_stack.active_layer
        if layer is None:
            return
        if self.on_opacity_changed:
            self.on_opacity_changed(layer, value)
        else:
            self._layer_stack.set_opacity(layer, value)

    def _sync_opacity_slider(self):
        """Update opacity slider to reflect the active layer."""
        layer = self._layer_stack.active_layer
        if layer is not None:
            self._opacity_slider.value = layer.opacity
        else:
            self._opacity_slider.value = 1.0

    def _on_tree_select(self, node: TreeNode):
        if self._updating:
            return
        layer = self._layer_from_node(node)
        if layer is not None:
            self._updating = True
            self._layer_stack.active_layer = layer
            self._opacity_slider.value = layer.opacity
            self._updating = False

    def _on_tree_drop(self, dragged: TreeNode, target: TreeNode | None,
                      position: str):
        dragged_layer = self._layer_from_node(dragged)
        if dragged_layer is None:
            return

        target_layer = self._layer_from_node(target) if target else None

        # Prevent dropping on self or own descendants
        if target_layer is not None:
            if target_layer is dragged_layer:
                return
            if target_layer in dragged_layer.all_descendants():
                return

        if position == "inside" and target_layer is not None:
            if self.on_move_layer:
                self.on_move_layer(dragged_layer, target_layer, 0)
            else:
                self._layer_stack.move_layer(dragged_layer, target_layer, 0)
        elif position == "above" and target_layer is not None:
            parent = target_layer.parent
            siblings = parent.children if parent else self._layer_stack._layers
            idx = siblings.index(target_layer) if target_layer in siblings else 0
            if self.on_move_layer:
                self.on_move_layer(dragged_layer, parent, idx)
            else:
                self._layer_stack.move_layer(dragged_layer, parent, idx)
        elif position == "below" and target_layer is not None:
            parent = target_layer.parent
            siblings = parent.children if parent else self._layer_stack._layers
            idx = siblings.index(target_layer) + 1 if target_layer in siblings else len(siblings)
            if self.on_move_layer:
                self.on_move_layer(dragged_layer, parent, idx)
            else:
                self._layer_stack.move_layer(dragged_layer, parent, idx)
        else:
            # root
            n = len(self._layer_stack._layers)
            if self.on_move_layer:
                self.on_move_layer(dragged_layer, None, n)
            else:
                self._layer_stack.move_layer(dragged_layer, None, n)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_add(self):
        if self.on_add_layer:
            self.on_add_layer()
        else:
            self._layer_stack.add_layer(self._layer_stack.next_name("Layer"))

    def _on_remove(self):
        layer = self._layer_stack.active_layer
        if layer is not None:
            if self.on_remove_layer:
                self.on_remove_layer(layer)
            else:
                self._layer_stack.remove_layer(layer)

    def _on_flatten(self):
        if self.on_flatten_layers:
            self.on_flatten_layers()
        else:
            self._layer_stack.flatten()

    # ------------------------------------------------------------------
    # Context actions
    # ------------------------------------------------------------------

    def _on_rename(self):
        layer = self._layer_stack.active_layer
        if layer is None:
            return
        self._show_rename_dialog(layer)

    def _show_rename_dialog(self, layer: Layer):
        if self._ui is None:
            return

        dlg = Dialog()
        dlg.title = "Rename Layer"
        dlg.buttons = ["OK", "Cancel"]
        dlg.default_button = "OK"
        dlg.cancel_button = "Cancel"

        input_box = TextInput()
        input_box.text = layer.name
        input_box.cursor_pos = len(layer.name)
        input_box.preferred_width = px(240)
        input_box.on_submit = lambda _text: dlg._on_button_click("OK")
        dlg.content = input_box

        def _apply(result: str):
            if result != "OK":
                return
            new_name = input_box.text.strip()
            if not new_name:
                return
            layer.name = new_name
            if self._layer_stack.on_changed:
                self._layer_stack.on_changed()

        dlg.on_result = _apply
        dlg.show(self._ui)
        self._ui.set_focus(input_box)
