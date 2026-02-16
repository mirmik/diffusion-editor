from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTreeWidget, QTreeWidgetItem, QAbstractItemView,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

from .layer import LayerStack, Layer, DiffusionLayer

_LAYER_ID_ROLE = Qt.ItemDataRole.UserRole

# QAbstractItemView.DropIndicatorPosition
_ON_ITEM = 0
_ABOVE_ITEM = 1
_BELOW_ITEM = 2
_ON_VIEWPORT = 3


class _DragTreeWidget(QTreeWidget):
    """QTreeWidget where we intercept drop and handle the move ourselves."""
    # dragged_layer_id, target_layer_id_or_None, indicator
    drop_requested = pyqtSignal(object, object, int)

    def dropEvent(self, event):
        dragged = self.selectedItems()
        if not dragged:
            return
        dragged_id = dragged[0].data(0, _LAYER_ID_ROLE)
        target_item = self.itemAt(event.position().toPoint())
        target_id = target_item.data(0, _LAYER_ID_ROLE) if target_item else None
        indicator = self.dropIndicatorPosition().value
        # Accept event but do NOT call super — we move layers in the model.
        # Defer the signal so Qt finishes drag-drop processing before we
        # modify the tree (otherwise Qt accesses destroyed C++ items → segfault).
        event.accept()
        QTimer.singleShot(0, lambda: self.drop_requested.emit(
            dragged_id, target_id, indicator))


class LayerPanel(QDockWidget):
    create_diffusion_requested = pyqtSignal()

    def __init__(self, layer_stack: LayerStack, parent=None):
        super().__init__("Layers", parent)
        self._layer_stack = layer_stack
        self._id_to_layer: dict[int, Layer] = {}
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setMinimumWidth(200)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        self._tree = _DragTreeWidget()
        self._tree.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tree.setHeaderHidden(True)
        self._tree.setIndentation(20)
        self._tree.setExpandsOnDoubleClick(False)
        layout.addWidget(self._tree)

        btn_layout = QHBoxLayout()
        self._add_btn = QPushButton("+")
        self._add_btn.setToolTip("New Layer")
        self._add_btn.setFixedWidth(40)
        self._remove_btn = QPushButton("-")
        self._remove_btn.setToolTip("Remove Layer")
        self._remove_btn.setFixedWidth(40)
        self._merge_btn = QPushButton("Flatten")
        self._merge_btn.setToolTip("Flatten all layers")
        btn_layout.addWidget(self._add_btn)
        btn_layout.addWidget(self._remove_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self._merge_btn)
        layout.addLayout(btn_layout)

        self._create_diff_btn = QPushButton("Create Diffusion Layer")
        layout.addWidget(self._create_diff_btn)

        self.setWidget(container)

        self._add_btn.clicked.connect(self._on_add)
        self._remove_btn.clicked.connect(self._on_remove)
        self._merge_btn.clicked.connect(self._on_flatten)
        self._create_diff_btn.clicked.connect(self.create_diffusion_requested.emit)
        self._tree.currentItemChanged.connect(self._on_selection_changed)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.drop_requested.connect(self._on_drop)
        self._layer_stack.changed.connect(self._sync_from_stack)

        self._updating = False

    def _layer_from_item(self, item: QTreeWidgetItem) -> Layer | None:
        layer_id = item.data(0, _LAYER_ID_ROLE)
        if layer_id is not None:
            return self._id_to_layer.get(layer_id)
        return None

    def _create_tree_item(self, layer: Layer) -> QTreeWidgetItem:
        name = f"[D] {layer.name}" if isinstance(layer, DiffusionLayer) else layer.name
        item = QTreeWidgetItem([name])
        item.setFlags(
            item.flags()
            | Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsDragEnabled
            | Qt.ItemFlag.ItemIsDropEnabled
        )
        item.setCheckState(0, Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
        item.setData(0, _LAYER_ID_ROLE, id(layer))
        self._id_to_layer[id(layer)] = layer
        for child in layer.children:
            item.addChild(self._create_tree_item(child))
        return item

    def _find_item_for_layer(self, layer: Layer) -> QTreeWidgetItem | None:
        target_id = id(layer)

        def _search(parent_item):
            for i in range(parent_item.childCount()):
                child = parent_item.child(i)
                if child.data(0, _LAYER_ID_ROLE) == target_id:
                    return child
                result = _search(child)
                if result is not None:
                    return result
            return None

        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            if item.data(0, _LAYER_ID_ROLE) == target_id:
                return item
            result = _search(item)
            if result is not None:
                return result
        return None

    def _sync_from_stack(self):
        self._updating = True
        self._id_to_layer.clear()
        self._tree.clear()
        for layer in self._layer_stack.layers:
            self._tree.addTopLevelItem(self._create_tree_item(layer))
        self._tree.expandAll()

        active = self._layer_stack.active_layer
        if active is not None:
            item = self._find_item_for_layer(active)
            if item is not None:
                self._tree.setCurrentItem(item)
        self._updating = False

    def _on_selection_changed(self, current, previous):
        if self._updating or current is None:
            return
        layer = self._layer_from_item(current)
        if layer is not None:
            self._layer_stack.active_layer = layer

    def _on_item_changed(self, item, column):
        if self._updating:
            return
        layer = self._layer_from_item(item)
        if layer is not None:
            visible = item.checkState(0) == Qt.CheckState.Checked
            self._layer_stack.set_visibility(layer, visible)

    def _on_drop(self, dragged_id, target_id, indicator):
        dragged = self._id_to_layer.get(dragged_id)
        if dragged is None:
            return
        target = self._id_to_layer.get(target_id) if target_id is not None else None

        # Can't drop on self or own descendant
        if target is not None:
            if target is dragged or target in dragged.all_descendants():
                return

        if indicator == _ON_ITEM and target is not None:
            # Drop ON item → make first child
            self._layer_stack.move_layer(dragged, target, 0)

        elif indicator == _ABOVE_ITEM and target is not None:
            parent = target.parent
            siblings = parent.children if parent else self._layer_stack._layers
            idx = siblings.index(target)
            if dragged in siblings and siblings.index(dragged) < idx:
                idx -= 1
            self._layer_stack.move_layer(dragged, parent, idx)

        elif indicator == _BELOW_ITEM and target is not None:
            parent = target.parent
            siblings = parent.children if parent else self._layer_stack._layers
            idx = siblings.index(target) + 1
            if dragged in siblings and siblings.index(dragged) < idx:
                idx -= 1
            self._layer_stack.move_layer(dragged, parent, idx)

        else:
            # OnViewport → move to end of root
            n = len(self._layer_stack._layers)
            if dragged in self._layer_stack._layers:
                n -= 1
            self._layer_stack.move_layer(dragged, None, n)

    def _on_add(self):
        self._layer_stack.add_layer(self._layer_stack.next_name("Layer"))

    def _on_remove(self):
        layer = self._layer_stack.active_layer
        if layer is not None:
            self._layer_stack.remove_layer(layer)

    def _on_flatten(self):
        self._layer_stack.flatten()
