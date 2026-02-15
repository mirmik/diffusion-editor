from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QAbstractItemView,
)
from PyQt6.QtCore import Qt, pyqtSignal

from .layer import LayerStack


class LayerPanel(QDockWidget):
    def __init__(self, layer_stack: LayerStack, parent=None):
        super().__init__("Layers", parent)
        self._layer_stack = layer_stack
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.setMinimumWidth(200)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        self._list = QListWidget()
        self._list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self._list)

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

        self.setWidget(container)

        self._add_btn.clicked.connect(self._on_add)
        self._remove_btn.clicked.connect(self._on_remove)
        self._merge_btn.clicked.connect(self._on_flatten)
        self._list.currentRowChanged.connect(self._on_selection_changed)
        self._list.itemChanged.connect(self._on_item_changed)
        self._layer_stack.changed.connect(self._sync_from_stack)

        self._updating = False

    def _sync_from_stack(self):
        self._updating = True
        self._list.clear()
        for layer in self._layer_stack.layers:
            item = QListWidgetItem(layer.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if layer.visible else Qt.CheckState.Unchecked)
            self._list.addItem(item)
        if 0 <= self._layer_stack.active_index < self._list.count():
            self._list.setCurrentRow(self._layer_stack.active_index)
        self._updating = False

    def _on_selection_changed(self, row):
        if self._updating:
            return
        if row >= 0:
            self._layer_stack._active_index = row

    def _on_item_changed(self, item):
        if self._updating:
            return
        row = self._list.row(item)
        visible = item.checkState() == Qt.CheckState.Checked
        self._layer_stack.set_visibility(row, visible)

    def _on_add(self):
        count = len(self._layer_stack.layers)
        self._layer_stack.add_layer(f"Layer {count}")

    def _on_remove(self):
        idx = self._layer_stack.active_index
        self._layer_stack.remove_layer(idx)

    def _on_flatten(self):
        self._layer_stack.flatten()
