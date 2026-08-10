"""Toolkit-neutral handlers for the editor's standard command inventory."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..document.commands import (
    AddLayerCommand,
    ClearSelectionCommand,
    FlattenLayersCommand,
    InvertSelectionCommand,
    RemoveLayerCommand,
    SelectAllCommand,
)
from .application import EditorApplication


class EditorCommandCoordinator:
    """Execute standard edit commands and project their enabled state."""

    def __init__(
            self,
            application: EditorApplication,
            *,
            fit_in_view: Callable[[], None] | None = None,
            request_remove_layer: Callable[[], None] | None = None,
            before_mutation: Callable[[], None] | None = None) -> None:
        self._application = application
        self._stack = application.layer_stack
        self._document = application.document
        self._fit_in_view = fit_in_view
        self._request_remove_layer = request_remove_layer
        self._before_mutation = before_mutation or (lambda: None)
        self._projected_states: dict[str, tuple[bool, bool]] = {}
        self.handlers: dict[str, Callable[[], None]] = {
            "edit.undo": self.undo,
            "edit.redo": self.redo,
            "edit.copy": self.copy,
            "edit.copy_visible": self.copy_visible,
            "edit.paste": self.paste,
            "selection.all": self.select_all,
            "selection.background": self.select_background,
            "selection.clear": self.clear_selection,
            "selection.invert": self.invert_selection,
            "layer.new": self.new_layer,
            "layer.remove": self.remove_layer,
            "layer.flatten": self.flatten_layers,
            "view.fit": self.fit,
        }
        self.refresh()

    def refresh(self) -> None:
        canvas_ready = self._stack.width > 0 and self._stack.height > 0
        active = self._stack.active_layer
        active_raster = (
            active is not None and active.accepts_pixel_edits
        )
        selection_bbox = self._stack.selection.bbox() if canvas_ready else None
        layers = self._stack.all_layers()
        can_remove = False
        if active is not None:
            removed_count = 1 + len(active.all_descendants())
            can_remove = len(layers) > removed_count
        states = {
            "edit.undo": self._application.history.can_undo,
            "edit.redo": self._application.history.can_redo,
            "edit.copy": active_raster and selection_bbox is not None,
            "edit.copy_visible": canvas_ready and selection_bbox is not None,
            "edit.paste": (
                canvas_ready and self._application.clipboard is not None
            ),
            "selection.all": canvas_ready,
            "selection.background": (
                canvas_ready
                and active is not None
                and not self._application.segmentation_controller.is_busy
            ),
            "selection.clear": (
                canvas_ready and not self._stack.selection.is_empty
            ),
            "selection.invert": canvas_ready,
            "layer.new": canvas_ready,
            "layer.remove": can_remove,
            "layer.flatten": (
                len(layers) > 1
                and all(layer.contributes_to_composite for layer in layers)
            ),
            "view.fit": canvas_ready and self._fit_in_view is not None,
        }
        for command_id, enabled in states.items():
            state = (bool(enabled), False)
            if self._projected_states.get(command_id) == state:
                continue
            self._projected_states[command_id] = state
            self._application.set_command_state(
                command_id,
                enabled=state[0],
                checked=state[1],
            )

    def undo(self) -> None:
        self._before_mutation()
        label = self._document.undo()
        if label is not None:
            self._application.set_status(f"Undo: {label}")
        self.refresh()

    def redo(self) -> None:
        self._before_mutation()
        label = self._document.redo()
        if label is not None:
            self._application.set_status(f"Redo: {label}")
        self.refresh()

    def select_all(self) -> None:
        self._execute(SelectAllCommand(), "Selected entire canvas")

    def select_background(self) -> None:
        self._before_mutation()
        layer = self._stack.active_layer
        if layer is None:
            self.refresh()
            return
        event = (
            self._application.segmentation_controller
            .start_select_background_selection(layer)
        )
        if event.status is not None:
            self._application.set_status(event.status)
        self.refresh()

    def clear_selection(self) -> None:
        self._execute(ClearSelectionCommand(), "Selection cleared")

    def invert_selection(self) -> None:
        self._execute(InvertSelectionCommand(), "Selection inverted")

    def new_layer(self) -> None:
        self._execute(
            AddLayerCommand(name=self._stack.next_name("Layer")),
            "New layer",
        )

    def remove_layer(self) -> None:
        self._before_mutation()
        if not self._can_remove_active():
            self.refresh()
            return
        if self._request_remove_layer is not None:
            self._request_remove_layer()
        else:
            layer = self._stack.active_layer
            if layer is not None:
                self._document.execute(RemoveLayerCommand(layer=layer))
                self._application.set_status("Layer removed")
        self.refresh()

    def flatten_layers(self) -> None:
        self._execute(FlattenLayersCommand(), "Layers flattened")

    def fit(self) -> None:
        if self._fit_in_view is not None:
            self._fit_in_view()

    def copy(self) -> None:
        self._before_mutation()
        layer = self._stack.active_layer
        bbox = self._stack.selection.bbox()
        if layer is None or not layer.accepts_pixel_edits or bbox is None:
            self._application.set_status("Copy: nothing selected")
            self.refresh()
            return
        x0, y0, x1, y1 = bbox
        patch = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
        ix0 = max(x0, layer.x)
        iy0 = max(y0, layer.y)
        ix1 = min(x1, layer.x + layer.width)
        iy1 = min(y1, layer.y + layer.height)
        if ix1 > ix0 and iy1 > iy0:
            src_x0 = ix0 - layer.x
            src_y0 = iy0 - layer.y
            dst_x0 = ix0 - x0
            dst_y0 = iy0 - y0
            width = ix1 - ix0
            height = iy1 - iy0
            patch[
                dst_y0:dst_y0 + height,
                dst_x0:dst_x0 + width,
            ] = layer.image[
                src_y0:src_y0 + height,
                src_x0:src_x0 + width,
            ]
        self._apply_selection_alpha(patch, bbox)
        self._application.clipboard = patch
        self._application.clipboard_pos = (x0, y0)
        self._application.set_status(f"Copied {x1 - x0}x{y1 - y0}")
        self.refresh()

    def copy_visible(self) -> None:
        self._before_mutation()
        bbox = self._stack.selection.bbox()
        if bbox is None:
            self._application.set_status("Copy Visible: nothing selected")
            self.refresh()
            return
        x0, y0, x1, y1 = bbox
        patch = self._stack.composite()[y0:y1, x0:x1].copy()
        self._apply_selection_alpha(patch, bbox)
        self._application.clipboard = patch
        self._application.clipboard_pos = (x0, y0)
        self._application.set_status(
            f"Copied {x1 - x0}x{y1 - y0} (visible)"
        )
        self.refresh()

    def paste(self) -> None:
        self._before_mutation()
        clipboard = self._application.clipboard
        if clipboard is None:
            self._application.set_status("Paste: clipboard is empty")
            self.refresh()
            return
        height, width = clipboard.shape[:2]
        position = self._application.clipboard_pos
        if position is None:
            x = (self._stack.width - width) // 2
            y = (self._stack.height - height) // 2
        else:
            x, y = position
        self._document.execute(AddLayerCommand(
            name="Floating Selection",
            image=clipboard.copy(),
            x=x,
            y=y,
            label="Paste",
        ))
        self._application.set_status(f"Pasted {width}x{height}")
        self.refresh()

    def _execute(self, command, status: str) -> None:
        self._before_mutation()
        self._document.execute(command)
        self._application.set_status(status)
        self.refresh()

    def _can_remove_active(self) -> bool:
        layer = self._stack.active_layer
        if layer is None:
            return False
        return (
            len(self._stack.all_layers())
            > 1 + len(layer.all_descendants())
        )

    def _apply_selection_alpha(
            self,
            patch: np.ndarray,
            bbox: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = bbox
        mask = self._stack.selection.data[y0:y1, x0:x1]
        alpha = patch[:, :, 3].astype(np.float32) * mask
        patch[:, :, 3] = np.clip(alpha, 0.0, 255.0).astype(np.uint8)
