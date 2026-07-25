"""Toolkit-neutral live status projection for the editor Canvas."""

from __future__ import annotations

from typing import Callable

from ..canvas.editor_canvas_controller import EditorCanvasController
from ..document.document_service import DocumentService
from ..document.layer_stack import LayerStack


def format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value}B"
    if value < 1024 * 1024:
        return f"{value / 1024:.0f}K"
    return f"{value / (1024 * 1024):.1f}M"


class CanvasStatusCoordinator:
    """Publishes live Canvas context after pointer motion.

    Application operation and error messages remain visible until the pointer
    moves over the Canvas again. The next motion replaces them with current
    document, layer, tool and memory state.
    """

    def __init__(
            self,
            layer_stack: LayerStack,
            document: DocumentService,
            canvas: EditorCanvasController,
            set_status: Callable[[str], None]) -> None:
        self._layer_stack = layer_stack
        self._document = document
        self._canvas = canvas
        self._set_status = set_status
        self._closed = False
        self._previous_mouse_moved = canvas.on_mouse_moved
        self._mouse_moved_callback = self._on_mouse_moved
        canvas.on_mouse_moved = self._mouse_moved_callback

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._canvas.on_mouse_moved is self._mouse_moved_callback:
            self._canvas.on_mouse_moved = self._previous_mouse_moved

    def status_text(self, x: int, y: int) -> str:
        width, height = self._layer_stack.width, self._layer_stack.height
        layer = self._layer_stack.active_layer
        layer_name = layer.name if layer is not None else "-"
        tool = self._canvas.brush_tool_mode.value
        brush_size = self._canvas.brush.size
        memory = (
            f"Hist:{format_bytes(self._document.memory_bytes())} "
            f"Cache:{format_bytes(self._layer_stack.cache_memory_bytes())}"
        )
        if 0 <= x < width and 0 <= y < height:
            position = f" | ({x},{y})"
        else:
            position = ""
        return (
            f"{width}x{height}{position} | {layer_name} | "
            f"{tool}:{brush_size}px | {memory}"
        )

    def _on_mouse_moved(self, x: int, y: int) -> None:
        if self._closed:
            return
        if self._previous_mouse_moved is not None:
            self._previous_mouse_moved(x, y)
        self._set_status(self.status_text(x, y))
