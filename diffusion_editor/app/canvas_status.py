"""Toolkit-neutral live status projection for the editor Canvas."""

from __future__ import annotations

import time
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
            set_status: Callable[[str], None],
            *,
            get_status: Callable[[], str] | None = None,
            clock: Callable[[], float] = time.monotonic,
            update_interval: float = 1.0 / 25.0) -> None:
        if update_interval < 0:
            raise ValueError("update_interval must be non-negative")
        self._layer_stack = layer_stack
        self._document = document
        self._canvas = canvas
        self._set_status = set_status
        self._get_status = get_status
        self._clock = clock
        self._update_interval = update_interval
        self._closed = False
        self._pending_position: tuple[int, int] | None = None
        self._last_publish_at = float("-inf")
        self._last_published: str | None = None
        self._history_revision: int | None = None
        self._cache_revision: int | None = None
        self._history_text = ""
        self._cache_text = ""
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
        memory = self._memory_status()
        if 0 <= x < width and 0 <= y < height:
            position = f" | ({x},{y})"
        else:
            position = ""
        return (
            f"{width}x{height}{position} | {layer_name} | "
            f"{tool}:{brush_size}px | {memory}"
        )

    def flush(self, *, force: bool = False) -> bool:
        """Publish the latest pointer position when its throttle is due."""
        if self._closed or self._pending_position is None:
            return False
        now = self._clock()
        if (
                not force
                and now - self._last_publish_at < self._update_interval):
            return False
        x, y = self._pending_position
        self._pending_position = None
        self._last_publish_at = now
        text = self.status_text(x, y)
        current = (
            self._get_status()
            if self._get_status is not None
            else self._last_published
        )
        self._last_published = text
        if text == current:
            return False
        self._set_status(text)
        return True

    def _memory_status(self) -> str:
        history_revision = self._document.memory_revision
        if history_revision != self._history_revision:
            self._history_revision = history_revision
            self._history_text = format_bytes(self._document.memory_bytes())
        cache_revision = self._layer_stack.cache_revision
        if cache_revision != self._cache_revision:
            self._cache_revision = cache_revision
            self._cache_text = format_bytes(
                self._layer_stack.cache_memory_bytes())
        return f"Hist:{self._history_text} Cache:{self._cache_text}"

    def _on_mouse_moved(self, x: int, y: int) -> None:
        if self._closed:
            return
        if self._previous_mouse_moved is not None:
            self._previous_mouse_moved(x, y)
        self._pending_position = (x, y)
        self.flush()
