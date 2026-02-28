"""DocumentService and command helpers for editor state mutations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .history import HistoryManager
from .layer_stack import LayerStack
from .commands import SnapshotCommand


@dataclass
class CallbackCommand:
    """Generic command backed by explicit do/undo/redo callbacks."""

    label: str
    do_fn: Callable[[], None]
    undo_fn: Callable[[], None]
    redo_fn: Callable[[], None]
    size_bytes: int = 0

    def do(self) -> None:
        self.do_fn()


class CommandBus:
    """Executes commands and registers them in history."""

    def __init__(self, history: HistoryManager):
        self._history = history

    def execute(self, command: CallbackCommand) -> None:
        command.do()
        self.push(command)

    def push(self, command: CallbackCommand) -> None:
        self._history.push_callbacks(
            label=command.label,
            undo_fn=command.undo_fn,
            redo_fn=command.redo_fn,
            size_bytes=command.size_bytes,
        )


class DocumentService:
    """Application-level façade over LayerStack/history mutations."""

    def __init__(self, layer_stack: LayerStack, history: HistoryManager,
                 apply_snapshot: Callable[[bytes], None]):
        self._layer_stack = layer_stack
        self._history = history
        self._apply_snapshot = apply_snapshot
        self._commands = CommandBus(history)

    def clear_history(self) -> None:
        self._history.clear()

    def undo(self) -> str | None:
        return self._history.undo()

    def redo(self) -> str | None:
        return self._history.redo()

    def memory_bytes(self) -> int:
        return self._history.memory_bytes()

    def set_history_memory_limit_bytes(self, max_memory_bytes: int) -> None:
        self._history.set_max_memory_bytes(max_memory_bytes)

    def execute_snapshot_action(self, label: str, action: Callable[[], None]) -> None:
        before = self._layer_stack.serialize_state()
        action()
        after = self._layer_stack.serialize_state()
        if before == after:
            return
        self._commands.push(CallbackCommand(
            label=label,
            do_fn=lambda: None,
            undo_fn=lambda: self._apply_snapshot(before),
            redo_fn=lambda: self._apply_snapshot(after),
            size_bytes=len(before) + len(after),
        ))

    def execute(self, command: SnapshotCommand) -> None:
        self.execute_snapshot_action(
            command.label,
            lambda: command.apply(self._layer_stack),
        )

    def push_callbacks(self, label: str,
                       undo_fn: Callable[[], None],
                       redo_fn: Callable[[], None],
                       size_bytes: int = 0) -> None:
        self._history.push_callbacks(
            label=label,
            undo_fn=undo_fn,
            redo_fn=redo_fn,
            size_bytes=size_bytes,
        )
