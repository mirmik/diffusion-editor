from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class HistoryEntry:
    label: str
    undo_fn: Callable[[], None]
    redo_fn: Callable[[], None]
    size_bytes: int = 0


class HistoryManager:
    def __init__(self, apply_snapshot: Callable[[bytes], None], max_entries: int = 50):
        self._apply_snapshot = apply_snapshot
        self._max_entries = max_entries
        self._undo_stack: list[HistoryEntry] = []
        self._redo_stack: list[HistoryEntry] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def clear(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()

    def push(self, label: str, before: bytes, after: bytes) -> None:
        if before == after:
            return
        self.push_callbacks(
            label=label,
            undo_fn=lambda: self._apply_snapshot(before),
            redo_fn=lambda: self._apply_snapshot(after),
            size_bytes=len(before) + len(after),
        )

    def push_callbacks(self, label: str, undo_fn: Callable[[], None],
                       redo_fn: Callable[[], None],
                       size_bytes: int = 0) -> None:
        self._undo_stack.append(HistoryEntry(
            label=label, undo_fn=undo_fn, redo_fn=redo_fn,
            size_bytes=size_bytes))
        if len(self._undo_stack) > self._max_entries:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def memory_bytes(self) -> int:
        """Estimated memory held by undo/redo entries."""
        total = 0
        for entry in self._undo_stack:
            total += entry.size_bytes
        for entry in self._redo_stack:
            total += entry.size_bytes
        return total

    def undo(self) -> str | None:
        if not self._undo_stack:
            return None
        entry = self._undo_stack.pop()
        entry.undo_fn()
        self._redo_stack.append(entry)
        return entry.label

    def redo(self) -> str | None:
        if not self._redo_stack:
            return None
        entry = self._redo_stack.pop()
        entry.redo_fn()
        self._undo_stack.append(entry)
        return entry.label
