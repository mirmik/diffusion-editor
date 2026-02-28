from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class HistoryEntry:
    label: str
    undo_fn: Callable[[], None]
    redo_fn: Callable[[], None]
    size_bytes: int = 0


class HistoryManager:
    def __init__(self, apply_snapshot: Callable[[bytes], None], max_entries: int = 50,
                 max_memory_bytes: int = 5 * 1024 * 1024 * 1024):
        if max_memory_bytes <= 0:
            raise ValueError("max_memory_bytes must be > 0")
        self._apply_snapshot = apply_snapshot
        self._max_entries = max_entries
        self._max_memory_bytes = max_memory_bytes
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

    @property
    def max_memory_bytes(self) -> int:
        return self._max_memory_bytes

    def set_max_memory_bytes(self, max_memory_bytes: int) -> None:
        if max_memory_bytes <= 0:
            raise ValueError("max_memory_bytes must be > 0")
        self._max_memory_bytes = max_memory_bytes
        self._enforce_limits()

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
        self._redo_stack.clear()
        self._enforce_limits()

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

    def _enforce_limits(self) -> None:
        while len(self._undo_stack) > self._max_entries:
            self._undo_stack.pop(0)
        while len(self._redo_stack) > self._max_entries:
            self._redo_stack.pop(0)
        while self.memory_bytes() > self._max_memory_bytes:
            if self._undo_stack:
                self._undo_stack.pop(0)
                continue
            if self._redo_stack:
                self._redo_stack.pop(0)
                continue
            break
