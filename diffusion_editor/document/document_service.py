"""DocumentService and command helpers for editor state mutations."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import Callable

from .history import HistoryManager
from .layer_stack import LayerStack
from .commands import SnapshotCommand


def _zip_snapshots_equal(before: bytes, after: bytes) -> bool | None:
    """Compare undo ZIPs by member payload, ignoring container metadata."""
    try:
        with (
            zipfile.ZipFile(io.BytesIO(before), "r") as before_archive,
            zipfile.ZipFile(io.BytesIO(after), "r") as after_archive,
        ):
            def member_key(info: zipfile.ZipInfo) -> tuple:
                return (
                    info.filename,
                    info.is_dir(),
                    info.file_size,
                    info.CRC,
                )

            before_members = sorted(
                before_archive.infolist(), key=member_key)
            after_members = sorted(
                after_archive.infolist(), key=member_key)
            if (
                [member_key(info) for info in before_members]
                != [member_key(info) for info in after_members]
            ):
                # ZIP central-directory CRC/size makes changed snapshots cheap:
                # payloads are read only for a potential semantic no-op.
                return False
            return all(
                before_archive.read(before_info)
                == after_archive.read(after_info)
                for before_info, after_info
                in zip(before_members, after_members)
            )
    except (zipfile.BadZipFile, EOFError, OSError, RuntimeError):
        return None


def _snapshots_semantically_equal(before: bytes, after: bytes) -> bool:
    if before == after:
        return True
    equal = _zip_snapshots_equal(before, after)
    return equal is True


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
        self._before_mutation_listeners: list[Callable[[], None]] = []

    def add_before_mutation_listener(
            self,
            listener: Callable[[], None]) -> Callable[[], None]:
        """Register a barrier that completes or cancels external live edits."""
        self._before_mutation_listeners.append(listener)
        removed = False

        def remove() -> None:
            nonlocal removed
            if removed:
                return
            removed = True
            try:
                self._before_mutation_listeners.remove(listener)
            except ValueError:
                pass

        return remove

    def prepare_mutation(self) -> None:
        """Leave every transient edit before mutating persistent state."""
        first_error: BaseException | None = None
        first_traceback = None
        for listener in tuple(self._before_mutation_listeners):
            try:
                listener()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                    first_traceback = exc.__traceback__
        if first_error is not None:
            raise first_error.with_traceback(first_traceback)

    def clear_history(self) -> None:
        self._history.clear()

    def undo(self) -> str | None:
        self.prepare_mutation()
        return self._history.undo()

    def redo(self) -> str | None:
        self.prepare_mutation()
        return self._history.redo()

    def memory_bytes(self) -> int:
        return self._history.memory_bytes()

    @property
    def memory_revision(self) -> int:
        return self._history.memory_revision

    def set_history_memory_limit_bytes(self, max_memory_bytes: int) -> None:
        self._history.set_max_memory_bytes(max_memory_bytes)

    def execute_snapshot_action(self, label: str, action: Callable[[], None]) -> None:
        self.prepare_mutation()
        before = self._layer_stack.serialize_state()
        try:
            action()
            after = self._layer_stack.serialize_state()
        except BaseException:
            # Commands may have changed pixels or structure before failing.
            # Restore the exact pre-command snapshot and never add history.
            self._apply_snapshot(before)
            raise
        if _snapshots_semantically_equal(before, after):
            return
        self._commands.push(CallbackCommand(
            label=label,
            do_fn=lambda: None,
            undo_fn=lambda: self._apply_snapshot_transactionally(before),
            redo_fn=lambda: self._apply_snapshot_transactionally(after),
            size_bytes=len(before) + len(after),
        ))

    def _apply_snapshot_transactionally(self, target: bytes) -> None:
        """Apply a history snapshot without leaving a partial restore behind."""
        current = self._layer_stack.serialize_state()
        try:
            self._apply_snapshot(target)
        except BaseException:
            self._apply_snapshot(current)
            raise

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
