"""Single-worker lifecycle primitives for background editor engines."""

from __future__ import annotations

from collections.abc import Callable
from queue import Empty, Queue
import threading
from typing import Generic, Literal, TypeVar
import uuid

from ..generation.types import EnginePollEvent


EventT = TypeVar("EventT")
TaskType = Literal[
    "load", "load_ip_adapter", "inference", "segmentation", "depth",
    "reconstruction",
]


class TaskCancelled(RuntimeError):
    """Raised by a cooperative operation after its cancellation token is set."""


class SingleWorkerEventQueue(Generic[EventT]):
    """Atomically owns one worker thread and its immutable outbound events."""

    def __init__(self) -> None:
        self._events: Queue[tuple[EventT, bool]] = Queue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._terminal_pending = False
        self._cancel = threading.Event()
        self._accepting = True

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return (
                self._thread is not None
                or (self._accepting and self._terminal_pending)
            )

    @property
    def cancellation_requested(self) -> bool:
        return self._cancel.is_set()

    def submit(
        self,
        operation: Callable[[threading.Event], EventT | None],
        *,
        name: str,
        discard_pending: bool = False,
    ) -> bool:
        with self._lock:
            if (
                not self._accepting
                or self._thread is not None
                or self._terminal_pending
            ):
                return False
            if discard_pending:
                while True:
                    try:
                        self._events.get_nowait()
                    except Empty:
                        break
            self._cancel.clear()
            thread = threading.Thread(
                target=self._run,
                args=(operation,),
                name=name,
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return True

    def _run(
        self,
        operation: Callable[[threading.Event], EventT | None],
    ) -> None:
        terminal_event: EventT | None = None
        try:
            terminal_event = operation(self._cancel)
        finally:
            current = threading.current_thread()
            with self._lock:
                # Enqueue the terminal event before publishing the idle state.
                # Otherwise a back-to-back submit can finish and enqueue its
                # terminal event in the window between these two transitions.
                if terminal_event is not None:
                    self._events.put((terminal_event, True))
                    self._terminal_pending = True
                if self._thread is current:
                    self._thread = None

    def emit(self, event: EventT) -> None:
        self._events.put((event, False))

    def poll_event(self) -> EventT | None:
        # Serialize terminal publication with polling so a controller never
        # observes that event until the worker slot has also been released.
        with self._lock:
            try:
                event, is_terminal = self._events.get_nowait()
            except Empty:
                return None
            if is_terminal:
                self._terminal_pending = False
            return event

    def drain_events(self) -> tuple[EventT, ...]:
        events: list[EventT] = []
        while (event := self.poll_event()) is not None:
            events.append(event)
        return tuple(events)

    def cancel(self) -> bool:
        with self._lock:
            if self._thread is None:
                return False
            self._cancel.set()
            return True

    def shutdown(self, timeout: float = 1.0) -> bool:
        with self._lock:
            self._accepting = False
            self._cancel.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        with self._lock:
            return self._thread is None


class EngineTaskQueue:
    """Single-worker queue that publishes standard generation engine events."""

    def __init__(self) -> None:
        self._worker: SingleWorkerEventQueue[EnginePollEvent] = (
            SingleWorkerEventQueue()
        )

    @property
    def is_busy(self) -> bool:
        return self._worker.is_busy

    def submit(
        self,
        task_type: TaskType,
        operation: Callable[[threading.Event], object | None],
        *,
        meta: object | None = None,
        job_id: str | None = None,
        name: str,
        on_error: Callable[[Exception], None] | None = None,
    ) -> bool:
        terminal_job_id = job_id or f"engine_job_{uuid.uuid4().hex}"

        def run(cancel: threading.Event) -> EnginePollEvent:
            result: object | None = None
            error: str | None = None
            try:
                result = operation(cancel)
                if cancel.is_set():
                    raise TaskCancelled("Operation cancelled")
            except Exception as exc:
                result = None
                if on_error is not None:
                    try:
                        on_error(exc)
                    except Exception:
                        pass
                error = str(exc)
            return EnginePollEvent(
                task_type=task_type,
                result=result,
                error=error,
                meta=meta,
                job_id=terminal_job_id,
            )

        return self._worker.submit(run, name=name)

    def poll_event(self) -> EnginePollEvent | None:
        return self._worker.poll_event()

    def emit(self, event: EnginePollEvent) -> None:
        """Publish a non-terminal event from the active worker operation."""
        self._worker.emit(event)

    def cancel(self) -> bool:
        return self._worker.cancel()

    def shutdown(self, timeout: float = 1.0) -> bool:
        return self._worker.shutdown(timeout)
