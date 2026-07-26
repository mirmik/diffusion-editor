"""Threaded bridge between nemor's synchronous loop and a UI dispatcher."""

from __future__ import annotations

import threading
from queue import Empty, Queue
from typing import Any

from nemor.core.session import Session
from nemor.core.agent import agent_loop


_CANCELLED_MESSAGE = "Agent operation cancelled"


class _RunGeneration:
    """Cancellation token that is never reused by a later submission."""

    def __init__(self, number: int) -> None:
        self.number = number
        self.stop_event = threading.Event()


class _MainThreadCall:
    """Exactly-once state machine for one UI-thread transaction.

    Moving from ``pending`` to ``running`` is the transaction boundary. A
    pending call can be dropped without invoking its function. Once running,
    the function is allowed to finish atomically; cancellation wins over its
    return value until the call has crossed the completion boundary.
    """

    _TERMINAL_STATES = frozenset({"cancelled", "failed", "succeeded"})

    def __init__(self, func, generation: _RunGeneration) -> None:
        self.generation = generation
        self._func = func
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._state = "pending"
        self._cancel_requested = False
        self._result: Any = None
        self._error: BaseException | None = None

    def cancel(self) -> None:
        signal = False
        with self._lock:
            if self._state == "pending":
                self._func = None
                self._state = "cancelled"
                self._error = RuntimeError(_CANCELLED_MESSAGE)
                signal = True
            elif self._state == "running":
                self._cancel_requested = True
        if signal:
            self._done.set()

    def fail_before_start(self, error: BaseException) -> None:
        signal = False
        with self._lock:
            if self._state == "pending":
                self._func = None
                self._state = "failed"
                self._error = error
                signal = True
        if signal:
            self._done.set()

    def execute(self) -> None:
        func = None
        signal = False
        with self._lock:
            if self._state != "pending":
                return
            if self.generation.stop_event.is_set():
                self._func = None
                self._state = "cancelled"
                self._error = RuntimeError(_CANCELLED_MESSAGE)
                signal = True
            else:
                self._state = "running"
                func = self._func
                self._func = None
        if signal:
            self._done.set()
            return

        try:
            result = func()
        except BaseException as exc:
            self._finish(error=exc)
        else:
            self._finish(result=result)

    def result(self):
        self._done.wait()
        with self._lock:
            if self._state not in self._TERMINAL_STATES:
                raise RuntimeError("Main-thread call completed inconsistently")
            error = self._error
            result = self._result
        if error is not None:
            raise error
        return result

    def _finish(
            self,
            *,
            result: Any = None,
            error: BaseException | None = None) -> None:
        signal = False
        with self._lock:
            if self._state != "running":
                return
            if (
                    self._cancel_requested
                    or self.generation.stop_event.is_set()):
                self._state = "cancelled"
                self._error = RuntimeError(_CANCELLED_MESSAGE)
            elif error is not None:
                self._state = "failed"
                self._error = error
            else:
                self._state = "succeeded"
                self._result = result
            signal = True
        if signal:
            self._done.set()


class AgentRunner:
    """Runs nemor's agent loop and emits events through defer or a queue.

    Native hosts pass ``defer``/``on_event``. Legacy hosts keep calling
    :meth:`poll` every frame.
    """

    def __init__(
            self,
            tool_registry,
            session: Session,
            config: dict,
            *,
            defer=None,
            on_event=None):
        self._tool_registry = tool_registry
        self._session = session
        self._base_config = config
        self._events: Queue[tuple[str, Any]] = Queue()
        self._main_calls: Queue[_MainThreadCall] = Queue()
        self._thread: threading.Thread | None = None
        self._state_lock = threading.Lock()
        self._generation_number = 0
        self._active_generation: _RunGeneration | None = None
        self._active_calls: set[_MainThreadCall] = set()
        self._accepting = True
        self._defer = defer
        self._on_event = on_event

    @property
    def is_busy(self) -> bool:
        with self._state_lock:
            return self._thread is not None

    def submit(self, user_input: str) -> None:
        """Submit a user message and start the agent loop in a thread."""
        with self._state_lock:
            if not self._accepting or self._thread is not None:
                return
            self._generation_number += 1
            generation = _RunGeneration(self._generation_number)
            self._active_generation = generation
            config = {
                **self._base_config,
                "_tool_registry": self._tool_registry,
                "_run_on_main_thread": (
                    lambda func: self._run_on_main_thread(
                        func, generation)),
            }
            thread = threading.Thread(
                target=self._run,
                args=(user_input, config, generation),
                name="agent-runner",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def cancel(self) -> None:
        with self._state_lock:
            generation = self._active_generation
            if generation is None:
                return
            generation.stop_event.set()
            calls = tuple(
                call for call in self._active_calls
                if call.generation is generation)
        for call in calls:
            call.cancel()

    def poll(self) -> list[tuple[str, Any]]:
        """Drain pending events for legacy hosts without a dispatcher."""
        self._drain_main_calls()
        events: list[tuple[str, Any]] = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except Empty:
                break
        return events

    def run_on_main_thread(self, func):
        """Run func from the UI polling thread and return its result."""
        with self._state_lock:
            generation = self._active_generation
        if generation is None:
            raise RuntimeError(_CANCELLED_MESSAGE)
        return self._run_on_main_thread(func, generation)

    def _run_on_main_thread(
            self, func, generation: _RunGeneration):
        call = _MainThreadCall(func, generation)
        with self._state_lock:
            if (
                    not self._accepting
                    or self._active_generation is not generation
                    or generation.stop_event.is_set()):
                call.cancel()
            else:
                self._active_calls.add(call)

        if generation.stop_event.is_set():
            call.cancel()
        if self._defer is None:
            self._main_calls.put(call)
        else:
            try:
                self._defer(call.execute)
            except RuntimeError as exc:
                call.fail_before_start(exc)
        try:
            return call.result()
        finally:
            with self._state_lock:
                self._active_calls.discard(call)

    def _drain_main_calls(self) -> None:
        while True:
            try:
                call = self._main_calls.get_nowait()
            except Empty:
                break
            call.execute()

    def shutdown(self, timeout: float = 1.0) -> None:
        with self._state_lock:
            self._accepting = False
            generation = self._active_generation
            if generation is not None:
                generation.stop_event.set()
            calls = tuple(self._active_calls)
            thread = self._thread
        for call in calls:
            call.cancel()
        if thread is not None:
            thread.join(timeout=timeout)

    def _emit(self, kind: str, value: Any) -> None:
        if self._defer is None or self._on_event is None:
            self._events.put((kind, value))
            return
        try:
            self._defer(lambda: self._deliver(kind, value))
        except RuntimeError:
            # Host shutdown closes the dispatcher before any view is destroyed.
            return

    def _deliver(self, kind: str, value: Any) -> None:
        with self._state_lock:
            accepting = self._accepting
        if accepting:
            self._on_event(kind, value)

    def _run(
            self,
            user_input: str,
            config: dict,
            generation: _RunGeneration) -> None:
        try:
            result = agent_loop(
                self._session, config,
                user_input=user_input,
                silent=True,
                stop_event=generation.stop_event,
                on_token=lambda t: self._emit_active(
                    generation, "delta", t),
                on_update=lambda: self._emit_active(
                    generation, "update", None),
                on_tool=lambda name, args, result: self._emit_active(
                    generation,
                    "tool",
                    {"name": name, "args": args, "result": result},
                ),
                on_thinking=lambda t: self._emit_active(
                    generation, "thinking", t),
            )
            if generation.stop_event.is_set():
                self._emit("cancelled", None)
            else:
                self._emit("result", result or "")
                self._emit("done", None)
        except Exception as e:
            if generation.stop_event.is_set():
                self._emit("cancelled", None)
            else:
                self._emit("error", str(e))
        finally:
            current = threading.current_thread()
            with self._state_lock:
                if self._thread is current:
                    self._thread = None
                    if self._active_generation is generation:
                        self._active_generation = None

    def _emit_active(
            self,
            generation: _RunGeneration,
            kind: str,
            value: Any) -> None:
        if not generation.stop_event.is_set():
            self._emit(kind, value)
