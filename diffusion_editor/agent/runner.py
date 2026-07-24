"""Threaded bridge between nemor's synchronous loop and a UI dispatcher."""

from __future__ import annotations

import threading
from queue import Empty, Queue
from typing import Any

from nemor.core.session import Session
from nemor.core.agent import agent_loop


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
        self._main_calls: Queue[tuple[Any, threading.Event, dict[str, Any]]] = Queue()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
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
            self._stop_event.clear()
            config = {
                **self._base_config,
                "_tool_registry": self._tool_registry,
                "_run_on_main_thread": self.run_on_main_thread,
            }
            thread = threading.Thread(
                target=self._run,
                args=(user_input, config),
                name="agent-runner",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def cancel(self) -> None:
        self._stop_event.set()

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
        done = threading.Event()
        box: dict[str, Any] = {}
        if self._defer is None:
            self._main_calls.put((func, done, box))
        else:
            try:
                self._defer(lambda: self._run_deferred_call(func, done, box))
            except RuntimeError as exc:
                box["error"] = exc
                done.set()
        while not done.wait(0.02):
            if self._stop_event.is_set():
                raise RuntimeError("Agent operation cancelled")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def _drain_main_calls(self) -> None:
        while True:
            try:
                func, done, box = self._main_calls.get_nowait()
            except Empty:
                break
            try:
                box["result"] = func()
            except Exception as exc:
                box["error"] = exc
            finally:
                done.set()

    def shutdown(self, timeout: float = 1.0) -> None:
        with self._state_lock:
            self._accepting = False
            self._stop_event.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)

    def _run_deferred_call(self, func, done, box) -> None:
        with self._state_lock:
            accepting = self._accepting
        if not accepting or self._stop_event.is_set():
            box["error"] = RuntimeError("Agent operation cancelled")
            done.set()
            return
        try:
            box["result"] = func()
        except Exception as exc:
            box["error"] = exc
        finally:
            done.set()

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

    def _run(self, user_input: str, config: dict) -> None:
        try:
            result = agent_loop(
                self._session, config,
                user_input=user_input,
                silent=True,
                stop_event=self._stop_event,
                on_token=lambda t: self._emit("delta", t),
                on_update=lambda: self._emit("update", None),
                on_tool=lambda name, args, result: self._emit(
                    "tool", {"name": name, "args": args, "result": result}),
                on_thinking=lambda t: self._emit("thinking", t),
            )
            self._emit("result", result or "")
            self._emit("done", None)
        except Exception as e:
            self._emit("error", str(e))
        finally:
            current = threading.current_thread()
            with self._state_lock:
                if self._thread is current:
                    self._thread = None
