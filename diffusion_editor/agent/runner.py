"""Threaded bridge between nemor's synchronous agent loop and SDL2 event loop."""

from __future__ import annotations

import threading
from queue import Empty, Queue
from typing import Any

from nemor.core.session import Session
from nemor.core.agent import agent_loop


class AgentRunner:
    """Runs nemor's agent_loop in a daemon thread, emits events to a queue.

    The main loop calls poll() every frame to drain events.
    """

    def __init__(self, tool_registry, session: Session, config: dict):
        self._tool_registry = tool_registry
        self._session = session
        self._base_config = config
        self._events: Queue[tuple[str, Any]] = Queue()
        self._main_calls: Queue[tuple[Any, threading.Event, dict[str, Any]]] = Queue()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def is_busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, user_input: str) -> None:
        """Submit a user message and start the agent loop in a thread."""
        if self.is_busy:
            return

        self._stop_event.clear()
        config = {
            **self._base_config,
            "_tool_registry": self._tool_registry,
            "_run_on_main_thread": self.run_on_main_thread,
        }

        self._thread = threading.Thread(
            target=self._run,
            args=(user_input, config),
            daemon=True,
        )
        self._thread.start()

    def cancel(self) -> None:
        self._stop_event.set()

    def poll(self) -> list[tuple[str, Any]]:
        """Drain all pending events. Call from main loop each frame."""
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
        self._main_calls.put((func, done, box))
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
        self.cancel()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self, user_input: str, config: dict) -> None:
        try:
            result = agent_loop(
                self._session, config,
                user_input=user_input,
                silent=True,
                stop_event=self._stop_event,
                on_token=lambda t: self._events.put(("delta", t)),
                on_update=lambda: self._events.put(("update", None)),
                on_tool=lambda name, args, result: self._events.put(
                    ("tool", {"name": name, "args": args, "result": result})
                ),
                on_thinking=lambda t: self._events.put(("thinking", t)),
            )
            self._events.put(("result", result or ""))
            self._events.put(("done", None))
        except Exception as e:
            self._events.put(("error", str(e)))
