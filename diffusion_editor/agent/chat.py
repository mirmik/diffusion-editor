"""Toolkit-neutral state and lifecycle for the built-in agent chat."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Protocol

from .config import (
    DEFAULT_AGENT_BASE_URL,
    DEFAULT_AGENT_MODEL,
    SYSTEM_PROMPT,
)

try:
    from nemor.core.session import Session as _NemorSession
except ImportError:  # pragma: no cover - optional local dependency
    _NemorSession = None

try:
    from .runner import AgentRunner as _AgentRunner
except ImportError:  # pragma: no cover - optional local dependency
    _AgentRunner = None


class AgentChatAction(str, Enum):
    CONNECT = "connect"
    SUBMIT = "submit"
    CANCEL = "cancel"
    CLEAR = "clear"


@dataclass(frozen=True)
class AgentChatIntent:
    action: AgentChatAction
    value: object = None


@dataclass(frozen=True)
class AgentChatState:
    transcript: str
    status: str
    input_text: str = ""
    available: bool = True
    busy: bool = False


class AgentChatPresentation(Protocol):
    def apply_agent_chat_state(self, state: AgentChatState) -> None: ...


RunnerFactory = Callable[..., Any]
SessionFactory = Callable[[], Any]


def _default_session_factory():
    if _NemorSession is None:
        raise RuntimeError("nemor is not installed")
    return _NemorSession("editor", system_prompt=SYSTEM_PROMPT)


def _default_runner_factory(*args, **kwargs):
    if _AgentRunner is None:
        raise RuntimeError("nemor is not installed")
    return _AgentRunner(*args, **kwargs)


class AgentChatCoordinator:
    """Own chat state while runners marshal events through the UI dispatcher."""

    def __init__(
            self,
            settings,
            tool_registry,
            layer_stack,
            document_service,
            *,
            defer: Callable[[Callable[[], None]], object],
            session_factory: SessionFactory = _default_session_factory,
            runner_factory: RunnerFactory = _default_runner_factory) -> None:
        self._settings = settings
        self._tool_registry = tool_registry
        self._layer_stack = layer_stack
        self._document_service = document_service
        self._defer = defer
        self._session_factory = session_factory
        self._runner_factory = runner_factory
        self._view: AgentChatPresentation | None = None
        self._runner = None
        self._session = None
        self._closed = False
        self._entries: list[str] = []
        self._assistant_buffer = ""
        self._assistant_active = False
        self._stopped = False

        try:
            self._session = self._session_factory()
            self._state = AgentChatState(
                transcript=(
                    "Configure the Agent Chat API in Edit -> Settings."),
                status="Ready",
            )
        except Exception:
            self._state = AgentChatState(
                transcript="Agent Chat is unavailable: nemor is not installed.",
                status="Unavailable",
                available=False,
            )

    @property
    def state(self) -> AgentChatState:
        return self._state

    def bind_view(self, view: AgentChatPresentation) -> None:
        self._require_open()
        self._view = view
        view.apply_agent_chat_state(self._state)

    def handle_intent(self, intent: AgentChatIntent) -> None:
        self._require_open()
        if intent.action == AgentChatAction.CONNECT:
            self.connect()
        elif intent.action == AgentChatAction.SUBMIT:
            self.submit(str(intent.value or ""))
        elif intent.action == AgentChatAction.CANCEL:
            self.cancel()
        elif intent.action == AgentChatAction.CLEAR:
            self.clear()

    def connect(self) -> None:
        if self._state.busy:
            return
        try:
            self._session = self._session_factory()
        except Exception as exc:
            self._publish(replace(
                self._state,
                status=f"Connection error: {str(exc)[:120]}",
                available=False,
            ))
            return
        self._shutdown_runner()
        self._entries.clear()
        self._assistant_buffer = ""
        self._assistant_active = False
        self._publish(AgentChatState(
            transcript="",
            status="Ready",
            available=True,
        ))

    def submit(self, text: str) -> None:
        text = text.strip()
        if not text or self._state.busy:
            return
        if self._session is None:
            self._publish(replace(
                self._state,
                status="Unavailable: nemor is not installed",
                available=False,
            ))
            return

        self._entries.append(f"You: {text}")
        self._assistant_buffer = ""
        self._assistant_active = True
        self._stopped = False
        self._publish(AgentChatState(
            transcript=self._render_transcript(),
            status="Waiting for agent...",
            input_text="",
            available=True,
            busy=True,
        ))
        try:
            self._shutdown_runner()
            self._runner = self._runner_factory(
                self._tool_registry,
                self._session,
                self._build_config(),
                defer=self._defer,
                on_event=self._on_runner_event,
            )
            self._runner.submit(text)
        except Exception as exc:
            self._assistant_active = False
            self._publish(replace(
                self._state,
                transcript=self._render_transcript(),
                status=f"Error: {str(exc)[:120]}",
                busy=False,
            ))

    def cancel(self) -> None:
        if self._runner is None or not self._state.busy:
            return
        self._runner.cancel()
        self._publish(replace(self._state, status="Stopping..."))

    def clear(self) -> None:
        if self._state.busy or not self._state.available:
            return
        self._shutdown_runner()
        self._session = self._session_factory()
        self._entries.clear()
        self._assistant_buffer = ""
        self._assistant_active = False
        self._publish(replace(
            self._state,
            transcript="",
            status="Cleared",
            input_text="",
        ))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._view = None
        self._shutdown_runner()

    shutdown = close

    def _on_runner_event(self, kind: str, value: Any) -> None:
        if self._closed:
            return
        if kind == "delta":
            self._assistant_buffer += str(value)
            self._publish(replace(
                self._state,
                transcript=self._render_transcript(),
                status="Generating...",
            ))
        elif kind == "tool":
            args = str(value.get("args", ""))[:80]
            line = f"[Tool: {value.get('name', '?')}({args})"
            result = value.get("result")
            if result:
                line += f" -> {str(result)[:120]}"
            self._assistant_buffer += f"\n{line}]\n"
            self._publish(replace(
                self._state,
                transcript=self._render_transcript(),
            ))
        elif kind == "result" and str(value).startswith("[Stopped"):
            self._assistant_buffer = f"(stopped: {value})"
            self._stopped = True
            self._publish(replace(
                self._state,
                transcript=self._render_transcript(),
            ))
        elif kind == "done":
            self._finish("Stopped" if self._stopped else "Ready")
        elif kind == "cancelled":
            self._finish("Stopped")
        elif kind == "error":
            self._finish(f"Error: {str(value)[:120]}")

    def _finish(self, status: str) -> None:
        if self._assistant_active and self._assistant_buffer:
            self._entries.append(f"Agent: {self._assistant_buffer}")
        self._assistant_buffer = ""
        self._assistant_active = False
        self._publish(replace(
            self._state,
            transcript=self._render_transcript(),
            status=status,
            busy=False,
        ))

    def _render_transcript(self) -> str:
        entries = list(self._entries)
        if self._assistant_active:
            entries.append(f"Agent: {self._assistant_buffer}")
        return "\n\n".join(entries)

    def _publish(self, state: AgentChatState) -> None:
        self._state = state
        if self._view is not None:
            self._view.apply_agent_chat_state(state)

    def _shutdown_runner(self) -> None:
        runner, self._runner = self._runner, None
        if runner is not None:
            runner.shutdown()

    def _build_config(self) -> dict:
        return {
            "server": str(self._settings.get(
                "agent_api_base_url", DEFAULT_AGENT_BASE_URL)),
            "model": str(self._settings.get(
                "agent_model", DEFAULT_AGENT_MODEL)),
            "auth_token": str(self._settings.get("agent_api_key", "")),
            "max_tokens": int(self._settings.get("agent_max_tokens", 4096)),
            "max_iterations": 15,
            "system_prompt": SYSTEM_PROMPT,
            "tools_enabled": True,
            "tools": {},
            "sampling": {
                "temperature": float(self._settings.get(
                    "agent_temperature", 0.7)),
            },
            "_layer_stack": self._layer_stack,
            "_document_service": self._document_service,
        }

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("agent chat coordinator is closed")
