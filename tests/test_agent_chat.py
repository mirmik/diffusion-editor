from __future__ import annotations

from dataclasses import replace
import importlib
import sys
import time
import types

from termin.gui_native import Rect, tc_ui_document_create, tc_ui_document_destroy

from diffusion_editor.agent.chat import (
    AgentChatAction,
    AgentChatCoordinator,
    AgentChatIntent,
    AgentChatState,
)
from diffusion_editor.app.native_agent_chat import NativeAgentChatPanel


class _Settings:
    def get(self, _key, default=None):
        return default


class _SessionFactory:
    def __init__(self):
        self.sessions = []

    def __call__(self):
        session = object()
        self.sessions.append(session)
        return session


class _FakeRunner:
    instances = []

    def __init__(
            self, registry, session, config, *, defer, on_event):
        self.registry = registry
        self.session = session
        self.config = config
        self.defer = defer
        self.on_event = on_event
        self.submitted = []
        self.cancelled = False
        self.shutdown_calls = 0
        self.instances.append(self)

    def submit(self, text):
        self.submitted.append(text)

    def cancel(self):
        self.cancelled = True

    def shutdown(self):
        self.shutdown_calls += 1

    def emit(self, kind, value=None):
        self.defer(lambda: self.on_event(kind, value))


class _View:
    def __init__(self):
        self.states = []

    def apply_agent_chat_state(self, state):
        self.states.append(state)


def _coordinator():
    pending = []
    sessions = _SessionFactory()
    _FakeRunner.instances.clear()
    coordinator = AgentChatCoordinator(
        _Settings(),
        "registry",
        "layers",
        "document",
        defer=pending.append,
        session_factory=sessions,
        runner_factory=_FakeRunner,
    )
    return coordinator, pending, sessions


def _drain(pending):
    while pending:
        pending.pop(0)()


def test_agent_chat_streams_tools_and_terminal_events_through_defer():
    coordinator, pending, _sessions = _coordinator()
    view = _View()
    coordinator.bind_view(view)

    coordinator.handle_intent(AgentChatIntent(
        AgentChatAction.SUBMIT, "inspect canvas"))
    runner = _FakeRunner.instances[-1]
    assert runner.submitted == ["inspect canvas"]
    assert coordinator.state.busy
    assert coordinator.state.input_text == ""
    assert runner.config["_layer_stack"] == "layers"

    runner.emit("delta", "Canvas ")
    runner.emit("delta", "is empty.")
    runner.emit("tool", {
        "name": "list_layers",
        "args": "{}",
        "result": "(no layers)",
    })
    assert "Canvas is empty" not in coordinator.state.transcript
    _drain(pending)
    assert "Agent: Canvas is empty." in coordinator.state.transcript
    assert "Tool: list_layers({}) -> (no layers)" in (
        coordinator.state.transcript)
    assert coordinator.state.status == "Generating..."

    runner.emit("done")
    _drain(pending)
    assert not coordinator.state.busy
    assert coordinator.state.status == "Ready"
    coordinator.close()


def test_agent_chat_cancel_clear_and_reconnect_replace_session_safely():
    coordinator, pending, sessions = _coordinator()
    coordinator.submit("long task")
    first = _FakeRunner.instances[-1]
    coordinator.cancel()
    assert first.cancelled
    assert coordinator.state.status == "Stopping..."
    first.emit("cancelled")
    _drain(pending)
    assert coordinator.state.status == "Stopped"
    assert not coordinator.state.busy

    coordinator.clear()
    assert coordinator.state.transcript == ""
    assert coordinator.state.status == "Cleared"
    cleared_session = sessions.sessions[-1]

    coordinator.submit("again")
    second = _FakeRunner.instances[-1]
    second.emit("error", "offline")
    _drain(pending)
    assert coordinator.state.status == "Error: offline"
    coordinator.connect()
    assert coordinator.state.status == "Ready"
    assert coordinator.state.transcript == ""
    assert sessions.sessions[-1] is not cleared_session
    assert second.shutdown_calls == 1
    coordinator.close()


def test_agent_chat_shutdown_ignores_already_deferred_events():
    coordinator, pending, _sessions = _coordinator()
    view = _View()
    coordinator.bind_view(view)
    coordinator.submit("work")
    runner = _FakeRunner.instances[-1]
    runner.emit("delta", "late")
    state_count = len(view.states)

    coordinator.close()
    _drain(pending)

    assert runner.shutdown_calls == 1
    assert len(view.states) == state_count


def test_unavailable_agent_chat_exposes_disabled_state():
    coordinator = AgentChatCoordinator(
        _Settings(),
        None,
        None,
        None,
        defer=lambda callback: callback(),
        session_factory=lambda: (_ for _ in ()).throw(
            RuntimeError("missing")),
        runner_factory=_FakeRunner,
    )
    assert not coordinator.state.available
    assert coordinator.state.status == "Unavailable"
    coordinator.close()


def test_native_agent_transcript_preserves_selection_and_manual_scroll():
    document = tc_ui_document_create()
    state = AgentChatState(
        transcript="\n".join(f"line {index}" for index in range(60)),
        status="Ready",
    )
    intents = []
    panel = NativeAgentChatPanel(
        document, state, intents.append, lambda: None)
    assert document.add_root(panel.widget.handle)
    try:
        document.layout_roots(Rect(0.0, 0.0, 320.0, 220.0))
        panel.transcript.scroll_y = 20.0
        appended = replace(state, transcript=state.transcript + "\nnext")
        panel.apply_agent_chat_state(appended)
        assert panel.transcript.scroll_y == 20.0

        panel.transcript.select(0, 4)
        selected = panel.transcript.selected_text
        deferred = replace(appended, transcript=appended.transcript + "\nlate")
        panel.apply_agent_chat_state(deferred)
        assert panel.transcript.selected_text == selected
        assert panel.transcript.model.text == appended.transcript

        panel.transcript.clear_selection()
        panel.poll()
        assert panel.transcript.model.text == deferred.transcript
    finally:
        panel.close()
        tc_ui_document_destroy(document)


def test_native_agent_controls_emit_typed_intents_and_follow_end():
    document = tc_ui_document_create()
    initial = AgentChatState(
        transcript="\n".join(f"line {index}" for index in range(60)),
        status="Ready",
    )
    intents = []
    panel = NativeAgentChatPanel(
        document, initial, intents.append, lambda: None)
    assert document.add_root(panel.widget.handle)
    try:
        document.layout_roots(Rect(0.0, 0.0, 320.0, 220.0))
        old_max = max(
            0.0,
            panel.transcript.content_height
            - panel.transcript.widget.bounds.height,
        )
        panel.transcript.scroll_y = old_max
        panel.apply_agent_chat_state(replace(
            initial, transcript=initial.transcript + "\nlast"))
        assert panel.transcript.scroll_y > old_max

        panel.input.text = "hello"
        panel.send_button.widget.enabled = True
        panel._submit(panel.input.text)
        assert intents[-1] == AgentChatIntent(
            AgentChatAction.SUBMIT, "hello")
        panel.apply_agent_chat_state(replace(initial, busy=True))
        assert not panel.send_button.widget.enabled
        assert panel.stop_button.widget.enabled
    finally:
        panel.close()
        tc_ui_document_destroy(document)


def test_agent_runner_uses_defer_for_events_and_main_thread_tools(monkeypatch):
    def agent_loop(
            _session, config, *, user_input, on_token, on_tool, **_kwargs):
        assert user_input == "hello"
        result = config["_run_on_main_thread"](lambda: "main result")
        on_token("stream")
        on_tool("tool", "{}", result)
        return "done"

    nemor = types.ModuleType("nemor")
    core = types.ModuleType("nemor.core")
    session = types.ModuleType("nemor.core.session")
    agent = types.ModuleType("nemor.core.agent")
    session.Session = object
    agent.agent_loop = agent_loop
    monkeypatch.setitem(sys.modules, "nemor", nemor)
    monkeypatch.setitem(sys.modules, "nemor.core", core)
    monkeypatch.setitem(sys.modules, "nemor.core.session", session)
    monkeypatch.setitem(sys.modules, "nemor.core.agent", agent)
    monkeypatch.delitem(
        sys.modules, "diffusion_editor.agent.runner", raising=False)
    runner_module = importlib.import_module("diffusion_editor.agent.runner")

    pending = []
    events = []
    runner = runner_module.AgentRunner(
        "registry",
        object(),
        {},
        defer=pending.append,
        on_event=lambda kind, value: events.append((kind, value)),
    )
    runner.submit("hello")
    deadline = time.monotonic() + 1.0
    while runner.is_busy and time.monotonic() < deadline:
        if pending:
            pending.pop(0)()
        else:
            time.sleep(0.001)
    _drain(pending)

    assert not runner.is_busy
    assert events == [
        ("delta", "stream"),
        ("tool", {
            "name": "tool",
            "args": "{}",
            "result": "main result",
        }),
        ("result", "done"),
        ("done", None),
    ]
    runner.shutdown()
