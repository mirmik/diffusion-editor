"""Agent chat panel powered by nemor-core agent loop with tool use."""

from __future__ import annotations

import threading
from typing import Any

from tcgui.widgets.button import Button
from tcgui.widgets.hstack import HStack
from tcgui.widgets.label import Label
from tcgui.widgets.panel import Panel
from tcgui.widgets.text_area import TextArea
from tcgui.widgets.text_input import TextInput
from tcgui.widgets.units import pct, px
from tcgui.widgets.vstack import VStack

from ..app.settings import Settings

try:
    from nemor.core.session import Session
    from .runner import AgentRunner
except ImportError:  # pragma: no cover - depends on optional local install
    Session = None
    AgentRunner = None


DEFAULT_AGENT_BASE_URL = "http://localhost:8080"
DEFAULT_AGENT_MODEL = "default"

SYSTEM_PROMPT = (
    "You are a built-in agent of Diffusion Editor — a layer-based image and texture "
    "generation tool. You can inspect and manipulate the document using the available "
    "tools: list layers, add/remove layers, toggle visibility, adjust opacity, "
    "query canvas info, and view the canvas as an image. "
    "All mutations go through the undo system. Be concise."
    "P.S. The project is in a prototype and experimental state, so we are mostly not drawing, but testing tools."
)


class AgentChatPanel(Panel):
    """Bottom chat panel with nemor-powered agent loop and tool use."""

    def __init__(self, settings: Settings, tool_registry,
                 layer_stack, document_service):
        super().__init__()
        self.preferred_width = pct(100)
        self.preferred_height = px(220)
        self.padding = 8
        self._settings = settings
        self._tool_registry = tool_registry
        self._layer_stack = layer_stack
        self._document_service = document_service

        self._session = (Session("editor", system_prompt=SYSTEM_PROMPT)
                         if Session is not None else None)
        self._runner: AgentRunner | None = None
        self._assistant_buffer = ""
        self._assistant_active = False

        root = VStack()
        root.spacing = 6
        root.preferred_width = pct(100)
        root.preferred_height = pct(100)

        header = HStack()
        header.spacing = 8

        title = Label()
        title.text = "Agent Chat"
        title.font_size = 13
        header.add_child(title)

        self._status = Label()
        self._status.text = "Not connected"
        self._status.font_size = 11
        self._status.stretch = True
        header.add_child(self._status)

        self._clear_btn = Button()
        self._clear_btn.text = "Clear"
        self._clear_btn.preferred_width = px(70)
        self._clear_btn.on_click = self.clear
        header.add_child(self._clear_btn)
        root.add_child(header)

        self._transcript = TextArea()
        self._transcript.read_only = True
        self._transcript.word_wrap = True
        self._transcript.preferred_width = pct(100)
        self._transcript.stretch = True
        self._transcript.text = "Configure the Agent Chat API in Edit -> Settings."
        root.add_child(self._transcript)

        input_row = HStack()
        input_row.spacing = 6

        self._input = TextInput()
        self._input.placeholder = "Ask the agent..."
        self._input.stretch = True
        self._input.on_submit = lambda _: self.submit()
        input_row.add_child(self._input)

        self._send_btn = Button()
        self._send_btn.text = "Send"
        self._send_btn.preferred_width = px(70)
        self._send_btn.on_click = self.submit
        input_row.add_child(self._send_btn)

        self._stop_btn = Button()
        self._stop_btn.text = "Stop"
        self._stop_btn.preferred_width = px(70)
        self._stop_btn.enabled = False
        self._stop_btn.on_click = self.cancel
        input_row.add_child(self._stop_btn)
        root.add_child(input_row)

        self.add_child(root)
        if self._session is None:
            self._transcript.text = "Agent Chat is unavailable: nemor is not installed."
            self._input.enabled = False
            self._send_btn.enabled = False
            self._clear_btn.enabled = False
            self._status.text = "Unavailable"

    def layout(self, x: float, y: float, width: float, height: float,
               viewport_w: float, viewport_h: float):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        inner_x = x + self.padding
        inner_y = y + self.padding
        inner_w = max(0.0, width - self.padding * 2)
        inner_h = max(0.0, height - self.padding * 2)
        for child in self.children:
            child.layout(inner_x, inner_y, inner_w, inner_h, viewport_w, viewport_h)

    def submit(self) -> None:
        if self._session is None or AgentRunner is None:
            self._status.text = "Unavailable: nemor is not installed"
            return
        text = self._input.text.strip()
        if not text or (self._runner is not None and self._runner.is_busy):
            return

        config = self._build_config()

        self._input.text = ""
        self._input.cursor_pos = 0
        self._assistant_buffer = ""
        self._assistant_active = True
        self._sync_transcript()
        self._set_busy(True, "Waiting for agent...")

        self._runner = AgentRunner(
            self._tool_registry, self._session, config,
        )
        self._runner.submit(text)

    def cancel(self) -> None:
        if self._runner is None or not self._runner.is_busy:
            return
        self._runner.cancel()
        self._status.text = "Stopping..."

    def clear(self) -> None:
        if Session is None:
            return
        if self._runner is not None and self._runner.is_busy:
            return
        self._session = Session("editor", system_prompt=SYSTEM_PROMPT)
        self._assistant_buffer = ""
        self._assistant_active = False
        self._transcript.text = ""
        self._status.text = "Cleared"

    def poll(self) -> None:
        if self._runner is None:
            return
        for kind, value in self._runner.poll():
            if kind == "delta":
                self._assistant_buffer += value
                self._status.text = "Generating..."
                self._sync_transcript()
            elif kind == "tool":
                tool_line = (
                    f"[Tool: {value['name']}("
                    f"{value['args'][:80]})"
                )
                if value["result"]:
                    result_preview = str(value["result"])[:120]
                    tool_line += f" -> {result_preview}"
                tool_line += "]"
                self._assistant_buffer += f"\n{tool_line}\n"
                self._sync_transcript()
            elif kind == "thinking":
                pass
            elif kind == "result":
                if value and value.startswith("[Stopped"):
                    self._assistant_buffer = f"(stopped: {value})"
                    self._sync_transcript()
            elif kind == "done":
                self._finish_assistant()
            elif kind == "cancelled":
                self._status.text = "Stopped"
                self._assistant_active = False
                self._set_busy(False)
                self._sync_transcript()
            elif kind == "error":
                self._status.text = f"Error: {str(value)[:120]}"
                self._assistant_active = False
                self._set_busy(False)

    def shutdown(self) -> None:
        if self._runner is not None:
            self._runner.shutdown()

    def _finish_assistant(self) -> None:
        self._assistant_buffer = ""
        self._assistant_active = False
        self._set_busy(False, "Ready")
        self._sync_transcript()

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        self._send_btn.enabled = not busy
        self._clear_btn.enabled = not busy
        self._stop_btn.enabled = busy
        if status is not None:
            self._status.text = status

    def _sync_transcript(self) -> None:
        lines: list[str] = []
        if self._session is None:
            self._transcript.text = "Agent Chat is unavailable: nemor is not installed."
            return
        for msg in self._session.messages:
            if msg.role == "system":
                continue
            if msg.role == "user":
                lines.append(f"You: {msg.content}")
            elif msg.role == "assistant" and msg.content:
                lines.append(f"Agent: {msg.content}")
            elif msg.role == "tool":
                lines.append(f"  [tool result: {msg.content[:150]}]")
        if self._assistant_active:
            lines.append(f"Agent: {self._assistant_buffer}")
        self._transcript.text = "\n\n".join(lines)
        if self._transcript._lines:
            self._transcript.cursor_line = max(0, len(self._transcript._lines) - 1)
            if self._transcript._lines[-1]:
                self._transcript.cursor_col = len(self._transcript._lines[-1])
            if self._transcript.height > 0:
                self._transcript._ensure_cursor_visible()

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
