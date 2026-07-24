"""termin-gui-native projection of the built-in agent chat."""

from __future__ import annotations

from typing import Callable

from termin.gui_native import TcDocument

from ..agent.chat import (
    AgentChatAction,
    AgentChatIntent,
    AgentChatState,
)


class NativeAgentChatPanel:
    """Selectable read-only transcript plus native chat controls."""

    def __init__(
            self,
            document: TcDocument,
            state: AgentChatState,
            on_intent: Callable[[AgentChatIntent], None],
            request_repaint: Callable[[], None]) -> None:
        self._on_intent = on_intent
        self._request_repaint = request_repaint
        self._closed = False
        self._syncing = False
        self._pending_state: AgentChatState | None = None
        self._connections: list[object] = []

        self.widget = document.create_vstack("NativeAgentChatPanel")
        self.widget.stable_id = "diffusion-editor.agent-panel"
        self.widget.set_layout_spacing(4.0)

        header = document.create_hstack("NativeAgentChatHeader")
        header.set_layout_spacing(4.0)
        title = document.create_label("Agent Chat", "NativeAgentChatTitle")
        title.stable_id = "diffusion-editor.agent.title"
        header.add_preferred_child(title)
        self.status = document.create_label("", "NativeAgentChatStatus")
        self.status.stable_id = "diffusion-editor.agent.status"
        header.add_flex_child(self.status, 1.0)
        self.connect_button = self._button(
            document, "Connect", "connect",
            lambda: self._emit(AgentChatAction.CONNECT))
        self.clear_button = self._button(
            document, "Clear", "clear",
            lambda: self._emit(AgentChatAction.CLEAR))
        header.add_fixed_child(self.connect_button.widget, 72.0)
        header.add_fixed_child(self.clear_button.widget, 60.0)
        self.widget.add_preferred_child(header)

        self.transcript = document.create_rich_text_view()
        self.transcript.widget.stable_id = (
            "diffusion-editor.agent.transcript")
        self.transcript.word_wrap = True
        self.transcript.show_scrollbar = True
        self.widget.add_flex_child(self.transcript.widget, 1.0)

        input_row = document.create_hstack("NativeAgentChatInputRow")
        input_row.set_layout_spacing(4.0)
        self.input = document.create_text_input("")
        self.input.widget.stable_id = "diffusion-editor.agent.input"
        self.input.placeholder = "Ask the agent..."
        self._connections.append(self.input.connect_submitted(
            lambda text, *_rest: self._submit(text)))
        self.send_button = self._button(
            document, "Send", "send",
            lambda: self._submit(self.input.text))
        self.stop_button = self._button(
            document, "Stop", "stop",
            lambda: self._emit(AgentChatAction.CANCEL))
        input_row.add_flex_child(self.input.widget, 1.0)
        input_row.add_fixed_child(self.send_button.widget, 58.0)
        input_row.add_fixed_child(self.stop_button.widget, 58.0)
        self.widget.add_preferred_child(input_row)

        self.apply_agent_chat_state(state)

    def apply_agent_chat_state(self, state: AgentChatState) -> None:
        if self._closed:
            return
        transcript_changed = state.transcript != self.transcript.model.text
        if transcript_changed and self.transcript.has_selection:
            self._pending_state = state
            self._apply_controls(state)
            self._request_repaint()
            return
        self._pending_state = None
        viewport_height = self.transcript.widget.bounds.height
        saved_scroll_y = self.transcript.scroll_y
        old_max_scroll = max(
            0.0, self.transcript.content_height - viewport_height)
        follow_end = (
            not self.transcript.has_selection
            and saved_scroll_y >= old_max_scroll - 2.0
        )
        self._syncing = True
        try:
            self._apply_controls(state)
            self.transcript.model.set_text(state.transcript)
            if follow_end:
                self.transcript.scroll_y = 1.0e9
            else:
                self.transcript.scroll_y = saved_scroll_y
        finally:
            self._syncing = False
        self._request_repaint()

    def poll(self) -> None:
        if (
                not self._closed
                and self._pending_state is not None
                and not self.transcript.has_selection):
            self.apply_agent_chat_state(self._pending_state)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._on_intent = lambda _intent: None
        self._connections.clear()

    def _button(self, document, text, suffix, callback):
        button = document.create_button(text)
        button.widget.stable_id = f"diffusion-editor.agent.{suffix}"
        self._connections.append(button.connect_clicked(callback))
        return button

    def _apply_controls(self, state: AgentChatState) -> None:
        self.status.text = state.status
        if state.input_text != self.input.text:
            self.input.text = state.input_text
            self.input.caret = len(state.input_text)
        self.input.widget.enabled = state.available and not state.busy
        self.send_button.widget.enabled = state.available and not state.busy
        self.connect_button.widget.enabled = not state.busy
        self.clear_button.widget.enabled = (
            state.available and not state.busy)
        self.stop_button.widget.enabled = state.busy

    def _submit(self, text: str) -> None:
        if not self._syncing and not self._closed:
            self._emit(AgentChatAction.SUBMIT, text)

    def _emit(self, action: AgentChatAction, value=None) -> None:
        if not self._syncing and not self._closed:
            self._on_intent(AgentChatIntent(action, value))
