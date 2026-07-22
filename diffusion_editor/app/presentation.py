"""Toolkit-neutral presentation contracts for the editor application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from ..document.layer import Layer


class StatusPresentation(Protocol):
    """Receives human-readable application status changes."""

    def set_status(self, text: str) -> None: ...


class CommandPresentation(Protocol):
    """Projects application command availability into a view toolkit."""

    def set_command_state(
            self, command_id: str, *, enabled: bool, checked: bool = False) -> None: ...


@dataclass(frozen=True)
class PanelUpdate:
    panel_id: str
    state: str
    payload: Mapping[str, Any] = field(default_factory=dict)


class PanelPresentation(Protocol):
    """Receives immutable panel state transitions from controllers."""

    def update_panel(self, update: PanelUpdate) -> None: ...


@dataclass(frozen=True)
class DialogRequest:
    dialog_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)


class DialogPresentation(Protocol):
    """Opens an application dialog without exposing toolkit widget types."""

    def show_dialog(self, request: DialogRequest) -> None: ...


class CanvasPresentation(Protocol):
    """Narrow application-facing surface of either Canvas implementation."""

    def fit_in_view(self) -> None: ...

    def get_composite_below(self, layer: Layer) -> np.ndarray | None: ...


@dataclass(frozen=True)
class ViewPorts:
    """Explicit collection of view projections bound to an application owner."""

    status: StatusPresentation | None = None
    commands: CommandPresentation | None = None
    panels: PanelPresentation | None = None
    dialogs: DialogPresentation | None = None
    canvas: CanvasPresentation | None = None


class HeadlessEditorPresentation:
    """Small bindable projection used by tests and future headless native hosts."""

    def __init__(self) -> None:
        self.status = "Ready"
        self.command_states: dict[str, tuple[bool, bool]] = {}
        self.panel_updates: list[PanelUpdate] = []
        self.dialog_requests: list[DialogRequest] = []

    def set_status(self, text: str) -> None:
        self.status = text

    def set_command_state(
            self, command_id: str, *, enabled: bool, checked: bool = False) -> None:
        self.command_states[command_id] = (enabled, checked)

    def update_panel(self, update: PanelUpdate) -> None:
        self.panel_updates.append(update)

    def show_dialog(self, request: DialogRequest) -> None:
        self.dialog_requests.append(request)

    def ports(self) -> ViewPorts:
        return ViewPorts(
            status=self,
            commands=self,
            panels=self,
            dialogs=self,
        )
