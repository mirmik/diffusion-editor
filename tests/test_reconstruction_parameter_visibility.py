from __future__ import annotations

from diffusion_editor.app.native_shell import NativeEditorView
from diffusion_editor.generation.types import (
    RECONSTRUCTION_BACKEND_PARAMETER_KEYS,
    ReconstructionBackend,
    ReconstructionParameters,
)


class _Widget:
    def __init__(self) -> None:
        self.visible = True
        self.enabled = True


class _Control:
    def __init__(self) -> None:
        self.widget = _Widget()
        self.value = 0.0
        self.selected_index = -1
        self.checked = False


def _view() -> NativeEditorView:
    view = NativeEditorView.__new__(NativeEditorView)
    keys = set().union(*RECONSTRUCTION_BACKEND_PARAMETER_KEYS.values())
    view.reconstruction_parameter_controls = {
        key: _Control() for key in keys
    }
    view.reconstruction_parameter_widgets = {
        key: _Widget() for key in keys
    }
    view._syncing_reconstruction_parameters = False
    view._request_repaint = lambda: None
    return view


def test_each_backend_shows_only_its_consumed_parameters() -> None:
    view = _view()
    for backend in ReconstructionBackend:
        view.update_reconstruction_parameters(
            ReconstructionParameters(backend=backend)
        )
        expected = RECONSTRUCTION_BACKEND_PARAMETER_KEYS[backend]
        assert {
            key for key, widget
            in view.reconstruction_parameter_widgets.items()
            if widget.visible
        } == expected
        assert {
            key for key, control
            in view.reconstruction_parameter_controls.items()
            if control.widget.enabled
        } == expected


def test_busy_keeps_applicable_rows_visible_but_disables_controls() -> None:
    view = _view()
    parameters = ReconstructionParameters(
        backend=ReconstructionBackend.SAM3D_OBJECTS
    )
    view.update_reconstruction_parameters(parameters, busy=True)
    assert {
        key for key, widget in view.reconstruction_parameter_widgets.items()
        if widget.visible
    } == RECONSTRUCTION_BACKEND_PARAMETER_KEYS[parameters.backend]
    assert all(
        not control.widget.enabled
        for control in view.reconstruction_parameter_controls.values()
    )
