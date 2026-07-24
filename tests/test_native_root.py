from __future__ import annotations

import os
import subprocess
import sys
import threading

import numpy as np
import pytest
from termin.gui_native import (
    CursorIntent,
    DynamicTextureOwnership,
    ModifierFlag,
    Point,
    PointerEvent,
    PointerEventType,
    WindowKey,
)

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.dialogs import FileDialogKind
from diffusion_editor.app.generation_panels import GenerationPanelKind
from diffusion_editor.app.layer_tree import LayerTreeAction, LayerTreeIntent
from diffusion_editor.app.native_root import NativeEditorRoot, WindowedNativeComposition
from diffusion_editor.app.presentation import ViewPorts
from diffusion_editor.canvas.brush import BrushToolMode


class _MemorySettings:
    def get(self, _key, default=None):
        return default

    def set(self, _key, _value):
        pass


class _Engine:
    model_info = {}

    def poll_event(self):
        return None

    def shutdown(self):
        pass


def _application() -> EditorApplication:
    engine = _Engine()
    return EditorApplication(
        settings=_MemorySettings(),
        engines=EngineSet(engine, engine, engine, engine, engine),
    )


class _FakeApplication:
    def __init__(self, trace):
        self.trace = trace
        self.running = True
        self.closed = False
        self.ports = None

    def bind_view(self, ports):
        self.ports = ports
        self.trace.append("bind-view")

    def poll(self):
        self.trace.append("poll")

    def request_stop(self):
        self.running = False

    def close(self):
        self.trace.append("application-close")
        self.running = False
        self.closed = True


class _FakeComposition:
    document = object()
    should_close = False

    def __init__(self, trace):
        self.trace = trace
        self.closed = False

    def pump_events(self):
        self.trace.append("pump-events")
        return 2

    def request_repaint(self):
        self.trace.append("request-repaint")

    def set_unhandled_key_handler(self, callback):
        self.trace.append(
            "shortcut-handler-set" if callback is not None
            else "shortcut-handler-clear"
        )

    def render_frame(self):
        self.trace.append("render-frame")
        return True

    def close(self):
        self.trace.append("composition-close")
        self.closed = True


class _FakeView:
    def __init__(self, _document, _request_repaint, trace):
        self.trace = trace

    def ports(self):
        return ViewPorts()

    def dispatch_shortcut(self, _key, _modifiers):
        return False

    def close(self):
        self.trace.append("view-close")


def _fake_view_factory(trace):
    return lambda document, request_repaint, _set_title, _handlers: _FakeView(
        document,
        request_repaint,
        trace,
    )


def test_native_root_import_does_not_load_legacy_or_editor_internal_hosts():
    code = """
import sys
from diffusion_editor.app.native_root import NativeEditorRoot
assert NativeEditorRoot is not None
assert not any(name == 'tcgui' or name.startswith('tcgui.') for name in sys.modules)
assert not any(name == 'termin.editor_native' or name.startswith('termin.editor_native.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_tick_drains_dispatcher_before_controller_poll_and_render():
    trace = []
    application = _FakeApplication(trace)
    composition = _FakeComposition(trace)
    root = NativeEditorRoot(
        application,
        composition,
        view_factory=_fake_view_factory(trace),
    )
    trace.clear()

    worker = threading.Thread(
        target=lambda: root.defer(lambda: trace.append("deferred"))
    )
    worker.start()
    worker.join()

    result = root.tick()

    assert trace == [
        "pump-events",
        "deferred",
        "poll",
        "render-frame",
    ]
    assert result.dispatched == 1
    assert result.dispatch_failed == 0
    assert result.dispatch_remaining == 0
    assert result.events == 2
    assert result.rendered is True
    root.close()


def test_close_joins_application_then_discards_callbacks_before_view_teardown():
    trace = []
    application = _FakeApplication(trace)
    composition = _FakeComposition(trace)
    root = NativeEditorRoot(
        application,
        composition,
        view_factory=_fake_view_factory(trace),
    )
    root.defer(lambda: trace.append("must-not-run"))
    trace.clear()

    root.close()
    root.close()

    assert root.discarded_on_close == 1
    assert trace == [
        "application-close",
        "shortcut-handler-clear",
        "view-close",
        "composition-close",
    ]
    with pytest.raises(RuntimeError, match="closed"):
        root.defer(lambda: None)
    with pytest.raises(RuntimeError, match="closed"):
        root.tick()


def test_windowed_composition_closes_borrowed_adapter_before_owned_resources(
    monkeypatch: pytest.MonkeyPatch,
):
    trace = []

    class Adapter:
        def wait_idle(self):
            trace.append("adapter-idle")

        def close(self):
            trace.append("adapter-close")

    class Document:
        valid = True

    class Manager:
        def contains(self, handle):
            assert handle == "window"
            return True

        def destroy_window(self, handle):
            assert handle == "window"
            trace.append("window-destroy")

        def close(self):
            trace.append("manager-close")

    class Session:
        def close(self):
            trace.append("session-close")

    document = Document()
    monkeypatch.setattr(
        "diffusion_editor.app.native_root.tc_ui_document_destroy",
        lambda value: trace.append("document-destroy")
        if value is document
        else None,
    )
    composition = WindowedNativeComposition.__new__(WindowedNativeComposition)
    composition._closed = False
    composition._adapter = Adapter()
    composition._document = document
    composition._manager = Manager()
    composition._handle = "window"
    composition._session = Session()

    composition.close()
    composition.close()

    assert trace == [
        "adapter-idle",
        "adapter-close",
        "document-destroy",
        "window-destroy",
        "manager-close",
        "session-close",
    ]


def test_real_offscreen_root_binds_application_and_renders(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TERMIN_SDK_SHADER_CACHE_ROOT", str(tmp_path / "shader-cache"))
    application = _application()

    with NativeEditorRoot.create_headless(
        application,
        width=96,
        height=64,
    ) as root:
        root.defer(lambda: application.set_status("Dispatched"))
        result = root.tick()

        assert result.dispatched == 1
        assert result.rendered is True
        assert root.view.root.stable_id == "diffusion-editor.root"
        assert root.view.menu_bar.widget.stable_id == "diffusion-editor.menu-bar"
        assert root.view.toolbar.widget.stable_id == "diffusion-editor.toolbar"
        assert root.view.status_bar.text == "Dispatched"
        assert root.agent_chat is not None
        assert root.agent_chat.widget.stable_id == (
            "diffusion-editor.agent-panel")
        assert root.view.agent_chat_view is root.agent_chat
        assert root.agent_chat_coordinator.state.status == "Unavailable"
        assert root.dialogs is not None
        assert root.dialog_coordinator is not None
        assert root.view.activate_command("edit.settings")
        assert root.dialogs.settings_dialog.open
        assert root.composition.document.overlay_count == 1
        assert root.tick().rendered is True
        assert root.dialogs.settings_dialog.activate("cancel")
        assert root.composition.document.overlay_count == 0
        assert root.view.activate_command("file.open")
        file_dialog = root.dialogs._file_dialogs[
            FileDialogKind.OPEN_FILE]
        assert file_dialog.open
        assert file_dialog.activate("cancel")

        root.composition.resize(160, 96)
        resized = root.tick()
        snapshot = root.composition.document.inspect_snapshot()
        root_item = next(
            item for item in snapshot["widgets"]
            if item["stable_id"] == "diffusion-editor.root"
        )
        assert resized.rendered is True
        assert (
            root_item["bounds"].width,
            root_item["bounds"].height,
        ) == (160.0, 96.0)

    assert application.closed
    assert root.composition.closed
    assert (
        application.shutdown_trace.index("native-agent-chat")
        < application.shutdown_trace.index("native-agent-chat-view")
        < application.shutdown_trace.index("diffusion-engine")
    )
    assert (
        application.shutdown_trace.index("native-dialog-coordinator")
        < application.shutdown_trace.index("native-dialogs")
        < application.shutdown_trace.index("diffusion-engine")
    )


def test_real_offscreen_canvas_renders_and_routes_image_space_paint(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(
        "TERMIN_SDK_SHADER_CACHE_ROOT", str(tmp_path / "shader-cache"))
    application = _application()
    image = np.zeros((32, 32, 4), dtype=np.uint8)
    application.layer_stack.init_from_image(image)

    with NativeEditorRoot.create_headless(
            application,
            width=640,
            height=360) as root:
        root.tick()
        assert root.canvas is not None
        assert (
            root.canvas.image_lease.ownership
            == DynamicTextureOwnership.OWNED
        )
        assert root.canvas_controls.widget.stable_id == (
            "diffusion-editor.canvas-controls")
        assert root.generation_panels.widget.stable_id == (
            "diffusion-editor.generation-panels")
        assert root.view.generation_panels_view is root.generation_panels
        assert root.layer_panel.widget.stable_id == (
            "diffusion-editor.layer-panel-content")
        assert root.view.layer_panel_view is root.layer_panel
        assert root.layer_panel.tree.selected_node == (
            root.layer_panel._stable_to_node[
                application.layer_stack.active_layer.id
            ]
        )
        root.layer_panel.opacity.value = 0.45
        assert application.layer_stack.active_layer.opacity == pytest.approx(
            0.45)
        root.layer_panel.visible.checked = False
        assert not application.layer_stack.active_layer.visible
        root.layer_panel.visible.checked = True
        assert application.layer_stack.active_layer.visible
        active_id = application.layer_stack.active_layer.id
        root.layer_tree_coordinator.handle_intent(LayerTreeIntent(
            LayerTreeAction.ATTACH_TOOL,
            layer_id=active_id,
            value="lama",
        ))
        assert application.layer_stack.active_layer.tool.tool_type == "lama"
        assert (
            root.generation_panels_coordinator.state.active_kind
            == GenerationPanelKind.LAMA
        )
        assert root.generation_panels.lama_group.widget.visible
        root.generation_panels.mask_eraser.checked = True
        assert (
            root.canvas.controller.brush_tool_mode
            == BrushToolMode.MASK_ERASER
        )
        assert root.canvas_controls.brush.tool_checkboxes[
            BrushToolMode.MASK_ERASER
        ].checked
        root.layer_tree_coordinator.handle_intent(LayerTreeIntent(
            LayerTreeAction.DETACH_TOOL,
            layer_id=active_id,
        ))
        assert application.layer_stack.active_layer.tool is None
        assert root.generation_panels.empty_label.visible
        root.canvas_controls.selection.rect_mode.checked = True
        assert root.canvas.widget.cursor_intent == CursorIntent.Crosshair
        root.canvas_controls.brush.tool_checkboxes[
            BrushToolMode.SMUDGE
        ].checked = True
        assert root.canvas.controller.brush_tool_mode == BrushToolMode.SMUDGE
        assert root.canvas.widget.cursor_intent == CursorIntent.Default
        root.canvas_controls.brush.tool_checkboxes[
            BrushToolMode.PAINT
        ].checked = True
        root.canvas_controls.brush.size.value = 5
        assert root.canvas.controller.brush.size == 5
        root.canvas.controller.brush.set_hardness(1.0)
        root.canvas.controller.brush.set_color(255, 0, 0, 255)
        brush_size = root.canvas.controller.brush.size
        root.composition.push_key(WindowKey.RIGHT_BRACKET)
        root.tick()
        assert root.canvas.controller.brush.size == brush_size + 5
        assert root.canvas_controls.brush.size.value == brush_size + 5
        widget_point = root.canvas.canvas.image_to_widget(Point(8, 8))
        zoom_before = root.canvas.canvas.zoom

        event = PointerEvent()
        event.x = widget_point.x
        event.y = widget_point.y
        event.type = PointerEventType.Wheel
        event.wheel_y = 1.0
        root.composition.document.dispatch_pointer_event(event)
        assert root.canvas.canvas.zoom > zoom_before
        assert not root.canvas.canvas.fit_mode

        anchor_before = root.canvas.canvas.image_to_widget(Point(0, 0))
        event.type = PointerEventType.Down
        event.button = 2
        root.composition.document.dispatch_pointer_event(event)
        event.type = PointerEventType.Move
        event.x = widget_point.x + 11
        event.y = widget_point.y + 7
        root.composition.document.dispatch_pointer_event(event)
        event.type = PointerEventType.Up
        root.composition.document.dispatch_pointer_event(event)
        anchor_after = root.canvas.canvas.image_to_widget(Point(0, 0))
        assert anchor_after.x == pytest.approx(anchor_before.x + 11)
        assert anchor_after.y == pytest.approx(anchor_before.y + 7)

        root.canvas.fit_in_view()
        widget_point = root.canvas.canvas.image_to_widget(Point(8, 8))

        event.button = root.canvas.controller.LEFT_BUTTON
        event.x = widget_point.x
        event.y = widget_point.y
        event.type = PointerEventType.Down
        root.composition.document.dispatch_pointer_event(event)
        event.type = PointerEventType.Up
        root.composition.document.dispatch_pointer_event(event)
        root.composition.resize(800, 480)
        root.tick()

        assert application.layer_stack.active_layer.image[8, 8, 3] > 0
        assert root.canvas.canvas.fit_mode
        assert root.composition.latest_frame_size == [800, 480]

    assert (
        root.canvas.image_lease.ownership
        == DynamicTextureOwnership.RELEASED
    )
    assert (
        root.canvas.overlay_lease.ownership
        == DynamicTextureOwnership.RELEASED
    )


def test_offscreen_routes_unhandled_shortcut_after_focused_widget(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TERMIN_SDK_SHADER_CACHE_ROOT", str(tmp_path / "shader-cache"))
    application = _application()
    activated = []

    with NativeEditorRoot.create_headless(
        application,
        width=320,
        height=200,
        command_handlers={
            "edit.redo": lambda: activated.append("redo"),
        },
    ) as root:
        text_input = root.composition.document.create_text_input("focus owner")
        root.view.canvas_host.add_stretch_child(text_input.widget)
        assert root.composition.document.set_focus(text_input.handle)
        root.tick()

        application.set_command_state("edit.redo", enabled=False)
        root.composition.push_key(
            WindowKey.Y,
            modifiers=int(ModifierFlag.Ctrl),
        )
        root.tick()
        assert activated == []

        application.set_command_state("edit.redo", enabled=True)
        root.composition.push_key(
            WindowKey.Y,
            modifiers=int(ModifierFlag.Ctrl),
        )
        result = root.tick()

        assert result.events == 1
        assert activated == ["redo"]
