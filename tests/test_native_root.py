from __future__ import annotations

import os
import subprocess
import sys
import threading

import pytest

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.native_root import NativeEditorRoot, WindowedNativeComposition
from diffusion_editor.app.presentation import ViewPorts


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
