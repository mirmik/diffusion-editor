from __future__ import annotations

import os
import subprocess
import sys
import tempfile
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
)
from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app import native_root as native_root_module
from diffusion_editor.app.dialogs import FileDialogKind
from diffusion_editor.app.generation_panels import GenerationPanelKind
from diffusion_editor.app.layer_tree import LayerTreeAction, LayerTreeIntent
from diffusion_editor.app.native_root import (
    NativeEditorRoot,
    WindowedNativeComposition,
    _is_composite_shape_checkpoint,
)
from diffusion_editor.app.native_shell import COMMAND_SPECS
from diffusion_editor.app.presentation import ViewPorts
from diffusion_editor.canvas.brush import BrushToolMode
from diffusion_editor.document.reconstruction import ReconstructionLayer
from diffusion_editor.generation.types import (
    ReconstructionStage,
    ReconstructionStageArtifact,
)


def test_composite_shape_checkpoint_is_not_a_standard_refine_source(tmp_path):
    standard = tmp_path / "standard.npz"
    composite = tmp_path / "composite.npz"
    np.savez_compressed(standard, coords=np.zeros((1, 4), dtype=np.int32))
    np.savez_compressed(
        composite,
        coords=np.zeros((1, 4), dtype=np.int32),
        composite_kind=np.asarray("enlarged_hr_geometry_v1"),
    )

    assert _is_composite_shape_checkpoint(standard) is False
    assert _is_composite_shape_checkpoint(composite) is True


class _MemorySettings:
    def __init__(self):
        self._recovery_dir = tempfile.TemporaryDirectory(
            prefix="diffusion-editor-native-root-test-"
        )

    def get(self, key, default=None):
        if key == "recovery_dir":
            return self._recovery_dir.name
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


def test_windowed_composition_creates_texture_lease_for_its_adapter(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = object()
    lease = object()
    composition = WindowedNativeComposition.__new__(WindowedNativeComposition)
    composition._adapter = adapter
    monkeypatch.setattr(
        native_root_module,
        "dynamic_texture_lease",
        lambda value: lease if value is adapter else None,
    )

    assert composition.create_texture_lease() is lease


def test_windowed_composition_renders_only_when_repaint_is_requested():
    class Adapter:
        def __init__(self):
            self.repaint_requested = False
            self.rendered = 0

        def render_frame(self):
            self.rendered += 1
            self.repaint_requested = False
            return True

    adapter = Adapter()
    composition = WindowedNativeComposition.__new__(WindowedNativeComposition)
    composition._adapter = adapter

    assert composition.render_frame() is False
    assert adapter.rendered == 0

    adapter.repaint_requested = True
    assert composition.render_frame() is True
    assert adapter.rendered == 1
    assert composition.render_frame() is False
    assert adapter.rendered == 1


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
        assert not root.view.agent_panel_visible
        assert not root.view.right_splitter.widget.visible
        assert root.view.activate_command("view.agent_panel")
        assert root.view.agent_panel_visible
        assert root.view.right_splitter.widget.visible
        assert root.view.activate_command("view.agent_panel")
        assert not root.view.agent_panel_visible
        assert not root.view.right_splitter.widget.visible
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
        assert {
            spec.action_id for spec in COMMAND_SPECS
        } <= set(root.view._command_handlers)
        assert root.canvas is not None
        assert (
            root.canvas.image_lease.ownership
            == DynamicTextureOwnership.OWNED
        )
        overlay = np.zeros((32, 32, 4), dtype=np.uint8)
        root.canvas._set_overlay(overlay)
        overlay_texture_id = root.canvas.overlay_lease.texture.id
        root.canvas._set_overlay(np.full_like(overlay, 23))
        assert root.canvas.overlay_lease.texture.id == overlay_texture_id
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
        active_id = application.layer_stack.active_layer.id
        active_node = root.layer_panel._stable_to_node[active_id]
        root.layer_panel._on_row_toggle(
            active_node, 0, root.layer_panel.model.node(active_node).item)
        assert not application.layer_stack.active_layer.visible
        root.layer_panel._on_row_toggle(
            active_node, 0, root.layer_panel.model.node(active_node).item)
        assert application.layer_stack.active_layer.visible
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
        model, command_id = root.canvas_controls.brush.tool_commands[
            BrushToolMode.MASK_ERASER]
        assert model.command(command_id).data.checked
        root.layer_tree_coordinator.handle_intent(LayerTreeIntent(
            LayerTreeAction.DETACH_TOOL,
            layer_id=active_id,
        ))
        assert application.layer_stack.active_layer.tool is None
        assert root.generation_panels.empty_label.visible
        root.layer_tree_coordinator.handle_intent(LayerTreeIntent(
            LayerTreeAction.ATTACH_TOOL,
            layer_id=active_id,
            value="instruct",
        ))
        ai_edit_tool = application.layer_stack.active_layer.tool
        assert ai_edit_tool.source_patch.size == (32, 32)
        assert (
            ai_edit_tool.patch_x,
            ai_edit_tool.patch_y,
            ai_edit_tool.patch_w,
            ai_edit_tool.patch_h,
        ) == (0, 0, 32, 32)
        root.layer_tree_coordinator.handle_intent(LayerTreeIntent(
            LayerTreeAction.DETACH_TOOL,
            layer_id=active_id,
        ))
        root.canvas_controls.selection.rect_mode.checked = True
        assert root.canvas.widget.cursor_intent == CursorIntent.Crosshair
        model, command_id = root.canvas_controls.brush.tool_commands[
            BrushToolMode.SMUDGE]
        root.canvas_controls.brush._on_tool_activated(
            0, command_id, model.command(command_id).data)
        assert root.canvas.controller.brush_tool_mode == BrushToolMode.SMUDGE
        assert root.canvas.widget.cursor_intent == CursorIntent.Default
        model, command_id = root.canvas_controls.brush.tool_commands[
            BrushToolMode.PAINT]
        root.canvas_controls.brush._on_tool_activated(
            0, command_id, model.command(command_id).data)
        root.canvas_controls.brush.size.value = 5
        assert root.canvas.controller.brush.size == 5
        root.canvas.controller.brush.set_hardness(1.0)
        root.canvas.controller.brush.set_color(255, 0, 0, 255)
        brush_size = root.canvas.controller.brush.size
        root.composition.push_key(ord("]"))
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
        assert (
            root.composition.document.pointer_capture
            == root.canvas._capture_relay.handle
        )
        event.type = PointerEventType.Move
        event.x = root.canvas.widget.bounds.x - 20
        event.y = root.canvas.widget.bounds.y - 20
        root.composition.document.dispatch_pointer_event(event)
        event.type = PointerEventType.Up
        root.composition.document.dispatch_pointer_event(event)
        assert not root.composition.document.pointer_capture

        painted = application.layer_stack.active_layer.image.copy()
        assert application.history.can_undo
        assert root.view.activate_command("edit.undo")
        assert application.layer_stack.active_layer.image[8, 8, 3] == 0
        assert root.view.activate_command("edit.redo")
        np.testing.assert_array_equal(
            application.layer_stack.active_layer.image,
            painted,
        )

        revision = application.history.memory_revision
        widget_point = root.canvas.canvas.image_to_widget(Point(20, 20))
        event.x = widget_point.x
        event.y = widget_point.y
        event.type = PointerEventType.Down
        root.composition.document.dispatch_pointer_event(event)
        root.canvas.cancel_pointer_interaction()
        assert not root.composition.document.pointer_capture
        np.testing.assert_array_equal(
            application.layer_stack.active_layer.image,
            painted,
        )
        assert application.history.memory_revision == revision

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
            ord("Y"),
            modifiers=int(ModifierFlag.Ctrl),
        )
        root.tick()
        assert activated == []

        application.set_command_state("edit.redo", enabled=True)
        root.composition.push_key(
            ord("Y"),
            modifiers=int(ModifierFlag.Ctrl),
        )
        result = root.tick()

        assert result.events == 1
        assert activated == ["redo"]


def test_offscreen_root_creates_selected_reconstruction_object(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(
        "TERMIN_SDK_SHADER_CACHE_ROOT", str(tmp_path / "shader-cache")
    )
    application = _application()
    image = np.zeros((16, 20, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    application.layer_stack.init_from_image(image)

    with NativeEditorRoot.create_headless(application) as root:
        class ViewportStub:
            mesh_count = 0

            def clear_model(self):
                pass

            def render_if_dirty(self):
                return False

        viewport = ViewportStub()
        root._ensure_reconstruction_viewport = lambda: viewport
        root.tick()
        background = application.layer_stack.active_layer
        before = application.layer_stack.composite()

        assert root.view.activate_command("layer.new_3d_reconstruction")
        root.tick()

        node = application.layer_stack.active_layer
        assert isinstance(node, ReconstructionLayer)
        root.view._change_reconstruction_parameter("steps", 20.0)
        root.view._change_reconstruction_parameter("resolution", 1280)
        root.view._change_reconstruction_parameter(
            "lr_conditioning_resolution", 1024
        )
        root.view._activate_reconstruction_refine("strength", 0.65)
        root.view._activate_reconstruction_refine("steps", 12.0)
        root.view._activate_reconstruction_refine(
            "hr_local_resolution", 1024
        )
        root.view._activate_reconstruction_refine("lr_steps", 13.0)
        root.view._activate_reconstruction_refine("lr_seed", 901.0)
        root.view._activate_reconstruction_refine("texture_steps", 14.0)
        root.view._activate_reconstruction_refine("texture_seed", 902.0)
        root.view._activate_reconstruction_refine("paint", True)
        assert node.generation_parameters.steps == 20
        assert node.generation_parameters.resolution == 1280
        assert node.generation_parameters.lr_conditioning_resolution == 1024
        assert node.refine_parameters.strength == 0.65
        assert node.refine_parameters.steps == 12
        assert node.refine_parameters.local_resolution == 1024
        assert node.lr_refine_parameters.steps == 13
        assert node.lr_refine_parameters.seed == 901
        assert node.texture_refine_parameters.steps == 14
        assert node.texture_refine_parameters.seed == 902
        assert root.canvas_controls_coordinator.selection_state.edit_mode
        assert root.canvas.controller.overlay_bridge.selection_as_mask
        root.view._activate_reconstruction_stage(
            ReconstructionStage.HR_COORDINATES
        )
        assert node.target_stage is ReconstructionStage.HR_COORDINATES
        assert root._presented_reconstruction_id == node.id
        assert root.layer_tree_coordinator.state.active_id == node.id
        assert root.layer_tree_coordinator.state.roots[0].node_type == (
            "reconstruction"
        )
        assert np.array_equal(application.layer_stack.composite(), before)
        assert application.status_text == f"Created: {node.name}"

        application.layer_stack.active_layer = background
        root.tick()
        assert root.view.reconstruction_mode is False
        assert not root.canvas_controls_coordinator.selection_state.edit_mode
        assert not root.canvas.controller.overlay_bridge.selection_as_mask

        application.layer_stack.active_layer = node
        root.tick()
        assert root.view.reconstruction_mode is True
        assert root.canvas.controller.overlay_bridge.selection_as_mask


def test_reconstruction_point_artifact_is_loaded_as_point_cloud(tmp_path) -> None:
    path = tmp_path / "preview.ply"
    path.write_bytes(b"ply\n")
    loaded = []

    class Viewport:
        def load_glb(self, _path):
            raise AssertionError("point preview must not be loaded as GLB")

        def load_point_cloud(self, point_path):
            loaded.append(point_path)

    root = NativeEditorRoot.__new__(NativeEditorRoot)
    viewport = Viewport()
    root._ensure_reconstruction_viewport = lambda: viewport

    artifact = ReconstructionStageArtifact(
        ReconstructionStage.POINT_CLOUD,
        str(path),
        "points",
    )

    assert root._load_reconstruction_artifact(artifact)
    assert loaded == [str(path)]


def test_latest_depth_cloud_is_uploaded_directly_and_keeps_3d_context() -> None:
    application = _application()
    image = np.zeros((3, 4, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    application.layer_stack.init_from_image(image)
    layer = application.layer_stack.active_layer
    positions = np.array(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    class Cloud:
        point_count = 2

    cloud = Cloud()
    cloud.positions = positions
    cloud.colors = colors
    cloud.confidence = np.array([1.0, 4.0], dtype=np.float32)
    application.latest_depth_point_cloud = cloud
    application._latest_depth_point_cloud_layer_ids = frozenset((layer.id,))
    uploads = []

    class Viewport:
        def load_point_cloud_data(
                self,
                uploaded_positions,
                uploaded_colors,
                *,
                fit_camera=None,
                confidence_colors=None,
                color_mode="image",
                confidence_legend="",
        ):
            uploads.append((
                uploaded_positions,
                uploaded_colors,
                fit_camera,
                confidence_colors,
                color_mode,
                confidence_legend,
            ))
            return len(uploaded_positions)

    class View:
        def __init__(self):
            self.contexts = []

        def set_reconstruction_context(self, visible, status=""):
            self.contexts.append((visible, status))

    root = NativeEditorRoot.__new__(NativeEditorRoot)
    root.application = application
    root.view = View()
    root.canvas = None
    root.canvas_controls_coordinator = None
    root._presented_reconstruction_id = "old-reconstruction"
    root._presented_depth_point_cloud_layer_id = None
    root._ensure_reconstruction_viewport = lambda: Viewport()

    root._view_depth_point_cloud()
    root._sync_reconstruction_selection(None)

    np.testing.assert_array_equal(uploads[0][0], positions)
    np.testing.assert_array_equal(uploads[0][1], colors)
    assert uploads[0][2] is True
    assert uploads[0][3].shape == (2, 3)
    np.testing.assert_allclose(
        uploads[0][3][0], np.array((48, 18, 59)) / 255.0)
    np.testing.assert_allclose(
        uploads[0][3][1], np.array((122, 4, 3)) / 255.0)
    assert uploads[0][4] == "confidence"
    assert "P2–P98" in uploads[0][5]
    assert root._presented_reconstruction_id is None
    assert root._presented_depth_point_cloud_layer_id == layer.id
    assert root.view.contexts[-1] == (True, "Depth point cloud")
    assert "2 colored points" in application.status_text
    assert "confidence colors" in application.status_text
    application.close()
