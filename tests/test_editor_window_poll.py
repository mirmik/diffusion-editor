import numpy as np
from PIL import Image
from types import SimpleNamespace

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.presentation import HeadlessEditorPresentation
from diffusion_editor.generation.diffusion_controller import DiffusionControllerEvent
from diffusion_editor.app.editor_window import EditorWindow
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.document.commands import AddLayerCommand


class _Engine:
    model_info = {"path": "model.safetensors"}

    def poll_event(self):
        return None

    def shutdown(self):
        pass


class _DiffusionController:
    def poll(self):
        return DiffusionControllerEvent(
            model_loaded_path="model.safetensors",
            status="Model loaded",
        )


def test_poll_diffusion_forwards_controller_model_loaded_event():
    engine = _Engine()
    application = EditorApplication(
        settings=_MemorySettings(),
        engines=EngineSet(engine, engine, engine, engine, engine),
    )
    application.diffusion_controller = _DiffusionController()
    presentation = HeadlessEditorPresentation()
    application.bind_view(presentation.ports())

    application.poll()

    assert presentation.panel_updates[-1].state == "model-loaded"
    assert presentation.panel_updates[-1].payload["path"] == "model.safetensors"
    assert presentation.status == "Model loaded"


class _MemorySettings:
    def __init__(self):
        self.values = {}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


class _Status:
    text = ""


class _CloseRecorder:
    def __init__(self, calls, name):
        self._calls = calls
        self._name = name

    def close(self):
        self._calls.append(self._name)


class _RecordingUI:
    def __init__(self, calls):
        self._calls = calls
        self._root = object()
        self._renderer = _CloseRecorder(calls, "ui-renderer")

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        self._root = value
        self._calls.append("ui-root")


def _export_window(image):
    window = object.__new__(EditorWindow)
    window._layer_stack = LayerStack()
    window._layer_stack.init_from_image(image)
    window._statusbar = _Status()
    return window


def test_export_image_path_adds_png_extension_by_default(tmp_path):
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[1, 2] = (10, 20, 30, 255)
    window = _export_window(image)
    path = tmp_path / "exported"

    assert window.export_image_path(str(path)) is True

    out_path = tmp_path / "exported.png"
    assert out_path.exists()
    out = np.array(Image.open(out_path).convert("RGBA"))
    assert tuple(out[1, 2]) == (10, 20, 30, 255)
    assert window._statusbar.text == "Exported: exported.png"


def test_export_image_path_rejects_unknown_extension(tmp_path):
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    window = _export_window(image)
    path = tmp_path / "exported.xyz"

    assert window.export_image_path(str(path)) is False

    assert not path.exists()
    assert "Unknown export extension '.xyz'" in window._statusbar.text


def test_export_image_path_uses_layer_stack_composite_not_canvas_buffer(tmp_path):
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[0, 0] = (255, 0, 0, 255)
    window = _export_window(image)
    window._canvas = object()
    path = tmp_path / "composite.png"

    assert window.export_image_path(str(path)) is True

    out = np.array(Image.open(path).convert("RGBA"))
    assert tuple(out[0, 0]) == (255, 0, 0, 255)


def test_legacy_open_commits_project_path_before_fallible_fit(tmp_path):
    engine = _Engine()
    application = EditorApplication(
        settings=_MemorySettings(),
        engines=EngineSet(engine, engine, engine, engine, engine),
    )
    application.layer_stack.init_from_image(
        np.zeros((4, 4, 4), dtype=np.uint8))
    project_path = tmp_path / "opened.deproj"
    application.layer_stack.save_project(str(project_path))
    application.document.execute(AddLayerCommand(name="Unsaved"))

    class Canvas:
        def cancel_pointer_interaction(self):
            pass

        def fit_in_view(self):
            raise RuntimeError("fit failed")

    window = object.__new__(EditorWindow)
    window.application = application
    window._document = application.document
    window._layer_stack = application.layer_stack
    window._canvas = Canvas()
    window.canvas_edit_coordinator = SimpleNamespace(discard=lambda: None)
    window._project_path = str(tmp_path / "old.deproj")
    window._statusbar = _Status()

    window._open_file_path_unchecked(str(project_path))

    assert window._project_path == str(project_path)
    assert len(window._layer_stack.layers) == 1
    assert not application.history.can_undo
    assert window._statusbar.text == "Opened: opened.deproj"
    application.close()


def test_close_releases_legacy_ui_before_host_graphics_shutdown():
    calls = []
    window = object.__new__(EditorWindow)
    window._closed = False
    window.application = _CloseRecorder(calls, "application")
    window.ui = _RecordingUI(calls)

    window.close()
    window.close()

    assert calls == ["application", "ui-root", "ui-renderer"]
