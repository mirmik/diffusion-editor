from pathlib import Path
import numpy as np
from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.dialogs import ApplicationDialogCoordinator
from diffusion_editor.app.editor_commands import EditorCommandCoordinator
from diffusion_editor.app.native_layer_panel import NativeLayerPanel
from diffusion_editor.app.layer_tree import LayerTreeCoordinator
from diffusion_editor.app.native_shell import COMMAND_SPECS
from diffusion_editor.document.svg_layer import SvgLayer


class Settings:
    def get(self, key, default=None):
        return default
    def set(self, key, value):
        pass


class Engine:
    model_info = {}
    def poll_event(self):
        return None
    def shutdown(self):
        pass


class Canvas:
    def fit_in_view(self):
        pass


class Dialogs:
    def __init__(self):
        self.files = []
        self.errors = []
    def show_file_dialog(self, spec, callback):
        self.files.append((spec, callback))
    def show_error(self, title, message):
        self.errors.append((title, message))


def test_svg_ui_import_replace_export_and_command_states(tmp_path):
    engine = Engine()
    app = EditorApplication(settings=Settings(), engines=EngineSet(*([engine]*5)))
    app.layer_stack.init_from_image(np.full((64, 64, 4), 255, dtype=np.uint8))
    app.reset_document_session(None)
    commands = EditorCommandCoordinator(app)
    dialogs = ApplicationDialogCoordinator(app, Canvas())
    view = Dialogs()
    dialogs.bind_view(view)
    path = tmp_path/'rooms.svg'
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect id="room" width="32" height="32" fill="red"/></svg>')
    assert app.command_states['layer.import_svg'] == (True, False)
    assert app.command_states['layer.rasterize_svg'] == (False, False)
    dialogs.command_handlers['layer.import_svg']()
    spec, callback = view.files.pop()
    assert spec.filters == 'SVG | *.svg'
    callback(str(path))
    layer = app.layer_stack.active_layer
    assert isinstance(layer, SvgLayer)
    commands.refresh()
    assert app.command_states['layer.rasterize_svg'] == (True, False)
    assert app.command_states['edit.clear_selected_pixels'] == (False, False)
    tree = LayerTreeCoordinator(app.layer_stack, app.document)
    item = NativeLayerPanel._item(tree.state.roots[0])
    assert item.subtitle == 'SVG · Vector layer'
    original = layer.svg_source
    path.write_text('<broken')
    dialogs.replace_svg_path(str(path), layer.id)
    assert view.errors and layer.svg_source == original
    path.write_text(original.replace('red', 'blue'))
    dialogs.replace_svg_path(str(path), layer.id)
    np.testing.assert_array_equal(layer.image[5,5], [0,0,255,255])
    target = tmp_path/'exported'
    dialogs.export_svg_path(str(target), layer.id)
    assert target.with_suffix('.svg').read_text() == layer.svg_source
    commands.handlers['layer.rasterize_svg']()
    assert app.layer_stack.active_layer.node_type == 'raster'
    commands.undo()
    assert app.layer_stack.active_layer is layer
    assert app.command_states['layer.replace_svg'] == (True, False)
    commands.handlers['layer.new_svg']()
    assert app.layer_stack.active_layer.node_type == 'svg'
    assert not app.layer_stack.active_layer.image.any()
    assert {'layer.new_svg','layer.import_svg','layer.replace_svg','layer.export_svg','layer.rasterize_svg'} <= {spec.stable_id for spec in COMMAND_SPECS}
    tree.close()
    dialogs.close()
    app.close()


def test_svg_native_canvas_updates_and_blocks_pixel_edits(tmp_path, monkeypatch):
    from diffusion_editor.app.native_root import NativeEditorRoot
    from diffusion_editor.canvas.brush import BrushToolMode
    monkeypatch.setenv('TERMIN_SDK_SHADER_CACHE_ROOT', str(tmp_path/'shader-cache'))
    monkeypatch.setenv('TERMIN_EDITOR_MCP', '0')
    engine = Engine()
    app = EditorApplication(settings=Settings(), engines=EngineSet(*([engine]*5)))
    with NativeEditorRoot.create_headless(app, width=800, height=600) as root:
        app.layer_stack.init_from_image(np.full((64,64,4), 255, dtype=np.uint8))
        layer_id = app.document.svg.add('<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect id="r" width="20" height="20" fill="red"/></svg>')
        root.tick()
        np.testing.assert_allclose(root.canvas.controller.get_composite()[5,5], [255,0,0,255], atol=1)
        app.document.svg.update_element('r', {'fill':'blue'})
        root.tick()
        np.testing.assert_allclose(root.canvas.controller.get_composite()[5,5], [0,0,255,255], atol=1)
        controller = root.canvas.controller
        before = app.document.svg.layer().image.copy()
        assert not root.canvas_controls_coordinator.brush_state.accepts_pixel_edits
        assert root.canvas_controls_coordinator.brush_state.can_move
        controller.set_brush_tool(BrushToolMode.PAINT)
        controller.pointer_down(5, 5, controller.LEFT_BUTTON, 0)
        controller.pointer_up(5,5)
        np.testing.assert_array_equal(app.document.svg.layer().image, before)
        controller.set_brush_tool(BrushToolMode.MOVE)
        controller.pointer_down(5, 5, controller.LEFT_BUTTON, 0)
        controller.pointer_move(10, 12)
        controller.pointer_up(10, 12)
        assert (app.document.svg.layer().x, app.document.svg.layer().y) == (5, 7)
        assert app.document.undo() == "Move Layer"
        assert (app.document.svg.layer().x, app.document.svg.layer().y) == (0, 0)
        app.document.svg.update(app.document.svg.source(), width=64, height=64)
        root.tick()
        np.testing.assert_allclose(controller.get_composite()[30,30], [0,0,255,255], atol=1)
        assert root.view.activate_command('layer.rasterize_svg')
        root.tick()
        assert app.layer_stack.active_layer.node_type == 'raster'
        assert root.view.activate_command('edit.undo')
        root.tick()
        assert app.document.svg.layer(layer_id).width == 64
