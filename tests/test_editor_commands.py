import numpy as np

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.editor_commands import EditorCommandCoordinator


class _Settings:
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
        settings=_Settings(),
        engines=EngineSet(engine, engine, engine, engine, engine),
    )


def test_standard_commands_project_state_and_round_trip_clipboard_history():
    application = _application()
    image = np.zeros((6, 8, 4), dtype=np.uint8)
    image[2:4, 3:6] = (10, 20, 30, 255)
    application.layer_stack.init_from_image(image)
    fits = []
    commands = EditorCommandCoordinator(
        application,
        fit_in_view=lambda: fits.append(True),
    )

    assert application.command_states["edit.undo"] == (False, False)
    assert application.command_states["selection.all"] == (True, False)
    assert application.command_states["selection.clear"] == (False, False)

    commands.handlers["selection.all"]()
    assert np.all(application.layer_stack.selection.data == 1.0)
    assert application.command_states["edit.undo"] == (True, False)
    assert application.command_states["selection.clear"] == (True, False)

    commands.handlers["edit.copy"]()
    np.testing.assert_array_equal(application.clipboard, image)
    assert application.clipboard_pos == (0, 0)
    assert application.command_states["edit.paste"] == (True, False)

    commands.handlers["edit.paste"]()
    assert len(application.layer_stack.all_layers()) == 2
    assert application.layer_stack.active_layer.name == "Floating Selection"
    assert application.status_text == "Pasted 8x6"

    commands.handlers["edit.undo"]()
    assert len(application.layer_stack.all_layers()) == 1
    assert application.command_states["edit.redo"] == (True, False)
    commands.handlers["edit.redo"]()
    assert len(application.layer_stack.all_layers()) == 2

    commands.handlers["view.fit"]()
    assert fits == [True]
    application.close()


def test_layer_and_selection_commands_cancel_active_mutation_first():
    application = _application()
    application.layer_stack.init_from_image(
        np.zeros((4, 4, 4), dtype=np.uint8)
    )
    cancelled = []
    commands = EditorCommandCoordinator(
        application,
        before_mutation=lambda: cancelled.append(True),
    )

    commands.handlers["layer.new"]()
    assert len(application.layer_stack.all_layers()) == 2
    assert application.layer_stack.active_layer.name == "Layer 0"

    commands.handlers["layer.flatten"]()
    assert len(application.layer_stack.all_layers()) == 1
    assert len(cancelled) == 2
    application.close()
