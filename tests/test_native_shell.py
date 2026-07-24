from __future__ import annotations

from termin.gui_native import (
    ModifierFlag,
    Rect,
    tc_ui_document_create,
    tc_ui_document_destroy,
)

from diffusion_editor.app.native_shell import NativeEditorView


EXPECTED_COMMANDS = (
    "file.new",
    "file.new_from_image",
    "file.open",
    "file.save",
    "file.save_as",
    "file.import",
    "file.export",
    "app.quit",
    "edit.undo",
    "edit.redo",
    "edit.redo.ctrl_y",
    "edit.copy",
    "edit.copy_visible",
    "edit.paste",
    "edit.settings",
    "selection.all",
    "selection.clear",
    "selection.invert",
    "layer.new",
    "layer.remove",
    "layer.flatten",
    "layer.detect",
    "view.fit",
)


def _snapshot_by_id(document):
    return {
        item["stable_id"]: item
        for item in document.inspect_snapshot()["widgets"]
        if item["stable_id"]
    }


def test_native_shell_snapshot_has_stable_layout_and_command_inventory():
    document = tc_ui_document_create()
    repaint_requests = []
    titles = []
    view = NativeEditorView(
        document,
        lambda: repaint_requests.append(True),
        titles.append,
        {},
    )
    try:
        document.layout_roots(Rect(0.0, 0.0, 1000.0, 700.0))
        snapshot = _snapshot_by_id(document)

        assert view.command_inventory == EXPECTED_COMMANDS
        assert {
            "diffusion-editor.root",
            "diffusion-editor.menu-bar",
            "diffusion-editor.toolbar",
            "diffusion-editor.main-splitter",
            "diffusion-editor.left-panel",
            "diffusion-editor.workspace-splitter",
            "diffusion-editor.canvas-host",
            "diffusion-editor.right-splitter",
            "diffusion-editor.layer-panel",
            "diffusion-editor.agent-panel",
            "diffusion-editor.status",
        } <= snapshot.keys()

        root = snapshot["diffusion-editor.root"]["bounds"]
        menu = snapshot["diffusion-editor.menu-bar"]["bounds"]
        toolbar = snapshot["diffusion-editor.toolbar"]["bounds"]
        main = snapshot["diffusion-editor.main-splitter"]["bounds"]
        status = snapshot["diffusion-editor.status"]["bounds"]
        assert (root.width, root.height) == (1000.0, 700.0)
        assert (menu.y, menu.height) == (0.0, 28.0)
        assert (toolbar.y, toolbar.height) == (28.0, 36.0)
        assert (main.y, main.height) == (64.0, 612.0)
        assert (status.y, status.height) == (676.0, 24.0)
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_native_shortcuts_and_command_state_use_app_owned_handlers():
    document = tc_ui_document_create()
    activated = []
    repaint_requests = []
    view = NativeEditorView(
        document,
        lambda: repaint_requests.append(True),
        lambda _title: None,
        {
            "file.save": lambda: activated.append("save"),
            "edit.redo": lambda: activated.append("redo"),
        },
    )
    try:
        ctrl = int(ModifierFlag.Ctrl)
        assert view.menu_bar.dispatch_shortcut(ord("s"), ctrl)
        assert activated == ["save"]

        view.set_command_state("file.save", enabled=False)
        assert not view.menu_bar.dispatch_shortcut(ord("s"), ctrl)
        assert activated == ["save"]

        view.set_command_state("file.save", enabled=True)
        assert view.menu_bar.dispatch_shortcut(ord("s"), ctrl)
        assert view.menu_bar.dispatch_shortcut(ord("y"), ctrl)
        assert activated == ["save", "save", "redo"]

        view.set_command_handler(
            "selection.all",
            lambda: activated.append("select-all"),
        )
        assert view.activate_command("selection.all")
        assert activated[-1] == "select-all"
        assert len(repaint_requests) == 2
    finally:
        view.close()
        tc_ui_document_destroy(document)


def test_native_status_and_window_title_are_presentation_state():
    document = tc_ui_document_create()
    repaint_requests = []
    titles = []
    view = NativeEditorView(
        document,
        lambda: repaint_requests.append(True),
        titles.append,
        {},
    )
    try:
        view.set_status("Saving")
        view.set_window_title("picture.png — Diffusion Editor")

        assert view.status_bar.text == "Saving"
        assert view.window_title == "picture.png — Diffusion Editor"
        assert titles == ["picture.png — Diffusion Editor"]
        assert repaint_requests == [True]
    finally:
        view.close()
        tc_ui_document_destroy(document)
