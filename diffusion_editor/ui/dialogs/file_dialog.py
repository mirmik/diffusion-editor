"""File dialogs via tcgui overlay layer."""

from typing import Callable

from tcgui.widgets.file_dialog_overlay import (
    show_open_file_dialog,
    show_save_file_dialog,
    show_open_directory_dialog,
)


def open_file_dialog(
    ui,
    on_result: Callable[[str | None], None],
    *,
    title: str = "Open",
    directory: str = "",
    filter_str: str = "",
) -> None:
    show_open_file_dialog(
        ui, on_result,
        title=title, directory=directory, filter_str=filter_str,
    )


def save_file_dialog(
    ui,
    on_result: Callable[[str | None], None],
    *,
    title: str = "Save",
    directory: str = "",
    filter_str: str = "",
) -> None:
    show_save_file_dialog(
        ui, on_result,
        title=title, directory=directory, filter_str=filter_str,
    )


def open_directory_dialog(
    ui,
    on_result: Callable[[str | None], None],
    *,
    title: str = "Select Directory",
    directory: str = "",
) -> None:
    show_open_directory_dialog(
        ui, on_result,
        title=title, directory=directory,
    )
