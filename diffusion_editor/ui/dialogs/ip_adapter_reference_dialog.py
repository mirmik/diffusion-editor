"""Dialog helpers for selecting an IP-Adapter reference layer."""

from __future__ import annotations

from collections.abc import Callable

from tcgui.widgets.combo_box import ComboBox
from tcgui.widgets.dialog import Dialog
from tcgui.widgets.label import Label
from tcgui.widgets.units import px
from tcgui.widgets.ui import UI
from tcgui.widgets.vstack import VStack

from ...document.layer import Layer
from ...document.layer_stack import LayerStack


def ip_adapter_reference_candidates(
        layer_stack: LayerStack,
        target_layer: Layer) -> list[Layer]:
    return [
        layer for layer in layer_stack.all_layers()
        if layer is not target_layer
    ]


def ip_adapter_reference_layer_label(
        layer_stack: LayerStack,
        layer: Layer) -> str:
    path = layer_stack.get_layer_path(layer)
    prefix = path if path is not None else "?"
    marks = []
    if not layer.visible:
        marks.append("hidden")
    if layer.patch_rect is not None:
        marks.append("patch")
    suffix = f" [{', '.join(marks)}]" if marks else ""
    return f"{prefix}: {layer.name}{suffix}"


def selected_reference_index(
        candidates: list[Layer],
        selected_layer_id: str | None) -> int:
    if selected_layer_id is None:
        return 0
    for index, candidate in enumerate(candidates):
        if candidate.id == selected_layer_id:
            return index
    return 0


def show_ip_adapter_reference_dialog(
        ui: UI,
        layer_stack: LayerStack,
        target_layer: Layer,
        selected_layer_id: str | None,
        on_selected: Callable[[Layer], None]) -> bool:
    candidates = ip_adapter_reference_candidates(layer_stack, target_layer)
    if not candidates:
        return False

    dlg = Dialog()
    dlg.title = "Pick IP-Adapter Reference Layer"
    dlg.buttons = ["OK", "Cancel"]
    dlg.default_button = "OK"
    dlg.cancel_button = "Cancel"

    content = VStack()
    content.spacing = 8

    label = Label()
    label.text = "Reference layer"
    label.font_size = 12
    content.add_child(label)

    combo = ComboBox()
    combo.items = [
        ip_adapter_reference_layer_label(layer_stack, candidate)
        for candidate in candidates
    ]
    combo.selected_index = selected_reference_index(
        candidates,
        selected_layer_id,
    )
    combo.preferred_width = px(320)
    content.add_child(combo)

    dlg.content = content

    def _apply(result: str) -> None:
        if result != "OK":
            return
        index = combo.selected_index
        if index < 0 or index >= len(candidates):
            return
        on_selected(candidates[index])

    dlg.on_result = _apply
    dlg.show(ui)
    ui.set_focus(combo)
    return True
