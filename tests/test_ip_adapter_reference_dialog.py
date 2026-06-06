import numpy as np

from diffusion_editor.ui.dialogs.ip_adapter_reference_dialog import (
    ip_adapter_reference_candidates,
    ip_adapter_reference_layer_label,
    selected_reference_index,
)
from diffusion_editor.document.layer import Layer
from diffusion_editor.document.layer_stack import LayerStack


def _rgba(width, height, color):
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:] = color
    return arr


def test_ip_adapter_reference_candidates_exclude_target_layer():
    stack = LayerStack()
    stack.init_from_image(_rgba(4, 4, (0, 0, 0, 255)))
    reference = Layer("Reference", 4, 4, _rgba(4, 4, (10, 20, 30, 255)))
    stack.insert_layer(reference)
    target = Layer("Diffusion", 4, 4, _rgba(4, 4, (0, 0, 0, 0)))
    stack.insert_layer(target)

    candidates = ip_adapter_reference_candidates(stack, target)

    assert target not in candidates
    assert reference in candidates
    assert stack.layers[-1] in candidates


def test_ip_adapter_reference_layer_label_includes_path_and_marks():
    stack = LayerStack()
    stack.init_from_image(_rgba(4, 4, (0, 0, 0, 255)))
    reference = Layer("Reference", 4, 4, _rgba(4, 4, (10, 20, 30, 255)))
    reference.visible = False
    reference.patch_rect = (1, 1, 3, 3)
    stack.insert_layer(reference)

    label = ip_adapter_reference_layer_label(stack, reference)

    assert label == "0: Reference [hidden, patch]"


def test_selected_reference_index_prefers_matching_layer_id():
    first = Layer("First", 1, 1)
    second = Layer("Second", 1, 1)

    assert selected_reference_index([first, second], second.id) == 1
    assert selected_reference_index([first, second], "missing") == 0
    assert selected_reference_index([first, second], None) == 0
