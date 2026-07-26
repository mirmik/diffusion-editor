import random

import numpy as np
import pytest

from diffusion_editor.document.layer import Layer
from diffusion_editor.document.layer_stack import LayerStack


def _stack() -> LayerStack:
    stack = LayerStack(tile_size=8)
    stack.init_from_image(np.zeros((8, 10, 4), dtype=np.uint8))
    stack.add_layer("Second")
    stack.add_layer("Third")
    assert stack.validate_invariants()
    return stack


def test_public_structure_views_are_read_only_tuples():
    stack = _stack()
    root = stack.layers[0]
    child = Layer("Child", 2, 2)
    stack.insert_layer(child)
    stack.move_layer(child, root, 0)

    assert isinstance(stack.layers, tuple)
    assert isinstance(root.children, tuple)
    with pytest.raises(AttributeError):
        stack.layers.append(Layer("Nope", 1, 1))
    with pytest.raises(AttributeError):
        root.children.append(Layer("Nope", 1, 1))
    with pytest.raises(RuntimeError, match="through LayerStack"):
        root.add_child(Layer("Nope", 1, 1))
    with pytest.raises(RuntimeError, match="through LayerStack"):
        root.remove_child(child)
    assert stack.validate_invariants()


def test_move_rejects_self_descendant_foreign_and_bad_index_before_mutation():
    stack = _stack()
    parent = stack.layers[0]
    child = Layer("Child", 2, 2)
    stack.insert_layer(child)
    stack.move_layer(child, parent, 0)
    before = stack.serialize_state()

    with pytest.raises(ValueError, match="itself or a descendant"):
        stack.move_layer(parent, parent, 0)
    with pytest.raises(ValueError, match="itself or a descendant"):
        stack.move_layer(parent, child, 0)
    with pytest.raises(ValueError, match="does not belong"):
        stack.move_layer(Layer("Foreign", 1, 1), None, 0)
    with pytest.raises(ValueError, match="does not belong"):
        stack.move_layer(child, Layer("Foreign Parent", 1, 1), 0)
    with pytest.raises(IndexError, match="out of range"):
        stack.move_layer(child, None, 99)

    assert stack.serialize_state() == before
    assert stack.validate_invariants()


def test_remove_rejects_last_root_subtree_and_foreign_layer():
    stack = LayerStack()
    stack.init_from_image(np.zeros((4, 4, 4), dtype=np.uint8))
    root = stack.active_layer
    child = Layer("Child", 2, 2)
    stack.insert_layer(child)
    stack.move_layer(child, root, 0)

    with pytest.raises(ValueError, match="last root"):
        stack.remove_layer(root)
    with pytest.raises(ValueError, match="does not belong"):
        stack.remove_layer(Layer("Foreign", 1, 1))

    assert stack.layers == (root,)
    assert stack.validate_invariants()


def test_insert_rejects_duplicate_ids_and_attached_subtree_root():
    stack = _stack()
    duplicate = Layer("Duplicate", 2, 2, layer_id=stack.layers[0].id)
    with pytest.raises(ValueError, match="already exists"):
        stack.insert_layer(duplicate)

    parent = Layer("Parent", 2, 2)
    attached = Layer("Attached", 2, 2)
    parent.add_child(attached)
    with pytest.raises(ValueError, match="must be detached"):
        stack.insert_layer(attached)
    assert stack.validate_invariants()


def test_active_and_solo_reject_stale_layer_objects():
    stack = _stack()
    foreign = Layer("Foreign", 1, 1)

    with pytest.raises(ValueError, match="does not belong"):
        stack.active_layer = foreign
    with pytest.raises(ValueError, match="does not belong"):
        stack.set_solo_layer(foreign)

    assert stack.validate_invariants()


def test_random_valid_moves_preserve_invariants_and_snapshot_roundtrip():
    rng = random.Random(12345)
    stack = _stack()
    for index in range(7):
        stack.add_layer(f"Layer {index}")

    for _ in range(100):
        layers = stack.all_layers()
        layer = rng.choice(layers)
        descendants = set(layer.all_descendants())
        possible_parents = [
            None,
            *(
                candidate
                for candidate in layers
                if candidate is not layer and candidate not in descendants
            ),
        ]
        new_parent = rng.choice(possible_parents)
        destination = (
            stack.layers if new_parent is None else new_parent.children
        )
        target_index = rng.randrange(len(destination) + 1)
        stack.move_layer(layer, new_parent, target_index)
        assert stack.validate_invariants()

    restored = LayerStack()
    restored.load_state(stack.serialize_state())
    assert restored.validate_invariants()
    assert [layer.id for layer in restored.all_layers()] == [
        layer.id for layer in stack.all_layers()
    ]


def test_document_replacement_and_removal_release_subtree_ownership():
    stack = _stack()
    old_root = stack.layers[0]

    stack.init_from_image(np.zeros((3, 4, 4), dtype=np.uint8))

    assert old_root._owner is None
    stack.insert_layer(old_root)
    assert old_root._owner is stack
    stack.remove_layer(old_root)
    assert old_root._owner is None
    assert stack.validate_invariants()
