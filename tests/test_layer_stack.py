"""Tests for LayerStack prefix-sum compositing."""

import numpy as np
import pytest

from diffusion_editor.layer import Layer, LayerStack


def _solid_image(w, h, r, g, b, a):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :] = [r, g, b, a]
    return img


def _make_stack(n_layers=5, w=64, h=64):
    """Create a stack with a red background and n_layers-1 semi-transparent layers."""
    stack = LayerStack()
    stack.on_changed = lambda: None
    bg = _solid_image(w, h, 255, 0, 0, 255)
    stack.init_from_image(bg)
    for i in range(n_layers - 1):
        img = _solid_image(w, h, 0, (i + 1) * 50, 0, 128)
        stack.add_layer(f"Layer {i}", img)
    return stack


# ---------- basic composite correctness ----------


class TestCompositeCorrectness:
    def test_single_layer(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(32, 32, 100, 200, 50, 255))
        result = stack.composite()
        assert result.shape == (32, 32, 4)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result[:, :, 0], 100)

    def test_two_opaque_layers(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(8, 8, 255, 0, 0, 255))
        stack.add_layer("top", _solid_image(8, 8, 0, 255, 0, 255))
        result = stack.composite()
        # Opaque green on top of red → green
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 255
        assert result[0, 0, 3] == 255

    def test_empty_stack(self):
        stack = LayerStack()
        result = stack.composite()
        assert result.shape == (1, 1, 4)

    def test_composite_idempotent(self):
        stack = _make_stack()
        a = stack.composite()
        b = stack.composite()
        np.testing.assert_array_equal(a, b)


# ---------- prefix cache behaviour ----------


class TestPrefixCache:
    def test_cache_built_on_first_composite(self):
        stack = _make_stack()
        assert all(stack._prefix_dirty)
        stack.composite()
        assert not any(stack._prefix_dirty)

    def test_cached_composite_returns_equal(self):
        stack = _make_stack()
        first = stack.composite()
        second = stack.composite()
        np.testing.assert_array_equal(first, second)

    def test_mark_top_layer_dirty(self):
        stack = _make_stack(5)
        stack.composite()
        stack.mark_layer_dirty(stack.layers[0])
        # Only last entry in compositing order should be dirty
        assert sum(stack._prefix_dirty) == 1
        assert stack._prefix_dirty[-1]

    def test_mark_bottom_layer_dirty(self):
        stack = _make_stack(5)
        stack.composite()
        stack.mark_layer_dirty(stack.layers[-1])
        assert all(stack._prefix_dirty)

    def test_mark_middle_layer_dirty(self):
        stack = _make_stack(5)
        stack.composite()
        # layers[2] is in the middle
        stack.mark_layer_dirty(stack.layers[2])
        n = len(stack.layers)
        comp_idx = n - 1 - stack.layers.index(stack.layers[2])
        dirty_count = sum(stack._prefix_dirty)
        expected_dirty = n - comp_idx
        assert dirty_count == expected_dirty

    def test_structural_change_rebuilds_caches(self):
        stack = _make_stack(3)
        stack.composite()
        assert not any(stack._prefix_dirty)
        stack.add_layer("new")
        assert all(stack._prefix_dirty)
        assert len(stack._prefix_caches) == len(stack.layers)


# ---------- visibility ----------


class TestVisibility:
    def test_toggle_visibility_invalidates(self):
        stack = _make_stack(3)
        stack.composite()
        stack.set_visibility(stack.layers[0], False)
        # At least one entry should be dirty
        assert any(stack._prefix_dirty)

    def test_hidden_layer_not_blended(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(8, 8, 255, 0, 0, 255))
        stack.add_layer("green", _solid_image(8, 8, 0, 255, 0, 255))
        stack.set_visibility(stack.layers[0], False)
        result = stack.composite()
        # Green hidden → red background
        assert result[0, 0, 0] == 255
        assert result[0, 0, 1] == 0

    def test_toggle_off_and_on(self):
        stack = _make_stack(3)
        before = stack.composite().copy()
        stack.set_visibility(stack.layers[0], False)
        stack.composite()
        stack.set_visibility(stack.layers[0], True)
        after = stack.composite()
        np.testing.assert_array_equal(before, after)


# ---------- opacity ----------


class TestOpacity:
    def test_set_opacity_invalidates(self):
        stack = _make_stack(3)
        stack.composite()
        stack.set_opacity(stack.layers[0], 0.5)
        assert any(stack._prefix_dirty)

    def test_zero_opacity_equals_hidden(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(8, 8, 255, 0, 0, 255))
        stack.add_layer("green", _solid_image(8, 8, 0, 255, 0, 255))

        stack.set_opacity(stack.layers[0], 0.0)
        r1 = stack.composite()

        stack.set_opacity(stack.layers[0], 1.0)
        stack.set_visibility(stack.layers[0], False)
        r2 = stack.composite()

        np.testing.assert_array_equal(r1, r2)


# ---------- exclude_layer (composite_below) ----------


class TestCompositeExcluding:
    def test_exclude_top_layer(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(8, 8, 255, 0, 0, 255))
        stack.add_layer("green", _solid_image(8, 8, 0, 255, 0, 255))
        below = stack.composite(exclude_layer=stack.layers[0])
        # Should be just the red background
        assert below[0, 0, 0] == 255
        assert below[0, 0, 1] == 0

    def test_exclude_bottom_layer(self):
        stack = LayerStack()
        stack.on_changed = lambda: None
        stack.init_from_image(_solid_image(8, 8, 255, 0, 0, 255))
        below = stack.composite(exclude_layer=stack.layers[0])
        # Nothing below the only layer → black/transparent
        assert below[0, 0, 3] == 0

    def test_get_prefix_below(self):
        stack = _make_stack(5)
        stack.composite()
        top = stack.layers[0]
        cache = stack.get_prefix_below(top)
        assert cache is not None
        assert cache.shape == (64, 64, 4)
        assert cache.dtype == np.uint8


# ---------- structural operations ----------


class TestStructuralOps:
    def test_add_layer(self):
        stack = _make_stack(3)
        stack.composite()
        stack.add_layer("extra")
        assert len(stack._prefix_caches) == len(stack.layers)
        assert all(stack._prefix_dirty)

    def test_remove_layer(self):
        stack = _make_stack(4)
        stack.composite()
        stack.remove_layer(stack.layers[0])
        assert len(stack._prefix_caches) == len(stack.layers)
        result = stack.composite()
        assert result.shape[0] == 64

    def test_move_layer(self):
        stack = _make_stack(4)
        r_before = stack.composite().copy()
        top = stack.layers[0]
        stack.move_layer(top, None, len(stack.layers))
        # After moving top to bottom, result should differ
        r_after = stack.composite()
        assert not np.array_equal(r_before, r_after)

    def test_flatten(self):
        stack = _make_stack(5)
        expected = stack.composite()
        stack.flatten()
        assert len(stack.layers) == 1
        result = stack.composite()
        np.testing.assert_array_equal(result, expected)

    def test_init_from_image(self):
        stack = _make_stack(5)
        stack.composite()
        new_img = _solid_image(32, 32, 0, 0, 255, 255)
        stack.init_from_image(new_img)
        assert len(stack._prefix_caches) == 1
        assert all(stack._prefix_dirty)
        result = stack.composite()
        assert result.shape == (32, 32, 4)


# ---------- mark_layer_dirty edge cases ----------


class TestMarkDirtyEdgeCases:
    def test_mark_unknown_layer(self):
        stack = _make_stack(3)
        stack.composite()
        orphan = Layer("orphan", 64, 64)
        # Should not crash, rebuilds all caches
        stack.mark_layer_dirty(orphan)
        assert all(stack._prefix_dirty)

    def test_mark_dirty_before_first_composite(self):
        stack = _make_stack(3)
        # All already dirty, should not crash
        stack.mark_layer_dirty(stack.layers[0])
        result = stack.composite()
        assert result.dtype == np.uint8
