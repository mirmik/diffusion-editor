"""Tests for history memory limits."""

from diffusion_editor.history import HistoryManager


def _noop():
    pass


def test_prunes_oldest_undo_entries_on_memory_limit():
    hm = HistoryManager(lambda _snap: None, max_entries=10, max_memory_bytes=12)

    hm.push_callbacks("A", _noop, _noop, size_bytes=6)
    hm.push_callbacks("B", _noop, _noop, size_bytes=6)
    hm.push_callbacks("C", _noop, _noop, size_bytes=6)

    assert hm.memory_bytes() == 12
    assert hm.undo() == "C"
    assert hm.undo() == "B"
    assert hm.undo() is None


def test_set_limit_trims_existing_history():
    hm = HistoryManager(lambda _snap: None, max_entries=10, max_memory_bytes=30)

    hm.push_callbacks("A", _noop, _noop, size_bytes=8)
    hm.push_callbacks("B", _noop, _noop, size_bytes=8)
    hm.push_callbacks("C", _noop, _noop, size_bytes=8)

    hm.set_max_memory_bytes(10)

    assert hm.memory_bytes() <= 10
    assert hm.undo() == "C"
    assert hm.undo() is None


def test_drops_too_large_entry():
    hm = HistoryManager(lambda _snap: None, max_entries=10, max_memory_bytes=4)

    hm.push_callbacks("huge", _noop, _noop, size_bytes=10)

    assert hm.memory_bytes() == 0
    assert hm.can_undo is False
