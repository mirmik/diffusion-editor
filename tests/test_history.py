"""Tests for history memory limits."""

import pytest

from diffusion_editor.document.history import HistoryManager


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


def test_failed_undo_keeps_entry_and_revision():
    should_fail = True
    events = []

    def undo():
        if should_fail:
            raise RuntimeError("undo failed")
        events.append("undo")

    hm = HistoryManager(lambda _snap: None)
    hm.push_callbacks("A", undo, _noop)
    revision = hm.memory_revision

    with pytest.raises(RuntimeError, match="undo failed"):
        hm.undo()

    assert hm.can_undo is True
    assert hm.can_redo is False
    assert hm.memory_revision == revision

    should_fail = False
    assert hm.undo() == "A"
    assert events == ["undo"]


def test_failed_redo_keeps_entry_and_revision():
    should_fail = False
    events = []

    def redo():
        if should_fail:
            raise RuntimeError("redo failed")
        events.append("redo")

    hm = HistoryManager(lambda _snap: None)
    hm.push_callbacks("A", _noop, redo)
    assert hm.undo() == "A"
    revision = hm.memory_revision
    should_fail = True

    with pytest.raises(RuntimeError, match="redo failed"):
        hm.redo()

    assert hm.can_undo is False
    assert hm.can_redo is True
    assert hm.memory_revision == revision

    should_fail = False
    assert hm.redo() == "A"
    assert events == ["redo"]


def test_failed_snapshot_undo_restores_post_command_state():
    state = {"value": b"after"}
    fail_before = True

    def apply_snapshot(snapshot):
        nonlocal fail_before
        state["value"] = snapshot
        if fail_before and snapshot == b"before":
            state["value"] = b"partial"
            raise RuntimeError("snapshot failed")

    hm = HistoryManager(apply_snapshot)
    hm.push("A", b"before", b"after")

    with pytest.raises(RuntimeError, match="snapshot failed"):
        hm.undo()

    assert state["value"] == b"after"
    assert hm.can_undo is True
    assert hm.can_redo is False

    fail_before = False
    assert hm.undo() == "A"
    assert state["value"] == b"before"
