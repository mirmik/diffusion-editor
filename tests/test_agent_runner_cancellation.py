from __future__ import annotations

import importlib
import sys
import threading
import time
import types

import pytest


class _DocumentProbe:
    """Small document-shaped probe for mutation atomicity assertions."""

    def __init__(self) -> None:
        self.pixels = bytearray([0])
        self.layers: list[str] = []
        self.history: list[str] = []

    def commit(self, label: str) -> str:
        self.pixels[0] += 1
        self.layers.append(label)
        self.history.append(f"commit:{label}")
        return label

    def assert_unchanged(self) -> None:
        assert self.pixels == bytearray([0])
        assert self.layers == []
        assert self.history == []

    def assert_committed_once(self, label: str) -> None:
        assert self.pixels == bytearray([1])
        assert self.layers == [label]
        assert self.history == [f"commit:{label}"]


def _load_runner(monkeypatch, loop):
    nemor = types.ModuleType("nemor")
    core = types.ModuleType("nemor.core")
    session = types.ModuleType("nemor.core.session")
    agent = types.ModuleType("nemor.core.agent")
    session.Session = object
    agent.agent_loop = loop
    monkeypatch.setitem(sys.modules, "nemor", nemor)
    monkeypatch.setitem(sys.modules, "nemor.core", core)
    monkeypatch.setitem(sys.modules, "nemor.core.session", session)
    monkeypatch.setitem(sys.modules, "nemor.core.agent", agent)
    monkeypatch.delitem(
        sys.modules, "diffusion_editor.agent.runner", raising=False)
    return importlib.import_module("diffusion_editor.agent.runner")


def _make_runner(runner_module, mode, pending, events):
    if mode == "legacy":
        return runner_module.AgentRunner("registry", object(), {})
    return runner_module.AgentRunner(
        "registry",
        object(),
        {},
        defer=pending.append,
        on_event=lambda kind, value: events.append((kind, value)),
    )


def _wait_for(predicate, message: str, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(message)
        time.sleep(0.001)


def _wait_until_scheduled(mode, runner, pending) -> None:
    if mode == "legacy":
        _wait_for(
            lambda: not runner._main_calls.empty(),
            "legacy main-thread call was not queued",
        )
    else:
        _wait_for(
            lambda: bool(pending),
            "native main-thread call was not deferred",
        )


def _dispatch_one(mode, runner, pending, returned_events) -> None:
    if mode == "legacy":
        returned_events.extend(runner.poll())
    else:
        pending.pop(0)()


def _drain(mode, runner, pending, events) -> list[tuple[str, object]]:
    if mode == "legacy":
        events.extend(runner.poll())
        return events
    while pending:
        pending.pop(0)()
    return events


@pytest.mark.parametrize("mode", ["legacy", "native"])
def test_cancel_before_dispatch_drops_document_mutation(
        monkeypatch, mode):
    document = _DocumentProbe()
    attempted = threading.Event()
    outcomes = []

    def loop(_session, config, *, on_tool, **_kwargs):
        attempted.set()
        try:
            result = config["_run_on_main_thread"](
                lambda: document.commit("cancelled"))
        except RuntimeError as exc:
            outcomes.append(("cancelled", str(exc)))
            # A tool loop may try to report a caught tool exception. The
            # cancelled generation must not turn it into tool output.
            on_tool("mutate", "{}", "must-not-be-reported")
            return "caught cancellation"
        outcomes.append(("success", result))
        return result

    runner_module = _load_runner(monkeypatch, loop)
    pending = []
    events = []
    runner = _make_runner(runner_module, mode, pending, events)

    runner.submit("cancel")
    assert attempted.wait(1.0)
    _wait_until_scheduled(mode, runner, pending)
    runner.cancel()
    _wait_for(lambda: not runner.is_busy, "cancelled runner stayed busy")

    delivered = _drain(mode, runner, pending, events)

    document.assert_unchanged()
    assert outcomes == [
        ("cancelled", "Agent operation cancelled"),
    ]
    assert delivered == [("cancelled", None)]
    runner.shutdown()


@pytest.mark.parametrize("mode", ["legacy", "native"])
def test_cancel_during_dispatch_finishes_atomic_mutation_as_cancelled(
        monkeypatch, mode):
    document = _DocumentProbe()
    entered_mutation = threading.Event()
    release_mutation = threading.Event()
    outcomes = []

    def mutation():
        # Entering the callback is the documented transaction boundary.
        entered_mutation.set()
        assert release_mutation.wait(1.0)
        return document.commit("atomic")

    def loop(_session, config, *, on_tool, **_kwargs):
        try:
            result = config["_run_on_main_thread"](mutation)
        except RuntimeError as exc:
            outcomes.append(("cancelled", str(exc)))
            on_tool("mutate", "{}", "must-not-be-reported")
            return "caught cancellation"
        outcomes.append(("success", result))
        return result

    runner_module = _load_runner(monkeypatch, loop)
    pending = []
    events = []
    returned_events = []
    runner = _make_runner(runner_module, mode, pending, events)
    runner.submit("cancel while running")
    _wait_until_scheduled(mode, runner, pending)

    dispatch = threading.Thread(
        target=_dispatch_one,
        args=(mode, runner, pending, returned_events),
    )
    dispatch.start()
    assert entered_mutation.wait(1.0)

    runner.cancel()
    assert runner.is_busy
    release_mutation.set()
    dispatch.join(1.0)
    assert not dispatch.is_alive()
    _wait_for(lambda: not runner.is_busy, "cancelled runner stayed busy")

    events[:0] = returned_events
    delivered = _drain(mode, runner, pending, events)

    document.assert_committed_once("atomic")
    assert outcomes == [
        ("cancelled", "Agent operation cancelled"),
    ]
    assert delivered == [("cancelled", None)]
    runner.shutdown()


@pytest.mark.parametrize("mode", ["legacy", "native"])
def test_cancel_after_call_completion_preserves_completed_result(
        monkeypatch, mode):
    document = _DocumentProbe()
    call_completed = threading.Event()
    release_agent = threading.Event()
    outcomes = []

    def loop(_session, config, **_kwargs):
        result = config["_run_on_main_thread"](
            lambda: document.commit("completed"))
        outcomes.append(("success", result))
        call_completed.set()
        assert release_agent.wait(1.0)
        return result

    runner_module = _load_runner(monkeypatch, loop)
    pending = []
    events = []
    returned_events = []
    runner = _make_runner(runner_module, mode, pending, events)
    runner.submit("complete then cancel")
    _wait_until_scheduled(mode, runner, pending)

    _dispatch_one(mode, runner, pending, returned_events)
    assert call_completed.wait(1.0)
    runner.cancel()
    release_agent.set()
    _wait_for(lambda: not runner.is_busy, "cancelled runner stayed busy")

    events[:0] = returned_events
    delivered = _drain(mode, runner, pending, events)

    document.assert_committed_once("completed")
    assert outcomes == [("success", "completed")]
    assert delivered == [("cancelled", None)]
    runner.shutdown()


@pytest.mark.parametrize("mode", ["legacy", "native"])
def test_shutdown_cancels_pending_call_and_leaves_no_live_target(
        monkeypatch, mode):
    document = _DocumentProbe()
    outcomes = []

    def loop(_session, config, **_kwargs):
        try:
            config["_run_on_main_thread"](
                lambda: document.commit("after shutdown"))
        except RuntimeError as exc:
            outcomes.append(("cancelled", str(exc)))
        return "stopped"

    runner_module = _load_runner(monkeypatch, loop)
    pending = []
    events = []
    runner = _make_runner(runner_module, mode, pending, events)
    runner.submit("shutdown")
    _wait_until_scheduled(mode, runner, pending)

    runner.shutdown()
    assert not runner.is_busy
    _drain(mode, runner, pending, events)
    runner.submit("must be ignored")
    time.sleep(0.01)

    document.assert_unchanged()
    assert outcomes == [
        ("cancelled", "Agent operation cancelled"),
    ]
    assert not runner.is_busy


@pytest.mark.parametrize("mode", ["legacy", "native"])
def test_cancelled_generation_cannot_run_after_resubmit(
        monkeypatch, mode):
    document = _DocumentProbe()
    outcomes = []

    def loop(_session, config, *, user_input, **_kwargs):
        try:
            result = config["_run_on_main_thread"](
                lambda: document.commit(user_input))
        except RuntimeError as exc:
            outcomes.append((user_input, str(exc)))
            return "cancelled"
        outcomes.append((user_input, result))
        return result

    runner_module = _load_runner(monkeypatch, loop)
    pending = []
    events = []
    runner = _make_runner(runner_module, mode, pending, events)

    runner.submit("old")
    _wait_until_scheduled(mode, runner, pending)
    runner.cancel()
    _wait_for(lambda: not runner.is_busy, "old generation stayed busy")

    runner.submit("new")
    if mode == "legacy":
        _wait_for(
            lambda: runner._main_calls.qsize() == 2,
            "new generation was not queued",
        )
        runner.poll()
    else:
        _wait_for(
            lambda: len(pending) >= 3,
            "new generation was not deferred",
        )
        while pending and runner.is_busy:
            pending.pop(0)()
    _wait_for(lambda: not runner.is_busy, "new generation stayed busy")
    _drain(mode, runner, pending, events)

    document.assert_committed_once("new")
    assert ("old", "Agent operation cancelled") in outcomes
    assert ("new", "new") in outcomes
    runner.shutdown()
