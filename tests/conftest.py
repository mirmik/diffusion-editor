from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_user_home(tmp_path, monkeypatch):
    """Keep tests away from production settings, caches and recovery files."""
    test_home = tmp_path / "home"
    test_home.mkdir()
    monkeypatch.setenv("HOME", str(test_home))
