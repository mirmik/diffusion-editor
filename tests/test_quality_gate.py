from __future__ import annotations

from pathlib import Path

import pytest

from diffusion_editor.quality_gate import (
    QualityGateError,
    verify_exact_lock,
    verify_installed_wheel_abis,
    verify_runtime_identity,
)


def test_current_test_runtime_is_canonical_cp314t_without_gil():
    soabi = verify_runtime_identity("pytest")
    assert soabi.startswith("cpython-314t-")


def test_exact_lock_reports_the_offending_line(tmp_path: Path):
    lock = tmp_path / "requirements.txt"
    lock.write_text("Pillow==12.3.0\nnumpy>=2\n", encoding="utf-8")

    with pytest.raises(
        QualityGateError,
        match=r"requirements\.txt:2: requirement is not an exact",
    ):
        verify_exact_lock(lock)


def test_exact_lock_rejects_recursive_or_index_options(tmp_path: Path):
    lock = tmp_path / "requirements.txt"
    lock.write_text("-r other.txt\n", encoding="utf-8")

    with pytest.raises(QualityGateError, match="not an exact distribution pin"):
        verify_exact_lock(lock)


class _Distribution:
    files = ("package/native.so",)

    def __init__(self, tag: str):
        self.metadata = {"Name": "native-package"}
        self._tag = tag

    def read_text(self, name: str):
        assert name == "WHEEL"
        return f"Wheel-Version: 1.0\nTag: {self._tag}\n"


def test_native_regular_cp314_wheel_tag_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "diffusion_editor.quality_gate.metadata.distributions",
        lambda: [_Distribution("cp314-cp314-linux_x86_64")],
    )

    with pytest.raises(
        QualityGateError,
        match=r"native-package: regular CPython wheel tag.*cp314-cp314",
    ):
        verify_installed_wheel_abis()


def test_native_cp314t_abi_wheel_tag_is_accepted(monkeypatch):
    monkeypatch.setattr(
        "diffusion_editor.quality_gate.metadata.distributions",
        lambda: [_Distribution("cp314-cp314t-linux_x86_64")],
    )

    checked = verify_installed_wheel_abis()
    assert checked == (
        "native-package: cp314-cp314t-linux_x86_64 (1 native file(s))",
    )
