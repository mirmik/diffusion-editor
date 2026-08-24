from __future__ import annotations

from pathlib import Path
import runpy


MODULE = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "scripts"
        / "qwen_view_names.py"
    )
)


def test_view_pattern_accepts_atomic_and_legacy_batch_names() -> None:
    pattern = MODULE["VIEW_PATTERN"]
    assert pattern.match("mv-eye-315.png").groups() == ("eye", "315")
    assert pattern.match(
        "eye-315-multiple-angles-on.png"
    ).groups() == ("eye", "315")
    assert pattern.match("contact-eye.png") is None
