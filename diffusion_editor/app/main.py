"""Command-line dispatcher for the legacy and native editor hosts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ..sdk_runtime import verify_application_environment


UI_CHOICES = ("legacy", "native")
DEFAULT_UI = "legacy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diffusion-editor")
    parser.add_argument(
        "--ui",
        choices=UI_CHOICES,
        default=DEFAULT_UI,
        help="UI implementation to launch (default: legacy)",
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="optional .deproj project or image to open",
    )
    return parser


def _run_legacy(path: str | None) -> int:
    from .legacy_main import main as legacy_main

    return legacy_main(path)


def _run_native(path: str | None) -> int:
    from .native_main import main as native_main

    return native_main(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verify_application_environment()
    if args.ui == "native":
        return _run_native(args.path)
    return _run_legacy(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
