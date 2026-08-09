"""Command-line entry point for the native editor."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ..sdk_runtime import verify_application_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diffusion-editor")
    parser.add_argument(
        "path",
        nargs="?",
        help="optional .deproj project or image to open",
    )
    return parser


def _run_native(path: str | None) -> int:
    from .native_main import main as native_main

    return native_main(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verify_application_environment()
    return _run_native(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
