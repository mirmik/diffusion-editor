"""Command-line entry point for Multiview Shape Studio."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .native_app import run_native


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="diffusion-editor-multiview")
    parser.add_argument(
        "project",
        nargs="?",
        help="optional *.mvstudio.json project to open",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_native(args.project)


if __name__ == "__main__":
    raise SystemExit(main())
