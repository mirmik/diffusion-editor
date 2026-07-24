#!/usr/bin/env python3
"""Import the supported UI-process dependency set without enabling the GIL."""

from __future__ import annotations

import importlib
from importlib import metadata
import sys


IMPORTS = (
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
    ("sdl2", "PySDL2"),
    ("yaml", "PyYAML"),
    ("diffusion_editor.app.main", "diffusion-editor"),
)

WORKER_ONLY = (
    "accelerate",
    "diffusers",
    "safetensors",
    "tokenizers",
    "torch",
    "torchvision",
    "transformers",
)


def _gil_enabled() -> bool:
    probe = getattr(sys, "_is_gil_enabled", None)
    return bool(probe()) if probe is not None else True


def main() -> int:
    if _gil_enabled():
        raise RuntimeError("main-process dependency probe started with the GIL enabled")

    imported: list[str] = []
    for module_name, distribution_name in IMPORTS:
        importlib.import_module(module_name)
        if _gil_enabled():
            raise RuntimeError(f"importing {module_name} enabled the GIL")
        imported.append(f"{distribution_name}=={metadata.version(distribution_name)}")

    loaded = sorted(name for name in WORKER_ONLY if name in sys.modules)
    if loaded:
        raise RuntimeError(
            "worker-only modules leaked into the UI process: "
            + ", ".join(loaded)
        )
    installed = []
    for distribution_name in WORKER_ONLY:
        try:
            installed.append(
                f"{distribution_name}=={metadata.version(distribution_name)}"
            )
        except metadata.PackageNotFoundError:
            pass
    if installed:
        raise RuntimeError(
            "worker-only distributions are installed in the UI environment: "
            + ", ".join(installed)
        )

    print("Main-process imports verified with the GIL disabled:")
    for item in imported:
        print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
