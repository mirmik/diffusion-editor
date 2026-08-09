#!/usr/bin/env python3
"""Import the supported UI-process dependency set without enabling the GIL."""

from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path
import sys

from diffusion_editor.quality_gate import (
    WORKER_ONLY_DISTRIBUTIONS,
    verify_main_environment,
    verify_runtime_identity,
)

IMPORTS = (
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
    ("sdl2", "PySDL2"),
    ("yaml", "PyYAML"),
    ("termin.dispatch", "termin-dispatch"),
    ("termin.gui_native", "termin-gui-native"),
    ("termin.mcp", "termin-mcp"),
    ("diffusion_editor.app.main", "diffusion-editor"),
)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    wheels = verify_main_environment(project_root)

    imported: list[str] = []
    for module_name, distribution_name in IMPORTS:
        importlib.import_module(module_name)
        verify_runtime_identity(f"import {module_name}")
        imported.append(f"{distribution_name}=={metadata.version(distribution_name)}")

    loaded = sorted(
        name for name in WORKER_ONLY_DISTRIBUTIONS if name in sys.modules
    )
    if loaded:
        raise RuntimeError(
            "worker-only modules leaked into the UI process: "
            + ", ".join(loaded)
        )

    print("Main-process imports verified with the GIL disabled:")
    for item in imported:
        print(f"- {item}")
    print(f"Native wheel ABI metadata verified: {len(wheels)} distribution(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
