#!/usr/bin/env python3
"""Load a GLB into the production windowed reconstruction viewport."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np

from diffusion_editor.app.application import EditorApplication
from diffusion_editor.app.native_root import NativeEditorRoot


class _SmokeSettings:
    def __init__(self, recovery_dir: str) -> None:
        self._recovery_dir = recovery_dir

    def get(self, key, default=None):
        return self._recovery_dir if key == "recovery_dir" else default

    def set(self, _key, _value) -> None:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", type=Path)
    args = parser.parse_args()
    if not args.glb.is_file():
        raise FileNotFoundError(args.glb)

    with tempfile.TemporaryDirectory(
        prefix="diffusion-editor-reconstruction-smoke-"
    ) as recovery:
        application = EditorApplication(settings=_SmokeSettings(recovery))
        application.layer_stack.init_from_image(
            np.full((32, 32, 4), (70, 100, 150, 255), dtype=np.uint8)
        )
        with NativeEditorRoot.create_windowed(
            application, width=960, height=600
        ) as root:
            viewport = root._ensure_reconstruction_viewport()
            stats = viewport.load_glb(str(args.glb))
            for _ in range(3):
                root.composition.request_repaint()
                root.tick()
            if not viewport.viewport.texture_id:
                raise RuntimeError("reconstruction viewport published no texture")
            print(
                "Reconstruction viewport smoke OK: "
                f"vertices={stats[0]} triangles={stats[1]} meshes={stats[2]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
