#!/usr/bin/env python3
"""Load a GLB into the production windowed reconstruction viewport."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np

from diffusion_editor.app.application import EditorApplication
from diffusion_editor.app.native_reconstruction_viewport import (
    RECONSTRUCTION_SHADING_MODES,
)
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
    parser.add_argument(
        "--mask",
        action="store_true",
        help="also render a source-face mask and its brush cursor",
    )
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
            for mode in RECONSTRUCTION_SHADING_MODES:
                viewport.set_shading_mode(mode)
                for _ in range(3):
                    root.composition.request_repaint()
                    root.tick()
                if not viewport.viewport.texture_id:
                    raise RuntimeError(
                        "reconstruction viewport published no texture "
                        f"in {mode!r} shading mode"
                    )
            if args.mask:
                first_item = viewport._mesh_items[0]
                vertex_count = min(len(first_item.positions), 128)
                viewport.set_refine_vertex_mask((tuple(
                    (index, (index + 1) / vertex_count)
                    for index in range(vertex_count)
                ),))
                viewport.set_refine_mask_visible(True)
                viewport.set_refine_mask_edit_enabled(True)
                viewport._refine_mask_cursor = (480.0, 300.0)
                for mode in ("flat", "smooth", "wireframe"):
                    viewport.set_shading_mode(mode)
                    for _ in range(3):
                        root.composition.request_repaint()
                        root.tick()
                    if not viewport.viewport.texture_id:
                        raise RuntimeError(
                            "reconstruction viewport published no texture with "
                            f"mask overlay in {mode!r} mode"
                        )
            print(
                "Reconstruction viewport smoke OK: "
                f"vertices={stats[0]} triangles={stats[1]} meshes={stats[2]} "
                f"shading={','.join(RECONSTRUCTION_SHADING_MODES)}"
                f" mask={args.mask}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
