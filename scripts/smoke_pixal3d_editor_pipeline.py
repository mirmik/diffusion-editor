#!/usr/bin/env python3
"""Run the production composite -> Pixal3D -> Termin viewport path."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image

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
    parser.add_argument("image", type=Path)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--hold", type=float, default=2.0)
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    rgba = np.array(
        Image.open(args.image).convert("RGBA"), dtype=np.uint8, copy=True
    )
    with tempfile.TemporaryDirectory(
        prefix="diffusion-editor-pixal3d-smoke-"
    ) as recovery:
        application = EditorApplication(settings=_SmokeSettings(recovery))
        application.layer_stack.init_from_image(rgba)
        with NativeEditorRoot.create_windowed(
            application, width=1200, height=760
        ) as root:
            root._start_reconstruction()
            if not root.reconstruction_controller.is_busy:
                raise RuntimeError(application.status_text)
            deadline = time.monotonic() + args.timeout
            while root.reconstruction_controller.is_busy:
                if time.monotonic() >= deadline:
                    root._cancel_reconstruction()
                    raise TimeoutError("Pixal3D editor pipeline timed out")
                result = root.tick()
                if not result.rendered and not result.events and not result.dispatched:
                    time.sleep(0.01)

            root.composition.request_repaint()
            root.tick()
            viewport = root.reconstruction_viewport
            if (
                viewport is None
                or viewport.mesh_count == 0
                or not viewport.viewport.texture_id
            ):
                raise RuntimeError(application.status_text)
            print(
                "Pixal3D editor pipeline smoke OK: "
                f"{viewport.mesh_count} mesh(es); {application.status_text}"
            )
            hold_until = time.monotonic() + max(args.hold, 0.0)
            while time.monotonic() < hold_until:
                root.tick()
                time.sleep(0.01)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
