#!/usr/bin/env python3
"""Run the parallel termin-gui-native root for a bounded number of frames."""

from __future__ import annotations

import argparse

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.native_root import NativeEditorRoot


class _MemorySettings:
    def get(self, _key, default=None):
        return default

    def set(self, _key, _value):
        pass


class _IdleEngine:
    model_info = {}

    def poll_event(self):
        return None

    def shutdown(self):
        pass


def _application() -> EditorApplication:
    engine = _IdleEngine()
    return EditorApplication(
        settings=_MemorySettings(),
        engines=EngineSet(engine, engine, engine, engine, engine),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("opengl", "vulkan"),
        default="vulkan",
        help="headless composition backend; windowed mode uses TERMIN_BACKEND",
    )
    args = parser.parse_args()
    if args.frames <= 0:
        parser.error("--frames must be positive")

    application = _application()
    factory = (
        NativeEditorRoot.create_windowed
        if args.windowed
        else NativeEditorRoot.create_headless
    )
    options = {} if args.windowed else {"backend": args.backend}
    with factory(application, width=160, height=100, **options) as root:
        root.defer(lambda: application.set_status("Native dispatcher ready"))
        rendered = 0
        for _ in range(args.frames):
            result = root.tick()
            rendered += int(result.rendered)
            if not application.running:
                break

    if rendered == 0:
        raise RuntimeError("native root did not render a frame")
    print(
        f"Diffusion Editor native root smoke OK: "
        f"{rendered} frame(s), {'windowed' if args.windowed else args.backend}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
