#!/usr/bin/env python3
"""Open an ASCII or binary PLY in Diffusion Editor's native point renderer."""

from __future__ import annotations

import argparse
import faulthandler
from pathlib import Path
import time

from tcbase import Action
from termin.gui_native import KeyCode

from diffusion_editor.app.native_reconstruction_viewport import (
    NativeReconstructionViewport,
)
from diffusion_editor.app.native_root import WindowedNativeComposition
from diffusion_editor.generation.ply_sequence import (
    PlySequence,
    discover_ply_files,
)
from diffusion_editor.native_ply import load_ply_points


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "View one PLY or browse a directory with Termin's native "
            "orbit/pan/zoom reconstruction viewport."
        ),
        epilog=(
            "Directory controls: Left/Right or P/N browse, PageUp/PageDown "
            "browse, Home/End jump to the first/last file."
        ),
    )
    parser.add_argument("source", type=Path, help="PLY file or directory")
    parser.add_argument(
        "--glob",
        dest="pattern",
        help=(
            "Directory glob, for example '*makehuman.ply'. By default the "
            "viewer prefers mesh-like PLYs over joint/point companions."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search subdirectories recursively.",
    )
    parser.add_argument(
        "--fit-each",
        action="store_true",
        help="Refit the orbit camera after every file change.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
        help="Rendered point diameter in pixels.",
    )
    parser.add_argument(
        "--coordinates",
        choices=("canonical-y-up", "colmap-y-down", "termin-z-up"),
        default="canonical-y-up",
        help=(
            "Input coordinate convention. Canonical head outputs are Y-up; "
            "the Termin viewport is Z-up."
        ),
    )
    parser.add_argument(
        "--smoke-frames",
        type=int,
        default=0,
        help="Close after this many rendered frames; zero runs interactively.",
    )
    args = parser.parse_args()
    args.source = args.source.expanduser().resolve()
    if not args.source.exists():
        parser.error(f"source does not exist: {args.source}")
    if args.source.is_file() and args.source.suffix.casefold() != ".ply":
        parser.error(f"source file is not PLY: {args.source}")
    if args.width <= 0 or args.height <= 0:
        parser.error("window dimensions must be positive")
    if args.point_size <= 0.0:
        parser.error("--point-size must be positive")
    if args.smoke_frames < 0:
        parser.error("--smoke-frames must be non-negative")
    return args


def _positions_for_termin(
    positions, coordinate_space: str
):
    if coordinate_space == "termin-z-up":
        return positions
    if coordinate_space == "colmap-y-down":
        # COLMAP camera/world convention near the initialized front camera:
        # +X right, +Y image-down, +Z forward. Convert to Termin's Z-up frame.
        converted = positions.copy()
        converted[:, 0] = positions[:, 0]
        converted[:, 1] = positions[:, 2]
        converted[:, 2] = -positions[:, 1]
        return converted
    # The learned canonical frame is right-handed: +X character-right,
    # +Y up, +Z towards the frontal camera.  Termin is Z-up and its fitted
    # front camera looks from +Y.  The extra X sign keeps character-right on
    # screen-right for that camera while preserving a proper rotation.
    converted = positions.copy()
    converted[:, 0] = -positions[:, 0]
    converted[:, 1] = positions[:, 2]
    converted[:, 2] = positions[:, 1]
    return converted


def main() -> int:
    args = _arguments()
    try:
        sequence = PlySequence(discover_ply_files(
            args.source,
            pattern=args.pattern,
            recursive=args.recursive,
        ))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    faulthandler.enable()
    composition = WindowedNativeComposition(
        title=f"PLY viewer — {sequence.current.name}",
        width=args.width,
        height=args.height,
    )
    viewport = None
    try:
        document = composition.document
        root = document.create_vstack("CanonicalPointCloudViewerRoot")
        root.stable_id = "canonical-pointcloud-viewer.root"
        root.set_layout_spacing(0.0)
        viewport = NativeReconstructionViewport(
            document,
            graphics_owner=composition.graphics,
            request_repaint=composition.request_repaint,
            resource_namespace="canonical-pointcloud-viewer",
        )
        viewport.set_point_cloud_size(args.point_size)
        root.add_flex_child(viewport.widget, 1.0)
        if not document.add_root(root.handle):
            raise RuntimeError("failed to mount the point-cloud viewport")

        def load_current(*, fit_camera: bool) -> int:
            path = sequence.current
            cloud = load_ply_points(path)
            point_count = viewport.load_point_cloud_data(
                _positions_for_termin(cloud.positions, args.coordinates),
                cloud.colors,
                fit_camera=fit_camera,
            )
            prefix = (
                f"[{sequence.index + 1}/{len(sequence.paths)}] "
                if len(sequence.paths) > 1 else ""
            )
            controls = " — Left/Right or P/N" if len(sequence.paths) > 1 else ""
            composition.set_window_title(
                f"{prefix}{path.name} — {point_count:,} points{controls}"
            )
            composition.request_repaint()
            print(
                f"{sequence.index + 1}/{len(sequence.paths)}  "
                f"{path}  ({point_count:,} points)",
                flush=True,
            )
            return point_count

        load_current(fit_camera=True)

        previous_keys = {
            KeyCode.Left.value,
            KeyCode.PageUp.value,
            KeyCode.P.value,
            ord("["),
        }
        next_keys = {
            KeyCode.Right.value,
            KeyCode.PageDown.value,
            KeyCode.N.value,
            ord("]"),
        }

        def browse(key: int, modifiers: int) -> bool:
            if modifiers != 0 or len(sequence.paths) <= 1:
                return False
            old_index = sequence.index
            if key in previous_keys:
                sequence.step(-1)
            elif key in next_keys:
                sequence.step(1)
            elif key == KeyCode.Home.value:
                sequence.home()
            elif key == KeyCode.End.value:
                sequence.end()
            else:
                return False
            if sequence.index == old_index:
                return True
            try:
                load_current(fit_camera=args.fit_each)
            except Exception as exc:
                failed = sequence.current
                sequence.index = old_index
                print(f"Cannot load {failed}: {exc}", flush=True)
                composition.set_window_title(
                    f"Cannot load {failed.name} — use another file"
                )
                composition.request_repaint()
            return True

        def browse_focused_viewport(
            key: int,
            scancode: int,
            action: int,
            modifiers: int,
        ) -> bool:
            del scancode
            if action != int(Action.PRESS):
                return False
            return browse(key, modifiers)

        viewport.set_key_handler(browse_focused_viewport)
        composition.set_unhandled_key_handler(browse)

        rendered_frames = 0
        while not composition.should_close:
            events = composition.pump_events()
            viewport.render_if_dirty()
            rendered = composition.render_frame()
            rendered_frames += int(rendered)
            if args.smoke_frames and rendered_frames >= args.smoke_frames:
                break
            if args.smoke_frames:
                # Interactive windows repaint only when dirty.  Keep smoke
                # mode advancing so values greater than one terminate too.
                composition.request_repaint()
            if not rendered and events == 0:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        composition.set_unhandled_key_handler(None)
        if viewport is not None:
            viewport.set_key_handler(None)
            viewport.close()
        composition.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
