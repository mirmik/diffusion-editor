#!/usr/bin/env python3
"""Render a synthetic depth-derived cloud through the production viewport."""

from __future__ import annotations

import tempfile

import numpy as np

from diffusion_editor.app.application import EditorApplication
from diffusion_editor.app.native_root import NativeEditorRoot
from diffusion_editor.generation.depth_point_cloud import (
    project_depth_point_cloud,
)
from diffusion_editor.generation.depth_visualization import colorize_confidence


class _SmokeSettings:
    def __init__(self, recovery_dir: str) -> None:
        self._recovery_dir = recovery_dir

    def get(self, key, default=None):
        return self._recovery_dir if key == "recovery_dir" else default

    def set(self, _key, _value) -> None:
        pass


def main() -> int:
    size = 256
    yy, xx = np.mgrid[:size, :size]
    nx = (xx - size * 0.5) / (size * 0.42)
    nz = (yy - size * 0.5) / (size * 0.42)
    radius_squared = nx * nx + nz * nz
    mask = (radius_squared <= 1.0).astype(np.uint8) * 255
    proximity = np.sqrt(np.clip(1.0 - radius_squared, 0.0, 1.0))
    depth = (2.0 - proximity * 0.3).astype(np.float32)
    image = np.zeros((size, size, 4), dtype=np.uint8)
    image[:, :, 0] = np.clip(xx, 0, 255).astype(np.uint8)
    image[:, :, 1] = np.clip(255 - yy, 0, 255).astype(np.uint8)
    image[:, :, 2] = 180
    image[:, :, 3] = mask
    intrinsics = np.array([
        [300.0, 0.0, size * 0.5],
        [0.0, 300.0, size * 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    cloud = project_depth_point_cloud(
        image,
        depth,
        mask=mask,
        confidence=(1.0 + proximity * 3.0).astype(np.float32),
        intrinsics=intrinsics,
    )
    confidence_preview = colorize_confidence(cloud.confidence)
    confidence_colors = np.ascontiguousarray(
        confidence_preview.rgb.astype(np.float32) / 255.0)

    with tempfile.TemporaryDirectory(
        prefix="diffusion-editor-depth-cloud-smoke-"
    ) as recovery:
        application = EditorApplication(settings=_SmokeSettings(recovery))
        application.layer_stack.init_from_image(image)
        with NativeEditorRoot.create_windowed(
            application, width=960, height=600
        ) as root:
            viewport = root._ensure_reconstruction_viewport()
            point_count = viewport.load_point_cloud_data(
                cloud.positions,
                cloud.colors,
                confidence_colors=confidence_colors,
                color_mode="confidence",
                confidence_legend="Synthetic confidence",
            )
            viewport.set_point_cloud_color_mode("image")
            viewport.set_point_cloud_color_mode("confidence")
            if viewport.point_cloud_color_mode != "confidence":
                raise RuntimeError("confidence point colors did not activate")
            root.view.set_reconstruction_context(True, "Depth point cloud")
            for _ in range(5):
                root.composition.request_repaint()
                root.tick()
            if not viewport.viewport.texture_id:
                raise RuntimeError(
                    "depth point-cloud viewport published no texture"
                )
            print(
                "Depth point-cloud viewport smoke OK: "
                f"points={point_count} stride={cloud.stride}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
