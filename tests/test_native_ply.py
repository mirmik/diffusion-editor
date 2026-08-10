from __future__ import annotations

import struct

import numpy as np

from diffusion_editor.native_ply import load_ply_points


def test_load_ascii_ply_points_with_default_colors(tmp_path) -> None:
    path = tmp_path / "points.ply"
    path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n0 1 2\n3 4 5\n",
        encoding="ascii",
    )

    cloud = load_ply_points(path)

    assert cloud.positions.dtype == np.float32
    assert cloud.positions.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert cloud.colors.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]


def test_load_binary_little_endian_ply_with_srgb_colors(tmp_path) -> None:
    path = tmp_path / "points.ply"
    header = (
        b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        b"end_header\n"
    )
    path.write_bytes(
        header
        + struct.pack("<fffBBB", 1.0, 2.0, 3.0, 255, 128, 0)
        + struct.pack("<fffBBB", -1.0, -2.0, -3.0, 0, 64, 255)
    )

    cloud = load_ply_points(path)

    assert cloud.positions.tolist() == [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]
    np.testing.assert_allclose(
        cloud.colors,
        np.asarray(
            [[1.0, 128.0 / 255.0, 0.0], [0.0, 64.0 / 255.0, 1.0]],
            dtype=np.float32,
        ),
    )
