"""Coordinate and camera math for canonical character point maps.

The canonical character frame is right handed: X points to the character's
right, Y points up, and Z points forward (towards the frontal camera).  Its
origin is the pelvis and one unit is the character's full height.

Camera matrices use the conventional computer-vision frame: X right, Y down,
Z forward.  Points are column vectors and transforms left-multiply them.
"""

from __future__ import annotations

from pathlib import Path
import struct
import zlib

import numpy as np


CANONICAL_FROM_BLENDER_AXES = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

CV_FROM_BLENDER_CAMERA = np.diag((1.0, -1.0, -1.0, 1.0))


def camera_distance_world(
    distance: float,
    *,
    character_height: float,
    unit: str,
) -> float:
    """Resolve an authored camera distance to Blender world units."""
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("camera distance must be positive and finite")
    if not np.isfinite(character_height) or character_height <= 0.0:
        raise ValueError("character_height must be positive and finite")
    if unit == "world":
        return float(distance)
    if unit == "character-height":
        return float(distance) * float(character_height)
    raise ValueError(f"unsupported camera distance unit: {unit}")


def normalized_camera_vector(
    *,
    camera_from_canonical: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int] | list[int],
) -> np.ndarray:
    """Encode the nominal camera in the 16-value form consumed by the head."""
    transform = np.asarray(camera_from_canonical, dtype=np.float32)
    calibration = np.asarray(intrinsics, dtype=np.float32)
    if transform.shape != (4, 4) or calibration.shape != (3, 3):
        raise ValueError("camera transform must be 4x4 and intrinsics 3x3")
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("image dimensions must be positive")
    width, height = image_size
    return np.concatenate(
        (
            transform[:3].reshape(-1),
            np.array(
                (
                    calibration[0, 0] / width,
                    calibration[1, 1] / height,
                    calibration[0, 2] / width,
                    calibration[1, 2] / height,
                ),
                dtype=np.float32,
            ),
        )
    )


def write_binary_mask_png(path: str | Path, mask: np.ndarray) -> None:
    """Write a dependency-free 8-bit grayscale PNG from a binary mask."""
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    pixels = np.ascontiguousarray((values != 0).astype(np.uint8) * 255)
    height, width = pixels.shape

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind)
        checksum = zlib.crc32(payload, checksum)
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", checksum & 0xFFFFFFFF)
        )

    scanlines = b"".join(b"\x00" + row.tobytes() for row in pixels)
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    encoded = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )
    Path(path).write_bytes(encoded)


def canonical_from_world(
    points: np.ndarray,
    *,
    pelvis_world: np.ndarray,
    character_height: float,
) -> np.ndarray:
    """Convert Blender world-space points to normalized canonical points."""
    values = np.asarray(points, dtype=np.float64)
    pelvis = np.asarray(pelvis_world, dtype=np.float64)
    if values.shape[-1] != 3 or pelvis.shape != (3,):
        raise ValueError("points and pelvis_world must end in three coordinates")
    if not np.isfinite(character_height) or character_height <= 0.0:
        raise ValueError("character_height must be positive and finite")
    return ((values - pelvis) @ CANONICAL_FROM_BLENDER_AXES.T) / character_height


def world_from_canonical_matrix(
    pelvis_world: np.ndarray,
    character_height: float,
) -> np.ndarray:
    """Return a homogeneous Blender-world-from-canonical transform."""
    pelvis = np.asarray(pelvis_world, dtype=np.float64)
    if pelvis.shape != (3,):
        raise ValueError("pelvis_world must have three coordinates")
    if not np.isfinite(character_height) or character_height <= 0.0:
        raise ValueError("character_height must be positive and finite")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        CANONICAL_FROM_BLENDER_AXES.T * float(character_height)
    )
    transform[:3, 3] = pelvis
    return transform


def camera_intrinsics(
    *,
    width: int,
    height: int,
    lens_mm: float,
    sensor_width_mm: float,
) -> np.ndarray:
    """Build centered pinhole intrinsics for square Blender pixels.

    The renderer deliberately uses horizontal sensor fit, unit pixel aspect,
    and zero camera shift.  Keeping those constraints explicit makes this
    compact formula exact rather than an approximate Blender calibration.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if lens_mm <= 0.0 or sensor_width_mm <= 0.0:
        raise ValueError("lens and sensor width must be positive")
    focal = float(lens_mm) * float(width) / float(sensor_width_mm)
    return np.array(
        [
            [focal, 0.0, width * 0.5],
            [0.0, focal, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def camera_from_canonical_matrix(
    *,
    world_from_camera_blender: np.ndarray,
    pelvis_world: np.ndarray,
    character_height: float,
) -> np.ndarray:
    """Return a CV-camera-from-canonical homogeneous transform."""
    world_from_camera = np.asarray(world_from_camera_blender, dtype=np.float64)
    if world_from_camera.shape != (4, 4):
        raise ValueError("world_from_camera_blender must be 4x4")
    camera_blender_from_world = np.linalg.inv(world_from_camera)
    return (
        CV_FROM_BLENDER_CAMERA
        @ camera_blender_from_world
        @ world_from_canonical_matrix(pelvis_world, character_height)
    )


def canonical_camera_ray_map(
    *,
    camera_from_canonical: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int] | list[int],
    grid_height: int,
    grid_width: int,
) -> np.ndarray:
    """Return canonical camera origin and unit ray direction per grid cell."""
    transform = np.asarray(camera_from_canonical, dtype=np.float64)
    calibration = np.asarray(intrinsics, dtype=np.float64)
    if transform.shape != (4, 4) or calibration.shape != (3, 3):
        raise ValueError("camera transform must be 4x4 and intrinsics 3x3")
    if len(image_size) != 2 or grid_height <= 0 or grid_width <= 0:
        raise ValueError("image and grid dimensions must be positive")
    image_width, image_height = image_size
    canonical_from_camera = np.linalg.inv(transform)
    origin = canonical_from_camera[:3, 3] / canonical_from_camera[3, 3]
    rows, columns = np.indices((grid_height, grid_width), dtype=np.float64)
    image_x = (columns + 0.5) * image_width / grid_width
    image_y = (rows + 0.5) * image_height / grid_height
    camera_directions = np.stack(
        (
            (image_x - calibration[0, 2]) / calibration[0, 0],
            (image_y - calibration[1, 2]) / calibration[1, 1],
            np.ones_like(image_x),
        ),
        axis=-1,
    )
    directions = camera_directions @ np.linalg.inv(transform[:3, :3]).T
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    origins = np.broadcast_to(origin, directions.shape)
    return np.concatenate((origins, directions), axis=-1).astype(np.float32)


def project_canonical_points(
    points: np.ndarray,
    *,
    intrinsics: np.ndarray,
    camera_from_canonical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project canonical points, returning image coordinates and CV depth."""
    values = np.asarray(points, dtype=np.float64)
    calibration = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(camera_from_canonical, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError("points must end in three coordinates")
    if calibration.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("intrinsics must be 3x3 and transform must be 4x4")
    flat = values.reshape(-1, 3)
    homogeneous = np.concatenate(
        (flat, np.ones((flat.shape[0], 1), dtype=np.float64)), axis=1
    )
    camera = (transform @ homogeneous.T).T[:, :3]
    projected = (calibration @ camera.T).T
    pixels = projected[:, :2] / projected[:, 2:3]
    return pixels.reshape(values.shape[:-1] + (2,)), camera[:, 2].reshape(
        values.shape[:-1]
    )


def pointmap_reprojection_error(
    canonical_xyz: np.ndarray,
    mask: np.ndarray,
    *,
    intrinsics: np.ndarray,
    camera_from_canonical: np.ndarray,
) -> np.ndarray:
    """Return per-hit distance from a point map to its source pixel center."""
    points = np.asarray(canonical_xyz)
    valid = np.asarray(mask, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError("canonical_xyz must have shape (height, width, 3)")
    if valid.shape != points.shape[:2]:
        raise ValueError("mask must match canonical_xyz image dimensions")
    projected, depth = project_canonical_points(
        points[valid],
        intrinsics=intrinsics,
        camera_from_canonical=camera_from_canonical,
    )
    rows, columns = np.nonzero(valid)
    expected = np.column_stack((columns + 0.5, rows + 0.5))
    if np.any(depth <= 0.0):
        raise ValueError("point map contains hits behind the camera")
    return np.linalg.norm(projected - expected, axis=1)
