"""Geometry helpers for experimental local high-resolution 3D refinement."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .types import ReconstructionRefinePlacement


_PIXAL_TO_CAMERA_BASIS = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)

# Raw Pixal geometry is first exported with the runner's 180-degree Y
# correction and is then converted by Termin from glTF Y-up to engine Z-up.
# Keep the resulting proper rotation explicit so placement authored in the
# viewport can be conjugated back into Pixal's canonical coordinates.
_ENGINE_FROM_PIXAL_PREVIEW = np.asarray(
    ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)


def quaternion_from_euler_degrees(
        x_degrees: float,
        y_degrees: float,
        z_degrees: float,
) -> tuple[float, float, float, float]:
    """Build an XYZ quaternion for the numeric local-fragment controls."""
    x, y, z = map(
        lambda value: math.radians(float(value)) * 0.5,
        (x_degrees, y_degrees, z_degrees),
    )
    sx, cx = math.sin(x), math.cos(x)
    sy, cy = math.sin(y), math.cos(y)
    sz, cz = math.sin(z), math.cos(z)
    return ReconstructionRefinePlacement(
        orientation=(
            sx * cy * cz + cx * sy * sz,
            cx * sy * cz - sx * cy * sz,
            cx * cy * sz + sx * sy * cz,
            cx * cy * cz - sx * sy * sz,
        )
    ).orientation


def euler_degrees_from_quaternion(
        orientation: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Return stable XYZ angles for the placement inspector."""
    x, y, z, w = ReconstructionRefinePlacement(
        orientation=orientation
    ).orientation
    sin_x_cos_y = 2.0 * (w * x - y * z)
    cos_x_cos_y = 1.0 - 2.0 * (x * x + y * y)
    x_angle = math.atan2(sin_x_cos_y, cos_x_cos_y)
    sin_y = 2.0 * (w * y + z * x)
    y_angle = math.asin(max(-1.0, min(1.0, sin_y)))
    sin_z_cos_y = 2.0 * (w * z - x * y)
    cos_z_cos_y = 1.0 - 2.0 * (y * y + z * z)
    z_angle = math.atan2(sin_z_cos_y, cos_z_cos_y)
    return tuple(map(math.degrees, (x_angle, y_angle, z_angle)))


def _quaternion_rotation_matrix(
        orientation: tuple[float, float, float, float],
) -> np.ndarray:
    x, y, z, w = ReconstructionRefinePlacement(
        orientation=orientation
    ).orientation
    return np.asarray((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)),
    ), dtype=np.float64)


def refine_placement_matrix(
        placement: ReconstructionRefinePlacement,
        pivot: tuple[float, float, float],
) -> np.ndarray:
    """Return the engine-space pivoted delta used by the live viewport."""
    pivot_value = np.asarray(pivot, dtype=np.float64)
    if pivot_value.shape != (3,) or not np.all(np.isfinite(pivot_value)):
        raise ValueError("refine placement pivot must contain three finite values")
    rotation = _quaternion_rotation_matrix(placement.orientation)
    linear = rotation * placement.scale
    translation = (
        np.asarray(placement.translation, dtype=np.float64)
        + pivot_value
        - linear @ pivot_value
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = linear
    result[:3, 3] = translation
    return result


def pixal_preview_point_to_engine(point) -> np.ndarray:
    value = np.asarray(point, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError("point must contain three components")
    return _ENGINE_FROM_PIXAL_PREVIEW @ value


def refine_placement_pivot_from_checkpoint(
        path: str | Path,
) -> tuple[float, float, float]:
    """Read the registered fragment center and expose it in engine Z-up."""
    with np.load(Path(path), allow_pickle=False) as saved:
        if str(saved.get("composite_kind", np.asarray("")).item()) != (
                "enlarged_hr_geometry_v1"):
            raise ValueError("checkpoint is not an enlarged HR refine proposal")
        bounds = np.asarray(saved["registration_bounds"], dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError("refine checkpoint has invalid registration bounds")
    return tuple(map(float, pixal_preview_point_to_engine(bounds.mean(axis=0))))


def apply_engine_refine_placement_to_pixal_points(
        points: np.ndarray,
        placement: ReconstructionRefinePlacement,
        pivot_pixal: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Apply a Z-up viewport delta to registered Pixal-space vertices."""
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    pivot = np.asarray(pivot_pixal, dtype=np.float64)
    pivot_engine = pixal_preview_point_to_engine(pivot)
    engine_matrix = refine_placement_matrix(placement, tuple(pivot_engine))
    basis = _ENGINE_FROM_PIXAL_PREVIEW
    pixal_linear = basis.T @ engine_matrix[:3, :3] @ basis
    pixal_translation = basis.T @ engine_matrix[:3, 3]
    transformed = (
        values.astype(np.float64, copy=False) @ pixal_linear.T
        + pixal_translation
    )
    return transformed.astype(values.dtype, copy=False)


@dataclass(frozen=True)
class LocalDetailTransform:
    """Uniform local-canonical to global-canonical affine transform."""

    scale: float
    translation: tuple[float, float, float]

    def apply(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points)
        offset = np.asarray(self.translation, dtype=values.dtype)
        return values * self.scale + offset

    def inverse(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points)
        offset = np.asarray(self.translation, dtype=values.dtype)
        return (values - offset) / self.scale


def project_pixal_points(
        points: np.ndarray,
        *,
        camera_angle_x: float,
        distance: float,
        image_size: tuple[int, int],
        mesh_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project raw Pixal3D canonical points like its projection conditioner."""
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if mesh_scale <= 0.0:
        raise ValueError("mesh_scale must be positive")

    aligned = values @ _PIXAL_TO_CAMERA_BASIS.T / mesh_scale
    transform = np.asarray((
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, -float(distance)),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ), dtype=np.float64)
    homogeneous = np.concatenate(
        (aligned, np.ones((len(aligned), 1), dtype=np.float64)), axis=1
    )
    camera = homogeneous @ np.linalg.inv(transform).T
    x_camera, y_camera, z_camera = camera[:, :3].T
    depth = -z_camera
    focal_pixels = (
        16.0 / np.tan(float(camera_angle_x) / 2.0) * width / 32.0
    )
    divisor = -z_camera + 1e-8
    pixels = np.stack((
        focal_pixels * x_camera / divisor + width / 2.0,
        -focal_pixels * y_camera / divisor + height / 2.0,
    ), axis=1)
    valid = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < height)
        & (depth > 0.0)
    )
    return pixels, depth, valid


def sample_mask_bilinear(mask: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Sample a normalized or uint8 image mask at pixel-center coordinates."""
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    points = np.asarray(pixels, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pixels must have shape [N, 2]")
    height, width = values.shape
    if width == 0 or height == 0:
        return np.zeros((len(points),), dtype=np.float32)

    floating = values.astype(np.float32)
    if np.issubdtype(values.dtype, np.integer):
        floating /= np.iinfo(values.dtype).max
    x = points[:, 0] - 0.5
    y = points[:, 1] - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    valid = (x1 >= 0) & (x0 < width) & (y1 >= 0) & (y0 < height)
    x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
    y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)
    sampled = (
        floating[y0c, x0c] * (1.0 - wx) * (1.0 - wy)
        + floating[y0c, x1c] * wx * (1.0 - wy)
        + floating[y1c, x0c] * (1.0 - wx) * wy
        + floating[y1c, x1c] * wx * wy
    )
    return np.where(valid, sampled, 0.0).astype(np.float32)


def local_roi_bounds(
        points: np.ndarray,
        weights: np.ndarray,
        *,
        threshold: float = 0.05,
        quantile: float = 0.01,
        padding: float = 0.15,
        minimum_extent: float = 1e-3,
) -> np.ndarray:
    """Return robust padded global bounds for points selected by a soft mask."""
    values = np.asarray(points, dtype=np.float64)
    mask_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if len(mask_weights) != len(values):
        raise ValueError("weights must match points")
    if not 0.0 <= quantile < 0.5:
        raise ValueError("quantile must be in [0, 0.5)")
    selected = values[mask_weights > threshold]
    if len(selected) < 3:
        raise ValueError("refine mask selects too little 3D geometry")
    lower = np.quantile(selected, quantile, axis=0)
    upper = np.quantile(selected, 1.0 - quantile, axis=0)
    center = (lower + upper) * 0.5
    extent = np.maximum(upper - lower, minimum_extent) * (1.0 + padding * 2.0)
    return np.stack((center - extent * 0.5, center + extent * 0.5))


def fit_local_detail_transform(
        local_points: np.ndarray,
        target_bounds: np.ndarray,
        *,
        quantile: float = 0.01,
) -> LocalDetailTransform:
    """Fit one uniform local-to-global transform using robust bounding cubes."""
    local = np.asarray(local_points, dtype=np.float64)
    target = np.asarray(target_bounds, dtype=np.float64)
    if local.ndim != 2 or local.shape[1] != 3 or len(local) < 3:
        raise ValueError("local_points must contain at least three 3D points")
    if target.shape != (2, 3) or np.any(target[1] <= target[0]):
        raise ValueError("target_bounds must have shape [2, 3] and positive size")
    lower = np.quantile(local, quantile, axis=0)
    upper = np.quantile(local, 1.0 - quantile, axis=0)
    local_extent = upper - lower
    if np.any(local_extent <= 1e-8):
        raise ValueError("local geometry is degenerate")
    target_extent = target[1] - target[0]
    scale = float(target_extent.max() / local_extent.max())
    local_center = (lower + upper) * 0.5
    target_center = (target[0] + target[1]) * 0.5
    translation = target_center - local_center * scale
    return LocalDetailTransform(scale, tuple(map(float, translation)))


def overlap_face_masks(
        base_vertices: np.ndarray,
        base_faces: np.ndarray,
        local_vertices: np.ndarray,
        local_faces: np.ndarray,
        target_bounds: np.ndarray,
        *,
        base_inner_radius: float = 0.72,
        local_outer_radius: float = 1.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose overlapping base/local faces around an ellipsoidal ROI collar."""
    bounds = np.asarray(target_bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    radii = np.maximum((bounds[1] - bounds[0]) * 0.5, 1e-8)

    def radii_for(vertices, faces):
        points = np.asarray(vertices, dtype=np.float64)
        indices = np.asarray(faces, dtype=np.int64)
        centroids = points[indices].mean(axis=1)
        return np.linalg.norm((centroids - center) / radii, axis=1)

    base_radius = radii_for(base_vertices, base_faces)
    local_radius = radii_for(local_vertices, local_faces)
    return base_radius >= base_inner_radius, local_radius <= local_outer_radius


def compose_local_detail_mesh(
        base_vertices: np.ndarray,
        base_faces: np.ndarray,
        local_vertices: np.ndarray,
        local_faces: np.ndarray,
        target_bounds: np.ndarray,
        *,
        base_inner_radius: float = 0.72,
        local_outer_radius: float = 1.08,
        include_counts: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, int, int]:
    """Build the honest base-plus-local preview used before final fusion."""
    base = np.asarray(base_vertices)
    local = np.asarray(local_vertices)
    base_indices = np.asarray(base_faces)
    local_indices = np.asarray(local_faces)
    keep_base, keep_local = overlap_face_masks(
        base,
        base_indices,
        local,
        local_indices,
        target_bounds,
        base_inner_radius=base_inner_radius,
        local_outer_radius=local_outer_radius,
    )
    vertices = np.concatenate((base, local), axis=0)
    faces = np.concatenate((
        base_indices[keep_base],
        local_indices[keep_local] + len(base),
    ), axis=0)
    if include_counts:
        return (
            vertices,
            faces,
            int(keep_base.sum()),
            int(keep_local.sum()),
        )
    return vertices, faces
