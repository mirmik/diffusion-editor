"""Geometry helpers for experimental local high-resolution 3D refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_PIXAL_TO_CAMERA_BASIS = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)


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


def merge_local_lr_latent(
        base_coords: np.ndarray,
        base_feats: np.ndarray,
        local_coords: np.ndarray,
        local_feats: np.ndarray,
        transform: LocalDetailTransform,
        target_bounds: np.ndarray,
        *,
        grid_resolution: int = 32,
        base_inner_radius: float = 0.72,
        local_outer_radius: float = 1.08,
        blend_strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compress a full-grid local LR latent into a global LR checkpoint.

    Pixal3D LR coordinates live on the sparse 32³ grid while decoded geometry
    lives in the canonical ``[-0.5, 0.5]`` cube.  The registered local latent
    is transformed through that canonical space, quantized back onto the base
    grid and blended through an ellipsoidal overlap collar.
    """
    base_xyz = np.asarray(base_coords)
    local_xyz = np.asarray(local_coords)
    base_values = np.asarray(base_feats)
    local_values = np.asarray(local_feats)
    bounds = np.asarray(target_bounds, dtype=np.float64)
    if base_xyz.ndim != 2 or base_xyz.shape[1] != 4:
        raise ValueError("base_coords must have shape [N, 4]")
    if local_xyz.ndim != 2 or local_xyz.shape[1] != 4:
        raise ValueError("local_coords must have shape [N, 4]")
    if base_values.ndim != 2 or len(base_values) != len(base_xyz):
        raise ValueError("base_feats must have shape [N, C]")
    if (
            local_values.ndim != 2
            or len(local_values) != len(local_xyz)
            or local_values.shape[1] != base_values.shape[1]
    ):
        raise ValueError("local_feats must match base feature width")
    if bounds.shape != (2, 3) or np.any(bounds[1] <= bounds[0]):
        raise ValueError("target_bounds must have shape [2, 3]")
    if grid_resolution < 2:
        raise ValueError("grid_resolution must be at least two")
    if not 0.0 < base_inner_radius < local_outer_radius:
        raise ValueError("overlap radii must be positive and ordered")
    if not 0.0 < blend_strength <= 1.0:
        raise ValueError("blend_strength must be in (0, 1]")

    denominator = float(grid_resolution - 1)
    center = bounds.mean(axis=0)
    radii = np.maximum((bounds[1] - bounds[0]) * 0.5, 1e-8)
    base_points = base_xyz[:, 1:].astype(np.float64) / denominator - 0.5
    local_points = local_xyz[:, 1:].astype(np.float64) / denominator - 0.5
    registered = transform.apply(local_points)
    base_radius = np.linalg.norm((base_points - center) / radii, axis=1)
    local_radius = np.linalg.norm((registered - center) / radii, axis=1)
    mapped = np.rint((registered + 0.5) * denominator).astype(np.int64)

    # Keep the base outside the replacement core.  Ordered dictionaries are
    # unnecessary here: sorting the final integer coordinates makes checkpoint
    # output deterministic across NumPy versions.
    values: dict[tuple[int, int, int, int], tuple[np.ndarray, float]] = {}
    for coord, feat, radius in zip(base_xyz, base_values, base_radius):
        if radius >= base_inner_radius:
            values[tuple(map(int, coord))] = (
                np.asarray(feat, dtype=np.float64).copy(), 0.0
            )

    # Several local tokens can collapse onto one global token.  Accumulate
    # their features before applying the collar blend.
    accumulated: dict[tuple[int, int, int, int], tuple[np.ndarray, int, float]] = {}
    for batch, xyz, feat, radius in zip(
            local_xyz[:, 0], mapped, local_values, local_radius):
        if not (np.all((xyz >= 0) & (xyz < grid_resolution))
                and radius <= local_outer_radius):
            continue
        key = (int(batch), int(xyz[0]), int(xyz[1]), int(xyz[2]))
        if key in accumulated:
            total, count, nearest_radius = accumulated[key]
            accumulated[key] = (total + feat, count + 1,
                                 min(nearest_radius, float(radius)))
        else:
            accumulated[key] = (
                np.asarray(feat, dtype=np.float64).copy(), 1, float(radius)
            )

    collar_width = local_outer_radius - base_inner_radius
    for key, (total, count, radius) in accumulated.items():
        local_feat = total / count
        collar = np.clip(
            (local_outer_radius - radius) / collar_width, 0.0, 1.0
        )
        local_weight = float(collar * blend_strength)
        if key in values:
            base_feat, _ = values[key]
            values[key] = (
                base_feat * (1.0 - local_weight)
                + local_feat * local_weight,
                local_weight,
            )
        elif local_weight > 0.0:
            values[key] = (local_feat, local_weight)

    if not values:
        raise ValueError("local LR merge produced an empty sparse latent")
    ordered = sorted(values)
    merged_coords = np.asarray(ordered, dtype=base_xyz.dtype)
    merged_feats = np.stack([values[key][0] for key in ordered]).astype(
        base_values.dtype, copy=False
    )
    return merged_coords, merged_feats
