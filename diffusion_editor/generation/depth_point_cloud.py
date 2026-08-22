"""Project monocular depth and source colors into a preview point cloud."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .types import DepthValueKind


DEFAULT_DEPTH_CLOUD_MAX_POINTS: int | None = None
DEFAULT_DEPTH_CLOUD_MASK_THRESHOLD = 0.5


@dataclass(frozen=True)
class DepthPointCloudData:
    positions: np.ndarray
    colors: np.ndarray
    source_size: tuple[int, int]
    stride: int
    depth_min: float
    depth_max: float
    value_kind: DepthValueKind
    intrinsics: np.ndarray
    approximate_camera: bool
    confidence: np.ndarray | None = None

    @property
    def point_count(self) -> int:
        return int(self.positions.shape[0])


def project_depth_point_cloud(
        image: np.ndarray,
        depth: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        confidence: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        value_kind: DepthValueKind = (
            DepthValueKind.DIRECT_SCALE_AMBIGUOUS),
        fallback_fov_y_degrees: float | None = None,
        max_points: int | None = DEFAULT_DEPTH_CLOUD_MAX_POINTS,
        mask_threshold: float = DEFAULT_DEPTH_CLOUD_MASK_THRESHOLD,
) -> DepthPointCloudData:
    """Unproject canonical model depth without affine depth remapping.

    The 2D mask may remain soft for antialiased previews. Point geometry is
    deliberately binary: confidence below ``mask_threshold`` is discarded so
    low-alpha model/Lanczos fringe does not become fully opaque 3D points.

    Direct depth is used exactly as returned by the model. Inverse-relative
    output is only reciprocated; its arbitrary scale is preserved instead of
    being normalized. It remains explicitly approximate when no model camera
    calibration is available.
    """

    source = np.asarray(image)
    depth_values = np.asarray(depth)
    if source.ndim != 3 or source.shape[2] not in (3, 4):
        raise ValueError("depth point-cloud image must be RGB or RGBA")
    if depth_values.ndim != 2 or depth_values.shape != source.shape[:2]:
        raise ValueError("depth point-cloud depth shape does not match image")
    if not np.issubdtype(source.dtype, np.number):
        raise ValueError("depth point-cloud image must be numeric")
    if not np.issubdtype(depth_values.dtype, np.number):
        raise ValueError("depth point-cloud depth must be numeric")
    source = source.astype(np.float32, copy=False)
    depth_values = depth_values.astype(np.float32, copy=False)
    if not np.isfinite(source).all() or not np.isfinite(depth_values).all():
        raise ValueError("depth point-cloud inputs must be finite")
    try:
        value_kind = DepthValueKind(value_kind)
    except ValueError as exc:
        raise ValueError("depth point-cloud value kind is invalid") from exc
    if max_points is not None:
        if isinstance(max_points, bool) or int(max_points) < 1:
            raise ValueError("depth point-cloud max_points must be positive")
        max_points = int(max_points)
    if (
            not math.isfinite(float(mask_threshold))
            or not 0.0 < float(mask_threshold) <= 1.0):
        raise ValueError("depth point-cloud mask threshold is invalid")
    mask_threshold = float(mask_threshold)

    confidence_values = None
    if confidence is not None:
        confidence_values = np.asarray(confidence)
        if confidence_values.shape != depth_values.shape:
            raise ValueError(
                "depth point-cloud confidence shape does not match depth")
        if not np.issubdtype(confidence_values.dtype, np.number):
            raise ValueError("depth point-cloud confidence must be numeric")
        confidence_values = confidence_values.astype(np.float32, copy=False)
        if not np.isfinite(confidence_values).all():
            raise ValueError("depth point-cloud confidence must be finite")

    if mask is None:
        mask_values = (
            source[:, :, 3]
            if source.shape[2] == 4
            else np.ones(depth_values.shape, dtype=np.float32)
        )
    else:
        mask_values = np.asarray(mask)
        if mask_values.shape != depth_values.shape:
            raise ValueError("depth point-cloud mask shape does not match depth")
        try:
            mask_values = mask_values.astype(np.float32, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("depth point-cloud mask must be numeric") from exc
        if not np.isfinite(mask_values).all():
            raise ValueError("depth point-cloud mask must be finite")
    if mask_values.size and float(mask_values.max()) > 1.0:
        mask_values = mask_values / 255.0
    mask_values = np.clip(mask_values, 0.0, 1.0)
    valid = (mask_values >= mask_threshold) & (depth_values > 0.0)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise ValueError("depth point-cloud mask contains no foreground")

    depth_min = float(depth_values[valid].min())
    depth_max = float(depth_values[valid].max())

    stride = (
        1
        if max_points is None
        else max(1, int(math.ceil(math.sqrt(valid_count / max_points))))
    )
    sampled_valid = valid[::stride, ::stride]
    while (
            max_points is not None
            and int(np.count_nonzero(sampled_valid)) > max_points):
        stride += 1
        sampled_valid = valid[::stride, ::stride]
    sampled_y, sampled_x = np.nonzero(sampled_valid)
    sampled_y = sampled_y * stride
    sampled_x = sampled_x * stride

    height, width = depth_values.shape
    approximate_camera = intrinsics is None
    if intrinsics is None:
        if fallback_fov_y_degrees is None:
            raise ValueError(
                "depth point-cloud projection requires camera intrinsics")
        fov = float(fallback_fov_y_degrees)
        if not math.isfinite(fov) or not 1.0 < fov < 179.0:
            raise ValueError("depth point-cloud fallback field of view is invalid")
        focal = 0.5 * float(height) / math.tan(math.radians(fov) * 0.5)
        camera_matrix = np.array([
            [focal, 0.0, width * 0.5],
            [0.0, focal, height * 0.5],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
    else:
        camera_matrix = np.asarray(intrinsics, dtype=np.float32)
        if (
                camera_matrix.shape != (3, 3)
                or not np.isfinite(camera_matrix).all()
                or camera_matrix[0, 0] <= 0.0
                or camera_matrix[1, 1] <= 0.0):
            raise ValueError("depth point-cloud intrinsics are invalid")
        camera_matrix = np.ascontiguousarray(camera_matrix)
    try:
        inverse_camera = np.linalg.inv(camera_matrix.astype(np.float64))
    except np.linalg.LinAlgError as exc:
        raise ValueError("depth point-cloud intrinsics are singular") from exc

    samples = depth_values[sampled_y, sampled_x].astype(np.float64)
    if value_kind is DepthValueKind.INVERSE_RELATIVE:
        camera_depth = 1.0 / samples
    else:
        camera_depth = samples
    pixels = np.column_stack((
        sampled_x.astype(np.float64),
        sampled_y.astype(np.float64),
        np.ones(sampled_x.shape, dtype=np.float64),
    ))
    camera_points = (pixels @ inverse_camera.T) * camera_depth[:, None]
    # Termin convention: X right, Y forward from the camera, Z up. This is a
    # rigid axis conversion of the model camera coordinates, not a rescale.
    positions = np.column_stack((
        camera_points[:, 0],
        -camera_points[:, 2],
        -camera_points[:, 1],
    ))
    colors = source[sampled_y, sampled_x, :3]
    if colors.size and float(colors.max()) > 1.0:
        colors = colors / 255.0
    colors = np.clip(colors, 0.0, 1.0)
    point_confidence = (
        np.ascontiguousarray(
            confidence_values[sampled_y, sampled_x], dtype=np.float32)
        if confidence_values is not None else None
    )
    return DepthPointCloudData(
        positions=np.ascontiguousarray(positions, dtype=np.float64),
        colors=np.ascontiguousarray(colors, dtype=np.float32),
        source_size=(width, height),
        stride=stride,
        depth_min=depth_min,
        depth_max=depth_max,
        value_kind=value_kind,
        intrinsics=camera_matrix,
        approximate_camera=approximate_camera,
        confidence=point_confidence,
    )
