"""Model-free scaled-orthographic structure and camera factorization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScaledOrthographicCamera:
    rotation: np.ndarray
    scale: float
    translation: np.ndarray


@dataclass(frozen=True)
class CameraFactorizationResult:
    points: np.ndarray
    cameras: tuple[ScaledOrthographicCamera, ...]
    singular_values: np.ndarray
    cost: float | None = None
    optimality: float | None = None
    evaluations: int | None = None

    def project(self) -> np.ndarray:
        return np.stack([
            camera.scale * (camera.rotation[:2] @ self.points.T).T
            + camera.translation
            for camera in self.cameras
        ])


def _symmetric_terms(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.array([
        left[0] * right[0],
        left[1] * right[1],
        left[2] * right[2],
        left[0] * right[1] + left[1] * right[0],
        left[0] * right[2] + left[2] * right[0],
        left[1] * right[2] + left[2] * right[1],
    ], dtype=np.float64)


def _metric_upgrade(motion: np.ndarray) -> np.ndarray:
    constraints = []
    for view in range(motion.shape[0] // 2):
        first = motion[2 * view]
        second = motion[2 * view + 1]
        constraints.append(
            _symmetric_terms(first, first)
            - _symmetric_terms(second, second)
        )
        constraints.append(_symmetric_terms(first, second))
    _u, _s, vt = np.linalg.svd(np.stack(constraints), full_matrices=False)
    values = vt[-1]
    metric = np.array([
        [values[0], values[3], values[4]],
        [values[3], values[1], values[5]],
        [values[4], values[5], values[2]],
    ], dtype=np.float64)
    if np.trace(metric) < 0.0:
        metric = -metric
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    largest = max(float(np.max(eigenvalues)), 1.0e-12)
    eigenvalues = np.clip(eigenvalues, largest * 1.0e-8, None)
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def _proper_rotation(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first / max(float(np.linalg.norm(first)), 1.0e-12)
    second = second - first * float(np.dot(first, second))
    second = second / max(float(np.linalg.norm(second)), 1.0e-12)
    third = np.cross(first, second)
    return np.stack((first, second, third))


def _canonicalize(
        points: np.ndarray,
        cameras: list[ScaledOrthographicCamera],
        reference_view: int) -> tuple[np.ndarray, list[ScaledOrthographicCamera]]:
    reference = cameras[reference_view]
    points = (reference.rotation @ points.T).T * reference.scale
    points -= np.mean(points, axis=0, keepdims=True)
    canonical = []
    for camera in cameras:
        canonical.append(ScaledOrthographicCamera(
            rotation=camera.rotation @ reference.rotation.T,
            scale=float(camera.scale / reference.scale),
            translation=np.asarray(camera.translation, dtype=np.float64),
        ))
    canonical[reference_view] = ScaledOrthographicCamera(
        rotation=np.eye(3, dtype=np.float64),
        scale=1.0,
        translation=np.asarray(
            cameras[reference_view].translation, dtype=np.float64),
    )
    return points, canonical


def factor_scaled_orthographic(
        observations: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        reference_view: int = 0) -> CameraFactorizationResult:
    """Factor complete 2D tracks without using point semantics or topology."""
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim != 3 or observations.shape[2] != 2:
        raise ValueError("observations must have shape (views, points, 2)")
    views, points_count, _coordinates = observations.shape
    if views < 3 or points_count < 4:
        raise ValueError("factorization requires at least 3 views and 4 points")
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations must be finite and complete")
    if not 0 <= reference_view < views:
        raise ValueError("reference view is out of range")
    if weights is None:
        weights = np.ones((views, points_count), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (views, points_count):
            raise ValueError("weights must have shape (views, points)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must be finite and non-negative")
    totals = np.sum(weights, axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("every view must contain positive weight")
    centroids = np.sum(
        observations * weights[:, :, None], axis=1) / totals
    centered = observations - centroids[:, None, :]
    measurement = centered.transpose(0, 2, 1).reshape(views * 2, points_count)
    u, singular_values, vt = np.linalg.svd(measurement, full_matrices=False)
    if singular_values.size < 3 or singular_values[2] <= 1.0e-12:
        raise ValueError("2D tracks do not contain a rank-3 explanation")
    root = np.diag(np.sqrt(singular_values[:3]))
    motion = u[:, :3] @ root
    structure = root @ vt[:3]
    upgrade = _metric_upgrade(motion)
    upgraded_motion = motion @ upgrade
    upgraded_structure = np.linalg.solve(upgrade, structure).T

    cameras = []
    for view in range(views):
        first = upgraded_motion[2 * view]
        second = upgraded_motion[2 * view + 1]
        scale = 0.5 * (
            float(np.linalg.norm(first)) + float(np.linalg.norm(second)))
        cameras.append(ScaledOrthographicCamera(
            rotation=_proper_rotation(first, second),
            scale=scale,
            translation=centroids[view],
        ))
    upgraded_structure, cameras = _canonicalize(
        upgraded_structure, cameras, reference_view)
    return CameraFactorizationResult(
        points=np.ascontiguousarray(upgraded_structure),
        cameras=tuple(cameras),
        singular_values=np.ascontiguousarray(singular_values),
    )


def reflect_depth(
        result: CameraFactorizationResult) -> CameraFactorizationResult:
    """Return the equally valid metric solution mirrored through world Z."""
    reflection = np.diag((1.0, 1.0, -1.0))
    cameras = []
    for camera in result.cameras:
        rows = camera.rotation[:2] @ reflection
        cameras.append(ScaledOrthographicCamera(
            rotation=_proper_rotation(rows[0], rows[1]),
            scale=camera.scale,
            translation=camera.translation.copy(),
        ))
    return CameraFactorizationResult(
        points=np.ascontiguousarray((reflection @ result.points.T).T),
        cameras=tuple(cameras),
        singular_values=result.singular_values.copy(),
        cost=result.cost,
        optimality=result.optimality,
        evaluations=result.evaluations,
    )


def camera_yaw_pitch_roll(rotation: np.ndarray) -> tuple[float, float, float]:
    """Report the Rz(roll) Rx(pitch) Ry(yaw) decomposition in degrees."""
    rotation = np.asarray(rotation, dtype=np.float64)
    pitch = np.arcsin(np.clip(rotation[2, 1], -1.0, 1.0))
    yaw = np.arctan2(-rotation[2, 0], rotation[2, 2])
    base = _rotation_z(0.0) @ _rotation_x(pitch) @ _rotation_y(yaw)
    residual = rotation @ base.T
    roll = np.arctan2(residual[1, 0], residual[0, 0])
    return tuple(float(np.degrees(value)) for value in (yaw, pitch, roll))


def _rotation_x(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((1, 0, 0), (0, cosine, -sine), (0, sine, cosine)))


def _rotation_y(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((cosine, 0, sine), (0, 1, 0), (-sine, 0, cosine)))


def _rotation_z(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)))


def refine_scaled_orthographic(
        initial: CameraFactorizationResult,
        observations: np.ndarray,
        *,
        weights: np.ndarray | None = None,
        reference_view: int = 0,
        soft_l1_scale: float = 0.005,
        max_evaluations: int = 1000) -> CameraFactorizationResult:
    """Robust bundle adjustment of free points and weak-perspective cameras."""
    try:
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise RuntimeError(
            "robust camera refinement requires SciPy; use .venv-pose"
        ) from exc
    observations = np.asarray(observations, dtype=np.float64)
    views, points_count, _coordinates = observations.shape
    if len(initial.cameras) != views or initial.points.shape != (points_count, 3):
        raise ValueError("initial factorization does not match observations")
    if weights is None:
        weights = np.ones((views, points_count), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    visible = weights > 0.0
    weighted = np.sqrt(np.clip(weights, 0.0, 1.0))
    moving_views = [view for view in range(views) if view != reference_view]
    rotations = np.stack([camera.rotation for camera in initial.cameras])
    rotvecs = Rotation.from_matrix(rotations[moving_views]).as_rotvec()
    log_scales = np.log([
        max(initial.cameras[view].scale, 1.0e-8)
        for view in moving_views
    ])
    translations = np.stack([
        camera.translation for camera in initial.cameras])
    packed = np.concatenate((
        initial.points.reshape(-1),
        rotvecs.reshape(-1),
        log_scales,
        translations.reshape(-1),
    ))

    point_values = points_count * 3
    rotation_values = len(moving_views) * 3
    scale_values = len(moving_views)

    def unpack(values: np.ndarray):
        offset = 0
        points = values[offset:offset + point_values].reshape(points_count, 3)
        points = points - np.mean(points, axis=0, keepdims=True)
        offset += point_values
        moving_rotations = Rotation.from_rotvec(
            values[offset:offset + rotation_values].reshape(-1, 3)
        ).as_matrix()
        offset += rotation_values
        moving_scales = np.exp(values[offset:offset + scale_values])
        offset += scale_values
        translations = values[offset:].reshape(views, 2)
        rotations = np.repeat(np.eye(3)[None, :, :], views, axis=0)
        scales = np.ones(views, dtype=np.float64)
        rotations[moving_views] = moving_rotations
        scales[moving_views] = moving_scales
        return points, rotations, scales, translations

    def residual(values: np.ndarray) -> np.ndarray:
        points, rotations, scales, translations = unpack(values)
        predicted = np.stack([
            scales[view] * (rotations[view, :2] @ points.T).T
            + translations[view]
            for view in range(views)
        ])
        differences = (predicted - observations) * weighted[:, :, None]
        return differences[visible].reshape(-1)

    optimized = least_squares(
        residual,
        packed,
        loss="soft_l1",
        f_scale=float(soft_l1_scale),
        max_nfev=int(max_evaluations),
        x_scale="jac",
    )
    points, rotations, scales, translations = unpack(optimized.x)
    cameras = tuple(
        ScaledOrthographicCamera(
            rotation=np.ascontiguousarray(rotations[view]),
            scale=float(scales[view]),
            translation=np.ascontiguousarray(translations[view]),
        )
        for view in range(views)
    )
    return CameraFactorizationResult(
        points=np.ascontiguousarray(points),
        cameras=cameras,
        singular_values=initial.singular_values.copy(),
        cost=float(optimized.cost),
        optimality=float(optimized.optimality),
        evaluations=int(optimized.nfev),
    )
