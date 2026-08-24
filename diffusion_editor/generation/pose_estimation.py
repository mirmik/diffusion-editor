"""Toolkit-neutral pose-estimation result and overlay rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class PoseEstimatorProfile:
    stable_id: str
    title: str
    layer_name: str
    keypoint_schema: str
    neural: bool


POSE_ESTIMATOR_PROFILES = (
    PoseEstimatorProfile(
        stable_id="dwpose",
        title="DWPose WholeBody (ONNX)",
        layer_name="Pose — DWPose",
        keypoint_schema="coco_wholebody_133",
        neural=True,
    ),
    PoseEstimatorProfile(
        stable_id="mediapipe-full",
        title="MediaPipe Pose Full",
        layer_name="Pose — MediaPipe",
        keypoint_schema="mediapipe_pose_33",
        neural=True,
    ),
    PoseEstimatorProfile(
        stable_id="silhouette-skeleton",
        title="Silhouette Skeleton (diagnostic)",
        layer_name="Pose — Silhouette Skeleton",
        keypoint_schema="silhouette_graph",
        neural=False,
    ),
)


def pose_estimator_profile(stable_id: str) -> PoseEstimatorProfile:
    for profile in POSE_ESTIMATOR_PROFILES:
        if profile.stable_id == stable_id:
            return profile
    raise ValueError(f"Unknown pose estimator profile: {stable_id}")


@dataclass(frozen=True)
class PoseKeypoint:
    name: str
    x: float
    y: float
    score: float
    z: float | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PoseKeypoint":
        return cls(
            name=str(value["name"]),
            x=float(value["x"]),
            y=float(value["y"]),
            score=float(value["score"]),
            z=(None if value.get("z") is None else float(value["z"])),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "score": self.score,
        }
        if self.z is not None:
            result["z"] = self.z
        return result


@dataclass(frozen=True)
class PoseConnection:
    start: str
    end: str
    group: str = "body"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PoseConnection":
        return cls(
            start=str(value["start"]),
            end=str(value["end"]),
            group=str(value.get("group", "body")),
        )

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start, "end": self.end, "group": self.group}


@dataclass(frozen=True)
class PoseSegment:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float = 1.0
    group: str = "diagnostic"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PoseSegment":
        return cls(
            x0=float(value["x0"]),
            y0=float(value["y0"]),
            x1=float(value["x1"]),
            y1=float(value["y1"]),
            score=float(value.get("score", 1.0)),
            group=str(value.get("group", "diagnostic")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x0": self.x0,
            "y0": self.y0,
            "x1": self.x1,
            "y1": self.y1,
            "score": self.score,
            "group": self.group,
        }


@dataclass(frozen=True)
class PoseInstance:
    keypoints: tuple[PoseKeypoint, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PoseInstance":
        points = value.get("keypoints")
        if not isinstance(points, list):
            raise ValueError("pose instance has no keypoints")
        return cls(tuple(PoseKeypoint.from_dict(point) for point in points))

    def to_dict(self) -> dict[str, Any]:
        return {"keypoints": [point.to_dict() for point in self.keypoints]}


@dataclass(frozen=True)
class PoseEstimationResult:
    profile_id: str
    width: int
    height: int
    keypoint_schema: str
    poses: tuple[PoseInstance, ...] = ()
    connections: tuple[PoseConnection, ...] = ()
    segments: tuple[PoseSegment, ...] = ()

    def __post_init__(self) -> None:
        pose_estimator_profile(self.profile_id)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("pose result dimensions must be positive")
        for pose in self.poses:
            names: set[str] = set()
            for point in pose.keypoints:
                if not point.name or point.name in names:
                    raise ValueError("pose keypoint names must be unique")
                names.add(point.name)
                values = (point.x, point.y, point.score)
                if not all(np.isfinite(value) for value in values):
                    raise ValueError("pose keypoints must be finite")
                if point.z is not None and not np.isfinite(point.z):
                    raise ValueError("pose keypoint depth must be finite")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PoseEstimationResult":
        if not isinstance(value, dict):
            raise ValueError("pose result must be an object")
        return cls(
            profile_id=str(value["profile_id"]),
            width=int(value["width"]),
            height=int(value["height"]),
            keypoint_schema=str(value["keypoint_schema"]),
            poses=tuple(
                PoseInstance.from_dict(item)
                for item in value.get("poses", [])
            ),
            connections=tuple(
                PoseConnection.from_dict(item)
                for item in value.get("connections", [])
            ),
            segments=tuple(
                PoseSegment.from_dict(item)
                for item in value.get("segments", [])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "width": self.width,
            "height": self.height,
            "keypoint_schema": self.keypoint_schema,
            "poses": [pose.to_dict() for pose in self.poses],
            "connections": [item.to_dict() for item in self.connections],
            "segments": [item.to_dict() for item in self.segments],
        }


@dataclass(frozen=True)
class PoseEstimationRequest:
    image: np.ndarray
    profile_id: str = "dwpose"

    def __post_init__(self) -> None:
        pose_estimator_profile(self.profile_id)
        image = np.asarray(self.image)
        if (
                image.dtype != np.uint8
                or image.ndim != 3
                or image.shape[2] not in (3, 4)
                or image.size == 0):
            raise ValueError("pose input must be a non-empty uint8 RGB/RGBA image")


_GROUP_COLORS = {
    "body": (0, 220, 255, 235),
    "face": (255, 210, 35, 210),
    "left_hand": (75, 255, 120, 230),
    "right_hand": (255, 80, 190, 230),
    "feet": (255, 145, 35, 230),
    "diagnostic": (255, 70, 70, 220),
}

_GROUP_SIZE_FACTORS = {
    "body": 1.0,
    "face": 0.35,
    "left_hand": 0.55,
    "right_hand": 0.55,
    "feet": 0.7,
    "diagnostic": 0.7,
}


def render_pose_overlay(
        result: PoseEstimationResult,
        *,
        confidence_threshold: float = 0.25) -> np.ndarray:
    """Render a transparent RGBA overlay while retaining raw coordinates."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("pose confidence threshold must be in [0, 1]")
    scale = max(1.0, min(result.width, result.height) / 512.0)
    line_width = max(2, int(round(3.0 * scale)))
    radius = max(3, int(round(4.0 * scale)))
    image = Image.new("RGBA", (result.width, result.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")

    for segment in result.segments:
        if segment.score < confidence_threshold:
            continue
        color = _GROUP_COLORS.get(segment.group, _GROUP_COLORS["diagnostic"])
        draw.line(
            (segment.x0, segment.y0, segment.x1, segment.y1),
            fill=color,
            width=line_width,
        )

    for pose in result.poses:
        points = {point.name: point for point in pose.keypoints}
        groups: dict[str, str] = {}
        for connection in result.connections:
            groups.setdefault(connection.start, connection.group)
            groups.setdefault(connection.end, connection.group)
            start = points.get(connection.start)
            end = points.get(connection.end)
            if (
                    start is None
                    or end is None
                    or min(start.score, end.score) < confidence_threshold):
                continue
            color = _GROUP_COLORS.get(
                connection.group, _GROUP_COLORS["body"])
            connection_width = max(
                1,
                int(round(
                    line_width
                    * _GROUP_SIZE_FACTORS.get(connection.group, 1.0)
                )),
            )
            draw.line(
                (start.x, start.y, end.x, end.y),
                fill=color,
                width=connection_width,
            )
        for point in pose.keypoints:
            if point.score < confidence_threshold:
                continue
            group = groups.get(point.name, "body")
            color = _GROUP_COLORS.get(group, _GROUP_COLORS["body"])
            point_radius = max(
                2,
                int(round(radius * _GROUP_SIZE_FACTORS.get(group, 1.0))),
            )
            alpha = int(round(90 + 165 * min(1.0, max(0.0, point.score))))
            fill = (color[0], color[1], color[2], alpha)
            draw.ellipse(
                (
                    point.x - point_radius,
                    point.y - point_radius,
                    point.x + point_radius,
                    point.y + point_radius,
                ),
                fill=fill,
                outline=(15, 15, 20, min(255, alpha + 20)),
                width=max(1, line_width // 2),
            )
    # ``np.asarray(PIL.Image)`` is C-contiguous but backed by immutable bytes.
    # The native tgfx binding requires a writable ndarray for texture upload,
    # so the overlay must own its storage rather than merely advertise C order.
    return np.array(image, dtype=np.uint8, order="C", copy=True)
