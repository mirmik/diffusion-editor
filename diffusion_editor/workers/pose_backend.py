"""Optional-dependency pose estimators used only by the isolated worker."""

from __future__ import annotations

from pathlib import Path
import gc
import os
from urllib.request import Request, urlopen

import numpy as np

from ..generation.pose_estimation import (
    PoseConnection,
    PoseEstimationResult,
    PoseInstance,
    PoseKeypoint,
    PoseSegment,
    pose_estimator_profile,
)


DWPOSE_DETECTOR_URL = (
    "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"
)
DWPOSE_MODEL_URL = (
    "https://huggingface.co/yzd-v/DWPose/resolve/main/"
    "dw-ll_ucoco_384.onnx"
)
MEDIAPIPE_FULL_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


COCO_BODY_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
COCO_FOOT_NAMES = (
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
)
COCO_WHOLEBODY_NAMES = (
    COCO_BODY_NAMES
    + COCO_FOOT_NAMES
    + tuple(f"face_{index}" for index in range(68))
    + tuple(f"left_hand_{index}" for index in range(21))
    + tuple(f"right_hand_{index}" for index in range(21))
)


def _connections(
        pairs: tuple[tuple[str, str], ...], group: str) -> list[PoseConnection]:
    return [PoseConnection(start, end, group) for start, end in pairs]


def _chain(prefix: str, indices: tuple[int, ...], group: str) -> list[PoseConnection]:
    return _connections(tuple(
        (f"{prefix}_{start}", f"{prefix}_{end}")
        for start, end in zip(indices, indices[1:])
    ), group)


def coco_wholebody_connections() -> tuple[PoseConnection, ...]:
    result = _connections((
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("nose", "left_eye"),
        ("nose", "right_eye"),
        ("left_eye", "left_ear"),
        ("right_eye", "right_ear"),
    ), "body")
    result += _connections((
        ("left_ankle", "left_heel"),
        ("left_ankle", "left_big_toe"),
        ("left_big_toe", "left_small_toe"),
        ("right_ankle", "right_heel"),
        ("right_ankle", "right_big_toe"),
        ("right_big_toe", "right_small_toe"),
    ), "feet")
    for indices in (
        tuple(range(0, 17)),
        tuple(range(17, 22)),
        tuple(range(22, 27)),
        tuple(range(27, 31)),
        tuple(range(31, 36)),
        (36, 37, 38, 39, 40, 41, 36),
        (42, 43, 44, 45, 46, 47, 42),
        (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48),
        (60, 61, 62, 63, 64, 65, 66, 67, 60),
    ):
        result += _chain("face", indices, "face")
    hand_chains = (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    )
    for side in ("left", "right"):
        for indices in hand_chains:
            result += _chain(f"{side}_hand", indices, f"{side}_hand")
    return tuple(result)


MEDIAPIPE_NAMES = (
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)


def mediapipe_connections() -> tuple[PoseConnection, ...]:
    body = (
        ("mouth_left", "mouth_right"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    )
    hands = (
        ("left_wrist", "left_thumb"),
        ("left_wrist", "left_index"),
        ("left_wrist", "left_pinky"),
        ("right_wrist", "right_thumb"),
        ("right_wrist", "right_index"),
        ("right_wrist", "right_pinky"),
    )
    feet = (
        ("left_ankle", "left_heel"),
        ("left_heel", "left_foot_index"),
        ("left_ankle", "left_foot_index"),
        ("right_ankle", "right_heel"),
        ("right_heel", "right_foot_index"),
        ("right_ankle", "right_foot_index"),
    )
    face = (
        ("nose", "left_eye_inner"),
        ("left_eye_inner", "left_eye"),
        ("left_eye", "left_eye_outer"),
        ("left_eye_outer", "left_ear"),
        ("nose", "right_eye_inner"),
        ("right_eye_inner", "right_eye"),
        ("right_eye", "right_eye_outer"),
        ("right_eye_outer", "right_ear"),
    )
    return tuple(
        _connections(body, "body")
        + _connections(hands[:3], "left_hand")
        + _connections(hands[3:], "right_hand")
        + _connections(feet, "feet")
        + _connections(face, "face")
    )


def _cache_dir() -> Path:
    configured = os.environ.get("DIFFUSION_EDITOR_POSE_MODEL_DIR")
    if configured:
        root = Path(configured).expanduser()
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
        root = root / "diffusion-editor" / "pose-models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _download(url: str, filename: str) -> Path:
    target = _cache_dir() / filename
    if target.is_file() and target.stat().st_size > 0:
        return target
    temporary = target.with_suffix(target.suffix + ".part")
    request = Request(url, headers={"User-Agent": "diffusion-editor/pose"})
    try:
        with urlopen(request, timeout=60) as source, temporary.open("wb") as sink:
            while chunk := source.read(1024 * 1024):
                sink.write(chunk)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


class PoseBackend:
    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._active_profile: str | None = None

    def estimate(
            self,
            profile_id: str,
            image_path: str | Path) -> PoseEstimationResult:
        pose_estimator_profile(profile_id)
        if self._active_profile not in (None, profile_id):
            for model in self._models.values():
                close = getattr(model, "close", None)
                if callable(close):
                    close()
            self._models.clear()
            gc.collect()
        self._active_profile = profile_id
        if profile_id == "dwpose":
            return self._estimate_dwpose(image_path)
        if profile_id == "mediapipe-full":
            return self._estimate_mediapipe(image_path)
        if profile_id == "silhouette-skeleton":
            return self._estimate_silhouette(image_path)
        raise ValueError(f"Unknown pose estimator profile: {profile_id}")

    def _estimate_dwpose(self, image_path: str | Path) -> PoseEstimationResult:
        try:
            import cv2
            from rtmlib import Custom
        except ImportError as exc:
            raise RuntimeError(
                "DWPose backend is unavailable. Run ./setup-pose-worker.sh."
            ) from exc

        model = self._models.get("dwpose")
        if model is None:
            detector = _download(
                DWPOSE_DETECTOR_URL, "dwpose-yolox_l.onnx")
            pose = _download(DWPOSE_MODEL_URL, "dwpose-ucoco-384.onnx")
            model = Custom(
                det_class="YOLOX",
                det=str(detector),
                det_input_size=(640, 640),
                pose_class="RTMPose",
                pose=str(pose),
                pose_input_size=(288, 384),
                to_openpose=False,
                backend="onnxruntime",
                device="cpu",
            )
            self._models["dwpose"] = model

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read pose input: {image_path}")
        height, width = image.shape[:2]
        keypoints, scores = model(image)
        keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)
        poses: list[PoseInstance] = []
        for person_points, person_scores in zip(keypoints, scores):
            count = min(
                len(COCO_WHOLEBODY_NAMES),
                len(person_points),
                len(person_scores),
            )
            points = []
            for index in range(count):
                x, y = map(float, person_points[index, :2])
                score = float(person_scores[index])
                if not np.isfinite((x, y, score)).all():
                    continue
                points.append(PoseKeypoint(
                    COCO_WHOLEBODY_NAMES[index], x, y, score))
            poses.append(PoseInstance(tuple(points)))
        return PoseEstimationResult(
            profile_id="dwpose",
            width=width,
            height=height,
            keypoint_schema="coco_wholebody_133",
            poses=tuple(poses),
            connections=coco_wholebody_connections(),
        )

    def _estimate_mediapipe(
            self, image_path: str | Path) -> PoseEstimationResult:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe backend is unavailable. Run ./setup-pose-worker.sh."
            ) from exc

        model = self._models.get("mediapipe-full")
        if model is None:
            model_path = _download(
                MEDIAPIPE_FULL_MODEL_URL, "pose_landmarker_full.task")
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(model_path)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=4,
                min_pose_detection_confidence=0.25,
                min_pose_presence_confidence=0.25,
            )
            model = mp.tasks.vision.PoseLandmarker.create_from_options(options)
            self._models["mediapipe-full"] = model

        image = mp.Image.create_from_file(str(image_path))
        width, height = image.width, image.height
        detected = model.detect(image)
        poses = []
        for landmarks in detected.pose_landmarks:
            points = []
            for index, landmark in enumerate(landmarks[:len(MEDIAPIPE_NAMES)]):
                visibility = float(
                    1.0 if landmark.visibility is None else landmark.visibility)
                presence = float(
                    1.0 if landmark.presence is None else landmark.presence)
                points.append(PoseKeypoint(
                    name=MEDIAPIPE_NAMES[index],
                    x=float(landmark.x) * width,
                    y=float(landmark.y) * height,
                    z=(None if landmark.z is None else float(landmark.z)),
                    score=max(0.0, min(1.0, visibility * presence)),
                ))
            poses.append(PoseInstance(tuple(points)))
        return PoseEstimationResult(
            profile_id="mediapipe-full",
            width=width,
            height=height,
            keypoint_schema="mediapipe_pose_33",
            poses=tuple(poses),
            connections=mediapipe_connections(),
        )

    def _estimate_silhouette(
            self, image_path: str | Path) -> PoseEstimationResult:
        try:
            import cv2
            from scipy.ndimage import convolve
            from skimage.morphology import skeletonize
        except ImportError as exc:
            raise RuntimeError(
                "Silhouette backend requires OpenCV, SciPy and scikit-image"
            ) from exc

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to read pose input: {image_path}")
        height, width = image.shape[:2]
        rgb = image[:, :, :3].astype(np.float32)

        def foreground_from_rgb() -> np.ndarray:
            border = np.concatenate((
                rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
            background = np.median(border, axis=0)
            distance = np.linalg.norm(rgb - background, axis=2)
            scaled = np.clip(distance, 0.0, 255.0).astype(np.uint8)
            _threshold, mask = cv2.threshold(
                scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return mask > 0

        foreground = foreground_from_rgb()
        if image.ndim == 3 and image.shape[2] == 4:
            alpha_foreground = image[:, :, 3] > 32
            alpha_coverage = float(np.mean(alpha_foreground))
            # A fully transparent/opaque alpha plane carries no silhouette.
            # This is common for generated editor layers, whose RGB remains
            # useful to the neural estimators even when alpha is degenerate.
            if 0.001 <= alpha_coverage <= 0.999:
                foreground = alpha_foreground

        mask = foreground.astype(np.uint8) * 255
        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
        if count <= 1:
            return PoseEstimationResult(
                profile_id="silhouette-skeleton",
                width=width,
                height=height,
                keypoint_schema="silhouette_graph",
            )
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_area = int(component_areas.max())
        minimum_area = max(64, int(round(largest_area * 0.01)))
        selected = []
        for item in range(1, count):
            area = int(stats[item, cv2.CC_STAT_AREA])
            left = int(stats[item, cv2.CC_STAT_LEFT])
            top = int(stats[item, cv2.CC_STAT_TOP])
            component_width = int(stats[item, cv2.CC_STAT_WIDTH])
            component_height = int(stats[item, cv2.CC_STAT_HEIGHT])
            component_x = left + component_width * 0.5
            component_y = top + component_height * 0.5
            if (
                    area >= minimum_area
                    and abs(component_x - width * 0.5) <= width * 0.48
                    and abs(component_y - height * 0.5) <= height * 0.5):
                selected.append(item)
        subject = np.isin(labels, selected)
        skeleton = skeletonize(subject)
        neighbours = convolve(
            skeleton.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            mode="constant",
        ) - skeleton.astype(np.uint8)
        special = skeleton & ((neighbours == 1) | (neighbours >= 3))
        ys, xs = np.nonzero(special)
        keypoints = tuple(
            PoseKeypoint(f"graph_{index}", float(x), float(y), 1.0)
            for index, (x, y) in enumerate(zip(xs, ys))
        )

        segments: list[PoseSegment] = []
        for y, x in zip(*np.nonzero(skeleton)):
            for dx, dy in ((1, 0), (0, 1), (1, 1), (-1, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and skeleton[ny, nx]:
                    segments.append(PoseSegment(
                        float(x), float(y), float(nx), float(ny),
                        group="diagnostic",
                    ))
        return PoseEstimationResult(
            profile_id="silhouette-skeleton",
            width=width,
            height=height,
            keypoint_schema="silhouette_graph",
            poses=(PoseInstance(keypoints),),
            segments=tuple(segments),
        )
