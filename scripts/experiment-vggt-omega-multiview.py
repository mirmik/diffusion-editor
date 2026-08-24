#!/usr/bin/env python3
"""Run a reproducible VGGT-Omega multi-view reconstruction experiment.

The model's depth, confidence, camera calibration, and world-space points are
saved as float32 arrays without coordinate normalization.  Display-only depth
and confidence colours are emitted separately, as are RGB and confidence-
coloured PLY clouds that Termin can open.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


DEFAULT_SOURCE = Path("/tmp/vggt-omega-source")
DEFAULT_CHECKPOINT = Path(
    "/tmp/vggt-omega-checkpoint/vggt_omega_1b_512.pt"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--vggt-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument(
        "--preprocess-mode",
        choices=("balanced", "max_size"),
        default="balanced",
    )
    parser.add_argument(
        "--replace-background",
        choices=("none", "white", "gray", "green"),
        default="none",
        help=(
            "Before inference, replace everything outside the largest central "
            "non-green component with a constant colour."
        ),
    )
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=20.0,
        help="Drop the lowest N%% confidence points from display clouds only.",
    )
    parser.add_argument(
        "--depth-edge-rtol",
        type=float,
        default=0.03,
        help="Reject display-cloud pixels crossing a relative depth edge; zero disables.",
    )
    parser.add_argument("--max-full-points", type=int, default=1_000_000)
    parser.add_argument("--max-subject-points", type=int, default=750_000)
    return parser.parse_args()


def _unproject_depth(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    """Official VGGT-Omega demo unprojection, retaining model world units."""

    depth = depth_map[..., 0]
    frames, height, width = depth.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (frames, height, width))
    y = np.broadcast_to(y[None], (frames, height, width))
    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]
    camera_points = np.stack(
        (
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ),
        axis=-1,
    )
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    ).astype(np.float32, copy=False)


def _depth_edge(depth: np.ndarray, rtol: float, kernel_size: int = 3) -> np.ndarray:
    if rtol <= 0.0:
        return np.zeros(depth.shape, dtype=bool)
    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)
    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y : y + depth.shape[1], x : x + depth.shape[2]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)
    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return relative_jump > rtol


def _subject_mask(rgb: np.ndarray) -> np.ndarray:
    """Select the largest central non-green component without dilation."""

    values = rgb.astype(np.int16)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    green_screen = (
        (green - red >= 12)
        & (green - blue >= 12)
        & (green * 100 >= red * 112)
        & (green * 100 >= blue * 112)
    )
    labels, count = ndimage.label(~green_screen)
    if count == 0:
        return np.zeros(green_screen.shape, dtype=bool)
    height, width = green_screen.shape
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    best_label = 0
    best_score = -1.0
    for label in np.argsort(sizes)[::-1][:32]:
        if sizes[label] < 64:
            break
        ys, xs = np.nonzero(labels == label)
        center_x = float(xs.mean()) / max(width - 1, 1)
        center_y = float(ys.mean()) / max(height - 1, 1)
        centrality = max(0.0, 1.0 - abs(center_x - 0.5) * 1.5)
        verticality = max(0.0, 1.0 - abs(center_y - 0.52))
        score = float(sizes[label]) * (0.35 + centrality * verticality)
        if score > best_score:
            best_label = int(label)
            best_score = score
    return labels == best_label


def _prepare_inference_inputs(
    inputs: list[Path],
    output_dir: Path,
    replacement: str,
) -> tuple[list[Path], list[int], list[np.ndarray]]:
    if replacement == "none":
        return inputs, [], []
    background = {
        "white": np.array((255, 255, 255), dtype=np.uint8),
        "gray": np.array((127, 127, 127), dtype=np.uint8),
        "green": np.array((64, 150, 72), dtype=np.uint8),
    }[replacement]
    prepared_dir = output_dir / "inference-inputs"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[Path] = []
    counts: list[int] = []
    masks: list[np.ndarray] = []
    for index, path in enumerate(inputs):
        with Image.open(path) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        mask = _subject_mask(rgb)
        composited = np.where(mask[..., None], rgb, background)
        target = prepared_dir / f"{index:02d}-{path.stem}.png"
        Image.fromarray(composited).save(target)
        prepared.append(target)
        counts.append(int(mask.sum()))
        masks.append(mask)
    return prepared, counts, masks


_TURBO_ANCHORS = np.array(
    (
        (48, 18, 59),
        (70, 107, 227),
        (40, 188, 235),
        (31, 234, 179),
        (164, 252, 60),
        (238, 208, 58),
        (251, 126, 33),
        (188, 44, 35),
        (122, 4, 3),
    ),
    dtype=np.float32,
)


def _turbo(values: np.ndarray, low: float, high: float, invert: bool = False) -> np.ndarray:
    if high - low <= np.finfo(np.float32).eps:
        normalized = np.full(values.shape, 0.5, dtype=np.float32)
    else:
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    if invert:
        normalized = 1.0 - normalized
    positions = np.linspace(0.0, 1.0, len(_TURBO_ANCHORS), dtype=np.float32)
    rgb = np.stack(
        [np.interp(normalized, positions, _TURBO_ANCHORS[:, c]) for c in range(3)],
        axis=-1,
    )
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _contact_sheet(
    panels: list[np.ndarray],
    labels: list[str],
    path: Path,
    columns: int = 4,
) -> None:
    if len(panels) == 0:
        return
    thumb_width = 320
    label_height = 24
    first_h, first_w = panels[0].shape[:2]
    thumb_height = max(1, round(first_h * thumb_width / first_w))
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, rows * (thumb_height + label_height)),
        "black",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        image = Image.fromarray(panel)
        image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_width
        y = (index // columns) * (thumb_height + label_height)
        sheet.paste(image, (x, y + label_height))
        draw.text((x + 5, y + 4), label, fill="white")
    sheet.save(path)


def _limit_indices(valid: np.ndarray, maximum: int) -> np.ndarray:
    indices = np.flatnonzero(valid)
    if maximum > 0 and len(indices) > maximum:
        take = np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)
        indices = indices[take]
    return indices


def _write_binary_ply(
    path: Path,
    positions: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
) -> None:
    if not (len(positions) == len(colors) == len(confidence)):
        raise ValueError("PLY arrays have inconsistent lengths")
    vertex = np.empty(
        len(positions),
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("confidence", "<f4"),
            ]
        ),
    )
    vertex["x"], vertex["y"], vertex["z"] = positions.T
    vertex["red"], vertex["green"], vertex["blue"] = colors.T
    vertex["confidence"] = confidence
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment raw VGGT-Omega world coordinates; colors are display-only\n"
        f"element vertex {len(vertex)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property float confidence\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(vertex.tobytes())


def _save_cloud_variants(
    output_dir: Path,
    stem: str,
    points: np.ndarray,
    rgb: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    maximum: int,
    conf_low: float,
    conf_high: float,
) -> int:
    indices = _limit_indices(valid.reshape(-1), maximum)
    flat_points = points.reshape(-1, 3)[indices]
    flat_rgb = rgb.reshape(-1, 3)[indices]
    flat_conf = confidence.reshape(-1)[indices]
    _write_binary_ply(
        output_dir / f"{stem}-rgb.ply",
        flat_points,
        flat_rgb,
        flat_conf,
    )
    confidence_rgb = _turbo(flat_conf, conf_low, conf_high)
    _write_binary_ply(
        output_dir / f"{stem}-confidence.ply",
        flat_points,
        confidence_rgb,
        flat_conf,
    )
    return len(indices)


def _save_camera_plot(
    extrinsic: np.ndarray,
    labels: list[str],
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    centers = -np.einsum("sji,sj->si", rotation, translation)
    forward = np.einsum(
        "sji,sj->si", rotation, np.broadcast_to((0.0, 0.0, 1.0), centers.shape)
    )
    figure = plt.figure(figsize=(9, 8), dpi=150)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(centers[:, 0], centers[:, 1], centers[:, 2], "o-", linewidth=1.5)
    scale = max(float(np.ptp(centers, axis=0).max()) * 0.12, 1e-4)
    axis.quiver(
        centers[:, 0], centers[:, 1], centers[:, 2],
        forward[:, 0], forward[:, 1], forward[:, 2],
        length=scale, normalize=True, color="tab:red",
    )
    for center, label in zip(centers, labels):
        axis.text(*center, label, fontsize=7)
    mid = (centers.min(axis=0) + centers.max(axis=0)) * 0.5
    half = max(float(np.ptp(centers, axis=0).max()) * 0.55, 1e-3)
    axis.set_xlim(mid[0] - half, mid[0] + half)
    axis.set_ylim(mid[1] - half, mid[1] + half)
    axis.set_zlim(mid[2] - half, mid[2] + half)
    axis.set_xlabel("world X")
    axis.set_ylabel("world Y")
    axis.set_zlabel("world Z")
    axis.set_title("VGGT-Omega predicted cameras (raw model coordinates)")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return centers.astype(np.float32), forward.astype(np.float32)


def main() -> int:
    args = _arguments()
    if len(args.inputs) < 2:
        raise SystemExit("VGGT-Omega joint inference needs at least two images")
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        raise SystemExit(f"missing input images: {missing}")
    if not (args.vggt_source / "vggt_omega/models/vggt_omega.py").is_file():
        raise SystemExit(f"VGGT-Omega source is missing: {args.vggt_source}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"VGGT-Omega checkpoint is missing: {args.checkpoint}")
    if not 0.0 <= args.confidence_percentile < 100.0:
        raise SystemExit("confidence percentile must be in [0, 100)")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffusion-editor-matplotlib")
    sys.path.insert(0, str(args.vggt_source))
    import torch
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for VGGT-Omega")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (
        inference_inputs,
        pre_inference_subject_pixels,
        pre_inference_subject_masks,
    ) = _prepare_inference_inputs(args.inputs, args.output_dir, args.replace_background)

    started = time.monotonic()
    print(f"Loading checkpoint: {args.checkpoint}", flush=True)
    model = VGGTOmega().eval()
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    del state_dict
    model = model.to("cuda")

    images = load_and_preprocess_images(
        [str(path) for path in inference_inputs],
        mode=args.preprocess_mode,
        image_resolution=args.image_resolution,
    ).to("cuda")
    print(f"Joint inference: tensor {tuple(images.shape)}", flush=True)
    inference_started = time.monotonic()
    with torch.inference_mode():
        predictions = model(images)
        extrinsic, intrinsic = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )
    torch.cuda.synchronize()
    inference_seconds = time.monotonic() - inference_started

    arrays: dict[str, np.ndarray] = {}
    for key in ("images", "depth", "depth_conf", "pose_enc"):
        value = predictions[key].detach().float().cpu().numpy()
        arrays[key] = value[0] if value.shape[0] == 1 else value
    for key, value in (("extrinsic", extrinsic), ("intrinsic", intrinsic)):
        array = value.detach().float().cpu().numpy()
        arrays[key] = array[0] if array.shape[0] == 1 else array
    del predictions, model, images
    torch.cuda.empty_cache()

    processed_float = np.ascontiguousarray(arrays["images"], dtype=np.float32)
    processed_rgb = np.ascontiguousarray(
        np.clip(np.transpose(processed_float, (0, 2, 3, 1)) * 255.0, 0, 255),
        dtype=np.uint8,
    )
    depth = np.ascontiguousarray(arrays["depth"], dtype=np.float32)
    confidence = np.ascontiguousarray(arrays["depth_conf"], dtype=np.float32)
    extrinsic_np = np.ascontiguousarray(arrays["extrinsic"], dtype=np.float32)
    intrinsic_np = np.ascontiguousarray(arrays["intrinsic"], dtype=np.float32)
    world_points = _unproject_depth(depth, extrinsic_np, intrinsic_np)

    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    raw_arrays = {
        **arrays,
        "world_points_from_depth": world_points,
    }
    for name, value in raw_arrays.items():
        np.save(raw_dir / f"{name}.npy", np.ascontiguousarray(value))

    depth_values = depth[..., 0] if depth.ndim == 4 else depth
    conf_values = confidence[..., 0] if confidence.ndim == 4 else confidence
    depth_low, depth_high = np.percentile(depth_values, (2.0, 98.0))
    conf_low, conf_high = np.percentile(conf_values, (2.0, 98.0))
    depth_panels = [
        _turbo(frame, float(depth_low), float(depth_high), invert=True)
        for frame in depth_values
    ]
    confidence_panels = [
        _turbo(frame, float(conf_low), float(conf_high))
        for frame in conf_values
    ]
    labels = [path.stem for path in args.inputs]
    _contact_sheet(processed_rgb, labels, args.output_dir / "inputs-processed.png")
    _contact_sheet(
        depth_panels,
        [f"{name} | near warm, far cold" for name in labels],
        args.output_dir / "depth-preview.png",
    )
    _contact_sheet(
        confidence_panels,
        [f"{name} | high warm, low cold" for name in labels],
        args.output_dir / "confidence-preview.png",
    )

    finite = (
        np.isfinite(world_points).all(axis=-1)
        & np.isfinite(depth_values)
        & np.isfinite(conf_values)
    )
    conf_threshold = float(np.percentile(conf_values[finite], args.confidence_percentile))
    display_valid = finite & (conf_values >= conf_threshold)
    edge_mask = _depth_edge(depth_values, args.depth_edge_rtol)
    display_valid &= ~edge_mask
    if pre_inference_subject_masks:
        target_size = (processed_rgb.shape[2], processed_rgb.shape[1])
        subject_masks = np.stack(
            [
                np.asarray(
                    Image.fromarray(mask).resize(target_size, Image.Resampling.NEAREST),
                    dtype=bool,
                )
                for mask in pre_inference_subject_masks
            ]
        )
    else:
        subject_masks = np.stack([_subject_mask(frame) for frame in processed_rgb])
    subject_valid = display_valid & subject_masks

    full_count = _save_cloud_variants(
        args.output_dir,
        "full-cloud",
        world_points,
        processed_rgb,
        conf_values,
        display_valid,
        args.max_full_points,
        float(conf_low),
        float(conf_high),
    )
    subject_count = _save_cloud_variants(
        args.output_dir,
        "subject-cloud",
        world_points,
        processed_rgb,
        conf_values,
        subject_valid,
        args.max_subject_points,
        float(conf_low),
        float(conf_high),
    )
    centers, forwards = _save_camera_plot(
        extrinsic_np, labels, args.output_dir / "camera-trajectory.png"
    )
    np.save(raw_dir / "camera_centers.npy", centers)
    np.save(raw_dir / "camera_forwards.npy", forwards)

    manifest = {
        "model": "facebook/VGGT-Omega",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_variant": "vggt_omega_1b_512.pt",
        "inputs": [str(path.resolve()) for path in args.inputs],
        "inference_inputs": [str(path.resolve()) for path in inference_inputs],
        "input_order": labels,
        "image_resolution": args.image_resolution,
        "preprocess_mode": args.preprocess_mode,
        "replace_background": args.replace_background,
        "pre_inference_subject_pixels": pre_inference_subject_pixels,
        "processed_shape": list(processed_rgb.shape),
        "depth_shape": list(depth.shape),
        "coordinates": (
            "raw model world coordinates from official depth unprojection; "
            "first predicted camera defines the gauge; global scale is not metric"
        ),
        "confidence_semantics": (
            "raw score 1+exp(logit), higher is better; not a calibrated probability"
        ),
        "confidence_percentile_for_display_clouds": args.confidence_percentile,
        "confidence_threshold_for_display_clouds": conf_threshold,
        "depth_edge_rtol_for_display_clouds": args.depth_edge_rtol,
        "full_cloud_points": full_count,
        "subject_cloud_points": subject_count,
        "raw_finite_points": int(finite.sum()),
        "subject_pixels_per_view": [int(mask.sum()) for mask in subject_masks],
        "depth_percentiles": {
            str(p): float(np.percentile(depth_values, p))
            for p in (0, 2, 10, 25, 50, 75, 90, 98, 100)
        },
        "confidence_percentiles": {
            str(p): float(np.percentile(conf_values, p))
            for p in (0, 2, 10, 20, 25, 50, 75, 90, 98, 100)
        },
        "camera_centers": centers.tolist(),
        "camera_forwards": forwards.tolist(),
        "inference_seconds": inference_seconds,
        "total_seconds": time.monotonic() - started,
        "notes": [
            "Raw arrays are uncompressed float32 .npy files.",
            "PLY filtering/downsampling changes point membership only, never coordinates.",
            "Depth/confidence preview colours use P2-P98 solely for display contrast.",
            "Subject masking is applied after joint inference and does not affect cameras/depth.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
