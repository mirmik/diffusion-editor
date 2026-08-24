#!/usr/bin/env python3
"""Validate and visualize a canonical point-map dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from diffusion_editor.training.canonical_pointmap import pointmap_reprojection_error


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--minimum-alpha-iou", type=float, default=0.97)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _colorize_xyz(xyz: np.ndarray, mask: np.ndarray) -> Image.Image:
    # With pelvis origin and height normalization, the useful range is close
    # to [-0.5, 0.5] on every axis.  Keeping one fixed mapping across all views
    # makes equal canonical positions visibly equal colors.
    encoded = np.clip(xyz + 0.5, 0.0, 1.0)
    encoded[~mask] = 0.0
    return Image.fromarray(np.round(encoded * 255.0).astype(np.uint8), "RGB")


def _colorize_normal(normal: np.ndarray, mask: np.ndarray) -> Image.Image:
    encoded = np.clip(normal * 0.5 + 0.5, 0.0, 1.0)
    encoded[~mask] = 0.0
    return Image.fromarray(np.round(encoded * 255.0).astype(np.uint8), "RGB")


def _contact_sheet(items: list[tuple[str, Image.Image]]) -> Image.Image:
    if not items:
        raise ValueError("cannot build an empty contact sheet")
    columns = min(6, len(items))
    rows = math.ceil(len(items) / columns)
    width, height = items[0][1].size
    label_height = 20
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)))
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(items):
        column = index % columns
        row = index // columns
        x = column * width
        y = row * (height + label_height)
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (0, 0, 0, 255))
            background.alpha_composite(image)
            image = background.convert("RGB")
        else:
            image = image.convert("RGB")
        sheet.paste(image, (x, y + label_height))
        draw.text((x + 4, y + 3), label, fill=(235, 235, 235))
    return sheet


def main() -> int:
    args = _arguments()
    root = args.dataset.resolve()
    manifest = _load_json(root / "manifest.json")
    if manifest.get("schema") != "diffusion-editor.canonical-pointmap-dataset":
        raise SystemExit("unsupported or missing dataset schema")
    diagnostics = (args.diagnostics_dir or root / "diagnostics").resolve()
    diagnostics.mkdir(parents=True, exist_ok=True)

    reports = []
    contacts: dict[str, list[tuple[str, Image.Image]]] = {
        "rgb": [],
        "mask": [],
        "xyz": [],
        "normal": [],
    }
    for view in manifest["views"]:
        identifier = view["id"]
        rgb_path = root / view["rgb"]
        geometry_path = root / view["geometry"]
        camera_path = root / view["camera"]
        camera = _load_json(camera_path)
        with np.load(geometry_path) as source:
            arrays = {name: source[name] for name in source.files}
        xyz = arrays["canonical_xyz"]
        normal = arrays["normal"]
        depth = arrays["depth"]
        mask = arrays["mask"].astype(bool)
        object_id = arrays["object_id"]
        height, width = mask.shape
        expected_vector = (height, width, 3)
        if xyz.shape != expected_vector or normal.shape != expected_vector:
            raise RuntimeError(f"{identifier}: invalid XYZ or normal shape")
        if depth.shape != mask.shape or object_id.shape != mask.shape:
            raise RuntimeError(f"{identifier}: invalid scalar map shape")
        if xyz.dtype != np.float32 or normal.dtype != np.float32:
            raise RuntimeError(f"{identifier}: XYZ and normal must be float32")
        if depth.dtype != np.float32 or arrays["mask"].dtype != np.uint8:
            raise RuntimeError(f"{identifier}: invalid depth or mask dtype")
        if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(normal)):
            raise RuntimeError(f"{identifier}: non-finite geometry")
        if not np.all(xyz[~mask] == 0.0) or not np.all(normal[~mask] == 0.0):
            raise RuntimeError(f"{identifier}: non-zero background geometry")
        if not np.all(depth[~mask] == 0.0) or np.any(depth[mask] <= 0.0):
            raise RuntimeError(f"{identifier}: invalid foreground/background depth")
        if np.any(object_id[mask] == 0) or np.any(object_id[~mask] != 0):
            raise RuntimeError(f"{identifier}: object IDs disagree with mask")
        normal_lengths = np.linalg.norm(normal[mask], axis=1)
        maximum_normal_error = float(np.max(np.abs(normal_lengths - 1.0)))
        if maximum_normal_error > 1.0e-4:
            raise RuntimeError(
                f"{identifier}: normal length error {maximum_normal_error:.6g}"
            )

        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
        transform = np.asarray(camera["camera_from_canonical"], dtype=np.float64)
        reprojection = pointmap_reprojection_error(
            xyz,
            mask,
            intrinsics=intrinsics,
            camera_from_canonical=transform,
        )
        with Image.open(rgb_path) as source:
            rgb = source.convert("RGBA")
        if rgb.size != (width, height):
            raise RuntimeError(f"{identifier}: RGB and geometry dimensions differ")
        alpha = np.asarray(rgb)[:, :, 3] >= 128
        intersection = int(np.logical_and(alpha, mask).sum())
        union = int(np.logical_or(alpha, mask).sum())
        alpha_iou = intersection / union if union else 1.0
        if alpha_iou < args.minimum_alpha_iou:
            raise RuntimeError(
                f"{identifier}: RGB alpha/raycast mask IoU {alpha_iou:.6f} "
                f"is below {args.minimum_alpha_iou:.6f}"
            )
        reports.append(
            {
                "id": identifier,
                "foreground_pixels": int(mask.sum()),
                "alpha_pixels": int(alpha.sum()),
                "alpha_mask_iou": alpha_iou,
                "reprojection_mean_pixels": float(reprojection.mean()),
                "reprojection_max_pixels": float(reprojection.max()),
                "normal_length_max_error": maximum_normal_error,
                "depth_min": float(depth[mask].min()),
                "depth_max": float(depth[mask].max()),
            }
        )
        contacts["rgb"].append((identifier, rgb.copy()))
        contacts["mask"].append(
            (identifier, Image.fromarray(mask.astype(np.uint8) * 255, "L"))
        )
        contacts["xyz"].append((identifier, _colorize_xyz(xyz, mask)))
        contacts["normal"].append((identifier, _colorize_normal(normal, mask)))

    for name, items in contacts.items():
        _contact_sheet(items).save(diagnostics / f"contact-{name}.png")
    summary = {
        "dataset": str(root),
        "views": reports,
        "view_count": len(reports),
        "foreground_pixels": {
            "minimum": min(item["foreground_pixels"] for item in reports),
            "maximum": max(item["foreground_pixels"] for item in reports),
        },
        "alpha_mask_iou": {
            "minimum": min(item["alpha_mask_iou"] for item in reports),
            "mean": float(np.mean([item["alpha_mask_iou"] for item in reports])),
        },
        "reprojection_max_pixels": max(
            item["reprojection_max_pixels"] for item in reports
        ),
    }
    (diagnostics / "validation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
