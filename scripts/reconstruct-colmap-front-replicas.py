#!/usr/bin/env python3
"""Run classical COLMAP SfM on independently generated views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--camera-mode",
        choices=("single", "auto"),
        default="single",
        help="Share one unknown intrinsic camera or estimate one per image",
    )
    parser.add_argument("--max-image-size", type=int, default=1600)
    parser.add_argument(
        "--name-contains",
        default="",
        help="Only include image filenames containing this text",
    )
    return parser.parse_args()


def _database_stats(path: Path) -> dict:
    with sqlite3.connect(path) as database:
        images = int(database.execute("select count(*) from images").fetchone()[0])
        keypoints = int(database.execute("select coalesce(sum(rows),0) from keypoints").fetchone()[0])
        pairs = database.execute(
            "select rows from two_view_geometries where rows > 0"
        ).fetchall()
    counts = [int(row[0]) for row in pairs]
    return {
        "images": images,
        "keypoints": keypoints,
        "verified_pairs": len(counts),
        "verified_pair_inliers_min": min(counts) if counts else 0,
        "verified_pair_inliers_median": sorted(counts)[len(counts) // 2] if counts else 0,
        "verified_pair_inliers_max": max(counts) if counts else 0,
    }


def main() -> int:
    args = parse_args()
    images = sorted(
        path for path in args.image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and args.name_contains in path.name
    )
    if len(images) < 2:
        raise SystemExit(f"need at least two images in {args.image_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    database_path = args.output_dir / "database.db"
    sparse_path = args.output_dir / "sparse"
    if database_path.exists() or sparse_path.exists():
        raise SystemExit(f"output is not empty: {args.output_dir}")

    import pycolmap

    started = time.monotonic()
    reader = pycolmap.ImageReaderOptions(camera_model="SIMPLE_RADIAL")
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = args.max_image_size
    extraction.sift.max_num_features = 16384
    extraction.sift.peak_threshold = 0.003
    camera_mode = (
        pycolmap.CameraMode.SINGLE
        if args.camera_mode == "single"
        else pycolmap.CameraMode.AUTO
    )
    print("Extracting SIFT features...", flush=True)
    pycolmap.extract_features(
        database_path,
        args.image_dir,
        image_names=[path.name for path in images],
        camera_mode=camera_mode,
        reader_options=reader,
        extraction_options=extraction,
        device=pycolmap.Device.cpu,
    )
    matching = pycolmap.FeatureMatchingOptions()
    matching.guided_matching = True
    matching.sift.max_ratio = 0.85
    print("Exhaustive matching...", flush=True)
    pycolmap.match_exhaustive(
        database_path,
        matching_options=matching,
        device=pycolmap.Device.cpu,
    )
    database_stats = _database_stats(database_path)
    print(json.dumps(database_stats, indent=2), flush=True)

    options = pycolmap.IncrementalPipelineOptions()
    options.min_model_size = 2
    options.multiple_models = True
    options.max_num_models = 10
    options.random_seed = 20260823
    options.mapper.init_min_num_inliers = 30
    options.mapper.init_min_tri_angle = 1.0
    options.mapper.abs_pose_min_num_inliers = 20
    options.mapper.abs_pose_min_inlier_ratio = 0.1
    options.mapper.filter_min_tri_angle = 0.5
    options.triangulation.min_angle = 0.5
    print("Incremental mapping...", flush=True)
    reconstructions = pycolmap.incremental_mapping(
        database_path,
        args.image_dir,
        sparse_path,
        options=options,
    )
    models = []
    for model_id, reconstruction in sorted(reconstructions.items()):
        model_dir = sparse_path / str(model_id)
        reconstruction.write(model_dir)
        ply_path = args.output_dir / f"sparse-model-{model_id}.ply"
        reconstruction.export_PLY(ply_path)
        models.append(
            {
                "id": int(model_id),
                "registered_images": int(reconstruction.num_reg_images()),
                "points3D": int(reconstruction.num_points3D()),
                "mean_observations_per_image": float(
                    reconstruction.compute_mean_observations_per_reg_image()
                ),
                "summary": reconstruction.summary(),
                "ply": str(ply_path),
            }
        )
    models.sort(key=lambda item: (item["registered_images"], item["points3D"]), reverse=True)
    report = {
        "schema": "diffusion-editor.colmap-front-replica-sfm",
        "schema_version": 1,
        "image_dir": str(args.image_dir.resolve()),
        "camera_mode": args.camera_mode,
        "name_contains": args.name_contains,
        "images": [path.name for path in images],
        "elapsed_seconds": time.monotonic() - started,
        "database": database_stats,
        "models": models,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if models else 2


if __name__ == "__main__":
    raise SystemExit(main())
