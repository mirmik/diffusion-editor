#!/usr/bin/env python3
"""Evaluate Qwen feature consistency on multiple complete elevation rings."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import runpy
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-similarity", type=float, default=0.50)
    parser.add_argument("--ransac-threshold", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _normalize(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32)
    norm = np.linalg.norm(features, axis=-1, keepdims=True)
    return (features / np.maximum(norm, 1e-8)).astype(np.float16)


def _grid_points(
    indices: np.ndarray,
    grid: tuple[int, int],
    size: tuple[int, int],
) -> np.ndarray:
    grid_width, grid_height = grid
    width, height = size
    y, x = np.divmod(indices, grid_width)
    return np.stack(
        (
            (x.astype(np.float32) + 0.5) * width / grid_width - 0.5,
            (y.astype(np.float32) + 0.5) * height / grid_height - 0.5,
        ),
        axis=1,
    )


def _mutual_matches(
    features0: np.ndarray,
    features1: np.ndarray,
    mask0: np.ndarray,
    mask1: np.ndarray,
    min_similarity: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids0 = np.flatnonzero(mask0.reshape(-1))
    ids1 = np.flatnonzero(mask1.reshape(-1))
    tensor0 = torch.from_numpy(
        features0.reshape(-1, features0.shape[-1])[ids0].astype(np.float32)
    ).to(device)
    tensor1 = torch.from_numpy(
        features1.reshape(-1, features1.shape[-1])[ids1].astype(np.float32)
    ).to(device)
    similarity = tensor0 @ tensor1.T
    best1 = similarity.argmax(dim=1)
    best0 = similarity.argmax(dim=0)
    source = torch.arange(len(ids0), device=device)
    mutual = best0[best1] == source
    scores = similarity[source, best1]
    keep = mutual & (scores >= min_similarity)
    selected0 = source[keep].cpu().numpy()
    selected1 = best1[keep].cpu().numpy()
    return (
        ids0[selected0],
        ids1[selected1],
        scores[keep].cpu().numpy(),
    )


class _Tracks:
    """Score-ordered union-find that forbids two tokens from one view."""

    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}
        self.members: dict[tuple[int, int], dict[int, int]] = {}

    def _add(self, node: tuple[int, int]) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.members[node] = {node[0]: node[1]}

    def find(self, node: tuple[int, int]) -> tuple[int, int]:
        self._add(node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            parent = self.parent[node]
            self.parent[node] = root
            node = parent
        return root

    def union(self, node0: tuple[int, int], node1: tuple[int, int]) -> bool:
        root0, root1 = self.find(node0), self.find(node1)
        if root0 == root1:
            return True
        members0, members1 = self.members[root0], self.members[root1]
        for view_index in members0.keys() & members1.keys():
            if members0[view_index] != members1[view_index]:
                return False
        if len(members0) < len(members1):
            root0, root1 = root1, root0
            members0, members1 = members1, members0
        self.parent[root1] = root0
        members0.update(members1)
        del self.members[root1]
        return True

    def view_count(self, node: tuple[int, int]) -> int:
        return len(self.members[self.find(node)])


def _edge_sets(views: list[dict]) -> dict[str, list[tuple[int, int]]]:
    lookup = {
        (str(view["elevation"]), int(view["azimuth_degrees"])): index
        for index, view in enumerate(views)
    }
    elevations = list(dict.fromkeys(str(view["elevation"]) for view in views))
    azimuths = sorted({int(view["azimuth_degrees"]) for view in views})
    edges: dict[str, list[tuple[int, int]]] = {}
    for elevation in elevations:
        ring = []
        for index, azimuth in enumerate(azimuths):
            next_azimuth = azimuths[(index + 1) % len(azimuths)]
            ring.append((lookup[(elevation, azimuth)], lookup[(elevation, next_azimuth)]))
        edges[f"ring-{elevation}"] = ring
    if len(elevations) >= 2:
        for lower, upper in zip(elevations, elevations[1:]):
            edges[f"vertical-{lower}-{upper}"] = [
                (lookup[(lower, azimuth)], lookup[(upper, azimuth)])
                for azimuth in azimuths
            ]
    return edges


def _contact_sheet(paths: list[Path], output: Path) -> None:
    columns = 2
    panel_width, panel_height = 960, 544
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (panel_width * columns, panel_height * rows), (8, 8, 8))
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            panel = source.convert("RGB")
            panel.thumbnail((panel_width, panel_height))
        x = (index % columns) * panel_width + (panel_width - panel.width) // 2
        y = (index // columns) * panel_height + (panel_height - panel.height) // 2
        sheet.paste(panel, (x, y))
    sheet.save(output)


def _view_sheet(views: list[dict], output: Path) -> None:
    cell_width, cell_height, label_height = 240, 272, 24
    elevations = list(dict.fromkeys(str(view["elevation"]) for view in views))
    azimuths = sorted({int(view["azimuth_degrees"]) for view in views})
    lookup = {
        (str(view["elevation"]), int(view["azimuth_degrees"])): view
        for view in views
    }
    sheet = Image.new(
        "RGB",
        (cell_width * len(azimuths), (cell_height + label_height) * len(elevations)),
        (8, 8, 8),
    )
    draw = ImageDraw.Draw(sheet)
    for row, elevation in enumerate(elevations):
        for column, azimuth in enumerate(azimuths):
            view = lookup[(elevation, azimuth)]
            with Image.open(view["image"]) as source:
                thumbnail = source.convert("RGB")
                thumbnail.thumbnail((cell_width, cell_height))
            x = column * cell_width + (cell_width - thumbnail.width) // 2
            y = row * (cell_height + label_height) + label_height
            sheet.paste(thumbnail, (x, y))
            draw.text(
                (column * cell_width + 5, y - label_height + 4),
                f"{elevation} / {azimuth:03d} deg",
                fill=(235, 235, 235),
            )
    sheet.save(output)


def main() -> int:
    args = _arguments()
    manifest_path = args.capture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    views = manifest["views"]
    if len(views) < 6:
        raise SystemExit("multi-ring analysis needs at least six views")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_keys = set(views[0]["feature_keys"])
    for view in views[1:]:
        feature_keys.intersection_update(view["feature_keys"])
    feature_keys = sorted(feature_keys)
    grid = tuple(int(value) for value in manifest["feature_grid"])
    size = tuple(int(value) for value in manifest["output_size"])
    width, height = size

    helper_path = Path(__file__).with_name("experiment-multiview-keypoints.py")
    helpers = runpy.run_path(str(helper_path))
    subject_mask = helpers["_subject_mask"]
    estimate_pair = helpers["_estimate_pair"]
    draw_pair = helpers["_draw_pair"]

    images = [np.asarray(Image.open(view["image"]).convert("RGB")) for view in views]
    masks = [
        cv2.resize(subject_mask(image, 3).astype(np.uint8), grid,
                   interpolation=cv2.INTER_NEAREST).astype(bool)
        for image in images
    ]
    archives = [np.load(view["features"]) for view in views]
    edge_groups = _edge_sets(views)
    edges = [edge for group in edge_groups.values() for edge in group]
    device = torch.device(args.device)
    focal = (0.5 * width) / np.tan(np.radians(34.0) * 0.5)
    intrinsics = np.array(
        ((focal, 0.0, (width - 1) * 0.5),
         (0.0, focal, (height - 1) * 0.5),
         (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )

    all_reports = []
    feature_cache: dict[str, dict[tuple[int, int], dict[str, object]]] = {}
    for feature_key in feature_keys:
        feature_maps = [archive[feature_key].astype(np.float32) for archive in archives]
        position_mean = np.mean(np.stack(feature_maps, axis=0), axis=0)
        feature_maps = [_normalize(feature_map - position_mean) for feature_map in feature_maps]

        pair_data: dict[tuple[int, int], dict[str, object]] = {}
        scored_edges = []
        locked_ratios = []
        for view0, view1 in edges:
            ids0, ids1, scores = _mutual_matches(
                feature_maps[view0], feature_maps[view1], masks[view0], masks[view1],
                args.min_similarity, device,
            )
            points0 = _grid_points(ids0, grid, size)
            points1 = _grid_points(ids1, grid, size)
            displacement = np.linalg.norm(points1 - points0, axis=1)
            locked_ratios.append(float(np.mean(displacement < 1.0)) if len(displacement) else 0.0)
            try:
                inliers, geometry = estimate_pair(
                    points0, points1, args.ransac_threshold, intrinsics
                )
            except cv2.error:
                inliers = np.zeros(len(points0), dtype=bool)
                geometry = {}
            data = {
                "ids0": ids0,
                "ids1": ids1,
                "scores": scores,
                "points0": points0,
                "points1": points1,
                "inliers": inliers,
                "geometry": geometry,
            }
            pair_data[(view0, view1)] = data
            for match_index in np.flatnonzero(inliers):
                scored_edges.append((
                    float(scores[match_index]), view0, int(ids0[match_index]),
                    view1, int(ids1[match_index]),
                ))

        tracks = _Tracks()
        for _score, view0, token0, view1, token1 in sorted(scored_edges, reverse=True):
            tracks.union((view0, token0), (view1, token1))

        supported_count = 0
        inlier_count = 0
        pair_reports = []
        for (view0, view1), data in pair_data.items():
            ids0 = data["ids0"]
            ids1 = data["ids1"]
            inliers = data["inliers"]
            supported = np.zeros(len(ids0), dtype=bool)
            for match_index in np.flatnonzero(inliers):
                node0 = (view0, int(ids0[match_index]))
                node1 = (view1, int(ids1[match_index]))
                supported[match_index] = (
                    tracks.find(node0) == tracks.find(node1)
                    and tracks.view_count(node0) >= 3
                )
            data["supported"] = supported
            supported_count += int(supported.sum())
            inlier_count += int(inliers.sum())
            pair_reports.append({
                "views": [view0, view1],
                "labels": [
                    [views[view0]["elevation"], views[view0]["azimuth_degrees"]],
                    [views[view1]["elevation"], views[view1]["azimuth_degrees"]],
                ],
                "match_count": int(len(ids0)),
                "magsac_inlier_count": int(inliers.sum()),
                "multiview_supported_count": int(supported.sum()),
                "similarity_median": float(np.median(data["scores"])) if len(ids0) else None,
                **data["geometry"],
            })

        track_histogram = Counter(len(members) for members in tracks.members.values())
        report = {
            "feature_key": feature_key,
            "match_count": int(sum(len(data["ids0"]) for data in pair_data.values())),
            "magsac_inlier_count": inlier_count,
            "multiview_supported_count": supported_count,
            "position_locked_ratio": float(np.mean(locked_ratios)),
            "track_length_histogram": dict(sorted(track_histogram.items())),
            "pair_reports": pair_reports,
        }
        all_reports.append(report)
        feature_cache[feature_key] = pair_data
        print(
            feature_key,
            "MAGSAC", inlier_count,
            "supported", supported_count,
            "tracks>=3", sum(count for length, count in track_histogram.items() if length >= 3),
            flush=True,
        )

    eligible = [report for report in all_reports if report["position_locked_ratio"] < 0.25]
    ranked = sorted(
        eligible or all_reports,
        key=lambda report: (
            report["multiview_supported_count"],
            report["magsac_inlier_count"],
        ),
        reverse=True,
    )
    best = ranked[0]
    pair_data = feature_cache[best["feature_key"]]

    contacts = {}
    for group_name, group_edges in edge_groups.items():
        paths = []
        for edge_index, (view0, view1) in enumerate(group_edges):
            data = pair_data[(view0, view1)]
            label0 = f"{views[view0]['elevation']} {int(views[view0]['azimuth_degrees']):03d}"
            label1 = f"{views[view1]['elevation']} {int(views[view1]['azimuth_degrees']):03d}"
            visualization = draw_pair(
                images[view0], images[view1], data["points0"], data["points1"],
                data["inliers"], data["supported"],
                f"Qwen {best['feature_key']} · {label0} -> {label1}",
                (
                    f"MNN {len(data['ids0'])} · MAGSAC {int(data['inliers'].sum())} · "
                    f"multi-view {int(data['supported'].sum())}"
                ),
            )
            path = args.output_dir / f"{group_name}-{edge_index:02d}.png"
            visualization.save(path)
            paths.append(path)
        contact_path = args.output_dir / f"contact-{group_name}.png"
        _contact_sheet(paths, contact_path)
        contacts[group_name] = str(contact_path.resolve())

    view_sheet_path = args.output_dir / "contact-views.png"
    _view_sheet(views, view_sheet_path)
    summary = {
        "capture_manifest": str(manifest_path.resolve()),
        "best_feature_key": best["feature_key"],
        "centering": "position-across-all-views",
        "green_definition": "MAGSAC inlier in conflict-free track spanning >=3 views",
        "edge_groups": {
            name: [list(edge) for edge in group] for name, group in edge_groups.items()
        },
        "contacts": contacts,
        "view_contact": str(view_sheet_path.resolve()),
        "reports": sorted(
            all_reports,
            key=lambda report: report["feature_key"],
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Best: {best['feature_key']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
