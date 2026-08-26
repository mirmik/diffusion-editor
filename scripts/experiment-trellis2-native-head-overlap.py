#!/usr/bin/env python3
"""Merge native TRELLIS.2 source/refined head meshes by exact overlap.

Nothing is remeshed, simplified, repaired, filled, textured, or UV-unwrapped.
In the overlap band an exactly coincident triangle belongs to the source mesh;
all other triangles belong to the refined mesh.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-slat", type=Path, required=True)
    parser.add_argument("--refined-slat", type=Path, required=True)
    parser.add_argument("--shape-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--overlap-fraction", type=float, default=0.40)
    parser.add_argument(
        "--trellis2-root",
        type=Path,
        default=Path("/home/mirmik/soft/TRELLIS.2"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/mirmik/soft/TRELLIS.2/models/TRELLIS.2-4B"),
    )
    return parser.parse_args()


def _load(path, sparse, torch):
    import numpy as np

    with np.load(path, allow_pickle=False) as saved:
        return sparse.SparseTensor(
            feats=torch.from_numpy(saved["feats"]).float().cuda(),
            coords=torch.from_numpy(saved["coords"]).int().cuda(),
        )


def _canonical_triangles(vertices, faces, np):
    triangles = vertices[faces]
    order = np.lexsort(
        (triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]),
        axis=1,
    )
    triangles = np.take_along_axis(triangles, order[:, :, None], axis=1)
    packed = np.ascontiguousarray(triangles).reshape(len(triangles), -1)
    signature_type = np.dtype((np.void, packed.dtype.itemsize * packed.shape[1]))
    return packed.view(signature_type).reshape(-1)


def main() -> int:
    args = _args()
    for path in (
        args.source_slat,
        args.refined_slat,
        args.shape_manifest,
        args.trellis2_root,
        args.model_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if not 0 < args.overlap_fraction <= 1:
        raise ValueError("overlap-fraction must be in (0, 1]")

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    sys.path.insert(0, str(args.trellis2_root.resolve()))

    import numpy as np
    import torch
    import trimesh
    from trellis2.modules import sparse
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    torch.set_grad_enabled(False)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(args.model_path))
    pipeline.low_vram = True
    pipeline.cuda()
    source_slat = _load(args.source_slat, sparse, torch)
    refined_slat = _load(args.refined_slat, sparse, torch)

    source = pipeline.decode_shape_slat(source_slat, args.resolution)[0][0]
    refined = pipeline.decode_shape_slat(refined_slat, args.resolution)[0][0]
    source_vertices = source.vertices.detach().float().cpu().numpy()
    source_faces = source.faces.detach().long().cpu().numpy()
    refined_vertices = refined.vertices.detach().float().cpu().numpy()
    refined_faces = refined.faces.detach().long().cpu().numpy()
    del source, refined, source_slat, refined_slat, pipeline
    torch.cuda.empty_cache()

    lower = min(source_vertices[:, 2].min(), refined_vertices[:, 2].min())
    upper = max(source_vertices[:, 2].max(), refined_vertices[:, 2].max())
    overlap_top = lower + args.overlap_fraction * (upper - lower)
    source_overlap = np.flatnonzero(
        source_vertices[source_faces, 2].max(axis=1) <= overlap_top
    )
    refined_overlap = np.flatnonzero(
        refined_vertices[refined_faces, 2].max(axis=1) <= overlap_top
    )

    source_signatures = _canonical_triangles(
        source_vertices, source_faces[source_overlap], np
    )
    refined_signatures = _canonical_triangles(
        refined_vertices, refined_faces[refined_overlap], np
    )
    _, source_common_local, refined_common_local = np.intersect1d(
        source_signatures,
        refined_signatures,
        assume_unique=False,
        return_indices=True,
    )
    source_common = source_overlap[source_common_local]
    refined_common = refined_overlap[refined_common_local]

    keep_refined = np.ones(len(refined_faces), dtype=bool)
    keep_refined[refined_common] = False
    manifest = json.loads(args.shape_manifest.read_text(encoding="utf-8"))
    center = np.asarray(manifest["local_center"], dtype=np.float32)
    scale = float(manifest["local_scale"])
    merged_vertices = np.concatenate((source_vertices, refined_vertices), axis=0)
    merged_vertices = merged_vertices / scale + center
    merged_faces = np.concatenate((
        source_faces[source_common],
        refined_faces[keep_refined] + len(source_vertices),
    ), axis=0)
    merged = trimesh.Trimesh(
        vertices=merged_vertices,
        faces=merged_faces,
        process=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.export(args.output)

    report = {
        "source_vertices": int(len(source_vertices)),
        "source_faces": int(len(source_faces)),
        "refined_vertices": int(len(refined_vertices)),
        "refined_faces": int(len(refined_faces)),
        "overlap_fraction": args.overlap_fraction,
        "source_overlap_faces": int(len(source_overlap)),
        "refined_overlap_faces": int(len(refined_overlap)),
        "exact_common_faces": int(len(source_common)),
        "merged_faces": int(len(merged_faces)),
        "local_center": center.tolist(),
        "local_scale": scale,
        "coordinate_space": "body-space editor Z-up",
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
