#!/usr/bin/env python3
"""Replace the native TRELLIS.2 body geometry inside a known local cube."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body-slat", type=Path, required=True)
    parser.add_argument("--head-body-space", type=Path, required=True)
    parser.add_argument("--shape-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
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


def _load_slat(path, sparse, torch):
    import numpy as np

    with np.load(path, allow_pickle=False) as saved:
        return sparse.SparseTensor(
            feats=torch.from_numpy(saved["feats"]).float().cuda(),
            coords=torch.from_numpy(saved["coords"]).int().cuda(),
        )


def _trellis_z_up_to_gltf_y_up(mesh, np):
    """Return a copy expressed in glTF's right-handed, Y-up coordinates."""
    result = mesh.copy()
    result.apply_transform(np.asarray((
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (0, -1, 0, 0),
        (0, 0, 0, 1),
    ), dtype=np.float64))
    return result


def main() -> int:
    args = _args()
    for path in (
        args.body_slat,
        args.head_body_space,
        args.shape_manifest,
        args.trellis2_root,
        args.model_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

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
    body_slat = _load_slat(args.body_slat, sparse, torch)
    body = pipeline.decode_shape_slat(body_slat, args.resolution)[0][0]
    body_vertices = body.vertices.detach().float().cpu().numpy()
    body_faces = body.faces.detach().long().cpu().numpy()
    del body, body_slat, pipeline
    torch.cuda.empty_cache()

    head_scene = trimesh.load(args.head_body_space, force="scene", process=False)
    head_parts = head_scene.dump(concatenate=False)
    if len(head_parts) != 1:
        raise ValueError(f"Expected one head mesh, got {len(head_parts)}")
    head = head_parts[0]
    head_vertices = np.asarray(head.vertices, dtype=np.float32)
    head_faces = np.asarray(head.faces, dtype=np.int64)

    manifest = json.loads(args.shape_manifest.read_text(encoding="utf-8"))
    center = np.asarray(manifest["local_center"], dtype=np.float32)
    scale = float(manifest["local_scale"])
    cube_min = center - 0.5 / scale
    cube_max = center + 0.5 / scale

    vertex_inside = np.logical_and(
        body_vertices >= cube_min,
        body_vertices <= cube_max,
    ).all(axis=1)
    face_inside = vertex_inside[body_faces].all(axis=1)
    kept_body_faces = body_faces[~face_inside]

    vertices = np.concatenate((body_vertices, head_vertices), axis=0)
    faces = np.concatenate((
        kept_body_faces,
        head_faces + len(body_vertices),
    ), axis=0)
    result_trellis = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    result_gltf = _trellis_z_up_to_gltf_y_up(result_trellis, np)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_gltf.export(args.output)

    print(json.dumps({
        "cube_min": cube_min.tolist(),
        "cube_max": cube_max.tolist(),
        "body_faces_before": int(len(body_faces)),
        "body_faces_removed": int(face_inside.sum()),
        "body_faces_kept": int(len(kept_body_faces)),
        "head_faces_added": int(len(head_faces)),
        "result_faces": int(len(faces)),
        "trellis_z_up_bounds": result_trellis.bounds.tolist(),
        "gltf_y_up_bounds": result_gltf.bounds.tolist(),
        "output_coordinates": "glTF 2.0 right-handed Y-up",
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
