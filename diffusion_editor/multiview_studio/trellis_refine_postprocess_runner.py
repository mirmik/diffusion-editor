#!/usr/bin/env python3
"""Repair and export a refined region after the heavy CUDA worker exits."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from trellis_refine_runner import (
    _mesh_contract,
    _postprocess_refined_mesh,
    _project_from_local_gltf,
)


def _emit(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: trellis_refine_postprocess_runner.py REQUEST.json"
        )
    request_path = Path(sys.argv[1]).expanduser().resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol") != 1:
        raise ValueError("unsupported region refine request protocol")

    output = Path(request["output_dir"]).expanduser().resolve()
    stage_result_path = output / "shape-stage-result.json"
    stage = json.loads(stage_result_path.read_text(encoding="utf-8"))
    if stage.get("shape_request_key") != request.get("shape_request_key"):
        raise ValueError("region shape-stage cache belongs to another request")

    import numpy as np
    import trimesh

    postprocess_resolution = int(
        stage.get(
            "actual_resolution",
            request.get("postprocess_resolution", int(request["resolution"])),
        )
    )
    _emit("[postprocess] applying Main mesh repair pipeline to refined region")
    refined_local_path, shared_postprocess = _postprocess_refined_mesh(
        Path(stage["postprocess_cache"]),
        output,
        dict(request["postprocess"]),
        postprocess_resolution,
    )
    refined_local = trimesh.load(
        refined_local_path, force="mesh", process=False
    )
    contract = _mesh_contract(refined_local, np)

    refined_project = _project_from_local_gltf(
        refined_local,
        np.asarray(stage["normalization_center_trellis"], dtype=np.float64),
        float(stage["normalization_scale"]),
        np,
    )
    refined_region_path = output / "refined-region.glb"
    refined_project.export(refined_region_path)

    result = dict(stage)
    result.update(
        {
            "status": "complete",
            "request_key": str(request["request_key"]),
            "shape_request_key": str(request["shape_request_key"]),
            "shape": str(refined_region_path.resolve()),
            "refined_region_local": str(refined_local_path.resolve()),
            "postprocess": {
                "resolution": postprocess_resolution,
                "shared_pipeline": shared_postprocess,
                "contract": contract,
            },
        }
    )
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _emit(f"[complete] standalone region: {refined_region_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
