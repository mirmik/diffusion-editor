#!/usr/bin/env python3
"""Lightweight cached mesh postprocess worker (no TRELLIS model loading)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from trellis_mesh_postprocess import run_mesh_postprocess


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: trellis_postprocess_runner.py REQUEST.json")
    request_path = Path(sys.argv[1]).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol") != 1:
        raise ValueError("unsupported postprocess request protocol")
    output, report = run_mesh_postprocess(
        Path(request["cache"]),
        Path(request["output_dir"]),
        dict(request["settings"]),
        int(request["actual_resolution"]),
        progress=lambda message: print(message, flush=True),
    )
    result_path = Path(request["result"]).resolve()
    result_path.write_text(
        json.dumps(
            {
                "protocol": 1,
                "status": "complete",
                "shape": str(output),
                "postprocess": report,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
