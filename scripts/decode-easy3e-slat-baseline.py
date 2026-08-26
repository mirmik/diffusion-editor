#!/usr/bin/env python3
"""Decode Easy3E's cached source SLAT without editing.

This produces the control model needed to separate VAE roundtrip damage from
changes introduced by FlowEdit/Repaint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--easy3e-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--slat-latent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    easy3e_root = args.easy3e_root.resolve()
    sys.path.insert(0, str(easy3e_root / "extensions" / "vox2seq"))
    sys.path.insert(0, str(easy3e_root))

    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("ATTN_BACKEND", "xformers")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    os.environ.setdefault("XFORMERS_DISABLED", "1")

    import torch
    import trellis.modules.sparse as sp
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    pipeline = TrellisImageTo3DPipeline.from_pretrained(str(args.checkpoint.resolve()))
    pipeline.cuda()
    cache = torch.load(args.slat_latent, map_location="cuda")
    slat = sp.SparseTensor(
        feats=cache["feats"].cuda(),
        coords=cache["coords"].cuda(),
    )
    with torch.no_grad():
        outputs = pipeline.decode_slat(slat, formats=["mesh", "gaussian"])
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0],
        outputs["mesh"][0],
        simplify=0,
        texture_size=1024,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    glb.export(args.output)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
