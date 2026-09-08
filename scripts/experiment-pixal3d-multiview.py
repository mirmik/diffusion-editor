#!/usr/bin/env python3
"""Run the official Pixal3D multiview entry point and record reproducibility data.

Use the existing Pixal3D/TRELLIS Python environment with CUDA access.
The optional alpha-only mode avoids loading RMBG for already masked inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--views', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-views', type=int)
    parser.add_argument('--resolution', type=int, choices=(1024, 1536), default=1024)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alpha-only', action='store_true')
    args = parser.parse_args()
    for name in ('root', 'model', 'views', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    from PIL import Image
    import numpy as np

    meta_path = args.views / 'transforms.json'
    meta = json.loads(meta_path.read_text())
    frames = meta['frames']
    if args.num_views is not None:
        if not 1 <= args.num_views <= len(frames):
            parser.error('--num-views must be between 1 and the number of frames')
        frames = frames[:args.num_views]
    if not frames:
        parser.error('no input views')
    inputs = {str(meta_path): hashlib.sha256(meta_path.read_bytes()).hexdigest()}
    for frame in frames:
        path = args.views / frame['file_path']
        with Image.open(path) as im:
            if args.alpha_only and (im.mode != 'RGBA' or np.all(np.asarray(im.getchannel('A')) == 255)):
                parser.error(f'alpha-only mode needs a non-opaque RGBA mask: {path}')
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    os.environ.setdefault('ATTN_BACKEND', 'sdpa')
    sys.path.insert(0, str(args.root))
    spec = importlib.util.spec_from_file_location('pixal3d_upstream_mv', args.root / 'inference_mv.py')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run in the GPU-enabled environment')
    if args.alpha_only:
        from pixal3d.pipelines import rembg

        class AlphaOnly:
            def __init__(self, **kwargs):
                pass

            def to(self, device):
                return self

            def cpu(self):
                return self

            def __call__(self, image):
                raise RuntimeError('Unexpected background removal in alpha-only mode')

        rembg.BiRefNet = AlphaOnly

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'command': sys.argv,
        'upstream_commit': subprocess.check_output(['git', '-C', str(args.root), 'rev-parse', 'HEAD'], text=True).strip(),
        'upstream_diff': subprocess.check_output(['git', '-C', str(args.root), 'diff'], text=True),
        'model_revision': (args.model / 'revision.txt').read_text().strip() if (args.model / 'revision.txt').exists() else None,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'attention_backend': os.environ['ATTN_BACKEND'],
        'inputs_sha256': inputs, 'num_views': len(frames),
        'resolution': args.resolution, 'seed': args.seed, 'low_vram': True,
        'alpha_only': args.alpha_only, 'status': 'running',
    }
    report_path = args.output.with_suffix('.run.json')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    start = time.monotonic()
    try:
        upstream.run_inference(
            views_dir=str(args.views), output_path=str(args.output),
            model_path=str(args.model), num_views=args.num_views,
            resolution=args.resolution, seed=args.seed, low_vram=True,
        )
        report['status'] = 'success'
        report['output_bytes'] = args.output.stat().st_size
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
