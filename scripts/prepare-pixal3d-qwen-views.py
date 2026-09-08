#!/usr/bin/env python3
"""Mask a front image and three Qwen cardinal views for approximate Pixal3D cameras."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import threading

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusion_editor.workers.segmentation_process import SegmentationProcessClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--front', required=True, type=Path)
    parser.add_argument('--qwen-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--size', type=int, default=1024)
    args = parser.parse_args()
    if args.size < 64:
        parser.error('--size must be at least 64')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'masks').mkdir(exist_ok=True)
    fov = math.radians(20)
    occupancy = 1 / 1.1
    distance = 1 / (2 * math.tan(fov / 2) * occupancy)
    frames, records = [], []
    client = SegmentationProcessClient()
    try:
        for angle in (0, 90, 180, 270):
            path = (args.front if angle == 0 else args.qwen_dir / f'mv-eye-{angle:03d}.png').resolve()
            with Image.open(path) as opened:
                image = opened.convert('RGB')
            mask_array = client.segment(np.asarray(image), threading.Event(), on_progress=print)
            if mask_array.shape != (image.height, image.width):
                raise RuntimeError(f'mask dimensions do not match {path}')
            ys, xs = np.nonzero(mask_array >= 16)
            if not len(xs):
                raise RuntimeError(f'empty foreground mask: {path}')
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if width > height * 1.1:
                raise RuntimeError(f'view too wide for shared height normalization: {path}')
            span = math.ceil(height * 1.1)
            left = round((bbox[0] + bbox[2] - span) / 2)
            top = round((bbox[1] + bbox[3] - span) / 2)
            crop = (left, top, left + span, top + span)
            mask = Image.fromarray(mask_array, 'L')
            mask.save(output / 'masks' / f'{angle:03d}.png')
            rgba = image.convert('RGBA')
            rgba.putalpha(mask)
            rgba = rgba.crop(crop).resize((args.size, args.size), Image.Resampling.LANCZOS)
            filename = f'view-{angle:03d}.png'
            rgba.save(output / filename)
            a = math.radians(angle)
            s, c = round(math.sin(a), 10), round(math.cos(a), 10)
            matrix = [[c, 0, s, distance * s], [s, 0, -c, -distance * c], [0, 1, 0, 0], [0, 0, 0, 1]]
            frames.append({'file_path': filename, 'name': f'azim{angle:03d}', 'transform_matrix': matrix})
            record = {'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'source_size': list(image.size), 'mask_bbox_threshold16': bbox,
                      'square_crop': list(crop), 'output': filename, 'azimuth_degrees': angle}
            records.append(record)
            print(json.dumps(record), flush=True)
    finally:
        client.shutdown()
    (output / 'transforms.json').write_text(json.dumps({'camera_angle_x': fov, 'mesh_scale': 1.0, 'frames': frames}, indent=2) + '\n')
    (output / 'preparation.json').write_text(json.dumps({
        'mask_backend': 'rembg/u2net', 'size': args.size,
        'camera_assumption': 'Uncalibrated cardinal orbit, zero elevation, centered subject of unit height; FOV 20 degrees.',
        'normalization': 'Per-view uniform square crop/pad, mask bbox height occupies 1/1.1 of image. No nonuniform stretching.',
        'camera_distance': distance, 'records': records,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
