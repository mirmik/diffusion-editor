"""Prepare masked views and nominal orbit cameras for Pixal3D multiview."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import threading

import numpy as np
from PIL import Image

from .model import Pixal3DSettings, ViewKey, ViewSlot


def orbit_camera(key: ViewKey, distance: float) -> list[list[float]]:
    a, e = math.radians(key.azimuth), math.radians(key.elevation_degrees)
    s, c, se, ce = math.sin(a), math.cos(a), math.sin(e), math.cos(e)
    return [[c, -s * se, s * ce, distance * s * ce],
            [s, c * se, -c * ce, -distance * c * ce],
            [0, ce, se, distance * se], [0, 0, 0, 1]]


def prepare_views(slots: tuple[ViewSlot, ...], output: Path, settings: Pixal3DSettings,
                  cancel: threading.Event, segment, on_progress=None) -> Path:
    """Use existing alpha when available, otherwise call the isolated segmenter.

    Nominal cameras are estimates, not calibration of generated images. All views
    participate, with eye-000 first. Uniform scaling preserves image proportions.
    """
    main = ViewKey('eye', 0)
    populated = [slot for slot in slots if slot.populated]
    if not any(slot.key == main for slot in populated):
        raise ValueError('Front view is required for Pixal3D')
    populated.sort(key=lambda slot: (slot.key != main, slot.key.elevation_degrees, slot.key.azimuth))
    for slot in populated:
        if not Path(slot.image_path).is_file():
            raise FileNotFoundError(f'Missing view {slot.key.stable_id}: {slot.image_path}')
    output.mkdir(parents=True, exist_ok=True)
    fov = math.radians(settings.fov)
    distance = 1.1 / (2 * math.tan(fov / 2))
    frames, records = [], []
    for index, slot in enumerate(populated, 1):
        if cancel.is_set():
            raise RuntimeError('Pixal3D view preparation cancelled')
        if on_progress:
            on_progress(f'Preparing Pixal3D view {index}/{len(populated)}: {slot.key.stable_id}')
        path = Path(slot.image_path).resolve()
        with Image.open(path) as source:
            rgba = source.convert('RGBA')
        alpha = np.asarray(rgba.getchannel('A'))
        existing_alpha = bool(np.any(alpha < 255))
        if not existing_alpha:
            alpha = np.asarray(segment(np.asarray(rgba.convert('RGB')), cancel, on_progress=on_progress), dtype=np.uint8)
            if alpha.shape != (rgba.height, rgba.width):
                raise ValueError(f'Mask dimensions do not match view {slot.key.stable_id}')
            rgba.putalpha(Image.fromarray(alpha))
        ys, xs = np.nonzero(alpha >= 16)
        if not len(xs):
            raise ValueError(f'Empty foreground mask for {slot.key.stable_id}')
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if settings.normalize_views:
            span = math.ceil(max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.1)
            left, top = round((bbox[0] + bbox[2] - span) / 2), round((bbox[1] + bbox[3] - span) / 2)
        else:
            span = max(rgba.size)
            left, top = (rgba.width - span) // 2, (rgba.height - span) // 2
        crop = (left, top, left + span, top + span)
        name = f'{slot.key.stable_id}.png'
        rgba.crop(crop).resize((1024, 1024), Image.Resampling.LANCZOS).save(output / name)
        frames.append({'file_path': name, 'name': slot.key.stable_id,
                       'transform_matrix': orbit_camera(slot.key, distance)})
        records.append({'id': slot.key.stable_id, 'source': str(path),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'existing_alpha': existing_alpha, 'source_size': list(rgba.size),
                        'mask_bbox': bbox, 'square_crop': crop})
    if cancel.is_set():
        raise RuntimeError('Pixal3D view preparation cancelled')
    manifest = output / 'transforms.json'
    manifest.write_text(json.dumps({'camera_angle_x': fov, 'mesh_scale': 1.0, 'frames': frames}, indent=2) + '\n')
    (output / 'preparation.json').write_text(json.dumps({
        'camera_model': 'nominal orbit, not calibrated', 'normalize_views': settings.normalize_views,
        'normalization': 'uniform square crop/pad; longest foreground dimension occupies 1/1.1',
        'distance': distance, 'views': records,
    }, indent=2) + '\n')
    return manifest
