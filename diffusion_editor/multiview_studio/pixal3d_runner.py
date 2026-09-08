"""Standalone GPU worker for the official Pixal3D multiview pipeline."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def run(request_path: Path):
    request = json.loads(request_path.read_text())
    if request.get('protocol') != 1:
        raise ValueError('Unsupported Pixal3D request protocol')
    output = request_path.resolve().parent
    root = Path(request['root'])
    settings = request['settings']
    os.environ.setdefault('ATTN_BACKEND', 'sdpa')
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable for Pixal3D multiview')
    spec = importlib.util.spec_from_file_location('pixal3d_mv', root / 'inference_mv.py')
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    from pixal3d.pipelines import rembg

    class MaskedInputOnly:
        def __init__(self, **kwargs):
            pass

        def to(self, device):
            return self

        def cpu(self):
            return self

        def __call__(self, image):
            raise ValueError('Expected a prepared alpha mask')

    # All views are masked before this worker starts; never load gated RMBG.
    rembg.BiRefNet = MaskedInputOnly
    started = time.monotonic()
    report = {'backend': 'pixal3d', 'status': 'running', 'settings': settings,
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
    try:
        views = upstream.load_views(str(output / request['views_dir']))
        upstream.check_main_view(views)
        pipeline = upstream.init_pipeline(request['model_path'], low_vram=True)
        print('Generating Pixal3D multiview geometry and PBR texture', flush=True)
        meshes, (_, _, resolution) = pipeline.run_mv(
            views, seed=settings['seed'], return_latent=True,
            pipeline_type=f"{settings['resolution']}_cascade", max_num_tokens=49152,
            sparse_structure_sampler_params={'steps': settings['steps']},
            shape_slat_sampler_params={'steps': settings['steps']},
            tex_slat_sampler_params={'steps': settings['steps']},
        )
        mesh = meshes[0]
        print('Exporting Pixal3D GLB', flush=True)
        glb = upstream.o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout, grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=settings['decimation_target'], texture_size=settings['texture_size'],
            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
        )
        # Official export followed by a half-turn around glTF Y: Studio uses
        # TRELLIS's front convention, opposite to standalone Pixal3D.
        glb.apply_transform(np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64))
        target = output / 'model.glb'
        glb.export(target, extension_webp=True)
        report.update(status='success', shape='model.glb', actual_resolution=int(resolution),
                      views=len(views['view_names']), output_bytes=target.stat().st_size)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        temporary = output / 'result.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(output / 'result.json')


if __name__ == '__main__':
    run(Path(sys.argv[1]).resolve())
