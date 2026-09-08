"""Standalone GPU worker for the official Pixal3D multiview pipeline."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def generate_geometry(pipeline, views, settings):
    """Run the official MV shape cascade, stopping before texture conditioning.

    Shape stages follow Pixal3D pixal3d_image_to_3d.py (upstream
    f7cf38429b0bd264f1995f0f8743a88b1c728b94); use its MV condition builders.
    """
    import torch
    from pixal3d.modules.sparse import SparseTensor

    self = pipeline
    image = views
    camera_angle_x = views['camera_angle_x']
    distance = views['camera_distance']
    mesh_scale = views['mesh_scale']
    hr_resolution = settings['resolution']
    max_num_tokens = 49152
    num_samples = 1
    sparse_structure_sampler_params = {'steps': settings['steps']}
    shape_slat_sampler_params = {'steps': settings['steps']}
    torch.manual_seed(settings['seed'])
    # ---- Stage 1: Sparse Structure (proj) ----
    cond_ss = self.get_proj_cond_ss(
        [image],
        camera_angle_x=camera_angle_x,
        distance=distance,
        mesh_scale=mesh_scale,
    )
    ss_res = 32
    coords = self.sample_sparse_structure(
        cond_ss, ss_res,
        num_samples, sparse_structure_sampler_params
    )
    del cond_ss
    torch.cuda.empty_cache()

    # ---- Stage 2: Shape LR 512 (proj) ----
    cond_shape_lr = self.get_proj_cond_shape(
        self.image_cond_model_shape_512, [image], coords,
        camera_angle_x=camera_angle_x,
        distance=distance,
        mesh_scale=mesh_scale,
    )
    lr_slat = self.sample_shape_slat(
        cond_shape_lr, self.models['shape_slat_flow_model_512'],
        coords, shape_slat_sampler_params
    )
    del cond_shape_lr
    torch.cuda.empty_cache()

    # ---- Stage 3a: Upsample LR → HR ----
    if self.low_vram:
        self.models['shape_slat_decoder'].to(self.device)
        self.models['shape_slat_decoder'].low_vram = True
    hr_coords = self.models['shape_slat_decoder'].upsample(lr_slat, upsample_times=4)
    if self.low_vram:
        self.models['shape_slat_decoder'].cpu()
        self.models['shape_slat_decoder'].low_vram = False

    lr_resolution = 512
    actual_hr_resolution = hr_resolution
    while True:
        grid_res = actual_hr_resolution // 16
        quant_coords = torch.cat([
            hr_coords[:, :1],
            ((hr_coords[:, 1:] + 0.5) / lr_resolution * (grid_res - 1)).round().int(),
        ], dim=1)
        hr_coords_unique = quant_coords.unique(dim=0)
        num_tokens = hr_coords_unique.shape[0]
        if num_tokens < max_num_tokens or actual_hr_resolution == 1024:
            break
        actual_hr_resolution -= 128

    actual_grid_res = actual_hr_resolution // 16
    del lr_slat, hr_coords, quant_coords
    torch.cuda.empty_cache()

    # ---- Stage 3b: Shape HR (proj) ----
    cond_shape_hr = self.get_proj_cond_shape(
        self.image_cond_model_shape_1024, [image], hr_coords_unique,
        camera_angle_x=camera_angle_x,
        distance=distance,
        mesh_scale=mesh_scale,
        grid_resolution_override=actual_grid_res,
    )
    noise_hr = SparseTensor(
        feats=torch.randn(hr_coords_unique.shape[0], self.models['shape_slat_flow_model_1024'].in_channels).to(self.device),
        coords=hr_coords_unique,
    )
    sampler_params_hr = {**self.shape_slat_sampler_params, **shape_slat_sampler_params}
    flow_model_hr = self.models['shape_slat_flow_model_1024']
    if self.low_vram:
        flow_model_hr.to(self.device)
    hr_slat = self.shape_slat_sampler.sample(
        flow_model_hr,
        noise_hr,
        **cond_shape_hr,
        **sampler_params_hr,
        verbose=True,
        tqdm_desc=f"Sampling HR shape SLat (proj, {actual_hr_resolution})",
    ).samples
    if self.low_vram:
        flow_model_hr.cpu()
    std = torch.tensor(self.shape_slat_normalization['std'])[None].to(hr_slat.device)
    mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(hr_slat.device)
    shape_slat = hr_slat * std + mean
    del cond_shape_hr, noise_hr, hr_slat, hr_coords_unique
    torch.cuda.empty_cache()

    meshes, _ = self.decode_shape_slat(shape_slat, actual_hr_resolution)
    return meshes, actual_hr_resolution


def run(request_path: Path):
    request = json.loads(request_path.read_text())
    if request.get('protocol') != 1:
        raise ValueError('Unsupported Pixal3D request protocol')
    output = request_path.resolve().parent
    root = Path(request['root'])
    settings = request['settings']
    os.environ.setdefault('ATTN_BACKEND', 'sdpa')
    sys.path.insert(0, str(root))
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
        if request.get('operation') == 'texture':
            from pixal3d_texture_runner import texture_mesh
            report.update(texture_mesh(upstream, request, output))
            report.update(status='success', output_bytes=(output / report['shape']).stat().st_size)
            return
        views = upstream.load_views(str(output / request['views_dir']))
        upstream.check_main_view(views)
        pipeline = upstream.init_pipeline(request['model_path'], low_vram=True)
        print('Generating Pixal3D multiview geometry', flush=True)
        with torch.no_grad():
            meshes, resolution = generate_geometry(pipeline, views, settings)
        mesh = meshes[0]
        print('Repairing and simplifying Pixal3D geometry', flush=True)
        import numpy as np
        from trellis_mesh_postprocess import run_mesh_postprocess
        cache = output / 'decoded-mesh.npz'
        np.savez_compressed(cache, vertices=mesh.vertices.cpu().numpy(), faces=mesh.faces.cpu().numpy())
        post_settings = {
            'fill_holes': True, 'fill_hole_perimeter': 0.03,
            'remesh': True, 'remesh_band': 1.0, 'remesh_project': 0.0,
            'simplify': True, 'decimation_target': settings['decimation_target'],
            'cleanup': True, 'final_repair': False,
            'remove_isolated_double_faces': True, 'remove_degenerate_faces': True,
            'exact_face_count': True, 'input_coordinates': 'gltf_y_up', 'fix_winding': True,
        }
        target, post_report = run_mesh_postprocess(cache, output, post_settings, int(resolution),
            progress=lambda message: print(message, flush=True))
        report.update(status='success', shape=target.name, textured=False,
                      actual_resolution=int(resolution), postprocess=post_report,
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
