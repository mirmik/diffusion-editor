"""Native Pixal3D MV texture sampling on an encoded, fixed input mesh."""
from pathlib import Path


def texture_mesh(upstream, request, output):
    import numpy as np
    import torch
    from pixal3d import models
    from pixal3d.modules import sparse
    from pixal3d.pipelines.trellis2_texturing import Trellis2TexturingPipeline
    from trellis_texture_runner import (
        _load_mesh, _shape_guide_subs, _bake_pbr_preserving_faces,
        _save_sparse, _save_guide_subs,
    )
    from trellis_refine_runner import _enable_chunked_encoder_mlp, _enable_chunked_decoder

    settings = request['texture_settings']
    resolution = 1024
    source, removed = _load_mesh(output / request['input_mesh'])
    # Studio Pixal3D geometry exports decoder coordinates directly. Keep that
    # frame for both camera conditioning and baking; do not use preprocess_mesh
    # (which would rotate it). Normalize only meshes outside the decoder cube.
    prepared = source.copy()
    center = np.zeros(3)
    scale = 1.0
    if not np.isfinite(source.vertices).all() or source.extents.max() <= 0:
        raise ValueError('Invalid input mesh bounds')
    if np.abs(source.vertices).max() >= 0.5:
        center = source.bounds.mean(axis=0)
        scale = 0.99999 / source.extents.max()
        prepared.vertices = (source.vertices - center) * scale

    print('[load] Pixal3D MV texture denoiser and decoder', flush=True)
    cls = upstream.Pixal3DMVImageTo3DPipeline
    cls.model_names_to_load = ['tex_slat_flow_model_1024', 'tex_slat_decoder']
    pipeline = cls.from_pretrained(request['model_path'], 'pipeline_mv.json')
    pipeline.low_vram = True
    pipeline._device = torch.device('cuda')
    pipeline.image_cond_model_tex_1024 = upstream.build_image_cond_model(upstream.IMAGE_COND_CONFIGS['tex_1024'])
    pipeline.models['shape_slat_encoder'] = models.from_pretrained(request['shape_encoder']).eval()
    encoder = pipeline.models['shape_slat_encoder']
    _enable_chunked_encoder_mlp(encoder, torch)
    decoder = pipeline.models['tex_slat_decoder']
    with torch.no_grad():
        print('[encode] current geometry, 1024', flush=True)
        # Shared shape VAE architecture, instantiated in the Pixal3D namespace.
        shape = Trellis2TexturingPipeline.encode_shape_slat(pipeline, prepared, resolution)
        guides = _shape_guide_subs(shape, len(decoder.blocks) - 1, sparse, torch)
        _save_sparse(output / 'encoded-shape-slat.npz', shape)
        _save_guide_subs(output / 'shape-guide-subs.npz', guides)
        views = upstream.load_views(str(output / request['views_dir']))
        upstream.check_main_view(views)
        print('[condition] all views together, Pixal3D projection', flush=True)
        condition = pipeline.get_proj_cond_shape(
            pipeline.image_cond_model_tex_1024, [views], shape.coords,
            camera_angle_x=views['camera_angle_x'], distance=views['camera_distance'],
            mesh_scale=views['mesh_scale'], grid_resolution_override=resolution // 16,
        )
        torch.manual_seed(settings['seed'])
        texture = pipeline.sample_tex_slat(condition, pipeline.models['tex_slat_flow_model_1024'],
                                           shape, {'steps': settings['steps']})
        _save_sparse(output / 'texture-slat.npz', texture)
        del condition, shape
        torch.cuda.empty_cache()
        print('[decode] Pixal3D PBR volume', flush=True)
        _enable_chunked_decoder(decoder, torch)
        decoder.to(pipeline.device)
        try:
            pbr = decoder(texture, guide_subs=guides)
            pbr.feats.mul_(0.5).add_(0.5)
        finally:
            decoder.cpu()
        del texture, guides
        torch.cuda.empty_cache()
        print('[bake] PBR onto current triangles without coordinate conversion', flush=True)
        textured = _bake_pbr_preserving_faces(prepared, pbr, resolution=resolution,
            texture_size=settings['texture_size'], device=pipeline.device, convert_to_gltf=False)
    textured.vertices = textured.vertices / scale + center
    textured.export(output / 'model.glb', extension_webp=True)
    return dict(shape='model.glb', textured=True, operation='texture',
                views=len(views['view_names']), actual_resolution=resolution,
                texture_settings=settings, input_faces=len(source.faces),
                output_faces=len(textured.faces), removed_degenerate_faces=removed,
                normalization_center=center.tolist(), normalization_scale=float(scale))
