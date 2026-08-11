# Pixal3D local 1536 geometry experiment

This experiment treats the selected image region as an independent Pixal3D
input, gives that region the complete 1536 shape grid, registers the decoded
local mesh back into the base mesh, and reconstructs a narrow band around their
overlap.

It deliberately does **not** replace the editor's shape checkpoint. Pixal3D
shape latents are tied to their canonical coordinate frame: scaling and merging
them directly produced nested or disconnected surfaces in the earlier
experiments. The current prototype therefore joins decoded geometry.

## Pipeline

1. Project base-mesh vertices into the saved conditioning camera and sample the
   refine mask.
2. Build robust padded 3D bounds from the selected base vertices.
3. Crop the conditioning image around the mask and run the ordinary Pixal3D
   sparse/LR/HR shape cascade at 1536 for that crop.
4. Fit a uniform local-to-global transform from robust mesh bounds.
5. Keep an ellipsoidal overlap collar from both meshes.
6. Optionally run CUDA narrow-band dual contouring over the overlap to produce
   one geometric surface.

The experimental entry point is
`diffusion_editor/workers/pixal3d_local_detail_runner.py`. Use `--fuse` to
produce `fused.glb`; `registration.glb` colors the base and local meshes
separately, and `fused-preview.glb` is a connected decimation suitable for fast
inspection. Full overlap export is opt-in with `--export-overlap` because these
diagnostics can occupy hundreds of megabytes.

## Current evidence

The automatic registration on the reference head experiment found scale
`0.2814` and translation `(-0.0001, 0.3521, -0.0254)`. The earlier manual fit
was scale `0.27` and translation `(0, 0.331, 0)`, so the mask-derived placement
is in the expected range. A full 12-step local mesh fused without a visible
silhouette gap. A two-step smoke mesh did not contain a usable face and
correctly produced a bad replacement, demonstrating that the local result needs
quality validation before it can be accepted automatically.

The geometry-only fused diagnostic remains large (about 6.5 million vertices
and 13 million triangles in the reference run) and may contain disconnected
components. UI integration, component cleanup, quality rejection, and a
compact persistent representation remain follow-up work. The separate texture
transfer experiment below operates on a controlled decimation of this result.

## Local texture transfer

The follow-up prototype keeps texture generation local as well. The
`pixal3d_local_texture_generator.py` runner takes a saved local shape latent,
subdivision guides and raw mesh through three isolated phases: Texture Flow,
streamed PBR decoding, and a local 4096-atlas GLB export. Separate processes
are intentional because retaining weights and the 10-million-voxel local PBR
volume together exceeds the useful VRAM margin.

`pixal3d_local_texture_transfer.py` then projects every fused target point to
the nearest triangles of the textured base and registered local meshes with a
CUDA BVH. It interpolates their barycentric UVs, samples both base-color maps,
and blends them in linear light using the same ellipsoidal overlap collar as
geometry fusion. The runner can retain diagnostic vertex colors or create a
new UV unwrap and bake the blend into an ordinary GLB atlas.

The reference run decoded 10,527,721 local PBR voxels and produced a fused
951k-triangle GLB with a 4096 atlas. Base colors remain outside the ROI and
local colors cover the head/neck core without a hard seam. The experiment only
transfers base color for now; metallic, roughness, alpha, normals, automatic
texture-quality rejection and UI integration remain production follow-ups.
