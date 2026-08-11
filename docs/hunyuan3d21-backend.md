# Hunyuan3D 2.1 reconstruction backend

Hunyuan3D 2.1 is an optional image-to-geometry-and-PBR backend. The editor runs
the installed ComfyUI custom-node implementation in an isolated subprocess; it
does not start ComfyUI and does not execute code inside a running editor.

The default local runtime is:

```text
ComfyUI:  /home/mirmik/soft/ComfyUI
Python:   /home/mirmik/soft/ComfyUI/venv/bin/python
Node:     /home/mirmik/soft/ComfyUI/custom_nodes/ComfyUI-Hunyuan3d-2-1
Shape:    models/diffusion_models/hunyuan3d-dit-v2-1.ckpt
VAE:      models/vae/hunyuan3d-vae-v2-1.ckpt
```

Paths can be overridden with:

```text
DIFFUSION_EDITOR_HUNYUAN3D21_ROOT
DIFFUSION_EDITOR_HUNYUAN3D21_PYTHON
DIFFUSION_EDITOR_HUNYUAN3D21_NODE_ROOT
DIFFUSION_EDITOR_HUNYUAN3D21_DIT
DIFFUSION_EDITOR_HUNYUAN3D21_VAE
DIFFUSION_EDITOR_GLTF_TRANSFORM
```

`gltf-transform` is required because the Hunyuan exporter writes data-URI
textures, while the current Termin native importer reads textures embedded in
GLB buffer views. The runner normalizes both the texture-stage preview and the
final GLB automatically.

## Stages and controls

The backend maps its native pipeline to the editor's existing stages:

- `HR shape flow`: image conditioning and DiT shape-latent sampling;
- `HR shape latent`: hierarchical VAE decode and raw geometry GLB;
- `Texture flow`: six-view PBR generation;
- `Texture latent`: UV bake, seam inpaint, and a textured preview GLB;
- `Final mesh`: normalized PBR GLB ready for the Termin viewport.

The artifact directory also retains the shape latent, preprocessed RGBA input,
six albedo/metal-roughness/normal/position views, bake masks, baked maps, and
final albedo and metal-roughness maps. The current stage panel exposes the raw
mesh and textured GLB previews; the future artifact-workspace UI can expose all
of the retained 2D intermediates without rerunning generation.

`Sampling steps` controls shape sampling. Hunyuan-specific controls set shape
guidance, octree decode resolution, texture steps, and texture guidance.
`Texture size` and `Final mesh faces` are shared controls. HR resolution,
camera FOV, LR conditioning, and Low VRAM do not apply to this backend.

FlashVDM decode is intentionally disabled: with the installed 2.1 checkpoint it
produced non-finite vertices, while the hierarchical marching-cubes decoder
produced a valid watertight mesh. Very low shape-step counts may produce no
surface; the runner reports this explicitly and suggests changing the seed or
increasing the step count.

## Verification

The editor-owned runner was tested on the local CUDA runtime through all
stages. A reduced smoke (`5` shape steps, octree `96`, one texture step,
texture `1024`) produced a normalized textured GLB readable by
`NativeGLBDocument`, with one mesh, 15,530 triangles, and an embedded PNG base
color texture. Unit tests use a fake subprocess runner and require no model or
GPU.

Masked geometry and texture refinement remain Pixal3D-only. Hunyuan's saved
shape latent is retained as a checkpoint for future stage reuse, but the current
Generate action still starts a new run.
