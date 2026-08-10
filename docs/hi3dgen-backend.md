# Hi3DGen reconstruction backend

Hi3DGen is an optional geometry-only image-to-3D backend. It first predicts a
normal map with StableNormal/YOSO and then conditions its sparse structure and
structured-latent samplers on that normal map. It does not generate textures.

The backend runs in an isolated subprocess and defaults to:

```text
root:          /home/mirmik/soft/Stable3DGen
python:        /home/mirmik/soft/Stable3DGen/venv/bin/python
model:         /home/mirmik/soft/Stable3DGen/weights/trellis-normal-v0-1
StableNormal:  /home/mirmik/soft/StableNormal
```

The root also contains `weights/yoso-normal-v1-8-1` and `weights/BiRefNet`.
Paths can be overridden per launch with:

```text
DIFFUSION_EDITOR_HI3DGEN_ROOT
DIFFUSION_EDITOR_HI3DGEN_PYTHON
DIFFUSION_EDITOR_HI3DGEN_MODEL
DIFFUSION_EDITOR_STABLE_NORMAL_ROOT
```

The tested RTX 5090 runtime uses Python 3.10, PyTorch 2.10/CUDA 13,
`spconv-cu126` 2.3.8, xFormers 0.0.35, diffusers 0.28, transformers 4.36,
and timm 1.x. DINOv2 uses PyTorch SDPA because current xFormers fp32 kernels
do not support Blackwell; Hi3DGen sparse attention continues to use xFormers.

## Stages and controls

The editor-owned runner publishes `Normal map`, `Sparse occupancy`,
`HR shape flow`, `HR shape latent`, and `Final mesh`. Sparse occupancy is a GLB
made from merged occupied boxes. HR shape latent is the full decoded geometry;
the final GLB is decimated to `Final mesh faces`.

`Sampling steps` controls sparse-structure flow. `Hi3D latent steps` controls
structured-latent flow. Both use `Hi3D guidance`; `Normal resolution` selects
512, 768, or 1024. Texture size, HR resolution, camera FOV, and low-VRAM
controls do not apply to this backend.

The normal PNG is retained as a stage artifact and the stage reports progress,
but the current 3D viewport cannot yet present 2D stage artifacts directly.

## Current boundary

Hi3DGen versions do not expose Pixal3D-compatible geometry or texture refine
checkpoints. Masked refinement is therefore disabled. Model startup currently
dominates a short run; keeping a warm worker is a possible follow-up once the
backend has proved useful in interactive testing.
