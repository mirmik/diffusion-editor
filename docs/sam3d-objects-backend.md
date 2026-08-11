# SAM 3D Objects reconstruction backend

SAM 3D Objects is an optional gated image-to-3D backend. The editor runs it in
its regular-CPython environment and keeps PyTorch/CUDA packages out of the
CPython 3.14t UI process.

The default local runtime is:

```text
root:        /home/mirmik/soft/sam-3d-objects
python:      /home/mirmik/soft/sam-3d-objects/venv/bin/python
checkpoints: /home/mirmik/soft/sam-3d-objects/checkpoints
```

These locations can be overridden with
`DIFFUSION_EDITOR_SAM3D_OBJECTS_ROOT`,
`DIFFUSION_EDITOR_SAM3D_OBJECTS_PYTHON`, and
`DIFFUSION_EDITOR_SAM3D_OBJECTS_CHECKPOINTS`.

After accepting the `facebook/sam-3d-objects` license, the checkpoint directory
must contain `pipeline.yaml` and its referenced checkpoint/config files. The
RTX 5090 runtime uses current Torch/CUDA, SDPA attention, `spconv`, `gsplat`,
and `nvdiffrast`. PyTorch3D is needed only for CPU-side structures and camera
math. Texture observation rendering uses `gsplat`; mesh UV baking uses
`nvdiffrast`.

The editor publishes these stages:

1. **Point cloud** — the colored MoGe pointmap.
2. **Sparse occupancy** — occupied cells selected by the sparse-structure
   diffusion model.
3. **HR shape flow** — structured-latent sampling progress.
4. **HR shape latent** — the raw decoded mesh.
5. **Texture latent** — a conventional colored point preview of the decoded
   Gaussian appearance. The full Gaussian PLY is retained as a checkpoint.
6. **Final mesh** — postprocessed, simplified mesh with a texture baked from
   100 Gaussian renders.

The SAM-specific controls are sparse and structured-latent step counts, their
two guidance scales, and the fraction of mesh faces removed by simplification.
The shared texture-size control selects the final baked texture resolution.

If the input composite already has transparency, its alpha channel is used as
the object mask. A fully opaque composite is passed through `rembg` first.
PLY stages remain in SAM's Z-up space; GLB artifacts are converted to glTF
Y-up so the editor importer restores the same Z-up orientation when switching
stages.

The current worker is process-per-generation. A cold representative run uses
about 18.5 GB of CUDA memory and spends most of its time loading weights.
Persistent workers, stage caching, and true anisotropic Gaussian-splat preview
are tracked separately.
