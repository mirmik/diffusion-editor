# TRELLIS.2 reconstruction backend

`3D Reconstruction` exposes `Pixal3D` and `TRELLIS.2` in its Backend control.
Pixal3D remains the default. The selected backend is persisted with generation
parameters, snapshotted when generation starts and recorded on every transient
`ReconstructionRun`, so versions and checkpoints cannot be confused by the UI.

The TRELLIS.2 backend runs in an isolated subprocess and defaults to the local
installation:

```text
root:   /home/mirmik/soft/TRELLIS.2
python: /home/mirmik/soft/TRELLIS.2/venv/bin/python
model:  /home/mirmik/soft/TRELLIS.2/models/TRELLIS.2-4B
```

These may be overridden per launch with:

```text
DIFFUSION_EDITOR_TRELLIS2_ROOT
DIFFUSION_EDITOR_TRELLIS2_PYTHON
DIFFUSION_EDITOR_TRELLIS2_MODEL
```

The editor-owned staged runner publishes the same stage protocol as Pixal3D:
sparse occupancy, LR shape flow/latent, LR-to-HR coordinates, HR shape
flow/latent, texture flow/latent and final textured PBR GLB. Existing viewport,
progress and per-Version artifact handling therefore remain backend-neutral.

TRELLIS.2 uses its native global image conditioning. `LR conditioning` and
camera FOV controls are disabled for this backend because they describe
Pixal3D's projected conditioning. Resolution, sampling steps, seed, token
fallback, final face budget, texture size and low-VRAM execution are shared.

## Current boundary

Masked geometry and texture refinement are currently Pixal3D-only. TRELLIS.2
runs intentionally publish no compatible refine checkpoints, and the native UI
disables masked-refine actions for those versions. Porting latent refinement
requires a separate conditioning and image-to-canonical-camera design rather
than treating the two checkpoint formats as interchangeable.
