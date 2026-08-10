# Pixal3D masked refinement backend

The first refinement slice deliberately defines a UI-neutral boundary. The
editor host supplies two same-sized images:

- a square, already preprocessed conditioning image;
- a soft grayscale mask in exactly the same image coordinates.

Canvas selection, crop overlays, brush controls, and conversion from document
coordinates into this preprocessed space belong to a future UI adapter. They
are not worker responsibilities.

## Base checkpoint

Once the base pipeline has sampled its normalized HR shape latent, the staged
runner writes a session-local, pickle-free NPZ checkpoint containing:

- protocol version;
- fixed sparse coordinates and normalized 32-channel features;
- actual HR resolution after token-budget fallback;
- camera FOV, distance, and mesh scale;
- SHA-256 of the preprocessed conditioning image;
- resolved Pixal3D model path.

The checkpoint remains owned by `Pixal3DProcessClient` and is deleted at engine
shutdown. Project persistence and reopening refinements remain separate work.

## Masked flow

The implementation is adapted from the validated experiment in
`~/project/Hunyuan3D-Omni-test/refine_pixal3d_shape_latent.py`.

The image mask is projected onto the existing HR sparse coordinates with the
same camera transform used by Pixal3D projection conditioning. Given normalized
base features `x0`, fixed noise `eps`, and flow time `t`, the frozen reference
trajectory is:

```text
sigma(t) = sigma_min + (1 - sigma_min) t
x_ref(t) = (1 - t) x0 + sigma(t) eps
```

Sampling starts at `strength` rather than pure noise. After each Euler step,
the predicted features are kept inside the soft token mask and blended with
`x_ref` outside it. At the end, the fully unselected part is restored exactly
to `x0`.

The current worker decodes a geometry-only GLB and writes another compatible
checkpoint, so refinements may be chained. Texture refinement/regeneration is
intentionally not part of this slice.

## Run model

`ReconstructionLayer` owns transient immutable `ReconstructionRun` values.
A base publication resets the run tuple. A masked-refine publication appends a
child with `parent_run_id` and makes it active. The compatibility fields
`glb_path` and mesh statistics continue to describe the active run.

The native host exposes `start_reconstruction_refine(node, condition, mask)`
as the integration point for the eventual panel. It snapshots both images,
binds the asynchronous result to the launching node and parent run, and never
overwrites the base run.

## Current limitations

- HR coordinates and occupancy are fixed, so large topology changes are not
  possible.
- A 2D projection mask selects every visible-depth token under its pixels; a
  later UI may add depth or canonical-space controls.
- The visible refined GLB is geometry-only.
- Runs and checkpoints are session-local and are reconstructed as one legacy
  base run after reopening a project.
- Detail-crop conditioning and its crop-to-body registration remain a separate
  extension.
