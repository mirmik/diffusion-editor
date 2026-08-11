# Pixal3D masked refinement

The editor supplies two same-sized canvas-space images to the worker:

- the original reconstruction source image;
- a soft grayscale mask painted with the document selection brush.

The worker resizes the pair together, obtains the same foreground alpha used by
Pixal3D, and applies an identical square subject crop to the source and mask.
This keeps the painted mask registered with the camera conditioning without
duplicating Pixal3D preprocessing in the editor process.

## Native UI

Selecting a 3D Reconstruction object exposes a `Masked refinement` section in
its contextual left panel. The 2D source canvas and 3D viewport remain visible
side by side. `Paint refine mask` enables the existing soft document selection
brush and its red overlay; erasing, brush size, hardness, flow, strength,
steps and seed are adjustable in the same panel. `Refine geometry in mask` and
`Refine texture in mask` snapshot the source and mask and start asynchronous
child runs. `Resize masked detail to 1024` is enabled by default and gives the
selected image region the full spatial conditioning resolution.

The `Version` control switches the viewport between `Base` and successive
`Refined N` runs. Switching a version also chooses the checkpoint used by the
next refinement.

Loading another stage or run preserves the orbit camera. Once the user clicks
a preview stage, incoming artifacts no longer replace that choice; automatic
stage following remains active only before an explicit selection.

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

After decoding the refined shape latent, the worker continues through a fresh
`Texture flow`, saves a texture checkpoint, decodes PBR voxels and bakes a
textured GLB. Geometry refinements may therefore be compared without losing
the ordinary Pixal3D texture pass.

## Masked LR refinement

The experimental pipeline can refine an accepted `lr_shape_flow` or
`lr_shape_latent` session checkpoint before HR coordinates are generated. The
saved LR features are denormalized model output, so the worker first maps them
back through Pixal3D's shape-latent mean and standard deviation. It projects
the canvas mask onto the fixed 32³ LR token grid, runs the same partial-noise
reference trajectory with `shape_slat_flow_model_512`, restores unselected
features exactly, and denormalizes the result again.

The operation publishes a decoded 512-resolution LR mesh preview and a new
pickle-free `lr_shape_latent` session checkpoint. Continuing through
`HR coordinates` uses the normal resume path, so the upsample is derived from
the most recently completed LR variant rather than from the original latent.

The experimental workspace retains Base LR and each Refined LR result as
separate session-local variants. The `Generate LR shape` artifact is always the
unrefined Base LR preview. On `Refine LR shape`, `Refine source` explicitly
chooses the checkpoint used by the next run and defaults to Base LR; selecting
another preview does not alter it. This permits several independent refines
from Base as well as an intentional refine chain from a chosen Refined LR
variant.

LR, HR geometry and texture refinement store independent sampling parameter
sets. Each operation therefore has its own strength, step count and seed rather
than inheriting the values last used by another refine phase. Mask painting is
intentionally shared: paint/erase mode, brush size, hardness, flow and clear
live in the separate `Mask tools` block and operate on the same canvas mask.

## Masked texture refinement

Every completed textured run owns two independent checkpoints:

- normalized HR shape SLat and fixed sparse coordinates;
- normalized texture SLat on the same coordinates.

`Refine texture in mask` keeps the selected shape fixed, projects the canvas
mask onto its sparse tokens and partially re-runs only Texture flow. After each
Euler step, texture features outside the soft token mask follow the saved
reference trajectory and are restored to the original texture latent at the
end. The PBR volume and GLB atlas are then decoded and baked again.

For additional detail, the worker extracts a padded square crop around the
mask and feeds it through the 1024 conditioning encoder. Its projection grid
first projects sparse 3D coordinates with the original full-frame camera, then
remaps those full-frame pixels into crop coordinates. The crop therefore acts
like the image editor's `Resize to 1024` without pretending that an off-centre
crop is a centred camera image. Full-frame projected features remain outside
the soft token mask; exact crop-projected features are blended inside it. Both
global token sequences remain available to the flow model. The same path is
used for masked geometry refine and masked texture refine.

## Run model

`ReconstructionLayer` owns transient immutable `ReconstructionRun` values.
A base publication resets the run tuple. A masked-refine publication appends a
child with `parent_run_id` and makes it active. The compatibility fields
`glb_path` and mesh statistics continue to describe the active run.

Each run also owns an immutable snapshot of its stage statuses, progress and
preview artifacts. Switching `Version` restores that complete snapshot and
keeps the currently selected preview stage when the target run has it. This
allows direct Base/refined comparisons at HR shape latent (and every other
available preview), rather than changing only Final mesh. Texture-only runs
inherit their parent's geometry-stage artifacts because their shape is fixed.

The native host snapshots both images, binds the asynchronous result to the
launching node and selected parent run, and never overwrites the base run.

## Current limitations

- HR coordinates and occupancy are fixed, so large topology changes are not
  possible.
- A 2D projection mask selects every visible-depth token under its pixels; a
  later UI may add depth or canonical-space controls.
- Texture latent preservation is local, but decoding, UV generation and baking
  rebuild the complete atlas and may introduce small differences outside the
  selected region.
- The mask currently reuses the document-level selection. It is transient and
  shared rather than stored separately on each reconstruction object.
- Runs and checkpoints are session-local and are reconstructed as one legacy
  base run after reopening a project.
