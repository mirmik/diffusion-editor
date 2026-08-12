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

## Enlarged HR geometry refine

HR refine treats the selected image region as an independent Pixal3D object:

1. project the soft mask onto the decoded base HR mesh and derive padded robust
   3D bounds;
2. crop the masked source, estimate its own camera and run a complete local
   sparse → LR → HR shape cascade at the requested HR resolution;
3. fit the decoded local mesh into the selected base bounds;
4. publish the registered local mesh as `Refine output · before merge`;
5. retain an ellipsoidal overlap collar, reconstruct one surface with CUDA
   narrow-band dual contouring and simplify it for the main viewport.

The result checkpoint is explicitly marked `enlarged_hr_geometry_v1` and keeps
the base and local HR latents plus their registration instead of pretending the
fused surface is one canonical Pixal3D latent. It cannot be used as the source
of another HR or texture-latent refine. The current result is geometry-only;
local texture generation and transfer must be connected separately before a
textured composite can be published.

## Masked LR refinement

LR refine is currently omitted from the user-facing workspace. The ordinary
LR generation still exposes its decoded 512-resolution preview and resumable
checkpoint as an approval point before HR.

The enlarged experiment can generate a masked crop independently on the full
32³ local grid and register its decoded mesh into the base LR geometry. It
cannot be flattened back into one standard Pixal3D LR latent: inserting even a
small number of local features changes global decoding and can remove the base
body. A private composite of base LR, local LR and registration is technically
possible, but it must be propagated independently through HR and texture before
it can become a real workflow. Until that continuation exists, exposing LR
refine would provide a preview which the next stage cannot truthfully consume.

HR geometry and texture refinement keep independent sampling parameter sets.
Mask painting is intentionally shared: paint/erase mode, brush size, hardness,
flow and clear live in the separate, always-visible `Mask tools` block near the
top of the workspace and operate on the same canvas mask.

The auxiliary `Refine output · before merge` viewport belongs to the atomic
enlarged-refine contract: it shows only the independently generated local
fragment that is subsequently registered and merged into the working model.
The internal crop, local camera, sparse/LR generation and registration steps
are not user-facing graph operations. The legacy HR and texture paths
deliberately publish no substitute artifact while they are migrated to
equivalent atomic local runners.

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

- LR refine is disabled because an enlarged local LR cannot be flattened into
  the canonical base latent without changing global decoding.
- Enlarged HR output is geometry-only. Texture generation/transfer for its
  composite checkpoint and repeated refinement of an existing composite are
  not connected yet.
- A 2D projection mask selects every visible-depth token under its pixels; a
  later UI may add depth or canonical-space controls.
- Texture latent preservation is local, but decoding, UV generation and baking
  rebuild the complete atlas and may introduce small differences outside the
  selected region.
- The mask currently reuses the document-level selection. It is transient and
  shared rather than stored separately on each reconstruction object.
- Runs and checkpoints are session-local and are reconstructed as one legacy
  base run after reopening a project.
