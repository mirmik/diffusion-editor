# Canonical point-map dataset

The first geometry-head kill-test uses the Blender Studio Rain v2 character in
its authored rest pose.  The dataset is deliberately generated without a room,
pose augmentation, view-dependent cropping, or generated Qwen images.

## Asset provenance

Rain v2 is distributed by Blender Studio under CC-BY.  Downloading is kept out
of Git and records the official source, license, and archive checksum:

```bash
python3 scripts/download-rain-v2.py --output-dir /tmp/rain-v2-cache
```

An already downloaded official archive can be verified and extracted without
network access:

```bash
python3 scripts/download-rain-v2.py \
  --output-dir /tmp/rain-v2-cache \
  --archive /tmp/rain_v22.zip
```

## Rendering

Embedded scripts in the Blend file are intentionally disabled.  A 24-view
eye-level ring at 256 by 256 pixels is generated with:

```bash
blender --background --disable-autoexec \
  /tmp/rain-v2-cache/rain-v2/rain_rig.blend \
  --python scripts/render-rain-canonical-dataset.py -- \
  --output-dir /tmp/rain-canonical-v1 \
  --views 24 \
  --resolution 256
```

Repeat `--elevation` to render multiple production rings in one dataset.  For
the low-angle, eye-level, and elevated Multiple Angles rows:

```bash
blender --background --disable-autoexec \
  /tmp/rain-v2-cache/rain-v2/rain_rig.blend \
  --python scripts/render-rain-canonical-dataset.py -- \
  --output-dir /tmp/rain-canonical-production-3rings \
  --views 8 \
  --elevation=-30 --elevation=0 --elevation=30 \
  --resolution 256
```

`--distance` retains its legacy Blender-world-unit meaning by default. For
assets authored at unrelated world scales, pass
`--distance-unit character-height`; for example, `--distance 2.2` then places
the camera 2.2 evaluated character heights from the pelvis and records both the
relative parameter and resolved world distance in the manifest.

For a fast two-view smoke test, pass explicit azimuths:

```bash
blender --background --disable-autoexec \
  /tmp/rain-v2-cache/rain-v2/rain_rig.blend \
  --python scripts/render-rain-canonical-dataset.py -- \
  --output-dir /tmp/rain-canonical-smoke \
  --azimuth 0 --azimuth 90 \
  --resolution 64
```

The same renderer can import a local FBX into a clean Blender scene.  Record a
stable identity and pose label explicitly:

```bash
blender --background --factory-startup --disable-autoexec \
  --python scripts/render-rain-canonical-dataset.py -- \
  --asset /path/to/character.fbx \
  --asset-name Character --identity-id character --pose-id authored-a-pose \
  --output-dir /tmp/character-canonical \
  --views 8 --elevation=-30 --elevation=0 --elevation=30 \
  --resolution 256
```

For an imported asset with no armature, the renderer estimates the pelvis at
the median mesh-object X/Y origin and half the evaluated height.  The manifest
marks this estimate and whether an armature survived import.  Ray supervision
is intersected with rendered alpha (0.5 by default), avoiding targets on
nominal geometry that is visually transparent in hair or clothing materials.

An opened Blend scene normally contributes all visible character meshes. If a
pack stores several visible variants side by side, use repeated `--only-mesh`
arguments as an exact allow-list. Unlike the historical additive
`--include-mesh`, strict selection hides every unlisted mesh before bounds,
RGB, mask and geometry supervision are computed:

```bash
blender --background --disable-autoexec character-pack.blend \
  --python scripts/render-rain-canonical-dataset.py -- \
  --opened-blend-character \
  --only-mesh Character.003 \
  --rig-name CharacterRig.003 --pelvis-bone pelvis \
  --output-dir /tmp/character-canonical \
  --azimuth 0 --azimuth 180 --resolution 128
```

Each view contains:

- `rgb.png`: transparent RGBA render;
- `mask.png`: binary center-sample surface mask;
- `geometry.npz`: `canonical_xyz`, `normal`, `depth`, `mask`, and `object_id`;
- `camera.json`: intrinsics and canonical-to-camera extrinsics.

The top-level `manifest.json` fixes the asset provenance, Blender version,
canonical frame, object IDs, rendering parameters, and relative artifact paths.

## Coordinate contract

Canonical coordinates are right handed: X points to the character's right,
Y points up, and Z points forward towards the frontal camera.  For Rain the
origin is the world-space head of `RIG-rain/DEF-Pelvis`; imported unrigged FBX
assets use the explicit estimate recorded in their manifest.  One unit is the
character's evaluated full height.

Camera coordinates follow the usual computer-vision convention: X right,
Y down, and Z forward.  Pixel coordinates start at the top-left, and rays pass
through pixel centers `(column + 0.5, row + 0.5)`.

The renderer rejects a view when its canonical point map does not reproject to
the sampled pixels within `1e-3` pixels.

Validate all numerical artifacts, compare the RGB alpha with the ray-cast
mask, and generate RGB/XYZ/normal contact sheets with:

```bash
python3 scripts/validate-canonical-pointmap-dataset.py \
  /tmp/rain-canonical-v1
```

## Frozen-Qwen readout and head overfit

The first neural kill-test uses a deterministic clean-target readout rather
than a generated image or an arbitrary denoising step.  Run it in the ML worker
environment with CUDA access:

```bash
.venv-workers/bin/python scripts/extract-qwen-canonical-features.py \
  /tmp/rain-canonical-v1 \
  --output-dir /tmp/rain-qwen-readout \
  --production-ring \
  --output-resolution 512
```

For the three-ring dataset, replace `--production-ring` with
`--production-elevation=-30 --production-elevation=0
--production-elevation=30`.  These values use the literal production phrases
`low-angle shot`, `eye-level shot`, and `elevated shot`.

The script keeps Qwen Image Edit 2511 and the Lightning 1.0 / Multiple Angles
0.9 adapters frozen.  It encodes the exact target render as a VAE latent,
concatenates the frontal and back conditioning latents exactly like the
production two-image pipeline, runs one transformer pass at normalized
timestep zero, and stores target-token LayerNorm features from blocks 15, 30,
45, and 59 as float16 arrays.  `--production-ring` selects only the eight
eye-level azimuth prompts supported by Multiple Angles: 0 through 315 degrees
in 45-degree increments.

Train only the small point-map head on selected views:

```bash
.venv-workers/bin/python scripts/train-canonical-pointmap-head.py \
  /tmp/rain-qwen-readout \
  --output-dir /tmp/rain-head-overfit \
  --steps 4000 \
  --target-resolution 64
```

Pass multiple compatible feature directories to train one head on a collection
of complete identities.  Equal-sized caches are naturally balanced; unequal
collections currently need an explicit sampling policy.  Each dataset
manifest should provide
`asset.identity_id` and `asset.pose_id`; sample diagnostics are prefixed with
the identity to avoid filename collisions:

```bash
.venv-workers/bin/python scripts/train-canonical-pointmap-head.py \
  /tmp/rain-features /tmp/april-features /tmp/arthur-features \
  --output-dir /tmp/multi-identity-head \
  --steps 12000 --timestep-conditioning
```

Evaluate an existing checkpoint without updating it by selecting held-out
features and passing `--steps 0 --checkpoint PATH`.

To train one head across the exact four-step production schedule, cache
teacher-forced noisy exact renders at native 64 by 64 target-token resolution:

```bash
.venv-workers/bin/python scripts/extract-qwen-canonical-features.py \
  /tmp/rain-canonical-production-3rings \
  --output-dir /tmp/rain-qwen-multitimestep \
  --production-elevation=-30 --production-elevation=0 \
  --production-elevation=30 \
  --output-resolution 1024 \
  --denoising-steps 4
```

The extractor asks the loaded FlowMatch scheduler for its shifted sigmas and
uses the same noise trajectory for all four states of a sample.  Train the
shared timestep-conditioned head with:

```bash
.venv-workers/bin/python scripts/train-canonical-pointmap-head.py \
  /tmp/rain-qwen-multitimestep \
  --output-dir /tmp/rain-head-multitimestep \
  --steps 12000 \
  --target-resolution 64 \
  --timestep-conditioning
```

For a feature-dependence control, evaluate the checkpoint without retraining
using `--steps 0 --checkpoint PATH --evaluation-feature-ablation zero` or
`--evaluation-feature-ablation spatial-roll`.

## Real denoising evaluation

Attach a trained head to the four actual Qwen denoising evaluations and run
the complete 24-view production grid with:

```bash
.venv-workers/bin/python \
  scripts/experiment-qwen-canonical-head-denoising.py \
  /tmp/rain-canonical-production-3rings \
  --checkpoint /tmp/rain-head-overfit/head.pt \
  --output-dir /tmp/rain-head-denoising-production \
  --elevation low --elevation eye --elevation elevated
```

The runner writes each generated image, the four head predictions, uniform and
uncertainty-weighted all-step fusion controls, XYZ/mask diagnostics, per-view
metrics, timings, and CUDA peaks.  It performs head inference online so that
the full Qwen activation maps are not retained.  Repeat with a different
`--seed` to test whether a checkpoint transfers beyond its teacher-forced noise
trajectory.

Analyze the saved predictions without rerunning Qwen:

```bash
.venv-workers/bin/python \
  scripts/analyze-qwen-canonical-head-denoising.py \
  /tmp/rain-head-denoising-production
```

The analysis writes `analysis.json`, a generated-image contact sheet, and an
XYZ contact sheet for every denoising step and fusion control.  Its projective
metric fits a free 3x4 camera independently for every view, avoiding the false
assumption that a semantic Qwen camera exactly matches the nominal Blender
camera.

## Point-cloud inspection

Fuse the foreground predictions from all saved views, voxelize them, and write
ASCII PLY clouds plus four-view projection sheets with:

```bash
./venv/bin/python scripts/export-canonical-head-pointcloud.py \
  /tmp/rain-head-denoising-production \
  --prediction step_1 \
  --output-dir /tmp/rain-pointcloud \
  --voxel-size 0.008
```

The prediction PLY contains canonical XYZ, sampled generated RGB, number of
raw samples, number of distinct source views, and mean predicted log error per
voxel.  A canonical-coordinate color version and an exact-reference cloud are
written beside it.  RGB and canonical-color PLY files for every individual
input view are written under `views/`.  The reference uses the same view IDs
and is resampled to the prediction grid before voxelization, so its cross-view
overlap is directly comparable.

Open any generated PLY in Diffusion Editor's existing native Termin point-cloud
renderer:

```bash
./venv/bin/python scripts/view-canonical-pointcloud.py \
  /tmp/rain-pointcloud/prediction-rgb.ply
```

The production reconstruction viewport supplies orbit, pan, zoom, circular
points, depth testing, and camera fitting; the demonstrator only loads the PLY
and keeps the normal native event loop running.  Canonical head clouds are
Y-up, so the viewer rotates them into Termin's Z-up coordinates by default.
Pass `--coordinates termin-z-up` only for a PLY that has already been converted.
Use `--point-size 1` to inspect the actual sampling density without the default
three-pixel circular splats visually merging neighboring points.
