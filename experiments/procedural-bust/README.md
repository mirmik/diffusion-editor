# Procedural SDF bust — experiment, 2026-10-02

The method works as a reproducible modelling loop: anatomical volumes and
profiles → sampled implicit field → OpenVDB quads → Blender → visual revision.
The result is a stylised blockout, not a finished anatomical sculpture.
No imported meshes, scans, textures, image generators or model weights were used.
The editor application and its runtime dependencies were not changed.

## Open the result

- [Final Blender scene](output/03-final/bust.blend)
- [Four final views](output/03-final/contact-sheet.jpg)
- [Initial draft / revised form / parameter variant](output/comparison.jpg)
- [Turned-head scene](output/04-turned/bust.blend)
- [Four variant views](output/04-turned/contact-sheet.jpg)
- [Recorded measurements](results.json)

`output/` contains local generated artifacts and is deliberately ignored by Git.
Each run has a `report.json` and a `source/` snapshot of its generator and actual
parameters. Preserve that directory when sharing the results. `source/neutral.json`
contains the parameters used for that particular run, including the turned variant.

## Environment and reproduction

Verified with the already installed Blender 5.2.1 LTS, its bundled NumPy and
`openvdb`, and CPU Cycles. No packages were installed. The regular project venv
is used for tests and Pillow contact sheets; Blender scripts run in Blender's
own Python, where `bpy` and `openvdb` are available.

From the repository root, build the final neutral study:

```bash
blender -b -t 12 --factory-startup --python-exit-code 1 \
  --python experiments/procedural-bust/build.py -- \
  --output experiments/procedural-bust/output/03-final \
  --voxel 0.0012 --resolution 1000 --samples 48
```

Build the parameter variant:

```bash
blender -b -t 6 --factory-startup --python-exit-code 1 \
  --python experiments/procedural-bust/build.py -- \
  --output experiments/procedural-bust/output/04-turned \
  --params experiments/procedural-bust/turned.json \
  --voxel 0.0015 --resolution 800 --samples 40
```

For an inexpensive preview, use `--voxel 0.0025 --resolution 640 --samples 24`.
Use `--no-render` for geometry only, or `--views front three-quarter` to limit
renders. Commands overwrite named artifacts in the chosen output directory;
choose a new directory to preserve a previous iteration.

```bash
./venv/bin/python -m unittest discover \
  -s experiments/procedural-bust -p 'test_*.py' -v
./venv/bin/python experiments/procedural-bust/preview.py \
  experiments/procedural-bust/output
```

The contact-sheet command expects the saved `01-draft`, `03-final`, and
`04-turned` runs. To reproduce the initial draft after a fresh checkout, its
historical source must be taken from the saved local `01-draft/source/` snapshot;
the current anatomical description intentionally produces the revised form.

## What is procedural

- `sdf.py`: ellipsoid distance approximation, oriented muscles, closed lofts,
  monotone cubic interpolation, smooth union and subtraction.
- `anatomy.py`: chest, shoulders, neck, skull/jaw profiles, cheeks, brow, ears,
  nose, lips, sockets and creases. Coordinates are metres; front is negative Y.
- `build.py`: bounded slab evaluation, narrow-band storage, quad extraction,
  alternating smoothing, limited fragment cleanup, topology checks, studio,
  four orthographic camera views, saved scene and timings.
- `neutral.json` / `turned.json`: shoulder width, neck extension, jaw width,
  and head yaw. Eyes follow the same head transform as the field.

The anatomy is one continuous extracted surface. Eyeballs are separate
parametric spheres, the plinth is a bevelled cylinder, and the studio includes
a ground plane and three area lights. The material has subtle procedural grain.
The topology checks apply to the anatomy surface, not a union of the whole scene.

Ellipsoid and loft fields approximate distance; they are not everywhere exact
Euclidean SDFs. Smooth operations and anisotropic head scaling also change the
distance property. That is adequate for this isosurface experiment but should
not be reused as an exact collision-distance oracle.

## Observations and measurements

| Run | Voxel | Anatomy vertices | Anatomy quads | Total elapsed |
| --- | --- | ---: | ---: | ---: |
| Initial draft, four 640px renders | 2.5 mm | 119,290 | 119,296 | 9.6 s |
| First form revision, four 800px renders | 1.5 mm | 303,906 | 303,908 | 39.1 s |
| Final neutral, four 1000px renders | 1.2 mm | 476,162 | 476,160 | 84.9 s |
| Turned variant, four 800px renders | 1.5 mm | 289,454 | 289,452 | 75.4 s |
| Revised geometry only, replay A / B | 2.5 mm | 109,444 | 109,442 | 7.4 / 5.9 s |

Some runs overlapped with other local builds, and sample counts differ. These
are observed wall times, not a controlled scaling benchmark. The final build
spent 59.7 seconds evaluating the field and 1.8 seconds extracting, smoothing
and cleaning the mesh. It used 12 CPU threads for Blender.

Visual revision was essential:

1. The initial draft exposed spherical cheeks and eyes, excessive nose/chin
   projection, horizontal bands at loft keys, and unintended neck/shoulder gaps.
2. Shape-preserving cubic interpolation removed the bands. Broader transitions
   filled the shoulder gaps; the neck was shortened and the face rebalanced.
3. More orbital tissue and a narrower aperture covered the eyeballs with lids.
   A nasal volume filled unintended gaps behind the nose bridge.
4. Fine sampling exposed a 16-vertex sliver at the suprasternal notch near
   `(0, -0.0444, 0.3216)` m. Cleanup removes only detached components with at most
   64 vertices and a bounding-box diagonal of at most five voxels, recording
   every removal. Larger detached components still fail validation.

Both final parameter sets have one connected anatomy component, no boundary
or non-manifold edges, no degenerate faces, positive signed volume, and Euler
characteristic 2. The initial drafts were closed meshes too, but contained
unintended handles: being watertight alone does not establish correct anatomy.

Four numerical tests passed: sphere distance including its centre, blend
depth/symmetry, interpolation without overshoot, and head/eye frame agreement.
Two independent 2.5 mm builds produced identical vertex-coordinate SHA-256
hashes and topology reports. The final `.blend` was reopened and its topology
and coordinate hash checked against its report.

## Evaluation and remaining scope

The technique is useful for fast, repeatable blockouts and parameterised static
models. Changing the four parameters genuinely changes the geometry; no vertex
editing or skinning is involved. The demonstrated yaw is 25 degrees, shoulder
and jaw multipliers are 0.88 / 0.86, and neck extension is 25 mm.

Visual quality remains the limiting factor. Eyelid rims, cheeks, jaw, lips and
ears are schematic; chest and clavicle transitions need anatomical reference.
Increasing the voxel resolution does not solve those design errors. A chosen
visual reference and another anatomy pass are recorded in Kanboard **#2788**.
The completed technical experiment is **#2787**.

There is no UV unwrap, animation rig, retopology, or editor UI integration.
Head rotation regenerates the surface; neck anatomy follows only approximately.
Only the two supplied parameter sets were visually validated, not every
combination within the accepted input ranges. Self-intersection and minimum
wall thickness are not certified. This is not presented as a game-ready or
print-ready asset.
