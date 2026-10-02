# Ellipsoids fitted to a Pixal3D head

Experiment ended at the user’s request on 2026-10-03. The final sculpt was
visually rejected; convincing character likeness was not achieved. All sources,
references and artifacts are preserved. No further modelling or review is pending.

Latest work uses the positive fit **as a base for surface modelling**, per the
user's clarification. See [sculpt scene](output/head-06-sculpt/head.blend),
[three views](output/head-06-sculpt/result.jpg),
[same-camera clay before/after](output/head-06-sculpt/before-after.jpg), and
[reference comparison](output/head-06-sculpt/reference-review.jpg).

`sculpt.py` loads the unchanged positive-only `head-01` scene, subdivides its
mesh once, and adds a shape key with localized surface deformations. It lifts
the low nose/mouth, shapes lip masses and their meeting line, changes cheeks,
chin and ear relief, and recesses the orbital skin. The exposed spherical eye
caps and brows are separate parametric surfaces. No CSG fitting result is used.
This is authored procedural sculpting guided visually by the accepted images,
not an automatic recovery of their anatomy. `head-03` through `head-05-sculpt`
are diagnostic iterations; complete eyeballs protruding through cheeks in
the first trial were replaced with exposed eye caps.

The final head has 547,298 vertices / 547,296 quads, one closed component,
Euler characteristic 2, positive volume, and no degenerate or non-manifold
faces. Reopening checks actual evaluated shape-key geometry, the unchanged
baseline, build source and three packed reference hashes. Eyes/brows remain
separate open surfaces; these checks do not prove all joins seamless or rule
out every self-intersection. Build peak RSS was 4,407,688 kB (about 4.2 GiB),
under the mandatory 12 GiB / zero-swap scope. Geometry and render checks are
not artistic acceptance: likeness remains approximate, lids are heavy,
cheek/ear planes remain soft, and the short neck cut is irregular.

For this pass use `sculpt.py -- <fresh-output> --size 900 --samples 48`, then
reopen with `finalize_sculpt.py` and `verify_sculpt.py`; all three Blender
commands require `run_bounded.py`. `preview_sculpt.py <output>` uses the
project venv. The build-time script is embedded in the blend and preserved in
`source/sculpt.py`; the live generator additionally persists reference image
fake users directly, a post-build fix applied by the finalizer to this artifact.
The original positive mesh and 51 ellipsoids remain hidden in the scene.
The sculpt shape key affects the head only; eye/brow objects are separate.

## Completed fitting experiments

Completed single-head feasibility experiment, Kanboard #2792. The fitted
representation is a smooth union of 51 ellipsoids, initialized from 39 manually
authored volumes and augmented by 12 automatically placed residual volumes.
Centres, radii and rotations were optimized; the target mesh was not deformed.

- [Result and individual ellipsoids](output/head-01/result.jpg)
- [Target / initial / fitted, three identical cameras](output/head-01/comparison.jpg)
- [Blender scene](output/head-01/ellipsoid-fit.blend)
- [Editable parameters](output/head-01/fitted.json)
- [Independent mesh evaluation](output/head-01/evaluation.json)
- [Reopened-scene check](output/head-01/verification.json)
- [Input images](output/head-01/pixal/views/inputs.jpg)

The scene contains `target`, `initial`, `fitted` and a hidden collection
`SOURCE | fitted ellipsoids`. Enable the collection to inspect its individual
objects. The final union is a saved mesh: editing a source ellipsoid does not
automatically rebuild that mesh. Parameter JSON retains centres, radii and
quaternions. Coordinates are Z-up/front -Y, with target head plus short neck
height normalized to one; no physical dimensions were recovered from images.

## Result

| Check | Initial | Fitted |
| --- | ---: | ---: |
| Volume IoU, 30,000 held-out uniform points | 59.67% | 99.34% |
| Bidirectional mean surface distance, fraction of head height | 5.659% | 0.106% |
| Target-to-mesh surface distance p95 | 10.413% | 0.290% |
| Connected components | 1 | 1 |
| Euler characteristic | 2 | 2 |
| Watertight, consistently oriented | yes | yes |

Surface evaluation uses 30,000 independently sampled points in each direction.
Optimization took 15.15 seconds on the installed RTX 5090 runtime (including
initial/final grids and fitting reports, excluding Pixal3D and Blender).
The final extracted mesh has 136,826 vertices / 136,824 quads. All 51 source
transforms and all three saved meshes match on reopening the Blender file.

This demonstrates fitting a compact volume representation. It is **not** a
claim of 99% character likeness. Eyes, lips, nostrils and inner ears are visibly
simplified; global volume IoU is insensitive to such small features. The fit
tracks the prepared Pixal3D target, including its proportional errors. There is
no rig, anatomical guarantee for individual parts, animation topology, or
full-body fit in this experiment.

The subsequent [CSG trial](output/head-02-csg/result.jpg) froze the 51-volume
base, optimized seven cuts, then added ten positive details. Independent mean
surface error fell from 0.105986% to 0.101818% of normalized head height,
about 3.9%; local ear errors fell 18–21%, eye errors 6–17%, mouth error 1.4%.
The result stayed visibly simplified and the user judged it weak. Its JSON
stores ordered union/subtract/union stages; `fit_csg.py` uses smooth max for
cuts. All 68 primitives and extracted meshes match after reopening. This
trial is preserved as evidence, not used as the new sculpt baseline.

## Target provenance and preparation

Three accepted bald references were used: the front crop from `Body.png` and
the two original profile views with hair removed by Qwen. The rejected new
rotations of `Body.png` were not used. `prepare_views.py` masks grey background
and clothing, keeps original face pixels, and crops a head with a short neck.
Cardinal cameras and per-view framing are approximate, not calibrated. The
accepted profile edits already had uncertain scalp proportions and changed
eyebrows; this remains uncertainty in the target.

The installed Pixal3D MV geometry worker ran seed 42, 12 steps, resolution 1024,
100,000 output triangles, without generating textures. Original GLB, decoded
geometry, settings, input hashes and worker logs are retained under `pixal/`.

The export is a thin, open-neck shell, with 16 boundary edges after welding.
Simple hole filling produced degenerate triangles; Blender voxel remeshing
retained a hollow shell and was discarded. Its early diagnostic images and
logs are not the final target.

The successful target preparation computes an unsigned-distance barrier at
grid step 0.0064 of original head height, caps the open neck on a declared
oblique plane, fills enclosed cells, and compensates for barrier thickness.
OpenVDB extracts the closed outer volume. This deliberately removes internal
surfaces and changes the neck base; narrow details may close. The mesh is then
centred and height-normalized. Before/after surface-change statistics, including
large deviations at the new neck cap, are in `target-report.json`. The prepared
surface has visible voxel ripples; the ellipsoid approximation smooths them.

## Fitting method

`field.py` implements an approximate ellipsoid distance and a log-sum-exp
smooth minimum, with fixed temperature 0.012. NumPy extraction/reference and
PyTorch training agree to 1.13e-7 on 2,000 probe points. This union differs from
the manual experiment's polynomial blend, and is used consistently throughout
this experiment. Approximate distances are not exact Euclidean SDFs.

Training uses 90,000 target surface points and 180,000 near-surface/uniform
volume samples. Adam optimizes centre, log-radius and normalized quaternion
parameters for 3,000 steps, then 1,500 steps after placing 12 residual volumes
at under-covered training surface points. Loss combines clamped field error,
surface residual, soft occupancy and weak parameter priors. Test points and
the final independent mesh-distance samples are not used to place primitives
or choose a checkpoint. Final meshes are extracted from the same field with
OpenVDB at step 0.005, followed by alternating Laplacian smoothing.

## Resource incident and required launcher

The user reported a hang and reboot during initial preparation. The previous
boot's kernel log confirms global OOM at 2026-10-02 23:28:25: Python PID 92498
had 55,022,400 kB anonymous RSS (about 52.5 GiB). The exact command is not
available. An unbatched CPU containment check on a two-million-triangle mesh
was a plausible high-allocation operation, not a proven identification of the
killed process. See `output/head-01/incident.json`, Kanboard #2793.

All expensive experiment entry points now require a cgroup with at most 12 GiB
RAM and zero swap. `run_bounded.py` creates that scope, sets MemoryHigh=10G,
CPUQuota=800%, and holds a lock to prevent simultaneous experiment jobs.
`limits.py` refuses an unbounded launch. CPU containment checks use batches of
eight points. Successful bounded preparation peaked below 1.2 GiB, and fitting
at about 1.96 GiB RSS. `/usr/bin/time -v` logs record each stage.

Use the project venv for preparation of images, the launcher and contact sheets;
use the already installed Pixal3D Python for CUDA/scientific dependencies.
Rtree is installed only in this experiment's ignored `vendor/` directory.
No editor or shared CUDA environment packages were changed.

From the repository root, run expensive commands through:

```bash
./venv/bin/python experiments/ellipsoid-fit/run_bounded.py \
  /home/mirmik/soft/TRELLIS.2/venv/bin/python -u \
  experiments/ellipsoid-fit/fit.py <fresh-prepared-run>
```

The ordered workflow is `prepare_views.py` → existing Studio
`pixal3d_runner.py` → `envelope.py` → Blender `extract_envelope.py` →
`prepare_target.py` → `fit.py` → Blender `render.py` → `evaluate.py` →
Blender `verify_scene.py` → project-venv `preview.py`.
Run the existing Pixal3D worker through the bounded launcher too. Run Blender
with `-b -noaudio -t 8 --python-exit-code 1 --python <script> -- <run>`;
verification instead loads `<run>/ellipsoid-fit.blend` before its script.
The legacy `regularize_target.py` records the discarded hollow-shell trial and
is not part of the successful workflow.

Generated outputs and dependencies are ignored by Git. `source/` in the run
snapshots the fitting-time scripts; postprocessing source and hashes are saved
separately by the final artifact manifest. Initial parameter JSON and all
sample arrays are retained for inspection.
