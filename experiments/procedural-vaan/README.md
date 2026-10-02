# Procedural Vaan — implicit volume experiments

The user ended this experimental direction on 2026-10-03 after rejecting the
latest surface sculpt. Sources and artifacts remain available; likeness was
not accepted and no further work or review is pending.

The latest continuation is [ellipsoid-fit](../ellipsoid-fit/README.md): Pixal3D
volume fitting has run, followed by a free surface-sculpt pass over the
positive-only fitted mesh. The earlier manual checkpoints below are preserved.

## Earlier manual ellipsoid-head checkpoint

At the user's request, the head was rebuilt in `ellipsoid_head.py` from
**38 positive ellipsoids and six ellipsoidal cuts**. The field uses smooth
unions/subtractions, without the old head loft or facial displacement.
`portrait.face` still provides separate parametric eyes, eyelid bands, lip
colour and ear trim, ray-fitted to the new volume. This is not a claim that
every face detail is an ellipsoid.

Three manual parameter trials are preserved. The latest checkpoint is
[head.blend](output/ellipsoid-head-03/head.blend), with
[three views](output/ellipsoid-head-03/head-review.jpg),
[same-camera comparison](output/ellipsoid-head-03/before-after.jpg) and
[raw construction / fused surface](output/ellipsoid-head-03/construction-review.jpg).
The `SOURCE | ellipsoid construction` collection stores inspectable positive
primitives and is hidden by default. Negative primitives are in the source and
build report. Baseline character artifacts remain untouched.

The reopened head has 249,866 vertices, one component, zero non-manifold edges,
Euler characteristic 2 and positive signed volume. All 306 eye clearance probes,
eight packed reference hashes, source hashes and construction transforms pass.
Width including ears / depth / height are 165.56 / 183.09 / 216.89 mm in the
model's authored scale. See [summary](ellipsoid-head-results.json) and
[verification](output/ellipsoid-head-03/verification.json).
Likeness has not been accepted. Eyelid joins, expression, lower-face proportions
and the neck transition remain approximate; technical checks do not certify
all separate surfaces as seamless.

During the pass, the user proposed fitting the ellipsoids to a Pixal3D multiview
mesh. That experiment is now executed in `../ellipsoid-fit`, including a
positive-only fit and a CSG trial. Use an unclothed,
hair-free reconstruction for anatomy: fitting the clothed character would copy
clothes and hair volume. The intended objective combines target surface error,
inside/outside volume agreement, and weak parameter constraints. Optimize the
same smooth union used for final extraction, initially keeping blending fixed;
add finer primitives where residual error remains. A fit would approximate the
Pixal3D geometry, not automatically correct its anatomical or identity errors.

Reproduce the manual checkpoint from the preserved source snapshot in a fresh
output directory, starting with `output/revision-3/vaan.blend`, and run
`build_ellipsoid_head.py -- --output <fresh-directory> --height 900 --samples 32
--views face face-side face-three-quarter` through Blender's `--python` option.
Reopen the resulting `head.blend` with `--python verify_ellipsoid_head.py`.
Generate contact sheets with the project venv and `preview_ellipsoid_head.py
<directory>`. Original build/source hashes are retained in the checkpoint.

## Earlier full-character revision 3

A static character built from NumPy implicit fields, OpenVDB and procedural
Blender surfaces. Revision 3 targets the flattened profile: cranial depth and
vault, ear placement, lower-face proportions, neck attachment and hair fit.
Revision 2's face/hair, hood, trousers, abdomen and footwear form the baseline.
The result remains a
stylised modelling study; it is not a finished reconstruction of the drawing.

## Review the result

- [Profile before / after, same camera](output/revision-3/before-after-profile.jpg)
- [Both profiles: reference / before / after](output/revision-3/profile-reference-review.jpg)
- [Skull before / after with hair hidden](output/revision-3/bare-head-progress.jpg)
- [Hair-free references and model](output/revision-3/anatomy-reference-review.jpg)
- [Measured head dimensions](output/revision-3/head-depth-comparison.json)
- [Face before / after, same camera](output/revision-3/before-after-face.jpg)
- [Four full-body views](output/revision-3/turnaround.jpg)
- [Five portrait views](output/revision-3/portrait-turnaround.jpg)
- [Blender scene](output/revision-3/vaan.blend)
- [Build report](output/revision-3/report.json)
- [Reopened-scene verification](output/revision-3/verification.json)
- [Tracked result summary](results.json)

Previous deliverables remain untouched in [output/final](output/final/) and
[output/revision-2](output/revision-2/).
Generated artifacts are local and Git-ignored. Every build snapshots the full
generator and its shared SDF kernel in `source/`, including source hashes.
Intermediate numbered directories document
visual iteration, not additional deliverables. The much earlier `03-refined`
run is explicitly rejected because of invalid object transforms.

## References

The user supplied the front drawing and asked to use both earlier generated
profiles. The four original images and four anatomy references (full bare front,
its crop, two edited profiles) are packed into the scene, retained with fake
users, and identified by SHA-256 hashes in the build report.

| View | Original file |
| --- | --- |
| Front | `/home/mirmik/Vaan/Front.png` |
| Right profile, facing left | `/home/mirmik/Vaan/views/mv-eye-090.png` |
| Left profile, facing right | `/home/mirmik/Vaan/views/mv-eye-270.png` |
| Back | `/home/mirmik/Vaan/Back.png` |
| Bare front | `/home/mirmik/Vaan/Body.png` |
| Same original right/left views, hair removed | `references/profile-hair-removal/` |

Qwen Image Edit 2511 with the default Lightning 4-step adapter, **without
Multiple Angles**, removed the scalp hair from crops of the existing profiles.
The prompt explicitly preserves the original pose, camera, facial silhouette
and clothing. See [before / after edits](references/profile-hair-removal/hair-removal-review.jpg),
[generation settings](references/profile-hair-removal/manifest.json) and
[visual review](references/profile-hair-removal/visual-review.json).
Qwen changed the eyebrows; their design is not used for modelling. The user
accepted these as useful references but considered the inferred scalp a little
too large. Scalp size is therefore constrained by the original front and original
hair silhouettes, rather than copied literally from the edits.

An earlier attempt rotated `Body.png` using Multiple Angles. The user rejected
those views because they changed the character. They remain under
`references/bare-head/qwen` for provenance and are **not fitting references**.
`output/13-bare-reference-fit` records the discarded trial; its portrait changes
were rolled back before fitting the original profiles.

These are artistic, generated references with inconsistent pose and framing,
not calibrated projections. The comparison helper crops the portrait references
for approximate matching. Before / after renders use identical camera settings.
No reference pixels are projected onto the model. Geometry does not use imported
character meshes, downloaded assets or learned reconstruction weights. Qwen is
used only to create supplementary 2D references; comparison sheets contain
unaltered reference/render pixels, apart from cropping and resizing.

## Reproduce

The installed Blender 5.2.1 includes NumPy and OpenVDB. No additional packages
were required. From the repository root, choose a fresh output directory:

```bash
blender -b -noaudio -t 12 --factory-startup --python-exit-code 1 \
  --python experiments/procedural-vaan/build.py -- \
  --output experiments/procedural-vaan/output/revision-3-rebuild \
  --body-voxel 0.0025 --head-voxel 0.0008 \
  --height 1600 --samples 80 \
  --views front right back left three-quarter face face-side \
          face-left face-three-quarter head-back

blender -b -noaudio -t 12 --factory-startup \
  experiments/procedural-vaan/output/revision-3-rebuild/vaan.blend \
  --python-exit-code 1 --python experiments/procedural-vaan/verify.py

./venv/bin/python experiments/procedural-vaan/preview.py \
  experiments/procedural-vaan/output/revision-3-rebuild \
  --baseline experiments/procedural-vaan/output/revision-2

blender -b -noaudio -t 12 --factory-startup \
  experiments/procedural-vaan/output/revision-3-rebuild/vaan.blend \
  --python-exit-code 1 --python experiments/procedural-vaan/inspect_head.py -- \
  --output experiments/procedural-vaan/output/revision-3-rebuild/diagnostic \
  --views face-side face-left face face-three-quarter head-back

./venv/bin/python experiments/procedural-vaan/profile_review.py \
  experiments/procedural-vaan/output/revision-3-rebuild \
  --baseline experiments/procedural-vaan/output/revision-2 \
  --bare-baseline experiments/procedural-vaan/output/revision3-analysis/baseline \
  --anatomy-references experiments/procedural-vaan/references/profile-hair-removal
```

Reference hair removal can be rerun separately with the existing editor worker:

```bash
PYTHONPATH=. \
DIFFUSION_EDITOR_QWEN_IMAGE_EDIT_MODEL=/home/mirmik/.cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots/6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9 \
./venv/bin/python experiments/procedural-vaan/remove_reference_hair.py \
  --output experiments/procedural-vaan/references/profile-hair-removal-rebuild \
  --seed 20261003
```

The local FP8 components and Lightning adapter were already installed. Loading
the adapter with global `HF_HUB_OFFLINE=1` failed; this separate worker defect
is tracked in Kanboard #2791. The successful run used the explicit local model
directory without that global flag. No new model weights were installed.

For faster previews keep the validated 2.5 mm body / 0.8 mm head sampling and
reduce image size, samples and number of views: `--height 1000 --samples 32
--views face face-side front`. `--no-render` saves geometry without rendering.
Coarser body sampling can introduce tiny handles between fingers; extraction
now rejects nonzero genus, instead of silently saving an invalid approximation.

Geometry runs in Blender's Python; comparison sheets use Pillow in the project
venv. Building geometry does not require the reference paths to exist, but
artifact verification expects the original references and checks the hashes of
every packed reference recorded in the report. `inspect_head.py` measures the
saved head mesh and hides hair only for diagnostic renders; it never saves those
visibility changes back to the scene. The source snapshot
contains its own `sdf.py` and can be run independently of the experiment tree.

## Construction

### Comparison with the supplied Fable / Opus technique

This implements the central implicit-volume approach, closest to the described
Opus/OpenVDB branch. It is a simplified implementation, not a full reproduction.
The comparison is with the user's quoted description: the linked Claude source
files could not be opened through the web tool during this review.

| Part | Implemented here | Difference from the description |
| --- | --- | --- |
| Body and head | NumPy implicit fields, smooth union/subtraction, sampled volumes, OpenVDB `convertToQuads`, alternating positive/negative Laplacian smoothing | Fewer anatomical volumes; mostly lofts plus analytic relief. Warped/ellipsoid fields approximate signed distance. No custom surface-nets implementation. |
| Resolution | 2.5 mm body and 0.8 mm head | Hands share the body grid, rather than a separate 0.9 mm grid. |
| Pose | Segment endpoints are specified before mesh extraction; no skinning | Endpoints are in world coordinates; no bone-frame hierarchy or inverse kinematics. |
| Hair | Closed grooved mesh locks, analytic scalp projection, fine curve filaments | No gravity/wind guide growth, general body-field collision integration, or large clumped child-hair system. Most visible volume comes from meshes. |
| Clothes | Separate SDF trousers, parametric vest/hood, surface-fitted trim and laces | No general body-offset garment construction or cloth simulation. |
| Face details | Skin relief belongs to the head field; eyes, lids and lip colour are parameter surfaces | A procedural hybrid, not every face detail extracted from one field. |

The flat profile was an error in the authored cranial proportions, not a
failure of volume extraction. Increasing vertex count alone would preserve
that incorrect shape. Similarity depends on the anatomical parameters and
multi-view review. Mesh density is not a measure of likeness.

Geometry uses NumPy and Blender's bundled OpenVDB, bpy/bmesh and mathutils.
Pillow is used outside Blender for cropping and comparison sheets. Qwen, when
used for additional reference angles, does not create or deform the mesh.

- `anatomy.py`: torso, arms, hands and trousers. Abdominal relief warps a common
  surface; trouser folds have unequal heights, widths and oblique directions.
- `portrait.py`: facial loft, continuous nasal/lip relief, orbital recesses,
  spherical eye apertures, skin eyelid bands, eyebrows, ear cartilage and hair.
- `details.py`: vest, curved leather panels, suspended hood, jewellery, pouches,
  cargo pockets, cuffs and sneakers with surface-projected crossing laces.
- `geometry.py`: approximate distance primitives, OpenVDB extraction, meshes,
  sweeps, curves, materials and topology checks.
- `build.py`: assembly, reference packing, audit, studio, cameras and artifacts.
- `verify.py`: saved-scene regression checks.
- `preview.py`, `profile_review.py`: render and reference comparison sheets.
- `inspect_head.py`: saved-mesh dimensions and diagnostic views without hair.
- `remove_reference_hair.py`: reproducible local Qwen reference edits.

Coordinates are in metres, with front towards negative Y. The head no longer
uses a corrective scale or translation: its landmarks are authored directly
in the same coordinates as the body. Head extraction uses 0.8 mm sampling;
body and trousers use 2.5 mm. Smoothing alternates positive and negative
Laplacian passes.

Measured on the saved skin mesh, revision 3 increases maximum head depth from
166.06 to 190.99 mm. Height changes from 219.00 to 216.86 mm; maximum width
including ears stays at 168.34 mm (previously 168.32 mm). These describe this
model's geometry, not an anatomical measurement recovered from the drawings.
The scene contains 828,930 raw mesh vertices. All three primary field meshes
pass the topology checks below. Ten standard renders and five hair-free
diagnostic views are saved alongside the scene.

Eyes are curved apertures inside blind orbital pockets. Eyelid surfaces join
the eyeball margins to the analytic skin surface. Iris, pupil and sclera share
one surface and a procedural shader with local eye UV coordinates. Nose,
philtrum, lip relief and chin are part of the head field; lip colour follows
that same surface.

Hair combines a continuous foundation, asymmetrical closed grooved locks and
fine tapered curve filaments. Buried lock samples project onto the foundation
to avoid its cutting off the middle of a lock. Fringe paths, side sweep and
nape flow are separately directed from the references. This is stylised hair
geometry, not simulated individual hairs.

The hood is a thickened parametric bag between the neckline and a turned lip.
It has a low suspended fold rather than the previous flat flap. Trouser volume
and compression folds were changed; the abdomen's separate pill-like bulges
were replaced with continuous relief. Laces now sample the shoe surface along
their full length, so the spline does not disappear into the vamp.

## Verification and limits

The three main extracted meshes must each have one connected component, no
non-manifold edges, positive signed volume and Euler characteristic 2. The
saved scene is reopened and its object, vertex, face and curve counts and mesh
bounds are compared with the build report. The check also confirms packed
reference hashes, 672 front-vest clearance probes and 306 eye aperture probes.
The audit checks finite mesh and curve coordinates and rejects empty meshes
or implausible mesh bounds.

These checks do not establish that every separate garment, hair lock and trim
piece is collision-free. The complete scene is not one watertight solid and
is not a print-ready or animation-ready asset. There is no rig, skinning,
production retopology or general body/clothing UV layout.

Remaining artistic differences are tracked in Kanboard **#2790**. Facial
identity and eyelid expression are still approximate; the hair has large
smooth locks and some visible layer separations. The collar and cloth remain
stiffer than the references, pockets are simplified, and body proportions and
materials need further art direction. Revision 3 is a completed improvement
pass that did not achieve an acceptable artistic result: the user judged the
result unsatisfactory. That full-character pass was stopped. The later ellipsoid-head experiment
and the proposed Pixal3D fitting direction are described above. The successful geometry checks do not constitute visual acceptance.
The initial experiment is recorded separately in **#2789**.
