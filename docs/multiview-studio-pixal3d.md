# Pixal3D multiview in Studio

Run `./run-multiview-studio.sh`, populate Front and any additional view cells,
select **Pixal3D multiview** in **Generation settings**, and click **Build
model**. All populated cells participate simultaneously, with Front as the
main camera. Four is not a hard limit; one view is also supported.

Pixal3D has separate persistent controls for seed, steps per stage, resolution
(1024/1536), FOV, framing normalization, final triangle count.
Build model runs only the shape stages and exports an untextured GLB. Switching to TRELLIS restores its own
settings. Existing manifests default to TRELLIS; new `shape_backend` and
`pixal3d` fields round-trip on save.

## Views and cameras

Existing non-opaque alpha is reused. RGB inputs use the editor's isolated
U2-Net worker. Pixal3D does not load gated RMBG. Missing files and empty masks
report errors without replacing the current model.

**Normalize view framing** uniformly crops/pads a square around each foreground
bounding box: its longest dimension occupies approximately 91% of the frame.
This matches the Vaan experiment for upright figures and preserves aspect
ratio. With normalization off, the complete image is square-padded and resized.
Original files are never modified.

Cameras follow each cell's azimuth and elevation (-30/0/+30 degrees), the
selected FOV, and an estimated distance matching normalized object size.
These are **nominal cameras, not calibrated Qwen poses**. Per-view camera
calibration/import is outside this integration. Inconsistent geometry,
framing, or perspective can degrade the result. More views cost memory/time.

Geometry postprocessing preserves decoder coordinates, matching Studio's
existing TRELLIS convention. The former PBR export swapped Y/Z and negated
Y; its subsequent Studio rotation cancelled that conversion. Neither rotation
is applied to direct geometry export.

## Runtime

Configure these environment variables if needed:

| Variable | Meaning |
| --- | --- |
| `DIFFUSION_EDITOR_PIXAL3D_PYTHON` | CUDA Python with Pixal3D/TRELLIS dependencies |
| `DIFFUSION_EDITOR_PIXAL3D_MV_ROOT` | Updated checkout containing `inference_mv.py` |
| `DIFFUSION_EDITOR_PIXAL3D_MV_MODEL` | Directory containing `pipeline_mv.json` and its MV checkpoints |
| `DIFFUSION_EDITOR_SEGMENTATION_PYTHON` | Optional segmentation worker Python override |

This checkout discovers the earlier experiment's
`.local/pixal3d-multiview/{upstream,model}` directories. If absent, defaults are
`/home/mirmik/soft/Pixal3D` and `/home/mirmik/soft/Pixal3D-hf-check`. Default
GPU Python is `/home/mirmik/soft/TRELLIS.2/venv/bin/python`.
A single-view-only install needs the updated source and MV weights. Missing
runtime files identify the relevant environment variable in the error message;
the integration does not install/upgrade the external runtime automatically.

The worker uses low-VRAM mode and SDPA by default (`ATTN_BACKEND` overrides it).
Qwen is unloaded before generation, and segmentation stops before Pixal3D
starts. **Cancel** stops the job even during silent checkpoint loading.
Failed/cancelled jobs do not replace the current model.

## Artifacts and other operations

Each job creates `shape-runs/pixal3d-<timestamp>-<unique>/` beside the project:
prepared RGBA views, `transforms.json`, preparation/source hashes,
`request.json`, `worker.log`, `result.json`, and `model.glb`.
Unsaved projects use the existing session workspace; artifacts are adopted
when saved. Prepared inputs and the relative result path remain usable after
that move.

**Reprocess cached mesh** and cached mesh settings belong to TRELLIS and are
hidden/disabled for Pixal3D builds. **Texture model** follows the selected
backend. With Pixal3D selected it uses the native MV texture denoiser, not
TRELLIS image conditioning or view-by-view sampling.

The **Pixal3D multiview texture generation** group has independent persistent
seed (43), steps (12), and texture size (2048) controls. All populated views
participate together, with Front required. FOV and framing come from the
Pixal3D geometry group. There is no Front warmup schedule. Geometry encoding
uses resolution 1024; guidance uses the MV checkpoint defaults (strength 1,
rescale 0, interval 0.6–0.9).

Texturing re-encodes the current geometry and bakes PBR onto its triangles.
It does not run shape diffusion or remesh the model. UV seams duplicate
vertices; exact zero-area input triangles are discarded as in the existing
texture exporter. Coordinates are preserved, including the Studio front
convention. Inputs inside the decoder cube retain their original frame;
meshes outside it are centered/scaled for encoding and restored on export.

The shared shape VAE encoder is loaded in the Pixal3D namespace from
`/home/mirmik/soft/TRELLIS.2/models/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16`.
Override its checkpoint prefix using `DIFFUSION_EDITOR_PIXAL3D_SHAPE_ENCODER`;
the `.json` and `.safetensors` files must both exist. The texture denoiser
and decoder come from Pixal3D's `pipeline_mv.json`.

Texture jobs are cancellable and save under `texture-runs/pixal3d-*`: a copy
of the input GLB, prepared views/cameras, request/result/log, encoded shape
and texture latents, subdivision guides, and the textured GLB. The project
keeps the geometry path and updates only the displayed textured result.
TRELLIS regional refine/texturing remains a separate, explicitly labelled
workflow. Pixal3D local region refine is not part of this integration.

Output-quality follow-ups: Kanboard #2290 (topology), #2294 (face/hand detail).
See the [Vaan experiment](pixal3d-vaan-qwen-experiment.md).

## Initial integration validation (2026-09-08, before geometry-only correction)

The production native application built Vaan from Front plus three Qwen views
through `build_shape()`: 1024, 12 steps, seed 42, target 100,000 faces, texture
1024. The saved project reopened with the same settings and rendered its GLB
in the Vulkan viewport. Frontal orientation was visually checked from a GPU
texture readback. Artifacts: `.local/pixal3d-multiview/studio-smoke/`, including
`vaan.mvstudio.json`, `smoke.json`, `viewport.png`, and `shape-runs/`.

Automated coverage includes legacy/default persistence, separate settings,
all 24 orbit rotations, Front ordering, alpha reuse/RGB segmentation, aspect
preservation, empty masks, silent-worker cancellation, worker errors and
missing results, backend dispatch, stale-result rejection, native panel
visibility, and busy-state controls.

## Geometry-only correction (2026-09-08)

Build model now runs sparse structure and the LR/HR shape cascade, decodes
shape, fills small holes and simplifies to the final triangle count. It skips
texture conditioning, sampling, decoding, UV unwrap and baking. Export uses
the decoded mesh directly, without the former PBR export remeshing pass.
Texturing remains a separate operation.

Validation: 83 Studio tests pass, including a worker regression that rejects
full-pipeline and PBR-export calls. A real GPU run with the four Vaan views
completed in 49.95 seconds. The GLB contains one mesh, zero textures/images
and no UV attributes; the log contains no texture sampling stage. Artifacts:
`.local/pixal3d-multiview/studio-smoke/geometry-only/`.

The follow-up orientation regression was corrected by removing the leftover
PBR compensation rotation from direct export. The worker test now checks
actual vertex axis directions, rather than only the rotation determinant.
A fresh four-view GPU build was opened in Studio's Vulkan viewport: Vaan is
upright and facing forward. Capture and project:
`.local/pixal3d-multiview/studio-smoke/geometry-upright/viewport.png` and
`.local/pixal3d-multiview/studio-smoke/upright.mvstudio.json`.

## Native texture validation (2026-09-08)

86 Studio tests pass, including backend dispatch, independent texture settings
and legacy persistence, request input snapshot, UI visibility and busy state.
The native application's Texture button completed a four-view Vaan job using
Pixal3D MV weights: seed 43, 12 steps, 1024px atlas, 196.93 seconds, peak CUDA
allocation 18.16 GB. All 98,731 triangles match the input exactly after
canonicalization of vertex/triangle ordering; bounds are identical. The GLB
contains base-color and metallic/roughness textures and renders upright,
facing forward in Studio's Vulkan viewport. The saved project preserves its
original geometry path and references the separate textured result.

Artifacts: `.local/pixal3d-multiview/studio-smoke/native-texture.mvstudio.json`,
`native-texture.png`, `native-texture-check.json` and
`texture-runs/pixal3d-20260908-235026-so1zrt0r/`.


## Shared repair and final triangle count (2026-09-09)

Pixal3D now feeds its **raw decoded geometry**, before decimation, into Studio's
shared TRELLIS mesh postprocessor. Sequence: small-hole filling → CuMesh
narrow-band remesh → simplification → topology cleanup → duplicate/degenerate
face removal → final face-count enforcement → consistent winding. MeshLib
final repair remains disabled. This uses the installed CuMesh implementation,
not the visualbruno fork from the previous experiment. Remesh padding follows
the TRELLIS recipe `(resolution + 3 * band) / resolution`, with band 1 and
project_back 0.

**Final triangles** (manifest key `pixal3d.decimation_target`) specifies the
count in the final exported GLB. CuMesh decimation may undershoot the target;
cleanup can change the count again. The final pass simplifies if necessary,
then subdivides existing triangles to fill the shortfall without changing
their geometric surface. It does not delete arbitrary faces or add new
surface details. Boundary-edge splitting handles parity when possible; an
impossible exact count reports an error (a closed triangulated manifold cannot
have an odd triangle count). The UI's normal 10,000-face increments are even.

The raw cache is `decoded-mesh.npz` in the job directory. Result GLBs are named
`shape-post-<key>.glb`; `result.json` includes every stage's counts and the final
postprocess report. Coordinate convention is explicit (`gltf_y_up` for Pixal)
so the shared TRELLIS exporter does not rotate an already aligned mesh.
Texture generation remains separate. The shared postprocess cache protocol
is version 3; existing TRELLIS recipes retain their settings and coordinate
conversion, while Pixal enables exact counts and winding correction.

GPU validation on the same raw Vaan mesh (3,285,876 triangles):

| Final target | Exported triangles | Nonmanifold edges (>2 faces) | Inconsistent winding edges | Zero-area faces | Boundary edges |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 100,000 | 0 | 0 | 0 | 1,170 |
| 250,000 | 250,000 | 0 | 0 | 0 | 2,530 |

Counts were independently measured after GLB reimport. Both variants remain
open meshes; residual topology investigation is tracked in #2290. The dense
intermediate remesh no longer determines final asset size. Native Vulkan
viewport checks confirm upright/front orientation; 91 automated tests pass,
including surface area/volume/winding conservation during exact-count
subdivision, boundary parity, impossible target rejection, and Pixal dispatch
to the shared postprocessor without texture generation.

Review projects: `.local/pixal3d-multiview/studio-smoke/shared-repair-100k.mvstudio.json`
and `shared-repair-250k.mvstudio.json`. Their matching directories contain the
raw cache (100k), post-request/post-result, final-glb-audit, logs and viewport
captures. The final winding-corrected checks are recorded in `post-result.json`.
