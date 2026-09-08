# Pixal3D multiview in Studio

Run `./run-multiview-studio.sh`, populate Front and any additional view cells,
select **Pixal3D multiview** in **Generation settings**, and click **Build
model**. All populated cells participate simultaneously, with Front as the
main camera. Four is not a hard limit; one view is also supported.

Pixal3D has separate persistent controls for seed, steps per stage, resolution
(1024/1536), FOV, framing normalization, target face count and texture size.
The result is a GLB with PBR textures. Switching to TRELLIS restores its own
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

Exports are aligned to Studio's existing TRELLIS front convention, a half-turn
around glTF Y from official standalone Pixal3D. Existing viewport and region
coordinate conventions remain valid.

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
hidden/disabled for Pixal3D builds. Separate **TRELLIS.2 texture generation**
and regional TRELLIS refine remain independent operations on the current
mesh. This integration adds full Pixal3D generation, not Pixal3D local refine.
The Pixal3D build itself already includes textures.

Output-quality follow-ups: Kanboard #2290 (topology), #2294 (face/hand detail).
See the [Vaan experiment](pixal3d-vaan-qwen-experiment.md).

## Validation (2026-09-08)

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
