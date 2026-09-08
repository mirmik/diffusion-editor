# Pixal3D multiview experiment (2026-09-08)

Board: task #2289. This is a standalone inference experiment, not a new editor
backend. StableGen is outside this experiment.

## Reproduction

- Upstream: <https://github.com/TencentARC/Pixal3D>, commit
  `f7cf38429b0bd264f1995f0f8743a88b1c728b94` (2026-09-01).
- Model: <https://huggingface.co/TencentARC/Pixal3D>, revision
  `b0cb2e1b794cab9aa0ac38a95d794a4d9337437f`, `pipeline_mv.json`.
- Isolated checkout and model directory: `.local/pixal3d-multiview/` (ignored).
  The older installation in `/home/mirmik/soft/Pixal3D` has local modifications
  and was left intact. The three existing decoder checkpoints are symlinked
  from `/home/mirmik/soft/Pixal3D-hf-check/ckpts`; four MV denoisers require
  approximately 22 GB of additional downloads.
- Python: `/home/mirmik/soft/TRELLIS.2/venv/bin/python`.
- GPU: RTX 5090, 32607 MiB; requires execution outside the filesystem sandbox
  in this session. CUDA and DINOv3/NAF loading were verified successfully.
- Input: official `assets/mv_images/example`, four RGBA views of a cyclops
  head. Input loader and canonical main-camera check passed.

From the editor repository root:

```bash
ATTN_BACKEND=sdpa /home/mirmik/soft/TRELLIS.2/venv/bin/python -u \
  scripts/experiment-pixal3d-multiview.py \
  --root .local/pixal3d-multiview/upstream \
  --model .local/pixal3d-multiview/model \
  --views .local/pixal3d-multiview/upstream/assets/mv_images/example \
  --output .local/pixal3d-multiview/results/official-four/model.glb \
  --resolution 1024 --seed 42 --alpha-only \
  > .local/pixal3d-multiview/results/official-four/inference.log 2>&1
```

Create the output directory before shell redirection. The wrapper records a
`model.run.json` with input hashes, upstream revision/diff, model revision,
runtime versions, elapsed time, PyTorch peak GPU memory, and success/failure.
It calls the official inference function with low-VRAM mode and its default
12-step samplers and GLB export settings. `--alpha-only` validates every input
mask and replaces only the unused background-removal constructor; no model
conditioning or inference algorithm is changed. Omit it for unmasked inputs
when the configured RMBG model is available.

## Input contract

The official [entry point](https://github.com/TencentARC/Pixal3D/blob/f7cf38429b0bd264f1995f0f8743a88b1c728b94/inference_mv.py)
requires images and `transforms.json`: per-view camera-to-world matrices,
horizontal FOV in radians, and object scale. Coordinates follow Blender/NeRF:
Z-up world, camera looking along local -Z, local +Y up. The main view should
be the canonical front camera at `(0, -distance, 0)`.

The example uses azimuths 0/90/180/270 degrees, zero elevation, a 20-degree FOV,
and distance 3.119205. Arbitrary crops or approximate camera poses can break
pixel alignment. RGBA masks are used directly; unmasked inputs require RMBG.

## Results

Both runs completed successfully, with no changes to upstream source code.
The control uses `--num-views 1` and output directory `results/official-one`.
It uses the same MV weights, not the separately trained single-view model.

| Run | Elapsed including initialization/export | Peak allocated GPU memory | GLB size | Triangles |
| --- | ---: | ---: | ---: | ---: |
| Four views | 163.28 s | 16.42 GiB | 33,514,520 bytes | 949,615 |
| First view only | 157.55 s | 14.26 GiB | 34,734,152 bytes | 969,126 |

Both runs used PyTorch 2.10.0+cu130, CUDA runtime 13.0, and SDPA. Timings
include first-use kernel autotuning for different sparse tensor sizes, so this
is not a warmed-up performance benchmark. PyTorch memory counters do not
include all driver/other-process allocations.

Both GLBs loaded in trimesh and Blender 5.2.1 and rendered at four cardinal
angles. Vertex coordinates were finite. The four-view output has one mesh,
626,905 exported vertices, and a PBR material. Each run directory contains
`model.glb`, `model.run.json`, `model.mesh.json`, `inference.log`, and `renders/`.

Open `.local/pixal3d-multiview/results/comparison.html` for a three-row gallery:
input views, one-view control, and four-view result. Rendering used the existing
`scripts/render-pixal3d-multiple-angles.py` through the local
`.local/pixal3d-multiview/render-cardinal.py` adapter, restricting its camera
list to 0/90/180/270 degrees at zero elevation, 512×512 pixels. Both outputs
use the same lighting and orthographic camera setup. Inputs use perspective
and different lighting; brightness and highlights are not a texture-fidelity
measurement.

Visual findings on this one object and seed:

- The one-view control invents a tooth-like second mouth on the back of the
  head and deforms the rear support.
- Four views remove that rear-face artifact and recover the cylindrical
  support and rear silhouette much closer to the supplied views.
- The front retains the cyclops eye, nose ring, teeth, and ears. Faceting and
  imperfect fine detail remain; this is not exact reconstruction.

The experiment supports testing the MV pipeline with our own consistent
views/cameras. It does not establish general quality across objects or
superiority over the separately trained single-view pipeline.

## Remaining topology investigation

Tracked in Kanboard #2290. Neither GLB reports watertight topology after
`trimesh.merge_vertices(merge_tex=True, merge_norm=True)`:

| Run | Boundary edges | Edges with more than two incident faces |
| --- | ---: | ---: |
| Four views | 117 | 275 |
| First view only | 42 | 610 |

For four views winding is consistent and no degenerate faces were detected.
These counts depend on welding precision and do not prove visible holes.
Localize the affected edges before choosing a repair or blaming multiview
conditioning; the control also has topology issues. Original textured GLBs
were preserved without mesh repair.
