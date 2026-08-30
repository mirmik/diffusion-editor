# Diffusers/Transformers worker

Diffusion, text-to-image generation, image editing, Grounding DINO, SAM 2.1,
and the depth models run in
a persistent subprocess using the shared
[model-worker environment](model-workers.md).

The `AI` menu exposes five depth profiles for the current composite:

- `DA3 Nested Giant Large 1.1`
  (`depth-anything/DA3NESTED-GIANT-LARGE-1.1`) is the default maximum-quality
  profile. It predicts metric depth and camera calibration at a 1008-pixel
  processing resolution, with the more accurate ray-pose path enabled.
- `DA3 Mono Large` (`depth-anything/DA3MONO-LARGE`) is a high-quality direct
  relative-depth comparison profile. It does not predict camera intrinsics,
  so the editor deliberately does not invent a 3D camera for it.
- `Apple Depth Pro` (`apple/DepthPro-hf`) predicts metric depth and focal
  length. The editor preserves both for calibrated unprojection.
- `Depth Anything V2 Large` is the large relative-depth comparison profile.
- `Depth Anything V2 Small` remains available as the fastest baseline.

`Depth Map — Character Only (DA3)` is a two-stage convenience workflow. It
first runs the existing foreground segmentation worker, then evaluates the
default DA3 Nested profile on the complete, unmasked RGB composite and only
then applies the frozen foreground mask to the preview and point cloud. The
depth worker request cannot contain the selection mask, so background scene
cues remain available throughout inference. The background
is transparent and does not participate in preview contrast normalization.

All ordinary depth commands also respect a non-empty canvas selection as an
output mask. This supports manual cleanup: select the background, invert the
selection so the character is selected, then run any depth profile. If the
composite already has transparency and there is no selection, its alpha is
used automatically.

After a depth result, `AI -> View Depth as Point Cloud` unprojects every valid
model-grid pixel directly into Termin's native GPU point-cloud viewport. DA3's
native output grid and matching intrinsics are retained without resizing the
depth field; source RGB and masks are resized to that grid instead. The visible
preview alone is resized back to the document canvas. For calibrated
direct-depth models it uses `P = depth * inverse(K) * [u, v, 1]`: there is no
min/max remapping, percentile-based thickness, recentering, or geometry scale.
The only coordinate change is a documented rigid axis conversion from camera
coordinates to Termin coordinates (`X right`, `Y forward`, `Z up`). The camera
origin and all model depth ratios therefore remain intact. No point-count
subsampling is applied by default and no PLY or GLB round-trip is used.

Depth Anything V2 is different: it produces inverse-relative values and no
camera calibration. Those values are reciprocated without an additional scale
normalization, and projection uses an explicitly approximate 55-degree
vertical field of view. Such a cloud is useful for comparing shape, but cannot
become metric or calibrated from one monocular result alone.

Left-drag orbits, middle/right-drag pans, and the wheel zooms. The soft
foreground alpha remains unchanged for the antialiased 2D preview. For 3D
geometry it is binarized at 50% confidence: alpha 127/255 is discarded and
128/255 is retained, so the weak segmentation fringe creates no points.

When a model supplies a per-pixel depth-confidence field, the point cloud
retains the raw float32 score aligned one-to-one with its points. The 3D panel
offers `Point color: Image / Confidence`; confidence is shown cold-to-warm
(low-to-high) using a 2nd–98th percentile display range stated in the panel.
This color contrast never changes or filters geometry or canonical confidence.
DA3 Nested Giant Large 1.1 supplies this field; the currently integrated DA3
Mono, Depth Pro, and Depth Anything V2 profiles do not.

The first run of each profile may download model weights; later runs reuse the
Hugging Face cache. Every result creates one visible false-color preview where
cold colors are farther and warm colors are closer. This RGB preview is only a
screen derivative: its display range uses the foreground's 2nd–98th percentile
so outliers do not wash out visible shape. It is never used as depth input.
The canonical artifact remains the unquantized float32 model output together
with its numerical convention and camera metadata; no grayscale uint8 “data
layer” is created.

DA3 uses the official ByteDance source pinned by `setup-workers.sh`. Its
minimal runtime overlay is locked separately in
`requirements-workers-da3.txt`; training, rendering, and full 3D dependencies
are deliberately excluded. The depth-only API may print an optional `gsplat`
warning, but `gsplat` is not required for this tool.

Requests and responses use bounded, versioned JSON-lines messages. Images and
masks cross the boundary as PNG files. Canonical depth and confidence use
strictly sized little-endian float32 planes; shape, dtype, finite values,
intrinsics, and paths are validated before entering the editor process.
Detections use a bounded JSON manifest plus optional mask PNGs. A failed
operation terminates the suspect process so the next request starts cleanly.

The deterministic smoke downloads no models:

```sh
./venv/bin/python scripts/smoke_ml_worker.py
```

A real smoke requires a local SDXL checkpoint and may download other model
weights:

```sh
.venv-workers/bin/python scripts/smoke_ml_worker.py \
  --real --sdxl /path/to/model.safetensors
```

The real smoke selects an available GPU by default. Pass `--device cpu` only
when a CPU run is intentional.

To download and exercise the depth profiles independently of the other ML
features:

```sh
./venv/bin/python scripts/smoke_depth_profiles.py
./venv/bin/python scripts/smoke_depth_profiles.py \
  da3-nested-giant-large-1.1 da3-mono-large v2-large
```

The native depth-cloud viewport has a synthetic GPU smoke:

```sh
xvfb-run -a ./venv/bin/python scripts/smoke_depth_point_cloud_viewport.py
```

CPU, CUDA, and ROCm installation are documented in the shared environment
guide.
