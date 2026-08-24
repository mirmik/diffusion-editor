# Rain canonical point-map head: first kill-test

Date: 2026-08-22

## Question

Does a frozen Qwen Image Edit 2511 transformer contain enough target-view and
conditioning-anchor information for a small trainable head to assign visible
target pixels to positions in one pelvis-centered canonical character frame?

This test uses no Qwen-generated images.  Exact Blender renders make the
supervision unambiguous and isolate the feature/readout hypothesis from
generation inconsistency.

## Dataset

- Asset: Blender Studio Rain v2, CC-BY.
- Pose: authored rest pose; no pose augmentation.
- Views: 24 eye-level azimuths at 15-degree intervals, 256 by 256 pixels.
- Camera orbit and target: `RIG-rain/DEF-Pelvis`.
- Canonical frame: X character-right, Y up, Z forward; pelvis origin; full
  evaluated character height equals one.
- Per view: RGBA, binary center-ray mask, canonical XYZ, canonical normal,
  CV-forward depth, object ID, intrinsics, and canonical-to-camera transform.

Renderer validation:

- generation time: 97.7 seconds for 24 views;
- foreground: 2,997 to 4,559 pixels per view;
- worst point-map reprojection error: `8.32e-5` pixels;
- RGB-alpha/raycast-mask IoU: minimum `0.98875`, mean `0.99081`;
- maximum unit-normal error: `5.96e-8`.

## Frozen-Qwen readout

The initial test used the front render as Qwen conditioning.  The corrected
production-ring rerun uses both front and back renders, matching the real
two-image Multiple Angles pipeline.  The exact target render is VAE-encoded as
the target latent.  One transformer pass is run at normalized timestep zero
with Lightning LoRA 1.0 and Multiple Angles LoRA 0.9 active and frozen.

The readout stores target-token `img_norm2` outputs after attention and before
AdaLN2 modulation from blocks 15, 30, 45, and 59.  The default 512-pixel
readout produces four 32 by 32 by 3,072 float16 maps, or 24 MiB per view.

Measured full-process peak for sequentially offloaded Qwen:

- CUDA allocated: 2.05 GiB;
- CUDA reserved: 2.21 GiB;
- 24-view 512-pixel extraction: 87.1 seconds and 577 MiB on disk.

## Head

The head has 1.69 million trainable parameters.  Each Qwen block is projected
with a learned 1x1 convolution, the results are fused with a camera embedding,
processed by a small residual convolutional decoder, and mapped to:

- canonical XYZ;
- foreground logit;
- predicted log error (uncertainty).

The final variant additionally gives every feature token its exact camera
origin and unit ray direction in canonical coordinates.  Qwen and both LoRAs
remain frozen; cached features ensure that only the head is trained.

## Results

All XYZ errors below are fractions of full character height.  `p95 max` is the
worst per-view 95th percentile, not a pooled percentile.

| Setup | Train median | Train p95 max | Held-out median | Held-out p95 max | Held-out within 5% | Held-out mask IoU |
|---|---:|---:|---:|---:|---:|---:|
| One view, native 32² | 1.10% | 1.86% | — | — | — | 1.000 |
| 8 views, 32² features → 64² | 0.87% | 2.26% | 3.05% | 22.60% | — | 0.932 |
| 16 views, 32² features → 64² | 0.88% | 2.16% | 2.39% | 20.05% | 82.11% | 0.948 |
| 8 views, native 64² features | 0.62% | 1.28% | 2.92% | 24.88% | 75.27% | 0.969 |
| 16 views + canonical rays | 0.88% | 2.24% | **1.91%** | **11.23%** | **89.71%** | **0.951** |

For the best ray-aware held-out run:

- worst per-view p90: 6.88% of height;
- uncertainty/log-error correlation: 0.50 mean across views;
- dropping the top 10% predicted-uncertainty pixels reduces worst p95 from
  11.23% to 9.18%;
- head training on 16 views took about 85 seconds and peaked at 139 MiB of
  allocated CUDA memory.

Increasing Qwen target-token resolution from 32² to 64² improved train error
and silhouette prediction but did not improve held-out XYZ.  Spatial
resolution is therefore not the primary generalization limit in this setup.

## Production-ring rerun

The intermediate-angle gate above tested continuous camera interpolation, but
production Multiple Angles requests eight discrete azimuths per elevation:
0, 45, 90, 135, 180, 225, 270, and 315 degrees.  The corrected experiment
therefore trains and evaluates the first capacity kill-test only on the exact
eye-level semantic prompts and conditions Qwen on both the front and back
renders.

| Conditioning | Mean median | Worst p90 | Worst p95 | Within 5% | Mask IoU |
|---|---:|---:|---:|---:|---:|
| Front only, canonical rays | 0.827% | 1.676% | 2.425% | ~100% | 1.000 |
| Front + back, canonical rays | **0.844%** | **1.575%** | **1.989%** | **~100%** | **1.000** |

The two-anchor run used a 32² Qwen feature grid, a 64² output map, 4,000 head
steps, and the same 1.69-million-parameter head.  Feature extraction took
40.7 seconds and peaked at 2.05 GiB CUDA allocated; head training took 42.9
seconds and peaked at 139 MiB.  The worst view was the 270-degree left side at
1.99% p95.  Dropping the top 10% predicted-uncertainty points lowers the worst
p95 to 1.60%; uncertainty/log-error correlation is 0.55.

## Three production elevation rings

The final synthetic capacity test adds the literal production `low-angle shot`
(-30 degrees) and `elevated shot` (+30 degrees) rings around the eye-level
ring, for 24 supported prompt/camera pairs.  The renderer now accepts repeated
`--elevation` arguments, and the extractor accepts repeated
`--production-elevation` selections.  Front and back eye-level renders remain
the two conditioning anchors.

| Training set / evaluated ring | Mean median | Worst p90 | Worst p95 | Within 5% | Mask IoU |
|---|---:|---:|---:|---:|---:|
| Eye-only baseline (8 views) | **0.844%** | **1.575%** | **1.989%** | ~100% | 1.000 |
| Three-ring head / eye level | 0.975% | 2.000% | 2.516% | 99.95% | 1.000 |
| Three-ring head / elevated +30 | 1.043% | 2.224% | 2.534% | ~100% | 1.000 |
| Three-ring head / low -30 | 1.045% | 2.084% | 2.581% | 99.95% | 1.000 |
| Three-ring head / all 24 | 1.021% | 2.224% | 2.581% | 99.97% | 1.000 |

The 24-view run used 12,000 steps, preserving the eye-only baseline's 500
epochs.  Rendering took 100.3 seconds with worst reprojection 1.06e-4 pixels;
feature extraction took 108.5 seconds; head training took 128.1 seconds.  Peak
CUDA allocations remained 2.05 GiB for Qwen and 139 MiB for the head.

Adding elevations broadens the exact-render support but does not improve fit:
eye-level median and worst p95 are both slightly worse than the eye-only
baseline.  The degradation is small relative to the threefold increase in
conditions and there is still no catastrophic tail, but further synthetic
camera expansion is no longer informative about the main deployment risk.

## Decision

The basic capacity hypothesis passes for the tested discrete production grid:
frozen Qwen features contain a real canonical-correspondence signal, and the
small head closely fits all 24 supported azimuth/elevation pairs without a
catastrophic tail.
The poor intermediate-angle interpolation is not a production acceptance gate.

This remains a one-character exact-render overfit, not a generalization result.
It does not yet show that the head tolerates Qwen-generated surface changes,
denoising-time features, new identities, or new poses.

## Real four-step denoising gate

The frozen three-ring head was then attached to the same four `img_norm2`
block taps during an actual Multiple Angles generation.  Qwen generated all 24
supported Rain views from the front and back anchors.  The head ran after each
of the four transformer evaluations, at scheduler timesteps approximately
1000, 766.7, 455.6, and 20.0.  No generated image or feature was used to
retrain the head.

Because a semantic Qwen camera is not pixel-identical to the Blender camera,
the generated views have no exact per-pixel XYZ ground truth.  Evaluation
therefore uses three complementary measurements:

- distance from predicted points to the union of exact Rain surfaces;
- coverage of that exact surface by the union of all 24 generated point maps;
- per-view reprojection residual after fitting an unconstrained projective
  camera to the predicted pixel/point correspondences.

All geometric distances remain fractions of full character height.  Image
residuals are fractions of image size.

| Denoising step | Points within 5% | Point→surface p95 | Surface coverage p95 | Mean reprojection median | Worst reprojection p95 |
|---|---:|---:|---:|---:|---:|
| 0, t≈1000 | 81.22% | 11.04% | 16.85% | 25.86% | 144.71% |
| 1, t≈767 | **93.52%** | 5.80% | **7.58%** | 3.96% | 18.37% |
| 2, t≈456 | 92.56% | 5.92% | 9.33% | **3.11%** | **15.65%** |
| 3, t≈20 | 95.11% | **4.97%** | 17.81% | 4.57% | 20.80% |

For this first head, trained only at `t=0`, step 0 was unusable and the middle
states transferred better than the others.  This is a measurement of domain
shift for that checkpoint, not evidence that any timestep is intrinsically
preferable.

This is a partial success, not a production pass.  At step 2, 92.56% of
predicted foreground points lie within 5% of Rain's exact surface, and the XYZ
contact sheet stays canonically colored while the rendered character rotates.
However, a 3.11% mean median reprojection residual and 15.65% worst-view p95
are too loose for dependable pixel-level fusion.  Predicted uncertainty is
also poorly calibrated under this domain shift: its correlation with
off-surface error is only 0.13.

The complete 24-view generation took 417 seconds.  Peak CUDA allocation was
22.29 GiB and peak reservation was 22.91 GiB on the 32-GiB worker GPU.

## Timestep-conditioned rerun

The next test removes the `t=0` training mismatch without selecting a
timestep in advance.  For every one of the 24 exact renders, its VAE latent
`x0` is mixed with a shared noise trajectory using the exact four-step
FlowMatch rule `(1-sigma) * x0 + sigma * noise`.  At 1024-pixel output the
shifted scheduler points are 1000.0, 766.709, 455.614, and 20.0, exactly
matching real generation.  All four states supervise one head, which receives
a Fourier timestep embedding in addition to the existing camera and ray
inputs.

The 96-sample feature cache uses native 64 by 64 target tokens and occupies
9.1 GiB.  Extraction took 481 seconds and peaked at 2.06 GiB CUDA allocated.
The 1.76-million-parameter head trained for 12,000 updates in 459 seconds and
peaked at 620 MiB allocated.

The exact teacher-forced fit stayed balanced rather than sacrificing one
state:

| Schedule step | Mean median XYZ | Worst p95 | Within 5% | Mask IoU |
|---|---:|---:|---:|---:|
| 0 | 1.91% | 5.27% | 97.22% | 1.000 |
| 1 | 1.79% | 5.67% | 97.10% | 1.000 |
| 2 | 1.71% | 5.31% | 98.06% | 1.000 |
| 3 | 1.75% | 4.75% | 97.80% | 1.000 |

Two complete generated-view evaluations were then run: the training seed
20260822 and an unseen noise seed 20260823.  The table reports the range across
those two seeds, not a selected best run.

| Prediction | Points within 5% | Point→surface p95 | Surface coverage p95 | Mean reprojection median | Worst reprojection p95 | RGB-mask IoU |
|---|---:|---:|---:|---:|---:|---:|
| Step 0 | 93.95–95.75% | 4.74–5.37% | 3.51–5.85% | 2.58–3.14% | 11.62–11.69% | 0.595–0.610 |
| Step 1 | 97.09–97.94% | 3.83–4.28% | 2.06–2.10% | 1.86–2.06% | 7.09–8.77% | 0.765–0.774 |
| Step 2 | 98.42–98.56% | 3.32–3.65% | 2.04–2.29% | 1.77–1.99% | 7.07–9.42% | 0.775–0.777 |
| Step 3 | 99.37–99.44% | 2.73–2.95% | 1.65–1.78% | 1.34% | 6.01–7.90% | 0.771–0.773 |
| Uniform all-step fusion | 98.68–99.14% | 3.08–3.42% | 3.03–3.05% | 1.44–1.68% | 6.06–6.52% | 0.760–0.763 |
| Confidence all-step fusion | 99.67–99.68% | 2.66–2.93% | 2.15–2.36% | 1.33–1.54% | 4.73–5.45% | 0.760–0.763 |

These values materially improve every state relative to the `t=0` head and
remain close on the unseen seed.  They still do not establish a universally
best timestep: the result is one head, one identity, one pose, and two noise
seeds.  The two simple fusions are reported as controls rather than an
architecture choice.

`RGB-mask IoU` is an additional controlled-vacuum check: the generated
foreground is approximated as `max(RGB) > 12` after resizing to the 64 by 64
head grid.  It is not a general segmentation metric, but it confirms that the
predicted mask follows generated pixels rather than only forming a coherent
standalone template.  The best old `t=0`-head mean was 0.700; the conditioned
head reaches approximately 0.77 on both seeds for steps 1–3.

Feature-dependence controls rule out the simplest camera-template
explanation.  On the same 96 exact samples, replacing Qwen features with zeros
raises mean median XYZ from 1.79% to 681% and mask IoU falls from 1.0 to zero.
Rolling every feature map by half an image gives 275% median and zero mask IoU.
The head therefore depends on spatial Qwen features rather than reconstructing
Rain only from the nominal camera and rays.

The same-seed 24-view generation took 396 seconds and peaked at 22.28 GiB CUDA
allocated.  Both generated contact sheets remain visually coherent and the
canonical XYZ coloring stays attached to the corresponding body regions.

The generated-domain gate now passes strongly enough to stop tuning on Rain.
It remains a correspondence proxy: semantic Qwen views have no exact generated
per-pixel Blender ground truth, and uncertainty/off-surface correlation is
still only about 0.14–0.19.

Next experiments, in order:

1. Add several identities and pose variation, holding out complete identities
   and poses rather than more Rain cameras.
2. Keep all four states available during that experiment; do not freeze a
   timestep or fusion policy from the single-character result.
3. Calibrate uncertainty on held-out generated identities and seeds.
4. If identity/pose transfer exposes camera drift, estimate a per-view camera
   correction and add explicit ray, reprojection, and cross-view losses.
5. Then compare alternative feature taps and integrate the resulting clouds in
   Termin.
