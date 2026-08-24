# Canonical point-map head: first identity-transfer gate

## Question

Does the timestep-conditioned head learn a transferable pixel-to-canonical
mapping, or only memorize Rain v2?  This experiment keeps Qwen Image Edit 2511,
Lightning 1.0, Multiple Angles 0.9, readout blocks 15/30/45/59, the four-step
FlowMatch schedule, camera rays, and the 24-view production grid frozen.

## Identity split

The new characters are local user MakeHuman FBX exports.  Their source paths
and SHA-256 hashes are stored in each dataset manifest; their original creator
and license provenance could not be verified, so the assets and caches remain
outside Git.

| Role | Identities | Use |
|---|---|---|
| train | Rain v2, April, Arthur | head optimization and best-epoch selection |
| held out | Termin | exact evaluation and real Qwen generation only |

The imported FBX files contain static meshes but no armature or vertex groups.
April, Arthur, and Termin are all in similar authored A-poses.  Their canonical
origin is therefore an explicit estimate at the median mesh-object X/Y origin
and 50% of evaluated character height.  This is an identity-transfer gate, not
a pose-transfer result.

Each identity has 8 azimuths by elevations -30/0/+30 degrees at 256 square.
The imported-material visibility mask retains exact ray hits only at pixels
whose rendered alpha is at least 0.5.  All three datasets pass validation with
RGB/mask IoU at least 0.9945 and reprojection error below 0.000114 pixels.

## Training

Each feature cache contains 24 views by 4 scheduler states at the native 64 by
64 target-token grid, or 96 samples and 9.66 GB of float16 arrays.  Termin is
absent from both training runs.

| Head | Initialization | Steps / LR | Train XYZ median | Within 5% | Mask IoU |
|---|---|---:|---:|---:|---:|
| scratch | random | 12,000 / 1e-3 | **2.63%** | **87.62%** | 0.968 |
| warm | Rain-only head | 6,000 / 3e-4 | 3.04% | 82.19% | **0.989** |

The scratch head is the primary result.  Warm-start retains cleaner Rain masks
but does not improve mixed-identity XYZ learning.

## Exact held-out Termin

All numbers aggregate the same 24 views and all four states.  Distances are in
units of full character height.

| Checkpoint | XYZ median | Worst p95 | Within 5% | Mask IoU |
|---|---:|---:|---:|---:|
| Rain-only baseline | 16.29% | 169.05% | 4.92% | 0.576 |
| warm transfer | 7.70% | 30.65% | 27.77% | 0.722 |
| scratch transfer | **5.94%** | 34.88% | **42.18%** | **0.738** |

Scratch improves the Rain-only baseline by about 2.7 times in median error and
8.6 times in the fraction within 5%.  Transfer is therefore real, but it is
still coarse and fails a practical exact-correspondence threshold.

Scratch by scheduler state:

| State | XYZ median | Worst p95 | Within 5% | Mask IoU |
|---:|---:|---:|---:|---:|
| 0 | 5.80% | 34.88% | 42.15% | 0.715 |
| 1 | **5.09%** | **18.11%** | **50.33%** | 0.740 |
| 2 | 5.31% | 18.24% | 46.50% | **0.755** |
| 3 | 7.56% | 25.97% | 29.73% | 0.743 |

No state is selected as a production policy from this one held-out identity.

## Real generated Termin views

One seed (`20260822`) was run at the eight supported eye-level azimuths.  Qwen
kept the character and coat visually coherent through the complete turntable.
The canonical coloring is stable by body region, confirming that the head is
not merely emitting a camera template.

| Prediction | Point-to-surface median | Surface p95 | Surface coverage p95 | Projective median | Worst projective p95 | RGB-mask IoU |
|---|---:|---:|---:|---:|---:|---:|
| step 0 | 1.73% | 5.27% | 5.32% | 3.54% | 10.44% | 0.573 |
| step 1 | 1.46% | **4.62%** | 5.01% | 2.78% | 7.81% | 0.794 |
| step 2 | 1.50% | 5.06% | **4.90%** | 2.94% | 8.28% | 0.831 |
| step 3 | 1.50% | 5.10% | 5.07% | 3.41% | 11.47% | **0.872** |
| confidence fusion | **1.44%** | 4.79% | 5.56% | **2.32%** | **7.18%** | 0.769 |

Nearest-surface metrics are optimistic because a wrong semantic
correspondence can still land on some nearby body surface.  Visual diagnostics
show the principal defect: the predicted mask often follows the body under the
long coat and omits wide coat tails or sleeves.  Fusion improves surface and
projective consistency but narrows the silhouette.

### Fused point-cloud check

The eight generated eye-level views were fused at a canonical voxel size of
0.008 character heights.  The cloud is recognizably the held-out Termin
character: head, torso, arms, legs, and the broad coat remain in their expected
canonical regions.  This is evidence that the output is more than a plausible
per-view color map.

Step 1 is the most useful visualization candidate.  Its XYZ span is
`0.648 x 0.955 x 0.266`, compared with `0.616 x 0.977 x 0.307` for the exact
same-view reference.  The final/fused cloud is cleaner but contracts to
`0.596 x 0.921 x 0.228`, agreeing with the earlier observation that confidence
fusion suppresses real coat and sleeve extent.

Cross-view coincidence remains weak.  At the chosen voxel size, 8.7% of step-1
voxels contain samples from at least two generated views, versus 30.1% for the
exact point maps after matching both the eight views and the 128x128 grid.
Thus the global canonical frame is meaningful, but the predicted surfaces are
still jittered or doubled between views; this cloud is diagnostic evidence,
not yet a reconstruction suitable for meshing.

## Decision

The experiment rejects the strong memorization-only hypothesis: mixed-identity
scratch training produces a large, repeatable improvement on a completely
held-out character and remains coherent on real Qwen generations.  It does not
yet establish production-quality correspondence, pose transfer, or calibrated
uncertainty.

Next work should expand scratch training to substantially more identities,
add genuinely rigged pose holdouts, and keep entire identities and poses out of
model selection.  The four scheduler states remain available; no timestep or
fusion rule is frozen from this run.

## Artifacts

- exact datasets: `/tmp/{april,arthur,termin}-canonical-production-3rings-1855`;
- feature caches: `/tmp/{april,arthur,termin}-qwen-multitimestep-production-24-1855`;
- scratch checkpoint: `/tmp/canonical-head-transfer-scratch-rain-april-arthur-1855/head.pt`;
- warm checkpoint: `/tmp/canonical-head-transfer-warm-rain-april-arthur-1855/head.pt`;
- held-out evaluations: `/tmp/canonical-head-*-eval-termin-1855`;
- generated run and contact sheets: `/tmp/termin-head-transfer-scratch-denoising-eye-1855`;
- step-1 cloud for the native viewer: `/tmp/termin-pointcloud-step1-1855/prediction-canonical.ply`;
- point-cloud projections and manifests: `/tmp/termin-pointcloud-*-1855`.
