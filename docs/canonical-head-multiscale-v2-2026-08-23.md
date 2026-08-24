# Canonical point-map head: multiscale v2

## Question

This experiment tests whether the frozen Qwen features were being limited by
the small local head.  It reuses the exact ten-identity split, 64 by 64 feature
caches, four denoising states, camera/ray conditioning and 12,000-update budget
from the scale-10 experiment.  No Qwen weights or cached features changed.

## Architecture

`multiscale-v2` projects the four 3,072-channel Qwen readouts independently,
then uses a conditioned encoder/decoder:

- 64 -> 32 -> 16 spatial encoder with residual FiLM conditioning;
- global multi-head attention at 16 by 16;
- skip-connected 16 -> 32 -> 64 decoder;
- learned transposed-convolution refinement to 128 and 256;
- auxiliary five-channel predictions at 64 and 128;
- separate XYZ, mask-logit and uncertainty branches at every output scale.

Camera and normalized FlowMatch timestep embeddings condition every residual
stage.  The final native output is 256 by 256 rather than a bilinearly enlarged
128 by 128 map.  The implementation keeps `local-v1` available and treats old
checkpoints without an architecture field as v1.

| Head | Parameters | Native target |
| --- | ---: | ---: |
| local v1 | 1.76 M | 128 square |
| multiscale v2 | 12.78 M | 256 square |

## Training behavior

The full run read the existing feature caches from the NVMe copy.  It processed
12,000 samples in 1,129 seconds (10.63 samples/s), peaked at 944 MB CUDA
allocated / 1.48 GB reserved and 2.83 GB host RSS.

The recipe was not fully stable.  Training improved through step 4,800, then
the mask branch collapsed around steps 5,000-7,200 before partly recovering as
the cosine learning rate fell.  Checkpoint selection is by mean full-epoch
loss, so `head.pt` contains the pre-collapse state at step 4,800.  A follow-up
recipe should lower the initial `5e-4` learning rate and/or rebalance the
primary and auxiliary mask losses; the present run must not be interpreted as
a clean 12,000-step optimum.

On the ten training identities, the selected multiscale checkpoint has 4.63%
aggregate XYZ median, 60.14% within 5%, and 0.796 mask IoU.  The v1 checkpoint
had 4.01%, 67.91%, and 0.868 respectively.  The larger head therefore did not
improve aggregate train fit under this recipe.

## Exact held-out results

All values aggregate the same 24 views and four scheduler states.  `p95` is the
worst per-sample p95, so it exposes catastrophic views that an average median
can hide.

| Identity | Head | XYZ median | Worst p95 | Within 5% | Mask IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| Jay | v1 | 9.08% | 58.97% | 24.94% | 0.846 |
| Jay | multiscale v2 | **7.58%** | **25.48%** | **29.71%** | 0.769 |
| Victoria | v1 | 6.09% | 37.22% | **40.20%** | 0.877 |
| Victoria | multiscale v2 | **6.06%** | **22.76%** | 37.35% | 0.817 |
| Termin | v1 | 5.96% | **22.46%** | 41.04% | 0.716 |
| Termin | multiscale v2 | **5.82%** | 23.04% | **45.19%** | 0.679 |

The deeper head materially reduces the geometric tail on Jay and Victoria and
slightly improves Termin's median.  Its mask branch is consistently worse.
This is evidence that Qwen's frozen readouts contain more transferable spatial
information than v1 could decode, but the current joint loss does not extract
it cleanly.

## Real generated rings

Rain, Jay and Termin were regenerated with the same seed (`20260822`), four
denoising steps and eight supported eye-level azimuths.  The table compares
confidence fusion from all four head readouts.

| Identity / head | Surface median | Surface p95 | Coverage p95 | Projective median | Worst projective p95 | RGB-mask IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rain v1 | 1.10% | 3.42% | 4.97% | 1.40% | 3.43% | 0.885 |
| Rain multiscale v2 | 1.11% | **3.19%** | **3.57%** | **0.97%** | **3.16%** | **0.947** |
| Jay v1 | 3.01% | 8.40% | 17.10% | 2.10% | 6.45% | 0.830 |
| Jay multiscale v2 | **1.35%** | **5.96%** | **13.56%** | **1.33%** | **4.69%** | **0.909** |
| Termin v1 | **1.49%** | 4.10% | 5.81% | 2.10% | 5.76% | 0.853 |
| Termin multiscale v2 | 1.55% | **3.81%** | **5.01%** | **1.59%** | **4.60%** | **0.881** |

Jay is the clearest gain: the old cloud was a broad doubled imprint, while v2
produces a recognizable single body with coherent hat, torso and legs.  Rain
and Termin improve primarily in silhouette and cross-view consistency.

## Point-cloud check

All confidence-fusion clouds use voxel size 0.008 character heights.

| Identity | Head | Voxels | Depth span | Voxels from 2+ views |
| --- | --- | ---: | ---: | ---: |
| Rain | v1 | 6,585 | 0.199 | 7.37% |
| Rain | multiscale v2 | 16,700 | 0.186 | **22.81%** |
| Jay | v1 | 4,222 | 0.198 | 3.27% |
| Jay | multiscale v2 | 11,069 | 0.168 | **17.50%** |
| Termin | v1 | 10,827 | 0.214 | 9.25% |
| Termin | multiscale v2 | 23,721 | 0.221 | **25.14%** |

Cross-view voxel agreement improves by roughly 2.7-5.4 times.  However the
clouds remain too shallow: predicted/reference depth spans are 0.186/0.421 for
Rain, 0.168/0.289 for Jay and 0.221/0.307 for Termin.  Multiscale decoding
reduces lateral smearing but does not solve canonical thickness.

## Decision

The experiment accepts the multiscale hypothesis in a limited sense: head
capacity was a real bottleneck for generated-domain consistency and held-out
geometric tails.  It does not yet produce reconstruction-quality geometry.
The next run should first stabilize optimization and branch balancing rather
than make the network still larger.  Missing depth remains a separate modeling
problem to diagnose after a stable recipe exists.

## Artifacts

- recipe: `data/canonical-multiscale-v2-2026-08-23.json`;
- implementation: `diffusion_editor/training/canonical_pointmap_head.py`;
- checkpoint and training metrics:
  `data/canonical-experiments/scale10-2026-08-23/training/scale10-multiscale-v2`;
- exact held-out metrics:
  `data/canonical-experiments/scale10-2026-08-23/evaluation/scale10-multiscale-v2/{jay,victoria,termin}`;
- generated rings:
  `data/canonical-experiments/scale10-2026-08-23/generated/{rain,jay,termin}-eye-multiscale-v2-seed20260822`;
- RGB PLY clouds:
  `data/canonical-experiments/scale10-2026-08-23/clouds/{rain,jay,termin}-eye-multiscale-v2-fusion/prediction-rgb.ply`.

## Verification

- four multiscale/factory unit tests pass under `.venv-workers`;
- the regular local suite reports 12 passed and four PyTorch-dependent skips;
- the full checkpoint was reloaded for all three exact held-out evaluations
  and all three generated Qwen rings;
- all exported PLY files were regenerated from the v2 manifests and visually
  inspected using their front/back/right/top projection sheets.
