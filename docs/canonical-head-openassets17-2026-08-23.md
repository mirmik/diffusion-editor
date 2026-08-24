# Canonical head: open-assets 17-identity continuation

## Question

This experiment tests whether the multiscale-v2 canonical point-map head is
primarily limited by identity diversity.  It continues the previously selected
ten-identity checkpoint on a mixed 17-identity training corpus and keeps five
complete identities out of training.

The seven added training identities are three Quaternius characters and four
strictly isolated OpenGameArt meshes.  Quaternius Wizard and OpenGameArt pack-1
identity 09 are new validation identities; Jay and Victoria remain validation
identities, and Termin remains the final test identity.

## Data and framing

Every new identity has 24 exact Blender views: eight azimuths at elevations
-30, 0 and +30 degrees.  Each view has four cached Qwen denoising states from
blocks 15, 30, 45 and 59, giving 96 samples per identity and 1,632 training
samples per epoch.

Two dataset defects had to be removed before training:

- opened `.blend` files now support an exact `only_meshes` allow-list, so a
  numbered OGA character cannot accidentally include sibling variants;
- camera distance can be expressed in character heights.  The new renders use
  2.2 character heights, avoiding crops on unusually large assets.

Blender is also launched with `--python-exit-code 1`, so a render-script
exception can no longer be mistaken for a successful dataset stage.

All nine added feature caches contain exactly 24 views, 96 manifest states and
96 `.npy` files.  The full recipe is
`data/canonical-openassets17-2026-08-23.json`.

## Training

The 12.78-million-parameter multiscale-v2 head resumed
`scale10-multiscale-v2-finetune-gate/checkpoints/epoch-002-step-001920.pt` and
trained for two epochs (3,264 updates) at `1e-4`.

| Epoch | Mean loss | Primary XYZ | Primary mask | Mean grad | Max grad |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.11085 | 0.001136 | 0.051999 | 1.325 | 8.641 |
| 2 | **0.08976** | **0.000684** | **0.041709** | **1.127** | **4.214** |

Training itself took 346 seconds at 9.43 samples/s.  Full runtime was 557
seconds because the trainer also wrote 3,264 per-sample diagnostic images.
Epoch 2 is the best full-epoch checkpoint.

## Exact held-out results

The baseline is the exact checkpoint used to initialize this run.  Lower XYZ
median and worst p95 are better; higher within-5% and mask IoU are better.

| Identity | XYZ median baseline -> new | Worst p95 baseline -> new | Within 5% baseline -> new | Mask IoU baseline -> new |
| --- | ---: | ---: | ---: | ---: |
| Jay | 7.41% -> **7.04%** | 27.77% -> **25.90%** | 36.87% -> **38.15%** | 0.841 -> **0.847** |
| Victoria | **5.28%** -> 5.56% | **24.66%** -> 24.98% | **49.21%** -> 49.11% | 0.867 -> **0.868** |
| Wizard | 9.23% -> **4.98%** | 38.69% -> **23.06%** | 19.51% -> **50.77%** | **0.880** -> 0.879 |
| OGA-09 | 11.91% -> **8.38%** | 37.51% -> **25.39%** | 7.90% -> **19.39%** | 0.931 -> **0.934** |
| Termin | 4.82% -> **4.34%** | 20.72% -> **20.02%** | 56.67% -> **61.21%** | **0.709** -> 0.707 |

The two genuinely new open-asset holdouts improve substantially.  Jay and
Termin also improve.  Victoria has a small geometry regression, while its mask
is unchanged within roughly one tenth of a percentage point.

Epoch 2 also beats epoch 1 on XYZ median for all four validation identities,
so the final checkpoint is preferred by exact validation rather than merely by
training loss.

## Real generated-domain gate

Wizard and OGA-09 were generated as complete eight-view eye-level Qwen rings
with seed 20260822.  Their generated block features were cached once and
replayed through the baseline, epoch-1 and epoch-2 heads.  Metrics below use
confidence fusion across all four denoising states.

| Identity / checkpoint | Surface median | Surface p95 | Coverage p95 | Projective median | Worst projective p95 | RGB-mask IoU | Voxels from 2+ views |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wizard baseline | 1.83% | 5.69% | 17.18% | 2.15% | 11.48% | **0.750** | **22.70%** |
| Wizard epoch 1 | 1.22% | 3.90% | 6.13% | 1.79% | 8.00% | 0.742 | 19.76% |
| Wizard epoch 2 | **1.09%** | **3.39%** | **5.60%** | **1.69%** | **7.49%** | 0.742 | 21.18% |
| OGA-09 baseline | 4.10% | 12.86% | 13.34% | 3.48% | 14.17% | 0.963 | **17.34%** |
| OGA-09 epoch 1 | 3.72% | 10.42% | 9.38% | 1.74% | 5.28% | 0.975 | 17.40% |
| OGA-09 epoch 2 | **3.49%** | **9.95%** | **8.63%** | **1.64%** | **4.64%** | **0.975** | 17.13% |

The declared seven-metric, two-identity generated-domain rank selects epoch 2:

| Candidate | Mean normalized rank |
| --- | ---: |
| baseline | 0.821 |
| epoch 1 | 0.536 |
| epoch 2 | **0.143** |

Unlike the previous long continuation, exact and generated-domain validation
agree here.  Epoch 2 is best on every generated geometry metric for both new
holdouts.  The baseline retains the best Wizard mask IoU and slightly stronger
cross-view coincidence, but those losses are small relative to the geometric
gain.

## Point-cloud check

| Identity | Voxels | Predicted depth | Reference depth | Voxels from 2+ views |
| --- | ---: | ---: | ---: | ---: |
| Wizard | 22,984 | 0.349 | 0.420 | 21.22% |
| OGA-09 | 35,928 | 0.315 | 0.446 | 17.12% |

Both RGB projection sheets are recognizable: Wizard preserves the hat, cloak,
torso and legs; OGA-09 preserves the large head, green clothing, arms and
stance.  This is materially beyond an anonymous silhouette.  The remaining
failure is also clear: contours are diffuse, cross-view voxel agreement is low
and both clouds are too shallow, especially OGA-09.

## Decision

The experiment accepts the identity-diversity hypothesis.  Seven additional
training identities materially improve exact and real generated geometry on
two unseen open characters without broadly damaging the old holdouts.  The
epoch-2 openassets17 checkpoint is the best candidate tested here.

This does not justify a larger head yet.  The next useful increment is more
clean character identities and generated-domain validation, while separately
addressing the persistent shallow-depth and cross-view-coincidence failures.

## Artifacts

- recipe: `data/canonical-openassets17-2026-08-23.json`;
- checkpoint and training report:
  `/home/mirmik/mnt-nvme/canonical-experiments/openassets17-2026-08-23/training/openassets17-multiscale-v2-finetune`;
- exact evaluation:
  `/home/mirmik/mnt-nvme/canonical-experiments/openassets17-2026-08-23/evaluation`;
- generated-domain gate:
  `/home/mirmik/mnt-nvme/canonical-experiments/openassets17-2026-08-23/generated-gate-wizard-oga09-seed20260822/gate.json`;
- viewable RGB clouds:
  `/home/mirmik/mnt-nvme/canonical-experiments/openassets17-2026-08-23/clouds/{quat-wizard,oga-pack1-09}-eye-openassets17-fusion/prediction-rgb.ply`.

## Verification

- strict mesh-selection, Blender-runner and relative-camera unit tests pass;
- all nine new exact datasets passed silhouette-IoU and reprojection checks;
- every new feature cache was structurally checked for 24 views and 96 states;
- baseline, epoch 1 and epoch 2 were reloaded independently for exact and
  generated-domain evaluation;
- both exported RGB and canonical projection sheets were visually inspected.
