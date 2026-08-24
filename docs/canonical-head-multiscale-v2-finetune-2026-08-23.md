# Canonical multiscale v2: stable low-LR continuation

## Question

The first multiscale-v2 run improved generated-domain coherence but collapsed
after step 4,800 with AdamW learning rate `5e-4`.  This experiment asks whether
the architecture merely needed stable optimization.  It resumes the selected
step-4,800 checkpoint for 12,000 additional samples with `lr=1e-4`, a fresh
AdamW optimizer and cosine decay.  The frozen Qwen features, ten-identity train
split, Jay/Victoria validation identities, Termin test identity and all other
loss weights remain unchanged.

## Diagnostics and stability

The trainer now records primary, 64-square auxiliary and 128-square auxiliary
losses per epoch, gradient norm before clipping, initial/best/final checkpoint
paths, and preserves `head-final.pt` separately from the best checkpoint.  It
raises immediately on a non-finite loss or gradient.

All 13 epochs improved monotonically.  Mean total loss fell from 0.09572 to
0.05558; mean primary XYZ loss fell from 0.000623 to 0.000262; mean primary mask
loss fell from 0.04727 to 0.02448.  Mean gradient norm fell from 0.892 to 0.611.
There were isolated clipped spikes, including a maximum of 18.21 in epoch two,
but no persistent mask collapse.  Step 12,000 is the best epoch and its state
is bit-identical to `head-final.pt` (maximum tensor difference zero).

The continuation processed 12,000 samples in 1,175 seconds at 10.22 samples/s.
Peak memory was 945 MB CUDA allocated, 1.48 GB CUDA reserved and 2.85 GB host
RSS.  This establishes `1e-4` as a stable continuation learning rate for the
current architecture and corpus.

## Exact results

| Corpus / identity | Head | XYZ median | Worst p95 | Within 5% | Mask IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| ten train identities | v1 | 4.01% | 43.94% | 67.91% | 0.868 |
| ten train identities | multiscale v2 | 4.63% | 45.26% | 60.14% | 0.796 |
| ten train identities | low-LR continuation | **2.46%** | **21.10%** | **88.45%** | **0.905** |
| Jay | v1 | 9.08% | 58.97% | 24.94% | 0.846 |
| Jay | multiscale v2 | 7.58% | **25.48%** | 29.71% | 0.769 |
| Jay | low-LR continuation | **7.19%** | 31.20% | **39.77%** | **0.869** |
| Victoria | v1 | 6.09% | 37.22% | 40.20% | 0.877 |
| Victoria | multiscale v2 | 6.06% | **22.76%** | 37.35% | 0.817 |
| Victoria | low-LR continuation | **4.99%** | 24.04% | **51.36%** | **0.889** |
| Termin | v1 | 5.96% | 22.46% | 41.04% | 0.716 |
| Termin | multiscale v2 | 5.82% | 23.04% | 45.19% | 0.679 |
| Termin | low-LR continuation | **4.22%** | **18.27%** | **63.15%** | **0.733** |

Stable continuation strongly improves exact geometry and recovers the mask
regression.  Jay's worst p95 is the one exception relative to the early v2
checkpoint, although it remains far below v1.

## Generated-domain results

The same deterministic Rain/Jay/Termin eye-level rings were regenerated with
seed `20260822`.  This is necessary because the canonical head reads Qwen's
intermediate denoising states during generation.  Values below are confidence
fusion over all four states.

| Identity / head | Surface median | Surface p95 | Coverage p95 | Projective median | Worst projective p95 | RGB-mask IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rain early v2 | 1.11% | 3.19% | 3.57% | 0.97% | 3.16% | **0.947** |
| Rain continued | **0.86%** | **2.64%** | **2.67%** | **0.84%** | **2.38%** | 0.912 |
| Jay early v2 | **1.35%** | **5.96%** | **13.56%** | **1.33%** | 4.69% | **0.909** |
| Jay continued | 1.50% | 6.54% | 14.62% | 1.45% | **3.93%** | 0.784 |
| Termin early v2 | 1.55% | 3.81% | 5.01% | 1.59% | 4.60% | **0.881** |
| Termin continued | **1.34%** | **3.64%** | **4.22%** | **1.40%** | **3.57%** | 0.847 |

Continuation improves generated XYZ on Rain and Termin but regresses the
generated silhouette on all three identities.  Jay also regresses on most
geometry metrics.  Exact held-out evaluation therefore does not reliably rank
generated-domain checkpoints; longer exact training causes identity-dependent
domain overfit even when it improves exact held-out identities.

## Cloud shape

| Identity | Head | Voxels | Depth / reference | Voxels from 2+ views |
| --- | --- | ---: | ---: | ---: |
| Rain | early v2 | 16,700 | 0.186 / 0.421 | **22.81%** |
| Rain | continued | 15,959 | **0.213 / 0.421** | 21.97% |
| Jay | early v2 | 11,069 | 0.168 / 0.289 | **17.50%** |
| Jay | continued | 9,408 | **0.169 / 0.289** | 15.04% |
| Termin | early v2 | 23,721 | 0.221 / 0.307 | **25.14%** |
| Termin | continued | 22,918 | **0.271 / 0.307** | 22.07% |

Rain becomes somewhat thicker and cleaner.  Termin recovers most of the
missing depth and is visibly more volumetric.  Jay does not recover depth and
loses silhouette completeness.  Cross-view voxel coincidence decreases
slightly for all three, so the continuation trades some exact view agreement
for thickness.

## Decision

The low learning rate solves the optimizer-collapse bug and proves that the
current head was under-optimized.  Simply extending exact training is not a
complete production recipe: it selects the final checkpoint while a real
generated holdout, Jay, has already become worse.  The continued checkpoint is
better for exact correspondence and Termin; the early v2 checkpoint remains
the safer generated-domain generalist for Jay and mask quality.

Further head growth is not justified by this result.  The next training gate
needs generated-domain checkpoint validation or a generated-like augmentation
that does not require exact XYZ, while keeping whole identities out of model
selection.  Until that gate exists, neither early nor continued v2 should be
declared the single production checkpoint.

## Artifacts

- recipe:
  `data/canonical-multiscale-v2-finetune-lr1e4-2026-08-23.json`;
- best/final-identical checkpoint and training diagnostics:
  `data/canonical-experiments/scale10-2026-08-23/training/scale10-multiscale-v2-finetune-lr1e4`;
- exact evaluation:
  `data/canonical-experiments/scale10-2026-08-23/evaluation/scale10-multiscale-v2-finetune-lr1e4/{jay,victoria,termin}`;
- generated rings:
  `data/canonical-experiments/scale10-2026-08-23/generated/{rain,jay,termin}-eye-multiscale-v2-finetune-lr1e4-seed20260822`;
- RGB clouds:
  `data/canonical-experiments/scale10-2026-08-23/clouds/{rain,jay,termin}-eye-multiscale-v2-finetune-lr1e4-fusion/prediction-rgb.ply`.

## Verification

- four PyTorch head tests pass under `.venv-workers`;
- local canonical/PLY suite: 12 passed, four PyTorch-dependent skips;
- syntax compilation and `git diff --check` pass;
- four-step CUDA resume smoke verified checkpoint loading, diagnostics and
  separate best/final outputs;
- full 12,000-step run contains no non-finite values or sustained collapse;
- best and final model tensors compare exactly;
- three exact held-out caches, three generated rings and three exported RGB
  clouds were reloaded, analyzed and visually inspected.
