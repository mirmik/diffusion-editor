# Generated-domain checkpoint gate — 2026-08-23

## Decision

Use `epoch02`, the checkpoint after 1,920 additional `lr=1e-4` updates, as the
current multiscale-v2 head. It is a substantially safer compromise than the
12,000-step final checkpoint: it preserves most of the Rain/Termin geometric
gain without the large generated-mask and Jay regressions.

Selected artifact:

`data/canonical-experiments/scale10-2026-08-23/training/scale10-multiscale-v2-finetune-gate/checkpoints/epoch-002-step-001920.pt`

The score difference between `epoch01` and `epoch02` is small (0.3901 versus
0.3846), so this is a working checkpoint, not evidence of a universal optimum.

## Predeclared split and rule

The split was recorded on Kanboard #1875 before generation or checkpoint
evaluation:

- validation identities: Jay and Victoria;
- fixed generated input: seed 20260822, eight eye-level azimuths, four denoising
  states per view;
- final test identity: Termin, not inspected until after checkpoint selection;
- descriptive train-domain control: Rain, not used for selection.

For each identity and checkpoint, confidence-fused predictions were compared by
seven metrics: surface median/p95, surface coverage p95, projective median/p95,
generated-image silhouette IoU, and the fraction of occupied voxels supported
by at least two views. Each identity/metric pair contributes one normalized
rank; all fourteen contributions have equal weight. The lowest mean rank wins,
with declared checkpoint order as the deterministic tie breaker.

## Retained checkpoints and feature cache

The deterministic sample order and training recipe were repeated for 12,000
updates. A complete model checkpoint was retained after every full 960-sample
epoch, producing 12 intermediate candidates at steps 960 through 11,520. The
partial final epoch is represented by `head-final.pt` rather than an additional
periodic file.

Qwen features were captured once per generated view/state as float16 arrays:

- `/home/mirmik/mnt-nvme/canonical-experiments/scale10-2026-08-23/generated-feature-caches/jay-eye-seed20260822` (3.1 GiB);
- `/home/mirmik/mnt-nvme/canonical-experiments/scale10-2026-08-23/generated-feature-caches/victoria-eye-seed20260822` (3.1 GiB).

All eight regenerated Jay PNG files were byte-identical to the earlier
multiscale-v2 run. Evaluation is resumable per checkpoint/identity and verifies
the checkpoint SHA-256 before accepting an existing result.

## Validation result

Lower rank score is better:

| candidate | initial | e01 | e02 | e03 | e04 | e05 | e06 | e07 | e08 | e09 | e10 | e11 | e12 | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mean rank | .5220 | .3901 | **.3846** | .6264 | .7473 | .4945 | .4945 | .5275 | .5220 | .4121 | .5165 | .4066 | .4945 | .4615 |

Key validation metrics (distances/reprojection are percentages of canonical or
image extent; IoU is unitless):

| identity/checkpoint | surface med | surface p95 | coverage p95 | projective med | mask IoU | multi-view ≥2 |
|---|---:|---:|---:|---:|---:|---:|
| Jay initial | 1.35 | 5.96 | 13.56 | 1.33 | .909 | 17.51% |
| Jay epoch02 | 1.38 | 5.86 | 13.66 | 1.51 | .894 | 16.39% |
| Jay final | 1.53 | 6.50 | 14.46 | 1.44 | .783 | 15.10% |
| Victoria initial | 1.76 | 5.03 | 5.72 | 1.64 | .903 | 18.49% |
| Victoria epoch02 | 1.43 | 4.29 | 5.37 | 1.52 | .904 | 20.29% |
| Victoria final | 1.29 | 4.08 | 5.29 | 1.45 | .879 | 22.46% |

The gate therefore detects the late Jay regression while recognizing that late
training continues to improve Victoria geometry. Epoch02 is the first candidate
that gives a useful Victoria gain without sacrificing Jay's silhouette.

## Blind Termin test and Rain control

Neither row affected checkpoint selection:

| identity/checkpoint | surface med | surface p95 | coverage p95 | projective med | mask IoU | multi-view ≥2 |
|---|---:|---:|---:|---:|---:|---:|
| Termin initial | 1.55 | 3.81 | 5.01 | 1.59 | .881 | 25.14% |
| Termin epoch02 | 1.56 | 3.73 | 4.52 | 1.45 | .878 | 23.33% |
| Termin final | 1.34 | 3.64 | 4.22 | 1.40 | .847 | 22.07% |
| Rain initial | 1.11 | 3.19 | 3.57 | .97 | .947 | 22.81% |
| Rain epoch02 | 1.02 | 2.88 | 3.03 | .90 | .936 | 22.53% |
| Rain final | .86 | 2.64 | 2.67 | .84 | .912 | 21.97% |

On Termin, epoch02 retains most coverage/projective improvement and nearly all
mask quality, while its multi-view agreement falls less than the final head. On
Rain, it retains a real geometric improvement with only a small mask and
multi-view cost. This is the intended checkpoint-gate behavior.

## Reproducibility caveat

The repeated training run used the same input feature bytes, seed, sample-order
SHA-256 (`606dfd2a...3bd4fbd`), and exact first-step loss. It was not bitwise
identical to the earlier CUDA run: small gradient differences accumulated, with
final exact-dataset median 2.475% versus 2.460%, mask IoU .90461 versus .90474,
and best epoch loss .055493 versus .055583. This is bounded CUDA numerical
nondeterminism rather than a corpus or checkpoint-retention change. Generated
checkpoint comparisons themselves are deterministic because they replay the
same persisted float16 Qwen features.

## Artifacts and verification

- gate report: `data/canonical-experiments/scale10-2026-08-23/generated-gate-jay-victoria-seed20260822/gate.json`;
- training report: `data/canonical-experiments/scale10-2026-08-23/training/scale10-multiscale-v2-finetune-gate/training.json`;
- Termin selected capture: `data/canonical-experiments/scale10-2026-08-23/generated/termin-eye-gate-selected-epoch02-seed20260822`;
- Termin selected cloud: `data/canonical-experiments/scale10-2026-08-23/clouds/termin-eye-gate-selected-epoch02-fusion`;
- Rain selected capture: `data/canonical-experiments/scale10-2026-08-23/generated/rain-eye-gate-selected-epoch02-seed20260822`;
- Rain selected cloud: `data/canonical-experiments/scale10-2026-08-23/clouds/rain-eye-gate-selected-epoch02-fusion`.

Verification completed:

- local venv: 14 passed and 4 torch-dependent skips across the gate, head,
  feature-loader and canonical-pointmap tests;
- worker venv: 8 corresponding unittest cases passed with PyTorch enabled;
- Python byte-compilation for the trainer, capture, evaluator, and gate modules;
- `git diff --check`;
- real CUDA checkpoint retention, cache capture, resumable replay, blind Termin
  test, Rain control, analysis, and point-cloud export.

## Next experiment

Keep the architecture and epoch02 checkpoint fixed. The next useful change is
targeted generated-domain robustness, especially mask calibration and the
identity-dependent Jay/Victoria trade-off. Enlarging the head before addressing
that would give the exact-domain optimizer more capacity to continue the same
late overfit.
