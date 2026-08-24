# Canonical head: camera ray-map ablation

## Question

The selected `multiscale-v2` head receives both a global 16D camera vector
through FiLM and a spatial six-channel camera ray map.  Generated Qwen views do
not necessarily match the nominal requested camera, so this experiment asks
whether the exact per-pixel ray map is redundant or actively harmful.

Only the ray map is removed.  The 16D camera FiLM conditioning and normalized
denoising timestep remain unchanged.

## Matched initialization and training

The selected open-assets checkpoint was not trained from scratch on all 17
identities.  To keep the ablation matched, the no-ray initializer was derived
from the same pre-openassets17 checkpoint used by the ray baseline:

- source: `scale10-multiscale-v2-finetune-gate`, epoch 2, step 1,920;
- source stem: `128 x 262 x 3 x 3` (`256` projected DiT channels plus `6` rays);
- no-ray stem: the first `256` input channels of the same tensor;
- every other model tensor is bit-identical to the source initializer;
- both variants then receive 3,264 updates over the same 17 identities with
  seed `20260822`, AdamW `1e-4`, cosine decay and batch size one.

The no-ray model has 12,771,247 parameters, exactly 6,912 fewer than the ray
baseline (`128 x 6 x 3 x 3`).

| Epoch 2 training aggregate | Rays | No ray |
| --- | ---: | ---: |
| Mean total loss | 0.089755 | 0.089966 |
| Primary XYZ loss | 0.0006842 | **0.0006798** |
| Primary mask loss | **0.041709** | 0.041871 |
| Mean gradient norm | 1.12695 | **1.12228** |

The curves are effectively equal.  A ray map is not required to fit the exact
training corpus when global camera FiLM remains available.

## Exact held-out geometry

Values are means across the same 24 views and four denoising states.  Lower
median/p95 is better; higher within-5% and mask IoU is better.

| Identity | Median rays -> no ray | Worst p95 rays -> no ray | Within 5% rays -> no ray | Mask IoU rays -> no ray |
| --- | ---: | ---: | ---: | ---: |
| Jay | **7.043%** -> 7.061% | 25.905% -> **25.584%** | 38.152% -> **38.352%** | **0.847329** -> 0.847318 |
| Victoria | **5.563%** -> 5.580% | **24.982%** -> 25.308% | 49.112% -> **49.159%** | **0.867875** -> 0.867573 |
| Wizard | 4.984% -> **4.970%** | 23.056% -> **22.423%** | 50.772% -> **51.004%** | **0.879091** -> 0.878814 |
| OGA-09 | **8.376%** -> 8.513% | 25.392% -> **25.332%** | **19.393%** -> 18.831% | **0.933866** -> 0.933576 |
| Termin | 4.340% -> **4.336%** | 20.017% -> **20.008%** | 61.210% -> **61.328%** | **0.706761** -> 0.706760 |

The differences are small and identity-dependent.  Exact cameras show no
material advantage for the explicit ray map.

## Generated Qwen views

The same cached Wizard and OGA-09 Qwen features are replayed through both
heads.  Thus image generation, feature tensors, view order and denoising states
are identical.  Confidence fusion uses all four states.

| Identity / metric | Rays | No ray | Relative no-ray change |
| --- | ---: | ---: | ---: |
| Wizard surface median | **1.089%** | 1.117% | +2.55% |
| Wizard surface p95 | **3.395%** | 3.427% | +0.96% |
| Wizard coverage p95 | **5.604%** | 5.751% | +2.62% |
| Wizard projective median | 1.689% | **1.668%** | -1.26% |
| Wizard projective p95 | 7.490% | **7.441%** | -0.64% |
| Wizard RGB-mask IoU | 0.742277 | **0.742687** | +0.06% |
| Wizard voxels from 2+ views | 21.180% | **21.449%** | +1.27% |
| OGA-09 surface median | 3.490% | **3.370%** | -3.42% |
| OGA-09 surface p95 | 9.948% | **9.912%** | -0.37% |
| OGA-09 coverage p95 | 8.628% | **8.434%** | -2.24% |
| OGA-09 projective median | 1.637% | **1.621%** | -0.97% |
| OGA-09 projective p95 | 4.636% | **4.533%** | -2.23% |
| OGA-09 RGB-mask IoU | 0.974604 | **0.974672** | +0.01% |
| OGA-09 voxels from 2+ views | **17.135%** | 16.926% | -1.22% |

The predeclared seven-metric gate selects no-ray with normalized rank `0.286`
versus `0.714` for rays.  This is a weak positive result rather than a large
quality jump: Wizard trades surface distance for slightly better projection
and view agreement, while OGA-09 improves most geometry metrics but loses a
small amount of multi-view voxel agreement.

## Cloud shape

| Identity | Variant | Voxels | Z depth | Reference Z depth | Voxels from 2+ views |
| --- | --- | ---: | ---: | ---: | ---: |
| Wizard | Rays | 22,993 | **0.3493** | 0.4198 | 21.180% |
| Wizard | No ray | 22,854 | 0.3448 | 0.4198 | **21.449%** |
| OGA-09 | Rays | 35,921 | 0.3153 | 0.4462 | **17.135%** |
| OGA-09 | No ray | 36,505 | **0.3187** | 0.4462 | 16.926% |

Projection sheets are visually almost indistinguishable.  Removing rays does
not solve the persistent shallow-depth or diffuse-contour failure.

## Interpretation

With the current discrete camera grid, DiT spatial features plus the global
camera FiLM vector already contain enough information for the head to learn the
view-to-canonical mapping.  The explicit ray map contributes no measurable
held-out benefit and may weakly hurt when a Qwen image deviates from the
nominal camera.

The practical evidence favors omitting the ray map in the next head recipe,
but it is not yet a clean from-scratch architecture verdict.  The initializer's
deeper weights were originally learned in a ray-conditioned head.  A matched
scratch run or another generated validation identity should confirm the choice
before replacing the selected production checkpoint.

## Artifacts

- recipe: `data/canonical-openassets17-no-ray-2026-08-23.json`;
- no-ray checkpoint: `/home/mirmik/mnt-nvme/canonical-experiments/openassets17-2026-08-23/training/openassets17-multiscale-v2-no-ray-finetune/head.pt`;
- exact evaluation: `/home/mirmik/mnt-nvme/canonical-experiments/openassets17-2026-08-23/evaluation/openassets17-multiscale-v2-no-ray-finetune`;
- generated gate: `/home/mirmik/mnt-nvme/canonical-experiments/openassets17-2026-08-23/generated-gate-ray-ablation-wizard-oga09-seed20260822/gate.json`;
- replayed captures, PLY clouds and projection sheets: `/home/mirmik/mnt-nvme/canonical-experiments/openassets17-2026-08-23/generated-replay-ray-ablation`.

## Verification

- converted initializer reloads with `ray_channels=0` and stem shape
  `128 x 256 x 3 x 3`;
- all non-stem tensors are byte-for-byte unchanged during conversion;
- 3,264 training updates complete without non-finite losses or gradients;
- all five exact held-out identities reload and evaluate independently;
- both generated identities replay the exact same cached Qwen tensors through
  both checkpoints;
- four RGB PLY clouds and their front/back/right/top projection sheets were
  exported and visually inspected.
