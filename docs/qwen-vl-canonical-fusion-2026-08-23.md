# Qwen2.5-VL canonical-head fusion probe

## Question

Does the Qwen2.5-VL understanding path inside Qwen Image Edit expose useful
front/back information that is lost when the canonical point-map head reads
only diffusion-transformer target tokens?

This is a bounded probe, not an architecture selection.  Termin is excluded
from all choices.

## What was captured

The exact Qwen Image Edit prompt template was reconstructed with two 384 px
conditioning images and the production view instruction.  From the complete
Qwen2.5-VL text encoder, the spatial image-token positions were retained after
hidden-state levels 7, 14, 21 and 28.  Each target view therefore has four
levels of two `14 x 14 x 3584` maps, one for front and one for back.  Layer 28
is the final Qwen2.5-VL output that Image Edit passes to its diffusion
transformer; the earlier levels test whether the final language blocks erase
useful locality.

The cache contains eight production eye-level views for 17 training identities
and four complete held-out identities: Jay, Victoria, Quaternius Wizard and
OGA-09.  It occupies 1,888,244,736 bytes on NVMe.

## Direct visual-tower signal check

Before training a head, exact Rain geometry was used to score target-45-degree
to front/back nearest-token correspondence directly.  Only target tokens with
a front/back surface sample within 7.5% of character height were scored.

| Feature | Canonical median | Canonical p95 | Within 5% |
| --- | ---: | ---: | ---: |
| Raw patch embedding | 12.32% | 76.94% | 15.7% |
| Visual block 10 | 8.03% | 36.67% | 35.3% |
| Visual block 14 | **6.45%** | 39.26% | 39.2% |
| Visual block 15 | 6.76% | 35.47% | **41.2%** |
| Visual block 31 | 13.45% | 46.21% | 15.7% |

The middle of the vision tower contains a real correspondence signal.  The
last vision block is substantially worse, so selecting the final layer by
default is not justified.

An earlier pairwise diagnostic had incorrectly treated the transparent black
background of exact Blender renders as foreground.  The analyzer now accepts
explicit masks and otherwise uses image alpha before falling back to the
generated-image background heuristic.

## Fusion head

`multiscale-vl-v3` upgrades the selected 12.78-million-parameter
`multiscale-v2` checkpoint without changing its initial prediction.  At the
`16 x 16` bottleneck, target tokens query all four front/back VL maps using
cross-attention.  Explicit source identity and two-dimensional coordinates are
added to the context.  A zero-initialized scalar gate makes the upgrade exactly
prediction-preserving before training.

For the controlled first test, the entire old head was frozen.  Only the
4,295,425 VL-fusion parameters were trained for two eye-level epochs: 1,088
updates at `3e-4`.  Mean epoch loss changed from 0.0766445 to 0.0765388.  The
learned gate reached -0.03925.

## Exact held-out result

All numbers below use the same 128 cached samples: four denoising states for
eight eye-level views of each held-out identity.

| Model | Mean XYZ median | Worst p95 | Within 5% | Mean mask IoU |
| --- | ---: | ---: | ---: | ---: |
| Frozen `multiscale-v2` | 6.349% | 21.483% | 41.929% | **0.89594** |
| Frozen base + VL fusion | 6.305% | **21.384%** | 42.270% | 0.89590 |

Per-identity XYZ median:

| Identity | Baseline | VL fusion |
| --- | ---: | ---: |
| Jay | **6.861%** | 6.892% |
| Victoria | 5.541% | **5.504%** |
| Wizard | 4.296% | **4.255%** |
| OGA-09 | 8.698% | **8.571%** |

The nominal change is small and mostly positive, but it is not sufficient to
claim that the head used VL content.

## Content ablations

| Held-out input | Mean XYZ median | Worst p95 | Within 5% | Mean mask IoU |
| --- | ---: | ---: | ---: | ---: |
| Real VL context | 6.3054% | **21.3838%** | 42.2698% | **0.89590** |
| Zeroed VL context | **6.2854%** | 21.4392% | **42.4028%** | 0.89520 |
| Spatially rolled VL context | 6.3054% | **21.3839%** | 42.2711% | **0.89590** |

Rolling every VL map by half its width and height has effectively no effect.
Zeroing the maps is not consistently worse and is slightly better on the two
aggregate central metrics.  Therefore the small nominal improvement comes
from a learned nearly static residual/position path, not demonstrably from the
Qwen2.5-VL image content.

## Interpretation

The frozen-base experiment separates two facts:

- middle visual-tower features do contain useful dense correspondence;
- simply attaching the final Qwen2.5-VL conditioning outputs to a frozen old
  bottleneck does not make the head consume that information.

## Matched joint-training test

The possibility that a frozen decoder simply could not interpret the new
residual was tested separately.  Two runs started from the same selected v2
checkpoint and used the same 17 identities, eight views, 1,088 updates, seed
and sample order
`d23ee916fd8e45df846d4b014dd8cbe6e3fcc39d786bc1c67fe0466f48987b13`.

Both runs trained the bottleneck, spatial self-attention, decoder, refinement
and prediction branches at `3e-5`.  The fusion run additionally trained
`vl_context` at `3e-4`; early DiT projectors and encoders remained frozen in
both runs.  The control had 8,635,247 trainable parameters and fusion had
12,930,672.

The training trajectories were effectively identical:

| Run | Epoch 1 loss | Epoch 2 loss | Epoch 2 XYZ loss | Epoch 2 mask loss |
| --- | ---: | ---: | ---: | ---: |
| Matched DiT-only control | 0.07640460 | **0.07543424** | **0.000462223** | **0.034298689** |
| Joint VL fusion | 0.07640496 | 0.07543436 | 0.000462229 | 0.034298775 |

Held-out comparison:

| Input | Mean XYZ median | Worst p95 | Within 5% | Mean mask IoU |
| --- | ---: | ---: | ---: | ---: |
| Matched DiT-only control | 6.335246% | **22.016147%** | 42.212451% | 0.8967330 |
| Joint VL fusion | 6.334598% | 22.018075% | 42.213255% | **0.8967433** |
| Joint fusion, zero VL | 6.334590% | 22.018179% | 42.213984% | 0.8967429 |
| Joint fusion, rolled VL | **6.334438%** | 22.019047% | **42.214721%** | 0.8967431 |

Per-identity XYZ medians also differ only in the fifth or sixth decimal place:

| Identity | Control | Joint VL |
| --- | ---: | ---: |
| Jay | 7.04993% | 7.04907% |
| Victoria | **5.95438%** | 5.95456% |
| Wizard | 4.09710% | 4.09606% |
| OGA-09 | 8.23957% | 8.23871% |

The learned VL gate is only -0.0008186.  Joint training therefore confirms the
frozen-base ablation: final Qwen2.5-VL conditioning content is not being used,
and the apparent differences are numerical noise rather than an architectural
gain.

## Decision after joint training

Do not continue scaling the final-Qwen2.5-VL-output branch in its current form.
The DiT features have already consumed the same prompt embeddings, so feeding
them to the head a second time appears redundant.

The remaining evidence-backed VL direction is a second-stage source/target
fusion using the earlier dense visual blocks 10-15: run the completed generated
target together with front/back through the shared vision tower and exploit the
measured same-feature-space correspondence.  This changes the reconstruction
contract from an in-denoising head to a post-generation pass and should be
treated as a separate architecture experiment rather than an automatic next
step.

## Artifacts

- conditioning cache:
  `/home/mirmik/mnt-nvme/canonical-experiments/qwen-vl-fusion-2026-08-23/conditioning`;
- fusion training:
  `/home/mirmik/mnt-nvme/canonical-experiments/qwen-vl-fusion-2026-08-23/training/frozen-crossattention-2epochs`;
- checkpoint SHA-256:
  `ec2b53bc52c8b06acf854fca4e9ee84ce52e56cd2ef3000e848f500a32904c1d`;
- baseline, real-VL, zero-VL and rolled-VL evaluations:
  `/home/mirmik/mnt-nvme/canonical-experiments/qwen-vl-fusion-2026-08-23/evaluation`;
- matched joint checkpoints:
  `/home/mirmik/mnt-nvme/canonical-experiments/qwen-vl-fusion-2026-08-23/training/{joint-control-2epochs,joint-fusion-2epochs}`;
- joint control SHA-256:
  `45cdbfb6f0d4ad28b1fbd8ae6bdb50d1f5772a937509274fcb1852ef0340c79e`;
- joint fusion SHA-256:
  `a8222cd691340ad871ddb4d3e95ebf9080a58bcec2fd03ba22243cc9179b8393`;
- direct visual-tower probe:
  `/home/mirmik/mnt-nvme/canonical-experiments/qwen-vl-fusion-2026-08-23/analysis/rain-target45-front-back.json`.

## Verification

- the VL extractor reconstructs the production Image Edit prompt template and
  validates both expanded image-token counts;
- the full text encoder loads locally in 15.49 GiB VRAM and emits finite
  `float16` caches;
- unit tests verify v2 checkpoint compatibility, exact prediction preservation
  at a closed gate, context sensitivity at an open gate and gradient flow;
- baseline and all ablations were independently reloaded and evaluated on the
  identical complete-identity holdout.
- matched joint runs have identical sample-order hashes and symmetric
  trainable old-module scopes; smoke predictions matched before training;
- zero and spatial-roll content ablations were independently rerun on the
  joint-trained checkpoint.
