# Canonical point-map head: ten-identity scaling gate

## Question

This experiment tests whether the existing 1.76-million-parameter,
timestep-conditioned canonical point-map head improves when the training split
grows from three to ten identities without changing its architecture.  Qwen
Image Edit 2511, Lightning 1.0, Multiple Angles 0.9, readout blocks
15/30/45/59, four FlowMatch states, nominal cameras and canonical camera rays
remain frozen.

## Split and exact data

| Split | Identities |
| --- | --- |
| train | Rain, April, Arthur, Atom, Barret, Kate, Snow, Ellie, Phil, Rex |
| validation | Jay, Victoria |
| test | Termin |

Every identity has 8 azimuths by elevations -30/0/+30 degrees at 256 square,
with all four Qwen scheduler states cached at a native 64 by 64 token grid.
The official Blender Studio characters are CC-BY assets.  Particle hair was
disabled because it does not produce an exact ray-castable surface; where
available, the hidden geometric hair fallback was explicitly included.
Blender characters were rendered with Workbench and subdivision level zero to
keep the exact image and ray-cast surface consistent.

All thirteen 24-view datasets pass the same validator.  RGB/mask IoU ranges
from 0.9864 to 0.9996 and reprojection error stays below 0.000114 pixels.  The
identity split is complete: neither Jay, Victoria nor Termin appears in head
optimization or checkpoint selection.

## Storage and loader

The realized experiment is self-contained under
`data/canonical-experiments/scale10-2026-08-23` and occupies 118 GiB.  A second
complete working copy is staged under
`/home/mirmik/mnt-nvme/canonical-experiments/scale10-2026-08-23`; `/tmp` is no
longer required to train or evaluate this experiment.

The lazy loader keeps a bounded LRU of four decoded 96 MiB samples.  The full
run peaked at 2.66 GB process RSS rather than attempting to hold the 92 GiB
train cache in process memory.  A fixed-seed eager-versus-cache-one control had
the same sample-order SHA-256, maximum logged loss difference below `7e-5`,
and 11.14 versus 11.72 samples/s on a one-view warm-cache test.

Cold direct reads measured about 0.53 GB/s from the project filesystem and up
to 5.9 GB/s from the NVMe working copy.  The first project-filesystem run was
stopped after step 3,800 at roughly 4 samples/s and retained as
`scale10-scratch-sata-aborted`.  The clean NVMe run completed 12,000 updates in
1,048 seconds at 11.45 samples/s.  It peaked at 650 MB CUDA allocated during
head training.

## Exact results

The ten-identity train fit is deliberately harder than the old three-identity
fit:

| Train corpus | XYZ median | Within 5% | Mask IoU |
| --- | ---: | ---: | ---: |
| Rain + April + Arthur | 2.63% | 87.62% | 0.968 |
| ten identities | 4.01% | 67.91% | 0.868 |

Fully held-out results aggregate 24 views and all four scheduler states:

| Identity | XYZ median | Worst p95 | Within 5% | Mask IoU |
| --- | ---: | ---: | ---: | ---: |
| Jay | 9.08% | 58.97% | 24.94% | 0.846 |
| Victoria | 6.09% | 37.22% | 40.20% | 0.877 |
| Termin | 5.96% | 22.46% | 41.04% | 0.716 |
| old three-identity head on Termin | 5.94% | 34.88% | 42.18% | 0.738 |

The expanded head reduces Termin's worst exact p95 but does not improve its
median, within-5 fraction or mask IoU.  Jay also shows that transfer quality is
still strongly identity-dependent.  Adding identities alone has therefore not
crossed the exact-correspondence gate.

## Real generated Termin ring

The same seed (`20260822`) and eight supported eye-level azimuths were rerun
with the new checkpoint.  Qwen retained a coherent character and turntable.
The confidence fusion comparison uses exactly the earlier post-hoc analysis:

| Head | Point-to-surface median | Surface p95 | Coverage p95 | Projective median | Worst projective p95 | RGB-mask IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| old three-identity | **1.44%** | 4.79% | **5.56%** | 2.32% | 7.18% | 0.769 |
| new ten-identity | 1.49% | **4.10%** | 5.81% | **2.10%** | **5.76%** | **0.853** |

The expanded corpus produces a meaningful generated-domain gain: cleaner
silhouettes, lower tail surface error and better projective consistency.  It
does not improve every metric; median nearest-surface distance and surface
coverage are slightly worse.  This differs from the exact Termin plateau and
suggests that identity diversity regularizes the head against Qwen generation
artifacts without yet solving exact semantic correspondence.

## Point-cloud check

Two RGB clouds were exported at voxel size 0.008 character heights:

| Prediction | Voxels | XYZ span | Voxels with at least 2 source views |
| --- | ---: | --- | ---: |
| confidence fusion | 10,827 | 0.696 x 0.930 x 0.214 | 9.25% |
| step 2 | 11,604 | 0.760 x 0.974 x 0.251 | 7.70% |
| exact reference | 8,853 | 0.615 x 0.975 x 0.307 | 30.07% |

Fusion is visually cleaner; step 2 retains more coat and arm width.  Both are
recognizable as Termin, but neither approaches exact cross-view coincidence.
The clouds remain diagnostic artifacts rather than meshable reconstructions.

### Simpler-character controls

The same generated eight-view ring was also run for Rain (a train identity)
and Jay (a held-out identity).  Their Qwen turntables are visually coherent,
so a failed cloud cannot be attributed merely to a failed rotation prompt.

| Identity / fusion | Point-to-surface median | Surface p95 | Coverage p95 | Projective median | RGB-mask IoU | Voxels with at least 2 source views |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rain, train | 1.10% | 3.42% | 4.97% | 1.40% | 0.885 | 7.37% |
| Jay, held out | 3.01% | 8.40% | 17.10% | 2.10% | 0.830 | 3.27% |
| Termin, held out | 1.49% | 4.10% | 5.81% | 2.10% | 0.853 | 9.25% |

Rain is readily recognizable but shallow and doubled between views.  Jay is
the strongest visual failure: its hat/head and arms smear laterally and its
height contracts even though the generated input views are clean.  Termin is
more coherent than Jay despite its more complicated coat.  Mesh complexity is
therefore not the dominant difficulty; out-of-distribution style and
proportions remain a stronger explanation.

## Decision

This experiment rejects a simple "more identities fixes correspondence"
hypothesis for the current head and 12,000-step budget.  It also shows that the
additional identities are useful: generated-domain silhouette and projective
metrics improve materially, and exact worst-case Termin p95 contracts.

No next architecture is selected here.  The result leaves at least three
plausible causes to separate: head capacity/optimization, the heterogeneous
Workbench-versus-material render domain, and missing pose supervision.  The
same frozen split and caches can support those ablations without another Qwen
feature extraction.

## Artifacts

- experiment spec: `data/canonical-identity-scale10-2026-08-23.json`;
- persistent realized root: `data/canonical-experiments/scale10-2026-08-23`;
- fast duplicate: `/home/mirmik/mnt-nvme/canonical-experiments/scale10-2026-08-23`;
- checkpoint and train metrics: `training/scale10-scratch` below the realized root;
- held-out metrics: `evaluation/{jay,victoria,termin}`;
- generated rings: `generated/{rain,jay,termin}-eye-seed20260822`;
- clouds: `clouds/{rain,jay,termin}-eye-{fusion,step2}`.
