# Per-view MakeHuman pose refinement — 2026-08-24

This is the second stage after the shared MakeHuman rig fit.  The body shape,
segment lengths, global body similarity, and all nine scaled-orthographic
cameras are loaded from the shared v3 report and frozen.  Each view starts from
the shared pose and independently optimizes only twelve major-bone axis-angle
rotations, with a prior toward the shared pose.

## Reproduction

```bash
./.venv-workers/bin/python scripts/refine-makehuman-dwpose-per-view-pose.py \
  /home/mirmik/mnt-nvme/canonical-experiments/makehuman-dwpose-frozen-camera-joint-fit-v3-seed1961831400/report.json \
  /home/mirmik/mnt-nvme/canonical-experiments/makehuman-dwpose-per-view-pose-refinement-seed1961831400 \
  --iterations-per-view 350 --evaluation-interval 25 \
  --pose-delta-prior-weight 0.15 --device cpu
```

## Result

Overall weighted joint RMS improves from `22.23 px` to `8.27 px`; the median
joint residual improves from `10.98 px` to `1.06 px`.  The camera and shape
SHA-256 values are identical before and after optimization.

| View | Shared RMS | Per-view RMS | Pose delta RMS |
|---|---:|---:|---:|
| low 000 | 22.28 px | 10.58 px | 8.42° |
| low 315 | 22.42 px | 4.32 px | 5.25° |
| low 045 | 20.87 px | 8.14 px | 5.21° |
| eye 000 | 6.24 px | 2.58 px | 4.04° |
| eye 315 | 7.80 px | 3.44 px | 5.84° |
| eye 045 | 12.20 px | 8.25 px | 4.98° |
| elevated 000 | 39.43 px | 5.14 px | 5.00° |
| elevated 315 | 24.10 px | 12.21 px | 7.79° |
| elevated 045 | 25.73 px | 12.82 px | 7.49° |

The result strongly supports the useful approximation that Qwen preserved one
body shape while changing the pose by several degrees between generated views.
It is not a claim that the generated bodies are geometrically identical.

The remaining large residuals are concentrated at chain endpoints: right wrist
in elevated 315 (`40.08 px`), left wrist in elevated 045 (`37.75 px`), and
several ankles (`23–26 px`).  Most other joints are within a few pixels.  This
pattern makes a tightly regularized per-view limb-length experiment more
informative than adding more rotational freedom indiscriminately.

The output directory contains the comparison report, shared and refined 3x3
grids, individual overlays, and nine separate posed mesh/joint PLY pairs.
