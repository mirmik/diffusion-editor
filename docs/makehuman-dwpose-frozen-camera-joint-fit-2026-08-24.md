# MakeHuman joint fit in frozen DWPose cameras — 2026-08-24

The experiment fits one shared MakeHuman shape and rig pose to twelve DWPose
joints observed in the verified nine-view Qwen batch.  It consumes the cameras
from the model-free scaled-orthographic factorization without optimizing any
camera parameter.

## What is optimized

- one global body similarity transform in the already fixed factorization world;
- shared rotations for clavicles, upper/lower arms, pelvis, upper/lower legs;
- MakeHuman `height`, `body_proportions`, torso dimensions, hip width, arm
  lengths, and upper/lower leg height modifiers.

The optimized joint targets are shoulders, elbows, wrists, hips, knees, and
ankles.  The mapping uses the corresponding MakeHuman bone heads.  All nine
views share the same pose; view-specific pose parameters are deliberately not
present.

## Reproduction

```bash
./.venv-workers/bin/python scripts/fit-makehuman-dwpose-joints.py \
  /home/mirmik/mnt-nvme/canonical-experiments/dwpose-camera-factorization-body17-seed1961831400/report.json \
  /home/mirmik/mnt-nvme/canonical-experiments/makehuman-dwpose-frozen-camera-joint-fit-v3-seed1961831400 \
  --targets-dir /home/mirmik/mnt-nvme/canonical-experiments/qwen-body-bald-native-makehuman/makehuman-source/makehuman/data/targets \
  --initial-report /home/mirmik/mnt-nvme/canonical-experiments/qwen-body-bald-native-makehuman/search-native-regional-calibrated-pass3/report.json \
  --iterations 300 --evaluation-interval 25 --device cpu
```

## Result

- weighted joint RMS: `52.91 px -> 22.23 px` at `1177 x 1337`;
- median joint error: `33.39 px -> 10.98 px`;
- 108 confident observations, twelve joints in nine views;
- camera SHA-256 before/after is identical, and the camera tensors pass an
  exact equality check;
- the best checkpoint is iteration 250.

Local MakeHuman segment changes, excluding the optimized global scale:

| Segment | Change |
|---|---:|
| upper arm | +1.53% |
| lower arm | +2.18% |
| upper leg | +0.80% |
| lower leg | -7.23% |

The eye-level ring fits best (`6.2–12.2 px` weighted RMS).  The elevated frontal
view remains the largest outlier (`39.4 px`), showing that a single rigid pose
cannot fully explain Qwen's per-view anatomical drift.  This is now a measurable
model mismatch rather than camera motion being silently absorbed by the fitter.

The persistent directory contains the full JSON report, initial and fitted 3x3
reprojection grids, individual overlays, a posed MakeHuman mesh PLY, a joint PLY,
and a MakeHuman shape `.mhm` file.
