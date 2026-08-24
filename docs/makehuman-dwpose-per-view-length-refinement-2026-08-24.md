# Per-view MakeHuman limb-length refinement — 2026-08-24

This third fitting stage starts from the nine independently refined poses.  It
keeps all cameras, the common MakeHuman shape parameters, and the global body
similarity frozen.  For each view it jointly optimizes a small correction to the
pose and four symmetric rig-segment scales: upper arm, lower arm, upper leg, and
lower leg.

The standard MakeHuman arm-length morph targets were not used for this pass.
Inspection showed that `upperarm_length` produces zero displacement at the rig
joint vertex groups and `lowerarm_length` produces only a very small one.  The
implementation therefore retargets bone-head chains directly and deforms the
exported rest mesh with the same skin weights.  Segment corrections are limited
to `+/-12%`.

## Reproduction

```bash
./.venv-workers/bin/python scripts/refine-makehuman-dwpose-per-view-lengths.py \
  /home/mirmik/mnt-nvme/canonical-experiments/makehuman-dwpose-per-view-pose-refinement-seed1961831400/report.json \
  /home/mirmik/mnt-nvme/canonical-experiments/makehuman-dwpose-per-view-length-refinement-seed1961831400 \
  --iterations-per-view 250 --evaluation-interval 25 \
  --max-length-fraction 0.12 --length-prior-weight 0.20 \
  --pose-correction-prior-weight 0.15 --device cpu
```

## Result

- rotations-only weighted RMS: `8.27 px`;
- rotations plus segment lengths: `1.76 px`;
- median residual: `1.06 px -> 0.085 px`;
- maximum residual: `40.08 px -> 8.07 px`;
- camera hashes and exact tensor comparisons remain unchanged.

The two formerly problematic wrists improve from approximately `38–40 px` to
`4–6 px`; ankle maxima improve from `23–26 px` to less than `0.4 px`.

The largest corrections have a coherent view-dependent pattern:

- elevated side views: upper arms `+9.6%/+11.2%`, lower arms approximately
  `+11.9%`;
- low and selected frontal views: lower legs approximately `+6–10%`;
- most remaining segment corrections are only a few percent.

This demonstrates that nine slightly different poses and limb proportions of a
common rig can approximate the generated images extremely well.  It does not
make the fitted 3D state unique: each view has more rotational and length degrees
of freedom than the twelve 2D joints can fully constrain.  The result should be
used as an image-matching initialization with strong priors, not as ground-truth
anatomy.

The persistent output contains the full per-view/per-joint comparison, a 3x3
overlay, and nine length-refined mesh/joint PLY pairs.
