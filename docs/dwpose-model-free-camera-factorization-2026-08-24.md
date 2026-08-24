# Model-free camera factorization from DWPose tracks

Date: 2026-08-24

## Question

Can nine Qwen views determine useful effective cameras before introducing a
MakeHuman body, bone lengths, or any other parametric geometry?

## Input

- accepted batch:
  `qwen-body-bald-verified-batch-seed1961831399`;
- nine images: three nominal azimuths by three nominal elevations;
- 17 standard COCO body points extracted independently by DWPose;
- no bone connections, lengths, symmetry, or MakeHuman data enter the fit.

## Method

The script `scripts/factor-dwpose-cameras.py` first performs rank-3 affine
factorization and a scaled-orthographic metric upgrade. It then jointly
refines free 3D points and one rotation, scale, and 2D translation per view
using confidence-weighted soft-L1 bundle adjustment. The eye-level front view
fixes the coordinate gauge. Nominal yaw is used only to select between the two
depth-reflected metric solutions; nominal angles are not part of the loss.

Reproduction:

```sh
./.venv-pose/bin/python scripts/factor-dwpose-cameras.py \
  /home/mirmik/mnt-nvme/canonical-experiments/\
qwen-body-bald-verified-batch-seed1961831399/manifest.json \
  /home/mirmik/mnt-nvme/canonical-experiments/\
dwpose-camera-factorization-body17-seed1961831400 \
  --track-set body17
```

## Result

- yaw correlation with the nominal 3x3 grid: `0.99637`;
- yaw mean absolute error: `3.62 degrees`;
- elevation correlation: `0.95817`;
- elevation mean absolute error: `7.00 degrees`;
- confidence-weighted reprojection RMS: `18.84 px` at image height `1337 px`;
- median reprojection error: `9.26 px`.

Mean recovered elevation by nominal ring:

| Nominal | Recovered mean |
| ---: | ---: |
| -30 | -16.74 |
| 0 | -2.03 |
| +30 | +30.04 |

The horizontal views are recovered consistently across all elevation rings.
The elevated and eye-level rings are also recovered close to their requested
values. The low-angle ring is systematically much weaker than its nominal
`-30 degrees`, which is evidence about the generated images rather than an
input camera prior.

The fourth-to-third centered measurement singular-value ratio is high
(`0.856` for the 17-point subset), so the nine observations are not an exact
single rigid scaled-orthographic scene. Nevertheless, robust fitting explains
the anatomical tracks to about 1.4% of image height and yields a stable camera
lattice. The residual non-rigidity should remain explicit in later mesh
fitting rather than being silently absorbed by the cameras.

`report.json` contains each full world-to-camera rotation matrix, normalized
scale and translation, so the result can initialize the later MakeHuman fit
without reconstructing matrices from the display Euler angles.

## Decision

Use the recovered cameras as an initialization for body/pose fitting. Keep
camera corrections bounded and retain per-view robust residuals, especially
for the low-angle ring. Do not force the nominal `-30/0/+30` elevations onto
the images.
