# Nine Qwen front replicas through COLMAP — 2026-08-23

## Setup

Nine Wizard images were independently generated with the identical Qwen
Multiple Angles request:

`<sks> front view eye-level shot medium shot …`

Only the seeds differ (`2026082300` through `2026082308`). No synthetic angle,
camera pose, or ordering was supplied to reconstruction. Exact Wizard front
and back renders were used as common conditioning images.

## Visual input

The outputs remain almost strictly frontal. Across seeds Qwen changes crop,
scale, background, hat silhouette, beard, clothing edges, and smaller anatomy.
There is little visible lateral parallax. Contact sheet:

`/home/mirmik/mnt-nvme/canonical-experiments/sfm-front9-wizard-2026-08-23/contact-front9.png`

## Classical SfM

COLMAP used CPU SIFT, exhaustive matching, geometric verification, and
incremental mapping. All 36 image pairs were verified, with 167–286 inliers
per pair. The image content is therefore matchable in 2D.

With one shared unknown camera, COLMAP could not form one scene. It split the
inputs into four models; the largest contains 4/9 images and 412 sparse points.

Allowing one unknown camera per generated image registers all 9 images in one
model with 314 points, mean track length 6.17, and mean reprojection error
1.27 px. This solution is geometrically degenerate:

- focal lengths range from about 921 to 6029 px;
- one radial distortion parameter reaches 101;
- one camera center lies roughly ten times farther away than the cluster;
- most recovered motion is along the optical axis and explains generated
  scale/crop changes rather than lateral camera parallax;
- predictions are identity/image-specific deformations, not observations of a
  single rigid surface.

## MVS decision

The installed pycolmap has no CUDA PatchMatch backend, and no external COLMAP
or OpenMVS dense executable is installed. More importantly, neither sparse
calibration is physically credible. Dense stereo or plane sweep on these poses
would only densify the false camera model, so it is not reported as a recovered
surface.

This is a useful negative result: repeated nominally frontal Qwen generations
contain many stable local correspondences, but their variation is dominated by
2D crop/scale and non-rigid hallucination. Unconstrained SfM can fit them only
by splitting the scene or by assigning implausible per-image intrinsics.

## Artifacts

- Generation manifest and images:
  `/home/mirmik/mnt-nvme/canonical-experiments/sfm-front9-wizard-2026-08-23`
- Shared-camera report:
  `/home/mirmik/mnt-nvme/canonical-experiments/sfm-front9-wizard-2026-08-23/colmap-single/report.json`
- Per-image-camera report:
  `/home/mirmik/mnt-nvme/canonical-experiments/sfm-front9-wizard-2026-08-23/colmap-auto/report.json`
- Largest 9-camera sparse cloud:
  `/home/mirmik/mnt-nvme/canonical-experiments/sfm-front9-wizard-2026-08-23/colmap-auto/sparse-model-2.ply`

View the sparse result in the native renderer:

```bash
./venv/bin/python scripts/view-canonical-pointcloud.py \
  /home/mirmik/mnt-nvme/canonical-experiments/sfm-front9-wizard-2026-08-23/colmap-auto/sparse-model-2.ply \
  --coordinates colmap-y-down --point-size 6
```
