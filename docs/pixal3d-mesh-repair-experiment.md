# Pixal3D: replaying the testmodeling mesh-processing recipe

Experiment date: 2026-09-09. User asked to recover the successful tool chain
from `~/test/testmodeling` and try it on the current Pixal3D model.

## Recovered recipe

The recommended control in
`/home/mirmik/test/testmodeling/garment_lab/vaan_visualbruno/README.md` uses
visualbruno/CuMesh, commit `d10e54c30ddd03d11472c1431693f985501c7966`:

1. Load the fork's Python `remeshing.py` against the existing compiled CuMesh
   backend in the TRELLIS.2 environment (no environment rebuild).
2. Narrow-band dual-contouring remesh: resolution 1024, band 1, bounds center,
   scale = largest bounds extent × 1.1, project_back = 0.
3. Keep inner layers (`remove_inner_faces=False`).
4. Drop exact zero-area triangles and unused vertices.

The old experiment then used Blender/StableGen projection and UV baking to
assess the improvement. That was a texturing validation stage, not mesh repair.
The old README recommends the control retaining inner layers; removing them
created many boundary edges. MeshLib final repair was disabled in related
runs because of a previously recorded degenerate-fan problem.

## Current application

Input is the geometry-only Vaan from the native Pixal3D Studio smoke, already
simplified to 98,731 triangles. It is the exact geometry used in the successful
native texturing test. This is not the raw undecimated neural output. Source
and remesher SHA256, backend path, parameters and metrics are in `report.json`.

| Metric | Input | Repaired |
| --- | ---: | ---: |
| Triangles | 98,731 | 4,807,646 |
| Boundary edges | 7,199 | 22 |
| Edges with >2 incident faces | 32,034 | 7,239 |
| Winding-inconsistent two-face edges | 16,329 | 0 |
| Edge-connected components | 28,180 | 2,129 |
| Faces in largest component | 18,856 | 4,500,926 |
| Exact zero-area faces | 0 | 0 |

Metrics use indexed geometry without an additional proximity weld. Remeshing
and export took 3.58 seconds, excluding the topology audit. Fourteen exact
zero-area output triangles were removed. The exported GLB reimport retains
face count and bounds; original input hash is unchanged.

Both models were opened in the real native Studio Vulkan viewport. The
character remains upright and facing forward, with a similar silhouette.
The comparison does not establish new anatomical detail: remeshing a coarse
input cannot recover detail discarded earlier. The output has much better
edge metrics but is still not watertight, and has roughly 49 times as many
triangles. Self-intersections and texture-bake improvement were not measured
in this trial. This supports further evaluation rather than making this
recipe the default generation postprocessor. Remaining topology investigation
is tracked in Kanboard #2290; this replay is #2305.

Repair changes topology and UVs. The trial deliberately operates on geometry
before texturing; it does not transfer the old PBR material. A separate Studio
project is provided for reviewing or texturing the repaired result.

## Artifacts and reproduction

Artifacts under `.local/pixal3d-multiview/repair-vaan-visualbruno/`:
`comparison.html`, `before.png`, `repaired.png`, `repaired.glb`,
`repaired.mvstudio.json`, `raw-remesh.npz`, and `report.json`.

```sh
/home/mirmik/soft/TRELLIS.2/venv/bin/python \
  scripts/experiment-pixal3d-mesh-repair.py \
  .local/pixal3d-multiview/studio-smoke/geometry-upright/model.glb \
  .local/pixal3d-multiview/repair-vaan-visualbruno-rerun
```

The output directory must be new. The script does not modify the old
experiment, input mesh, model weights, or production Studio pipeline.
