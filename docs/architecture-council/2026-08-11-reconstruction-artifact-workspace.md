# Reconstruction artifact workspace

- Date: 2026-08-11
- Status: Accepted
- Kanboard decision: #1490

## Context

The existing reconstruction tool presents one linear set of backend stages and
stores one active preview artifact per stage. Completed base and masked-refine
runs provide useful version comparison, but they cannot represent independent
variants of sparse occupancy, LR shape, HR shape, local 1536 geometry, or
texture processing. Re-running a later target also repeats compatible earlier
work because resumable checkpoints are not first-class inputs.

The experimental Pixal3D local-detail path makes the limitation especially
visible. A useful refinement exposes and approves several intermediate results:
the mask and 3D ROI, the cropped and upscaled conditioning image, the isolated
local mesh, local-to-global registration, overlap and fusion diagnostics, the
locally textured mesh, transferred texture maps, and final assembly. LR shape
refinement likewise needs its own variants and approval before LR-to-HR
coordinate upsampling.

At the same time, the current reconstruction panel is usable for Pixal3D,
TRELLIS.2, SPAR3D, Hi3DGen, Hunyuan3D 2.1, and SAM 3D Objects. Replacing its
data model and controls in place would expose every backend to a long partial
migration.

## Decision

Reconstruction experiments will use an immutable operation-and-artifact DAG
behind a staged user interface.

The DAG contains:

- backend pipeline definitions declaring operations, ports, parameters,
  preview capabilities, checkpoint capabilities, and supported refine modes;
- immutable operation records identified by their input artifact IDs, complete
  parameter snapshot, model identity, worker protocol version, and seed;
- immutable artifacts for images, masks, volumes, point clouds, latents,
  checkpoints, meshes, textures, and diagnostics;
- explicit accepted variants that feed subsequent operations, independently
  from the artifact currently selected for preview;
- content-derived operation fingerprints used for reuse and targeted
  recomputation.

Large user-facing stages are presentation groups, not storage records. Each
group can expand into separately executable and inspectable operations. In
particular, the planned Pixal3D path includes sparse variants, LR generation
and LR refine, LR-to-HR coordinates, HR generation and refine, local ROI and
conditioning upscale, isolated local geometry, registration, fusion, geometry
post-processing, local texture generation, texture transfer, and final
assembly.

Advancing between material checkpoints is manual by default through an
explicit acceptance action. Preview selection never changes the accepted
input. Refinement creates a sibling or child variant rather than overwriting an
artifact. Historical downstream results remain valid members of their old
branch; `stale` is only a projection relative to the current draft inputs and
parameters.

The new UI will be introduced as a parallel experimental reconstruction
workspace. It shares existing engines, workers, artifact loaders, and viewport
implementations but initially has its own graph state and controls. The legacy
panel remains the default and keeps its existing execution path for every
backend. Users can enter and leave the experimental workspace without
converting or mutating the legacy run.

The parallel boundary is deliberately at the document/presentation layer, not
at model execution. There will not be a second Pixal3D worker implementation or
a forked backend pipeline.

## Rationale

An immutable DAG preserves comparable variants and allows exact cache keys.
It also represents geometry and texture as partially independent branches that
can be assembled in different combinations. A staged presentation provides
this power without requiring users to edit a free-form node graph.

Keeping the experimental workspace beside the legacy tool limits regression
risk while the graph contract is incomplete. Sharing workers prevents the two
interfaces from acquiring different model semantics or fixes.

## Alternatives considered

### Mutate the current linear pipeline in place

This minimizes the initial UI work but makes stage-local variants, multi-output
operations, and targeted reuse increasingly ad hoc. It also risks regressing
all backends during a Pixal3D-led migration.

### Store only a tree of completed runs

A run tree preserves history but duplicates common prefixes and poorly models
independent geometry and texture branches. Intermediate images and diagnostics
remain subordinate files rather than reusable graph inputs.

### Build a completely separate reconstruction tool and workers

This maximizes isolation but duplicates cancellation, error handling, model
loading, backend fixes, and artifact publication. The accepted design isolates
the new document and UI contract while sharing execution infrastructure.

### Expose a free-form node editor

It represents the graph directly but makes routine staged refinement harder to
understand. The stored graph remains general; the first UI is an ordered,
expandable pipeline browser.

## Consequences and risks

- Legacy runs require an adapter before they appear as graph branches.
- Backend definitions must not claim a refine operation that their worker
  cannot resume.
- Artifact lifecycle and project persistence remain separate work; the first
  slice may be session-local.
- Fingerprints must include model and worker protocol identity or cache reuse
  will be unsafe.
- Local-detail outputs can be very large, so persistent storage needs bounded
  cleanup and explicit missing-artifact states.
- Experimental and legacy controls may temporarily expose different feature
  sets. The workspace must be visibly marked experimental and must never
  silently rewrite legacy state.

## Follow-up work

- #1487: introduce the operation/artifact graph and legacy adapter.
- #1486: build the parallel pipeline, variant, and artifact browser.
- #1489: add content-addressed reuse and targeted invalidation.
- #1447: resume generation from compatible stage checkpoints.
- #1488: publish local upscale, geometry, registration, fusion, and texture
  diagnostics.
- #1430: persist graph metadata and manage project-owned artifacts.

