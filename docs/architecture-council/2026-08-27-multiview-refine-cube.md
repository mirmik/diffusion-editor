# Multiview refine cube

- Date: 2026-08-27
- Status: Accepted
- Kanboard decision: #1944

## Context

The latest TRELLIS.2 head-refine experiments already proved a concrete local
cube transform.  They clipped the head above a horizontal neck plane, fitted
the clipped geometry uniformly into `[-0.5, 0.5]^3`, encoded that local mesh,
kept the lower Shape-SLat band on the source trajectory, refined the remaining
tokens, and mapped the result back with the inverse uniform transform.  The
experiment stored `local_center` and `local_scale`; the inverse side length is
the cube size.

Multiview Studio now needs the user, rather than a hard-coded top-of-body
fraction, to choose which part is refined.  That selection must be visible on
the current model, persist in the project, and remain distinct from later
placement of a generated replacement patch.

## Decision

One refine operation owns one user-confirmed, axis-aligned cube in the local
coordinate system of the source geometry.  The persistent representation is:

- center: three finite geometry-local coordinates;
- side: one positive finite side length;
- source geometry fingerprint;
- explicit coordinate-space/version marker;
- draft or confirmed state.

The editor enters an explicit `Add refine region` mode.  A click on the mesh
raycasts the current displayed geometry and creates a draft cube at the chosen
part with a model-relative initial size.  The user then adjusts its center and
uniform size in the 3D workspace, checks it from arbitrary orbit-camera views,
and explicitly confirms or cancels it.  Refinement must not start from an
unconfirmed draft.

The implementation provides a visible translucent/wire cube, compact direct
glTF-axis translation arrows, a uniform-size handle, and numeric center/size
fields. The numeric rows stay hidden until an ROI exists, preserving 3D
workspace before selection. A head auto-fit may later be offered as an
optional shortcut, never as the authoritative selection.

Worker code converts the geometry-local cube to the TRELLIS local domain with
one uniform transform.  For a cube side `s`, `local_scale = 1 / s` and
`p_local = (p - center) * local_scale`.  glTF Y-up to TRELLIS Z-up conversion
is explicit at the worker boundary.

The selected cube is the local generation domain, not the final seam policy.
Frozen anchor bands, feathering, overlap collars, cut planes and patch welding
remain pipeline parameters derived inside that domain.  Likewise, per-view 2D
conditioning rectangles are separate authored data; a projected cube may
provide an initial suggestion but is not assumed to match generated views.

## Rationale

An axis-aligned cube and uniform scale reproduce the successful experiment
without anisotropic distortion or a new rotated Shape-SLat coordinate
contract.  Direct selection on the model satisfies the product requirement
that the user chooses the part to refine.  Separating selection, seam policy
and generated-patch placement prevents one transform from silently serving
three incompatible purposes.

Binding the cube to the source geometry fingerprint makes stale selections
detectable after rebuilding or replacing geometry.  Retaining ordinary
geometry-local coordinates keeps rendering, picking and serialization simple;
normalization remains an execution detail.

## Alternatives considered

### Automatic head preset only

This exactly reproduces the first experiment but cannot select clothes, limbs
or arbitrary defects and does not give the user authority over the operation.
It remains suitable only as an optional initializer.

### Screen-space rectangle as the primary selector

A rectangle is convenient in one view but has ambiguous depth and can select
unrelated overlapping surfaces.  Per-view rectangles remain useful for image
conditioning, not as the canonical 3D ROI.

### Rectangular or oriented box

Independent extents or rotation are more general, but the proven path assumes
a uniform cube and a fixed local vertical direction.  Supporting them now
would require new normalization, anchor-mask and assembly semantics without
evidence that the added degrees of freedom improve the first head workflow.

## Consequences and risks

- The custom Multiview viewport needs mesh raycasting and a rendered ROI
  overlay before the operation is usable.
- Mouse capture must distinguish ROI picking/manipulation from orbit and pan.
- A cube becomes stale when its source geometry fingerprint changes; the UI
  must require explicit rebinding rather than silently transforming it.
- A single cube can include disconnected or occluded geometry.  This is a
  downstream extraction/quality problem, not a reason to make screen-space
  selection authoritative.
- Axis alignment may be inefficient for long diagonal limbs.  Oriented boxes
  can be reconsidered after the axis-aligned workflow is measured on more than
  the head case.

## Follow-up work

- #1923: persist and edit the cube, author per-view rectangles, run local
  multiview geometry refine and assemble the replacement.
- #1943: use the confirmed cube as the spatial domain for local PBR texture
  refinement without changing geometry outside the region.
- Reuse the existing refine-patch placement concepts only after local
  generation; ROI selection itself does not own patch translation/rotation.

## Related artifacts

- `scripts/experiment-trellis2-local-face-cube-refine.py`
- `scripts/assemble-trellis2-native-head-cube.py`
- `scripts/experiment-trellis2-native-head-overlap.py`
- `scripts/experiment-trellis2-texture-local-face-cube.py`
- `scripts/assemble-trellis2-refined-head-pbr.py`

## Implementation status

The selector is implemented in Multiview Studio. It includes geometry-bound
serialization, mesh raycasting, a translucent/wire viewport overlay, direct
glTF X/Y/Z translation arrows, a uniform-size handle, conditional numeric
controls, and
Confirm/Cancel/Clear. The current user workflow is documented in
`docs/multiview-studio-refine.md`.

Shape-SLat and PBR processing are deliberately not launched by this slice;
they remain follow-up work in #1923 and #1943.
