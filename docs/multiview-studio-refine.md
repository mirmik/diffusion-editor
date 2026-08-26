# Multiview Studio refine regions

Multiview Studio stores one refine region for the current source geometry.  In
the first implemented interaction slice, that region is an axis-aligned cube
with a geometry-local center and one uniform side length.

## Selecting a cube

1. Build or load the model in the 3D workspace.
2. Press **Pick cube** and click the part of the model to refine.
3. Drag the red **X**, green **Y**, or blue **Z** arrow to move the cube.
4. Drag the magenta diagonal arrow or its endpoint to resize uniformly.
5. Once the cube exists, optional numeric **X**, **Y**, **Z**, and **Size**
   fields appear for exact adjustment. They stay hidden before selection.
6. Orbit the camera to check the region from several sides, then press
   **Confirm** when the cube covers the intended part.

A draft cube is orange and a confirmed cube is green. **Cancel** restores the
cube that existed before the current edit; **Clear** removes the selection.
The cube remains visible while the camera is orbited.

The selection is serialized in `.mvstudio.json` in glTF Y-up geometry-local
coordinates. It contains a SHA-256 fingerprint of the source geometry. If the
geometry file changes or a different generated mesh replaces it, the editor
discards the stale cube instead of applying it to an incompatible model.

The arrow names also refer to glTF coordinates. The Termin viewport is Z-up,
so the display conversion is explicit: glTF `(x, y, z)` is shown as viewport
`(x, -z, y)`. Consequently the green glTF Y arrow appears along viewport +Z,
and the blue glTF Z arrow appears along viewport -Y. The manipulator and the
numeric fields therefore edit the same stored glTF components even though the
gizmo is drawn in Termin coordinates.

This slice selects and persists the spatial domain only. It does not yet run
Shape-SLat geometry refine or PBR texture refine. Those operations must accept
only a confirmed cube and own their separate conditioning crops, anchor/seam
policy, cache key, and patch acceptance flow.

The complete rationale and worker-space transform are recorded in
[`architecture-council/2026-08-27-multiview-refine-cube.md`](architecture-council/2026-08-27-multiview-refine-cube.md).
