# Multiview Studio refine regions

Multiview Studio stores up to eight confirmed refine regions for the current
source geometry. Each region is an axis-aligned cube with a geometry-local
center and one uniform side length. A separate draft cube is the region
currently being positioned.

## Selecting a cube

1. Build or load the model in the 3D workspace.
2. Press **Pick cube** and click the part of the model to refine.
3. Drag the red **X**, green **Y**, or blue **Z** arrow to move the cube.
4. Drag the magenta diagonal arrow or its endpoint to resize uniformly.
5. Once the cube exists, optional numeric **X**, **Y**, **Z**, and **Size**
   fields appear for exact adjustment. They stay hidden before selection.
6. Orbit the camera to check the region from several sides, then press
   **Confirm** when the cube covers the intended part. The mesh subset is
   added to the **Meshes** collection to the right of the 3×8 view matrix.

The **Main** icon represents the complete model. A single click on **Region 1**
through **Region 8** returns to the complete model and selects that region's
cube for numeric or gizmo editing. Double-clicking an icon opens its subset
mesh in the same large 3D viewport. Double-clicking **Main** returns to the
complete model without a selected cube. The UI does not create a second 3D
viewport.

The displayed subset reuses the source GLB vertex buffer and vertex layout.
Positions, authored normals, UVs, tangents, and compatible material textures
remain unchanged; only triangles outside the cube are removed from the index
buffer. Boundary triangles are kept whole when their bounds intersect the
cube. This deliberately avoids cutting polygons and inventing new vertex
attributes at the preview boundary.

## Painting a refine mask

Double-click a confirmed region. **Show mask** controls only the red overlay;
**Paint mask** separately enables or disables brush input. Enabling painting
also makes the overlay visible. Drag with the left mouse button to paint; hold
**Shift** while dragging to erase. The mouse wheel changes the brush radius
from 4 to 256 screen pixels, and the yellow circle shows the current footprint.

Mask state is independent from model shading. **Flat**, **Smooth**, and
**Wireframe** never hide the mask or enable painting. In surface modes the
weighted mask is drawn as a translucent, depth-tested red surface over the
unchanged model. In **Wireframe**, the complete model keeps its normal wire
color and weighted edges blend towards red, including occluded edges from the
x-ray stroke. Turn **Paint mask** off to orbit the camera while keeping the
overlay visible, then inspect the opposite side of the region.

The first mask tool is deliberately an x-ray brush. It projects every source
vertex and does not consult the depth buffer. A single stroke therefore affects
visible and occluded vertices, including the back of the mesh. Camera orbit is
suspended while a left-button mask stroke is active; it returns after turning
**Paint mask** off.

Each source vertex stores a mask weight from zero to one. The initial brush
profile uses a fixed smoothstep falloff: weight is one at the circle center and
falls smoothly to zero at its edge. Painting applies the maximum of the current
and brush weights; erasing multiplies the current value by the inverse brush
weight. The GPU interpolates the three vertex weights across each triangle, so
opacity represents a real continuous mask value rather than a display-only
tint.

Each region persists non-zero `(vertex index, weight)` pairs per source mesh
together with the same geometry fingerprint as its cube. Older projects that
contain binary source-face masks remain readable: vertices belonging to those
faces are promoted to weight one and the next stroke saves the weighted format.
Reopening the project, switching region icons, or moving and reconfirming a
region preserves the weights.

## Selecting conditioning view patches

Selecting **Main** keeps the normal generation, texturing, mesh-postprocess,
and output controls in the left sidebar. Selecting **Region 1** through
**Region 8** replaces that sidebar with the operation state for that region:
cube size, mask summary, assigned conditioning views, and the patch controls
for the currently selected view.

Select a populated card in the 3×8 view matrix and drag a rectangle directly
over its full-size preview. The rectangle becomes that view's conditioning
patch for the selected region. **Use full view** assigns the entire image and
**Remove patch** removes only this region's assignment. The same operations
are available from the view card's context menu. Left, top, width, and height
percentages provide exact adjustment after a rectangle exists.

Assigned cards show `patch W×H%` in the matrix. Patch rectangles are stored in
normalized image coordinates, separately for every refine region, so they do
not depend on the source image resolution and round-trip in `.mvstudio.json`.
Clearing a view also removes its patches. Replacing incompatible source
geometry removes the regions and their patch groups together. This is a manual
contract: cube projection and automatic crop detection are intentionally not
part of the current slice.

The **X**, **Y**, **Z**, and **Size** rows are visible only while a cube is
actually shown in the 3D scene. Moving a confirmed region turns it back into a
draft; the next **Confirm** updates that region instead of appending a new one.

A draft cube is orange and a confirmed cube is green. **Cancel** restores the
cube that existed before the current edit; **Clear** removes the selection.
The cube remains visible while the camera is orbited.

The draft and confirmed regions are serialized in `.mvstudio.json` in glTF
Y-up geometry-local coordinates. Each contains a SHA-256 fingerprint of the
source geometry. If the geometry file changes or a different generated mesh
replaces it, the editor discards stale cubes instead of applying them to an
incompatible model.

The arrow names also refer to glTF coordinates. The Termin viewport is Z-up,
so the display conversion is explicit: glTF `(x, y, z)` is shown as viewport
`(x, -z, y)`. Consequently the green glTF Y arrow appears along viewport +Z,
and the blue glTF Z arrow appears along viewport -Y. The manipulator and the
numeric fields therefore edit the same stored glTF components even though the
gizmo is drawn in Termin coordinates.

## Running standalone geometry refine

Once the region has a non-empty vertex mask and at least one populated view
patch, **Refine standalone region** becomes available in the region sidebar.
The same panel owns seed, step count, strength, CFG, Shape-SLat resolution, and
preview face target. Running it materializes the existing subset, encodes a new
source Shape-SLat, alternates the manually assigned patches through the sampler
steps, and restores the source trajectory outside the weighted token mask.

Boundary triangles remain whole, so their outer vertices may lie beyond the
authored selection cube. The worker therefore fits the complete extracted
subset uniformly into the encoder's `[-0.5, 0.5]^3` domain with a small margin,
and records that reversible center and scale in `result.json`. This preserves
the fringe without clipping it at the cube boundary.

Each distinct input/settings hash is cached under `refine-runs/`. The output
contains the exact source-region artifact, source and refined Shape-SLat files,
the token mask, cropped/prepared conditioning images, and a separate refined
region GLB. Successful outputs appear as additional `R1 · Refined 1`, etc.
mesh cards. A single click still selects the parent cube; a double click opens
the standalone result in the one shared 3D viewport. Re-running identical
inputs reuses the cached result.

The operation never cuts or modifies Main.

## Texturing a standalone refined region

Double-click an `R… · Refined …` card to select the standalone geometry result.
The region sidebar then exposes **Texture seed**, **Texture steps**, **Texture
latent**, **Texture size**, and **Texture refined region**. The button is enabled
only while that refined GLB still exists and the parent region has at least one
valid view patch.

This operation deliberately re-encodes the refined GLB to Shape-SLat before
sampling texture attributes: the geometry has changed, so the source Shape-SLat
from geometry refine is not treated as its final texture-domain shape. Only the
manually assigned region patches participate. Each normalized patch rectangle
is cropped from its view before TRELLIS.2 preprocessing, and the patches cycle
uniformly through all texture steps without a separate Front warmup.

The worker bakes base color, metallic, roughness, and alpha into a new UV atlas
and stores `textured.glb`, encoded Shape-SLat, texture SLat, prepared patches,
and `result.json` below the geometry result's `texture-runs/` directory. The
`R… · Refined …` card is marked `PBR` and subsequently opens the textured GLB;
its original untextured refined GLB remains recorded and unchanged. Repeating
the same geometry, patches, settings, seed, and model path is a cache hit.

This is texture generation for an isolated replacement part. It is not yet
masked retexturing of an already textured assembled model, does not preserve
an earlier UV layout, and does not perform assembly or seam processing.

Returning an accepted region to Main is a separate later stage. Registration,
replacement-core deletion, seam/CuMesh work, topology processing, and final
assembled-model texturing do not belong to isolated region refinement.

The complete rationale and worker-space transform are recorded in
[`architecture-council/2026-08-27-multiview-refine-cube.md`](architecture-council/2026-08-27-multiview-refine-cube.md).
