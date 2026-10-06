# Modelling and analysing any part in the Workbench

The Workbench is a feature-based modeller with an FE solver behind it. A part is a **feature history**, evaluated as
constructive solid geometry (CSG) by `Archimedes/cadkernel.js`, as in SolidWorks or CATIA Part Design: each boss adds
material, each cut removes it, in tree order. Any part you can build that way can be meshed and analysed.

## Features

| Feature | What it does | Parameters |
|---|---|---|
| **Sketch** | closed profiles on the Front (XY), Top (XZ) or Right (YZ) plane at an offset, or on any planar face | rectangles (corner radius), circles, polylines with a fillet per corner; a profile inside another is a hole |
| **Extrude** | sweeps a sketch: boss or cut | depth; normal, reverse or mid-plane |
| **Revolve** | spins a sketch about its vertical or horizontal axis: boss or cut | angle (partial revolves get end faces) |
| **Box, Cylinder, Sphere** | primitives: boss or cut | position, sizes, axis |
| **Hole** | cylindrical cut from an entry point | diameter, direction, blind depth or through all |
| **Linear / Circular pattern** | copies of an earlier feature | direction and spacing, or axis, centre, count and angle |
| **Mirror** | a mirrored copy of an earlier feature | plane (x / y / z = offset) |
| **Imported (STL)** | a watertight triangle mesh as a body: boss or cut | scale, move. Its faces are facets grouped by normal |

Insert features from the toolbar in the viewport (Geometry stage), the **Insert** menu, the right panel, or the
right-click menu on a face. With a planar face selected:

- **Sketch** and **Extrude** sketch on that face;
- **Hole** goes into it;
- **Box, Cylinder, Sphere** are placed on it.

Extrude or Revolve with no free sketch starts one, and creates the feature when the sketch is OK'd.

Every feature opens in its **PropertyManager** (right panel): the model previews live, **✓ OK** (Enter) keeps the edit
and **✕ Cancel** (Esc) undoes it. A new feature that is cancelled disappears. In the feature tree:

- **Double-click** a feature to edit it.
- **Right-click** to suppress, delete or move it up or down.
- A sketch sits under the feature that uses it.

## Sketcher

Opening a sketch turns the view normal to its plane and shows a grid.

| Tool | Key | Use |
|---|---|---|
| Select | S | click an entity; drag its vertices, corners, centre or radius point; Delete removes it |
| Line | L | click points; click the first point, double-click or Enter to close; Esc cancels |
| Rectangle | R | two opposite corners |
| Circle | C | centre, then a point on the circle |
| Sketch fillet | | click a polyline corner (or a rectangle) to round it with the fillet radius |

Points snap to the grid step and to existing points. Every entity's coordinates can also be typed in the panel.
Pan and orbit keep working with the middle and right buttons.

## Editing on the model

| Do | Result |
|---|---|
| hover / click a face | pre-highlight / select (Ctrl or Shift adds); the owning feature lights up in the tree |
| drag the face's **arrow** | resize live, snapping to the grid step (Shift: 1 mm). See below for what each face drags |
| double-click a face, or click a dimension label | the **Modify** box: type (live preview), wheel or ↑↓ to step, Enter ✓, Esc ✕ |

What dragging a face changes:

- the end face of an extrusion: its depth;
- the side face of an extrusion: moves that sketch edge;
- a fillet face: its radius;
- a box face: that side of the box;
- a cylinder wall or sphere: the radius;
- a hole wall: the diameter.

## Supports and loads (stage 03)

Select faces (Ctrl or Shift to add more), then **+ Fixed**, **+ Roller**, **+ Force** or **+ Pressure** (panel,
context toolbar or right-click).

| Item | Meaning |
|---|---|
| Fixed | all displacements zero on the faces |
| Roller | normal displacement zero; free to slide in the face |
| Force | a total force vector (Fx, Fy, Fz), shared over the faces by area |
| Pressure | normal pressure; positive pushes into the part, negative pulls |
| Gravity | self weight along −Y (on / off); a uniform temperature step is set in stage 02 |

Items reference faces by name (for example `Extrude1 · side (line 2)`), so they stay attached when you change
dimensions and remesh. If a face disappears, the item says so and the log warns before solving.

## Materials (stage 02)

Library: Al 7075-T6, Al 6061-T6, Steel 4340, Steel S355, Ti-6Al-4V, CFRP, Concrete C30 and PLA. **Custom material**
lets you type E, ν, ρ, yield strength and expansion coefficient; it is saved with the part.

## Files

| Command | Does |
|---|---|
| File ▸ New Part / New from template | blank part, or the L-bracket, cantilever, plate with a hole, I-beam, flanged shaft |
| File ▸ Open Part / STL… (Ctrl+O) | a saved part (`.archpart.json`) or an STL to start from |
| File ▸ Save Part (Ctrl+S) | downloads the features, study, material and mesh settings |
| Import STL… | adds an STL body to the current part |
| Export STL | the meshed surface |
| Export Results (CSV) | node coordinates, displacements, stresses, strain energy |

## Mesh and solver

The part is meshed with structured hex8 elements on a grid fitted to its bounding box:

- **Spacing:** faces at the box limits land exactly on cell boundaries.
- **Curved and oblique faces:** these become a fine staircase.
- **Samples per cell:** up to n³ points are sampled per cell, so that thin and curved features are captured fairly.
- **Size:** set the element size and the element budget in stage 04. Use at least two elements across thin walls for
  bending.

The in-browser solver (Jacobi-preconditioned CG, 2×2×2 Gauss) runs static analysis with recovery of stress, strain
and reactions, plus optional modes.

The DOLFINx bridge takes the parametric L-bracket only. Any other part solves in the browser, and the log says so.

Checks (tests/cadkernel_test.js, run by `make test`):

- **Face naming:** the bracket's faces are named correctly.
- **Old mesher:** the voxel mesh matches the old bracket mesher exactly (2250 elements at 6 mm).
- **Holes:** sketch holes, revolve, patterns, mirror and blind holes all work.
- **STL:** the parity inside test and the facet grouping work.
- **Edits:** Instant3D edits move the sketch geometry.

In the browser:

- the cantilever template's root stress is 50.6 MPa against My/I = 50 MPa;
- the plate with a hole is 73.5 MPa against about 73 MPa from Kt ≈ 2.2 on the net section.

## Navigation

| | SolidWorks (default) | CATIA (View menu or right-click) |
|---|---|---|
| rotate | left drag or middle drag | middle + left / right drag |
| pan | Ctrl + middle drag, or right drag | middle drag |
| zoom | wheel (at the cursor), Shift + middle drag | wheel (at the cursor), Ctrl + middle drag |
| zoom to fit | **F**, or double-click the middle button | same |

| Key | Does |
|---|---|
| Space | view orientation: Front, Back, Left, Right, Top, Bottom, Isometric, Normal to |
| Ctrl+1 … Ctrl+7, Ctrl+8 | standard views, Normal to the selected face (if the browser lets the page have them) |
| ←→↑↓ | rotate 15° (Shift: 90°); Ctrl + arrows pan |
| Ctrl+Z / Ctrl+Y | undo / redo model edits (features, sketches, dimensions, studies, material) |
| Ctrl+Shift+Z | previous view |
| Delete | delete the selected feature, study item or sketch entity |
| Esc | cancel the entity being drawn, the Modify box or the feature edit, then clear the selection |
