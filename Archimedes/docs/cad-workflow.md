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
| **Imported (STL)** | a triangle mesh as a body: boss or cut | scale, move. Its faces are facets grouped by normal. A mesh with holes or missing facets is still read: inside / outside is voted over three ray directions, and the feature warns |
| **Loft** | blends between two closed profiles on parallel planes (a square to a circle, say) | the two sketches |
| **Sweep** | moves a closed profile along an open path sketch (lines and fillet arcs) | profile and path sketches |
| **Fillet / Chamfer** (3D) | rounds or bevels the edges between faces: convex edges lose material, concave edges gain it | radius or distance; the edges |
| **Shell** | hollows the part to a wall thickness, leaving the selected faces open | thickness; the open faces |

Insert features from the toolbar in the viewport (Geometry stage), the **Insert** menu, the right panel, or the
right-click menu on a face. With a planar face selected:

- **Sketch** and **Extrude** sketch on that face;
- **Hole** goes into it;
- **Box, Cylinder, Sphere** are placed on it.

Extrude or Revolve with no free sketch starts one, and creates the feature when the sketch is OK'd.

For **Fillet / Chamfer**, select two faces that meet (Ctrl+click) for the edge between them, or one face for all its
edges, then **Use selected faces** in the PropertyManager. For **Shell**, the selected faces are the openings. Loft
takes the last two free closed sketches, and Sweep a closed profile plus an open path. The PropertyManager's **CHECK**
section lists anything the kernel could not do exactly (for example a fillet on an edge it cannot classify).

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
| Path (open) | P | an open polyline for Sweep; Enter, double-click or right-click finishes it |
| Rectangle | R | two opposite corners |
| Circle | C | centre, then a point on the circle |
| Sketch fillet | | click a polyline corner (or a rectangle) to round it with the fillet radius |

Points snap to the grid step and to existing points. Every entity's coordinates can also be typed in the panel.
Pan and orbit keep working with the middle and right buttons.

### Relations and driving dimensions

Lines you draw close to horizontal or vertical get that relation automatically. Select points, lines or circles with
the Select tool (Shift adds) and the panel's **RELATIONS & DIMENSIONS** lists what fits the selection:

| Relations | Dimensions |
|---|---|
| fix, coincident, horizontal, vertical, parallel, perpendicular, equal, tangent, point on line | length, distance, horizontal / vertical distance, radius, diameter, angle |

A dimension drives the geometry: click its label on the canvas (or type in the panel) and the sketch is re-solved
(Levenberg–Marquardt on all the relations at once). Dragging a point moves it through the solver too, so relations hold
while you drag. The sketch colour gives its state: **white** fully defined (0 degrees of freedom left), **cyan**
under-defined (the panel shows how many DOF remain), **red** over-defined or conflicting. Select a relation in the
list and press Delete to remove it.

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

Select faces (Ctrl or Shift to add more), then **+ Fixed**, **+ Roller**, **+ Contact**, **+ Force** or **+ Pressure**
(panel, context toolbar or right-click).

| Item | Meaning |
|---|---|
| Fixed | all displacements zero on the faces |
| Roller | normal displacement zero; free to slide in the face |
| Contact | compression-only support: pushes back on the face but lets it lift off (Python core) |
| Force | a total force vector (Fx, Fy, Fz), shared over the faces by area |
| Pressure | normal pressure; positive pushes into the part, negative pulls |
| Gravity | self weight along −Y (on / off); a uniform temperature step is set in stage 02 |

Items reference faces by name (for example `Extrude1 · side (line 2)`), so they stay attached when you change
dimensions and remesh. If a face disappears, the item says so and the log warns before solving.

## Materials (stage 02)

Library: Al 7075-T6, Al 6061-T6, Steel 4340, Steel S355, Ti-6Al-4V, CFRP, Concrete C30 and PLA. **Custom material**
lets you type E, ν, ρ, yield strength, hardening modulus H (for nonlinear analysis; the library materials use E/100)
and expansion coefficient; it is saved with the part.

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

The part is meshed with hex8 elements on a grid fitted to its bounding box, then **body-fitted**:

- **Spacing:** faces at the box limits land exactly on cell boundaries.
- **Curved and oblique faces:** the skin nodes are projected onto the real CAD surfaces (alternating projections, so
  nodes on an edge settle on both faces), spread evenly along each surface, and checked: an element whose Jacobian
  would invert is relaxed back toward the grid. Moved elements get their own isoparametric stiffness, volume and
  load integrals. Stage 04 ▸ **SURFACE** switches back to the plain voxel staircase; the mesh panel shows the number
  of fitted nodes and the minimum scaled Jacobian.
- **Samples per cell:** up to n³ points are sampled per cell, so that thin and curved features are captured fairly.
- **Size:** set the element size and the element budget in stage 04. Use at least two elements across thin walls for
  bending.

Stage 05 ▸ **ANALYSIS** picks the analysis, and **ENGINE** where it runs:

| Analysis | What it solves | Engines |
|---|---|---|
| Static | linear elastic, small strain: displacement, stress, strain energy, reactions | all |
| Nonlinear | J2 plasticity with isotropic hardening H, Newton–Raphson over the load increments, step cutting; adds the PEEQ field | Python |
| Modal | the lowest natural frequencies and mode shapes, lumped mass | Python, browser (on a coarse level) |
| Buckling | linear buckling load factors on the applied loads, and the buckled shapes | Python |

Contact supports work in every analysis on the Python core.

| Engine | |
|---|---|
| **AUTO** (default) | DOLFINx for the parametric bracket when that core is attached, otherwise the Python core when the page is served by the demo server, otherwise the browser |
| **DOLFINX** | the FEniCSx container (`docker compose up` in `Archimedes/backend`): any part, posted as the Workbench's own mesh (`solve_mesh`), linear static, CG + GAMG |
| **PYTHON** | `archimedes_fe.meshsolver` behind `POST /api/fe/solve`: hex8 with Wilson–Taylor incompatible modes (no shear locking in bending), SuperLU or AMG-preconditioned CG by matrix size and shape |
| **BROWSER** | Jacobi-preconditioned CG in the tab, static and modal; nothing to install |

`POST /api/fe/solve` takes `{nodes, elems, fixed, f, material, analysis, n_modes, contact, dT, steps}` (arrays as JSON
lists or base64 float32 / int32, element nodes in the order n = a + 2b + 4c) and returns the nodal fields base64-encoded;
`GET /api/fe/info` says what the core can do.

Checks:

- **Kernel** (tests/cadkernel_test.js, 18 tests, run by `make test`): face naming, the bracket mesher, holes, revolve,
  patterns, mirror, STL parity (also with a facet missing), edge fillets and chamfers (convex and concave), shell, loft,
  sweep, the sketch constraint solver (dimensions, DOF count, conflicts, dragging) and the surface projection.
- **FE core** (tests/test_meshsolver.py, tests/test_fe_endpoint.py): cantilever deflection and bending stress, the
  patch test on distorted elements, thermal stress, modal and buckling against Euler–Bernoulli, plastic strain against
  (σ − σy)/H, compression-only contact, error handling.
- **In the browser**: the cantilever template's root stress is 50.6 MPa against My/I = 50 MPa. The plate with a hole on
  the body-fitted mesh converges steadily with the element size: 72.2, 75.0 and 77.4 MPa at 4, 3 and 2 mm (nodal
  peak, sampled at the corners), against about 74.7 MPa from Kt on the net section. The voxel staircase jumps around
  (58.5, 71.3, 83.3 MPa).
- **Through the Workbench on the Python core**: cantilever modes 90.9 and 135.6 Hz against 90.6 and 135.9 Hz
  (Euler–Bernoulli); a 20 × 20 × 1000 mm column buckles at 6.915 × 1 kN against π²EI / 4L² = 6.909 kN; a contact
  support carries the whole load, with the lifted part of the face released.

Big solves take time: the 2 mm plate (17 880 elements, 70k DOF) takes about 35 s in the tab and about 17 s on the
Python core (AMG-preconditioned CG; about half of it is assembling the incompatible-mode elements). A 7 000-element
cantilever bent into yield over 6 increments takes about 27 s on the Python core.

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
