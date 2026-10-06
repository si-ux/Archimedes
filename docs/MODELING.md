# Building and solving a member by gesture

The browser workbench (`make demo` → port 8000) has three phases, shown as tabs at the top of the viewport.
Gestures mean different things in each phase. Mouse and keyboard always work as well.

| Phase | What you do | Gestures |
|---|---|---|
| ① Model | create the member | two-hand pinch-and-pull, or draw a circle; enter numbers by fingers |
| ② Loads & supports | add supports and loads, then solve | fist / pinch / V held still, point-and-flick, palm sweep, two palms, thumbs-up |
| ③ Results | explore the results | rotate, zoom, palm-plane section cut, probe, V-swipe fields (see `DESIGN.md`) |

## ① Model

### Increments first
On entering Model (or with **⚙️ Increments…** in the toolbar) three prompts set the step sizes. Answer each with
1–5 fingers, by clicking a line on the card, or by typing:

| Fingers | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| length step (mm) | 10 | 50 | **100** | 250 | 500 |
| section step (mm) | 5 | 10 | **25** | 50 | 100 |
| load step (kN, kN/m) | 0.5 | **1** | 5 | 10 | 50 |

Bold = default. Every length, section and load you enter afterwards snaps to its step. The two-hand pull pre-fills
the length to the nearest step. The chosen steps are shown in the left panel under *Model*.

| Gesture | Result |
|---|---|
| 🤏🤏 **Both hands pinch** (index + thumb), pull apart, hold still 0.6 s | rectangular member. A horizontal pull gives a **beam**, a vertical pull a **column**. The pull distance pre-fills the length |
| ☝️ **Draw a circle** in the air with the index finger | circular member. Then: 1 finger = column, 2 = beam; diameter; depth (length) |
| ✊ fist drag · 🤏 one-hand pinch | rotate · zoom |

### Entering numbers with your fingers
Every number prompt (the card at the bottom of the viewport) can be answered three ways: with fingers, by typing, or with the OK button.

**To accept a value: show 🙌 both open palms and hold about a second** (the bar on the card turns green), or press
Enter. With nothing typed, the same gesture accepts the default shown on the card.

| Do | Means |
|---|---|
| hold up *n* fingers, still, 0.8 s | digit *n* (0 = fist; use two hands for 6–9: 5 + 3 = 8) |
| change the count, or lower the hand | needed before the same digit can be entered again (3-0-0 = 3, fist, open, fist) |
| 🙌 both open palms, 0.8 s | **accept** |
| 🤏 one-hand pinch, move up / down | nudge the value by one increment per step (about 0.6 palm heights) |
| swipe one open hand to the left | delete the last digit |
| ✊✊ two fists, 1 s | cancel |

The big number on the card is what the camera currently counts. The ring shows the hold.

### When fingers are misread
- The camera panel shows a dot per finger (T I M R P, lit = extended) and the count, so you can see what the
  tracker sees.
- Finger states come from MediaPipe's *world* landmarks (metric 3D), so a tilted hand or a hand far from the camera
  counts the same. Each finger has a hysteresis band: a half-bent finger keeps its last state instead of flickering.
- Spread the fingers a little, keep the palm facing the camera, and keep the hand inside the frame. **Calibrate** and
  *Collect training data* (right panel) adapt the classifier to your hands.

## ② Loads & supports

Point at the member: the yellow ring shows where the hand lands (and *x*/*z* in metres). Positions within 6% of an end
snap onto it.

| Gesture | Adds | Notes |
|---|---|---|
| ✊ fist **held still** 0.8 s | **fixed** support | dragging the fist rotates the view instead |
| 🤏 pinch **held still** | **pinned** support | moving the pinch zooms instead |
| ✌️ V **held still** | **roller** support | |
| ☝️ point at a spot, hold 0.6 s, then **flick** | **point load** in the flick direction | snapped to the world axis that looks closest on screen; no flick within 1.5 s = straight down (−Z). Then enter kN |
| ✋ **sweep a flat palm** along the span, then stop | **uniform load** over the swept span | the palm faces the way the load acts (palm down = −Z, gravity). Then enter kN/m |
| 🙌 **both flat palms** at the two ends, hold 0.8 s | **trapezoidal load** between the hands | then enter the start and end intensity (kN/m) |
| 👍 thumbs-up **held still** 1.2 s | **solve** | |

All six directions (±X, ±Y, ±Z) are reachable by gesture. The prompt card also has ±X/±Y/±Z buttons to correct
the direction before you answer. The left panel lists every support and load, plus any reason the model can't be
solved yet ("rollers alone let the beam slide…").

### What the supports restrain
| | Beam (along x) | Column (along z) |
|---|---|---|
| fixed | every node of the cross-section, x y z | same |
| pinned | bottom edge line: x y z (free to rotate) | base: centre line x y z; above the base: x y only (it can still shorten) |
| roller | bottom edge line: y z (slides along the beam) | centre line: x y (lateral guide) |

## Section cut
The cut is an exact plane, not whole elements removed. The surface is clipped on the GPU, and the cap is each crossed
element's own slice (field values interpolated along its edges), so the cut face is flat at any angle. The
Workbench's 2D renderer does the same in software: crossed elements are clipped polygon by polygon, and their slice
polygons form the cap. Axis cuts in the Workbench are exact too, wherever the slider puts them.

## Switching between the modeller and the Workbench
The *Modeller · Workbench ↗ · VR/AR ↗* links (top left) and the Workbench's **← Gesture modeller** button (top centre)
open each page in its own named tab. Switching brings the other tab forward with its state as you left it. If a tab is
reloaded, the modeller's model and results come back from the server (they are kept per browser).

## The solver (`archimedes_fe`)

The browser demo uses a small solver that runs in a Codespace without Docker. The DOLFINx backend in `Archimedes/backend` is the
authoritative one.

- **8-node bricks with Wilson–Taylor incompatible modes** (statically condensed), on a structured voxel grid of the member.
  Plain bricks lock in bending, while these match beam theory with 6 elements through the depth.
  Circular sections are voxelised (8 × 8 cells across).
- Static: sparse LU (SuperLU, banded node order). Modal: shift-invert Lanczos (ARPACK) with lumped mass. The first 6 modes are computed.
- Results at the nodes: Gauss-point strains extrapolated to the corners and averaged at shared nodes.

| Check (tests/test_fe.py) | Result |
|---|---|
| cantilever tip deflection vs Timoshenko | within 1% |
| bending stress at mid-span | exact to 0.1% |
| simply supported UDL, mid-span deflection | within 6% (shear included in FE, not in 5wL⁴/384EI) |
| first two cantilever frequencies vs Euler–Bernoulli | within 6% |
| short column axial stress vs P/A | within 2% |
| sum of reactions vs applied loads | exact |

**Result fields:** von Mises, Tresca, S1/S3 principal, S11 S22 S33 S12 S13 S23 (MPa);
U, U1 U2 U3 (mm); equivalent strain and E11 E22 E33 E12 E13 E23 (µε, tensor shear); strain energy density (kJ/m³);
mode shapes 1–6 with frequencies (animated). Indices 1, 2, 3 = world x, y, z.

**Columns:** the left panel reports K (from the end supports), KL/r, the Euler load, the squash load A·f, and whether
the column is *short* (crushing governs) or *long* (buckling governs). The demos include one of each.

## Demos

| Demo | Member | Supports | Loads |
|---|---|---|---|
| Continuous beam | 300 × 500 concrete, 12 m | pinned + 3 rollers (3 × 4 m spans) | 20 kN/m UDL, 60 kN point mid-span |
| Cantilever | 150 × 300 steel, 3 m | fixed | 30 kN at the tip, 0→15 kN/m trapezoidal |
| Long column | Ø150 steel, 6 m | pinned–pinned | 200 kN axial, 2 kN lateral at mid-height |
| Short column | 400 × 400 concrete, 2 m | fixed–pinned | 1500 kN axial, 20 kN lateral |

## VR / AR (`/xr`)

Every solve is published to the server's scene endpoints:

| Endpoint | Gives |
|---|---|
| `GET /api/scene.glb?field=von_mises` | glTF binary of the solved member, coloured by any field (`U3`, `S11`, `mode1`…) on the deformed shape, plus support glyphs. Opens in any glTF viewer |
| `GET /api/scene.json` | version, model, statistics, field ranges |
| `GET /xr` | WebXR viewer: **VR** (Quest browser etc.): trigger or pinch = next field, grip = rotate. **AR** (Android Chrome): tap a surface to place the model. **Desktop**: orbit with the mouse. It reloads automatically after each new solve |

WebXR needs HTTPS. Codespaces' forwarded ports already are. To open `/xr` on a headset, make port 8000 public in the
Ports tab, or sign in to GitHub in the headset browser. iPhone Safari has no WebXR, so download the `.glb` instead
(an AR viewer app can show it).
