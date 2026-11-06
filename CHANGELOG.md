# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) from its first tagged release.

## [Unreleased]

## [1.0.0] - 2026-11-06

First tagged release.

### Gestures: wizards, two-hand vocabulary, visible cursors
- Model wizard: choose a drawing plane (plan, side or front), then a rectangle or circle, size it by pulling two
  pinched hands apart (snapped to the section increment, locked by holding still), extrude the same way, confirm with
  a thumbs-up. On the front plane, point at the start and end of the member line on a grid first. Step chips, choice
  cards, a live readout, a sketch grid and a 3D preview show every step.
- Loads wizard: choose what by finger count (fixed, pinned, roller, point, uniform, trapezoidal), point at the start
  (and end) on the member, point the load direction, then type the value. Each pick is confirmed by holding still or
  by pinching the other hand; two fists go back a step.
- Results: one hand does the precise work, both hands the big moves. Two pinches zoom (exactly with the hands'
  distance) and pan; two fists turn; two pointing hands compare two points; both thumbs up reset. A lone pinch does
  nothing, so pinch and point are never mixed up.
- The section cut is always straight and moves in steps; the Workbench uses its X / Y / Z cut and evaluates it on the
  fitted node positions.
- The probe draws a ring cursor snapped onto the model in both apps; the field change is a dial with an on-screen
  strip; confirmations play a short tick.
- Finger counting reads the thumb from its span across the palm, which tells a tucked thumb from a raised one; the
  Workbench sends MediaPipe world landmarks.
- Still modes are refused while the other hand is raised; a second hand never cancels an active mode unless it makes
  the two-hand version of it.

### Gestures (earlier in this cycle)
- Faster and more accurate: pose changes register in 2 frames, modes arm in 0.1 to 0.2 s (was 0.25 to 0.8 s),
  position filters no longer lag (probe cursor lag 81 ms → 21 ms, error 4.1 % → 1.1 %), and the Workbench camera
  follows the hand with a frame-rate independent ease.
- Two-hand control: pinch with both hands to zoom by the distance between them and rotate by moving them; pinch with
  the other hand to pin the probe value; show the other open palm to tilt the section cut.
- The section cut is straight by default: its axis is chosen from the palm's first frames and locked, so the plane no
  longer staggers with hand tremor. Tilting is opt-in.
- Number entry: a finger count answers a choice question (the load, length and section increments) directly; digits
  need 0.55 s (was 0.8 s); pinch and hold deletes a digit; the keyboard (typing, Backspace) is no longer overwritten
  by the camera frames, and typed text is passed to the gesture side. An answer's finger count must change before the
  next question reads a new one.
- `scripts/gesture_bench.py` measures response time and accuracy on synthetic hands; `scripts/record_demo.py` records
  the demos in `docs/media/` with Playwright and ffmpeg.

### Fixed
- A two-hand grab no longer crashes the engine when one hand leaves the frame.

### Added
- Apache License 2.0 (`LICENSE`, `NOTICE`).
- Project documentation for contributors: README overview with screenshots, contributing guide, code of conduct,
  security policy, issue and pull request templates.
- Continuous integration on GitHub Actions: Ruff, pytest on Python 3.10 to 3.12, and the CAD kernel tests on Node.
- Dependabot updates for pip packages and GitHub Actions.

## 2026-10-07: body-fitted meshing, Python FE core and 3D features

### Added
- Workbench: body-fitted hex8 mesh. Surface nodes are projected onto the CAD surfaces, spread evenly and checked for
  inverted elements; the voxel mesh remains an option.
- Workbench: nonlinear (J2 plasticity), modal and linear buckling analyses, compression-only contact supports and a
  plastic strain (PEEQ) field.
- Workbench: engine choice (AUTO, DOLFINX, PYTHON, BROWSER).
- Python FE core `archimedes_fe.meshsolver` behind `POST /api/fe/solve` and `GET /api/fe/info`: incompatible-mode
  hex8, sparse direct or AMG-preconditioned CG, Newton–Raphson plasticity, active-set contact, shift-invert modal
  and linear buckling.
- CAD kernel: edge fillets and chamfers, shell, loft, sweep, open path sketches.
- Sketcher: constraint solver with geometric relations and driving dimensions.
- DOLFINx backend: `solve_mesh` op, so any part can be solved as a posted mesh.

### Changed
- Ground grid is centred under the part, clipped at the camera, and can be switched off.
- STL import tolerates holes and missing facets.

### Fixed
- A solve no longer blocks the page while it runs.
- The load-increment slider no longer over-scales results when fewer increments are set.

## 2026-10-06: CAD modelling, Python console and gesture fixes

### Added
- Workbench: feature-based CAD kernel (`cadkernel.js`) that models and analyses any part: sketches, extrude,
  revolve, primitives, holes, patterns, mirror and STL bodies.
- Workbench: SolidWorks / CATIA style interaction (PropertyManager, drag handles, Modify box, standard views,
  undo / redo).
- Gesture modeller: Abaqus-style Python console with custom scripts and a replayable journal.

### Fixed
- More robust finger-count reading, load increments, exact section cuts, and switching between the two pages.

## 2026-10-05: gesture FE modelling

### Added
- FE core (`archimedes_fe`) with an incompatible-mode brick solver and modal analysis.
- Gesture modelling of beams and columns, with supports, loads and a results panel.
- WebXR viewer (`/xr`) and GLB export of results.

## 2026-10-03: first version

### Added
- Computer-vision gesture layer: landmarks, features, pose classifier, filters, intent state machine, calibration
  and quality hints.
- Browser demo (MediaPipe in WebAssembly, Python over WebSocket) and desktop app.
- Archimedes Workbench FE pre/post-processor with a DOLFINx backend, wired to the gesture layer.

[Unreleased]: https://github.com/si-ux/Archimedes/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/si-ux/Archimedes/releases/tag/v1.0.0
