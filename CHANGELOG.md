# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) from its first tagged release.

## [Unreleased]

### Added
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
