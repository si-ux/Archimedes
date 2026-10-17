# Archimedes Workbench

A browser-based part modeller and finite-element pre/post-processor. It is one of the two apps in this repository;
see the [main README](../README.md) for the overview and setup.

The Workbench is served by the demo server at `/workbench/`:

```bash
make demo        # from the repository root, then open http://localhost:8000/workbench/
```

The page also opens directly from disk (`Archimedes Workbench.dc.html`). It then solves in the browser only, since
the Python FE core and the gesture engine come from the server.

## Contents of this folder

| Path | What it is |
|---|---|
| `Archimedes Workbench.dc.html` | The Workbench application: UI, mesher, in-browser solver, results viewer |
| `cadkernel.js` | The feature-based CAD kernel: sketches, constraint solver, features, faces, voxelizer, surface projection. A UMD module that also runs in Node (`tests/cadkernel_test.js`) |
| `support.js` | The small template runtime the page is built on |
| `Archimedes Site.dc.html` | A one-page project overview site |
| `backend/` | The FEniCSx / DOLFINx solver core and its WebSocket bridge, run in Docker ([README](backend/README.md)) |
| `docs/cad-workflow.md` | **User guide:** features, sketcher, loads, mesh, analyses and engines |
| `docs/gestures.md` | The Workbench gesture vocabulary |
| `docs/assistant-backend.md` | Design notes for the planned design assistant |
| `uploads/` | Reference material: the FEniCSx build guide and a sketch |
| `.devcontainer/`, `.vscode/` | A dev container and VS Code tasks for working on the DOLFINx backend |

## Solver engines

| Engine | Needs | Analyses |
|---|---|---|
| Browser | nothing | static, modal |
| Python core | the demo server (`make demo`) | static, nonlinear, modal, buckling, contact |
| DOLFINx | Docker: `cd backend && docker compose up` | static (any part); the parametric bracket also gets gmsh tetrahedra |

**AUTO** (the default) picks the best engine that is available. Details are in
[`docs/cad-workflow.md`](docs/cad-workflow.md#mesh-and-solver).

## Working on the DOLFINx backend

DOLFINx has no Windows build and is awkward to install natively, so it lives in a container:

- **Run it:** `cd backend && docker compose up --build` (or the VS Code task *Solver: start bridge*).
- **Develop inside it:** open this folder in VS Code and choose **Dev Containers: Reopen in Container**. That gives
  you `import dolfinx`, IntelliSense and breakpoints in the solver.

The bridge is bound to `127.0.0.1:8791` on purpose: it runs whatever study it receives. Don't expose it on a network.
