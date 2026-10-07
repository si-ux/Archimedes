# Archimedes · gesture result viewer (Atomcamp final project, Computer Vision track)

Explore finite-element results with your hands: rotate, zoom, **cut a section with your palm**, probe stress
values and switch fields, using an ordinary webcam. Hand landmarks come from MediaPipe. Everything after that
(features, trained pose classifier, few-shot personalisation, temporal smoothing, intent state machine, One Euro
filtering, quality checks) is in this repo and is unit-tested.

Read [`docs/DESIGN.md`](docs/DESIGN.md) for the interaction design and the CV reasoning behind it.

## Quick start in a GitHub Codespace (no webcam needed on the server)

1. Open the repo in a Codespace. On first start, `.devcontainer/setup.sh` installs everything.
   For an existing Codespace, run `bash .devcontainer/setup.sh` once.
2. `make demo`
3. Open the forwarded **port 8000** in your browser and click **Enable camera**, or **Watch demo** to see a
   scripted tour with no camera.

This works in a Codespace because the browser opens *your* webcam and runs MediaPipe locally. Only ~1 KB of
landmark coordinates per frame goes to the Python engine in the Codespace. Video never leaves your machine.

First time with the camera? Press **Calibrate (C)**. In 30 seconds it:
- detects your camera's left/right convention,
- sets an interaction box that fits your comfortable reach,
- records ~30 frames of each pose so recognition adapts to your hand (`data/personal_samples.npz`).

## Build, load and solve a member by gesture

The browser demo is now a small gesture FE workbench, with three phases shown as tabs:

1. **Model**: pinch with both hands and pull apart to extrude a rectangular beam (horizontal pull) or column
   (vertical pull), or draw a circle with your index finger for a round member. Enter the dimensions by holding up
   fingers: one digit at a time, both palms = OK.
2. **Loads & supports**: hold a fist (fixed), pinch (pinned) or V (roller) still on the member. Point and flick for
   a point load, sweep a flat palm for a uniform load, hold both palms for a trapezoidal load, all in ±X/±Y/±Z.
   Thumbs-up solves.
3. **Results**: von Mises, Tresca, principal stresses, S11…S23, U1…U3, strains E11…E23, strain energy and
   six animated mode shapes. The left panel shows the legend, the mesh details (elements, nodes, DOF, cell size,
   solver timings), the model with column slenderness checks, and the reactions. A view cube at the top right sets
   views by mouse.

Demos: continuous beam, cantilever, long column and short column. The solver (`archimedes_fe`) is an
incompatible-mode brick solver checked against beam theory. **VR/AR**: `/xr` (WebXR) and `/api/scene.glb`.
Details are in [`docs/MODELING.md`](docs/MODELING.md).

**Python console** (🐍 Python, or the ` key): Abaqus-style commands for everything above, such as
`beam(b=300, h=500, L=6000); pinned(at=0); roller(at=1); uniform(10); solve(); peak('U3')`. It also runs your own
scripts (*Import script…*, `run('file.py')`, `import` from `fe_scripts/`) and journals every gesture and mouse edit as
a replayable command. See [`docs/SCRIPTING.md`](docs/SCRIPTING.md).

## Archimedes Workbench with live gestures

The Workbench (`Archimedes/`, FE pre/post-processor with a DOLFINx backend) is
served by the same demo server, with the gesture engine wired in:

```bash
make demo        # then open port 8000 at /workbench/ and press G
```

`/workbench/?tour` runs a camera-free gesture tour. The vocabulary and the
reasons for it are in [`Archimedes/docs/gestures.md`](Archimedes/docs/gestures.md).

The Workbench models and analyses any part, the way SolidWorks or CATIA do. It has:

- a feature tree: sketches with fillets and holes, extrude and revolve (boss or cut), loft, sweep, 3D edge fillets
  and chamfers, shell, box, cylinder, sphere, hole, linear and circular patterns, mirror, and imported STL bodies;
- a PropertyManager (✓ / ✕) for every feature, a sketcher with relations and driving dimensions (a constraint
  solver), and Instant3D drag arrows and a Modify box on any face;
- supports and loads on any picked faces (fixed, roller, compression-only contact, force, pressure, gravity,
  temperature);
- a body-fitted hex8 mesh (skin nodes on the real CAD surfaces) and static, nonlinear (plasticity), modal and
  buckling analyses on the Python FE core, the DOLFINx container or in the browser;
- a material library plus a custom material, templates, open / save, and STL import and export;
- standard views, pan and zoom at the cursor (SolidWorks or CATIA buttons), an optional ground grid (View ▸ Ground
  Grid, or # Grid in the viewport), and undo / redo.

See [`Archimedes/docs/cad-workflow.md`](Archimedes/docs/cad-workflow.md).
In the Workbench, thumbs-up held still for 1.2 s runs the solve instead of
taking a snapshot.

## Gestures

| Hold… | to… | then… |
|---|---|---|
| ✊ fist | rotate | drag, like grabbing the model |
| 🤏 pinch | zoom | move up / down |
| ✋ flat palm (still) | section cut | your palm *is* the cutting plane; close the hand to lock |
| ☝️ point | probe | the fingertip is the cursor; hold still to pin a value |
| ✌️ V | change field | swipe left / right |
| 👍 thumbs-up (still) | snapshot | — |
| 🙌 both palms | reset | — |

Relax your hand to stop. A relaxed or passing hand never moves the model. Mouse and keys always work too:
drag, wheel, hover, click, `1-4`, `X`, `S`, `U`, `R`.

## Desktop app (on your own computer)

```bash
pip install -r requirements.txt pyvista
python apps/desktop_demo.py                                   # camera window with skeleton + HUD
python apps/desktop_demo.py --pyvista result.vtu --field VonMises --field U
```

`archimedes_gestures/pyvista_bridge.py` is the hook for the existing Archimedes PyVista/Qt viewer. Call
`bridge.apply(pipeline.view)` on the main thread. Section cuts use GPU clipping planes.

## Project workflow (maps to the capstone phases)

| Phase | Do | Command |
|---|---|---|
| 1 Landmarks + logging | record labelled 3-s clips per pose, one subject ID per person | web demo → *Collect training data* |
| 2 Static classifier | extract HaGRID landmarks, train, evaluate on unseen people | `python scripts/extract_hagrid.py …` then `make train` → `reports/static_pose.md` |
| 3 Dynamic gestures | swipe detector + DTW few-shot baseline; add a GRU/1D-CNN on Jester/IPN | `archimedes_gestures/dynamic.py` |
| 4 Continuous control | tune One Euro: jitter vs lag on recorded `still_palm` / `moving_palm` clips | `python scripts/evaluate_filter.py` |
| 5 Robustness | dim light, clutter, sleeves, second person; read the HUD hints | web demo |
| 6 User study | 4 tasks, gestures vs mouse, 6-10 classmates | web demo (both inputs built in) |

`make train-smoke` and `make filter-smoke` run the same scripts on synthetic hands, so you can check that the
pipeline works before you have data. Don't report those numbers.

## Layout

```
archimedes_gestures/   landmarks, features, classifier, intent engine, modelling engine, finger-count numbers,
                       filters, quality, calibration, view_state, recorder, tracker (MediaPipe), pyvista_bridge,
                       synthetic hand model
archimedes_fe/         member model, brick FE solver (static + modal), general hex8 mesh solver (meshsolver:
                       static, plasticity, contact, modal, buckling), demos, model session, GLB export,
                       scripting (Python console commands, journal)
fe_scripts/            example console scripts and a helper module
Archimedes/            Workbench page, cadkernel.js (feature-based CAD kernel), DOLFINx backend, docs
webdemo/               FastAPI + WebSocket server, workbench page (MediaPipe WASM + three.js), /xr WebXR viewer
apps/desktop_demo.py   OpenCV/MediaPipe desktop app, optional PyVista window
scripts/               train_static.py, extract_hagrid.py, evaluate_filter.py
tests/                 95 tests (+ 18 CAD-kernel tests in Node), no camera or GPU needed:  make test
```
