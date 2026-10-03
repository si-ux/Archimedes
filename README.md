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

## Archimedes Workbench with live gestures

The Workbench (`Archimedes/`, FE pre/post-processor with a DOLFINx backend) is
served by the same demo server, with the gesture engine wired in:

```bash
make demo        # then open port 8000 at /workbench/ and press G
```

`/workbench/?tour` runs a camera-free gesture tour. The vocabulary and the
reasons for it are in [`Archimedes/docs/gestures.md`](Archimedes/docs/gestures.md).
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
archimedes_gestures/   landmarks, features, classifier, intent engine, filters, quality, calibration,
                       view_state, recorder, tracker (MediaPipe), pyvista_bridge, synthetic hand model
webdemo/               FastAPI + WebSocket server, static page (MediaPipe WASM + three.js)
apps/desktop_demo.py   OpenCV/MediaPipe desktop app, optional PyVista window
scripts/               train_static.py, extract_hagrid.py, evaluate_filter.py
tests/                 26 tests, no camera or GPU needed:  make test
```
