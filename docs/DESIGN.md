# Design: a gesture layer people can actually use

This is the computer-vision side of Archimedes. Hands control how **results** are viewed and, in the browser
workbench, also build simple members: geometry, supports and loads (see [`MODELING.md`](MODELING.md)). Real
projects still bring geometry from CAD/BIM and solve in FEniCSx. The gesture modeller covers single prismatic
members, for teaching and quick checks. This document explains how the earlier plan was changed
to make it easier to use, and why each change makes sense from a computer-vision point of view.

## 1. What changed from the first plan

| First plan | Now | Why (CV / usability) |
|---|---|---|
| Two-hand split: the left hand picks the mode, the right hand controls | **One hand by default.** The pose picks the mode and the motion is the control | Two raised arms tire people out within minutes. MediaPipe also tracks worse when hands overlap or hide each other. With one hand, users learn 6 shapes instead of a 2×N grid |
| Separate clutch hand | **Relaxing the hand is the clutch.** A mode starts only after its pose has been held briefly | Removes accidental input ("Midas touch") without asking for a second hand. Repositioning works like lifting a mouse |
| Modes chosen by finger count (two vs three fingers) | **Poses that are easy to tell apart**: fist, flat palm, point, V, thumbs-up, pinch | Counting fingers fails under self-occlusion and side views. The chosen poses differ in finger *shape*, not just count |
| Orbit by twisting the wrist | **Orbit by dragging a fist** ("grab the model") | Monocular depth (z) from MediaPipe is noisy, so wrist roll and pitch are unreliable. 2D translation of the palm is the steadiest signal a webcam gives |
| Raw landmarks drive motion | Motion follows the **rigid palm** (wrist + 4 knuckles) | Fingertips move whenever the pose changes. The knuckles hardly move, so opening the hand to release no longer kicks the view |
| Per-frame classification | **EMA plus enter/exit hysteresis**, and N frames to switch | A single misclassified frame can never change mode |
| Lock the section on the release frame | Lock the plane from **~150 ms before** release | Frames during a pose change are the noisiest, so they never end up in the result |
| Errors are silent | **Plain-language quality hints**: too dark, blurry, hand at edge, too far or near, too fast | Most "it doesn't work" cases are input problems, and naming the problem fixes it in one step |
| One model for everyone | **30-second calibration plus a few-shot personal adapter (k-NN)** | Hands, cameras and rooms differ. About 30 frames per pose make the model fit *this* user, with no retraining |
| Left/right handedness taken as given | **Detected during calibration** ("raise your right hand") | MediaPipe's left/right label depends on whether the frame was mirrored, which differs between camera paths |
| Desktop app only | **Browser demo** (webcam and MediaPipe in WASM, Python brain over WebSocket) **plus** the desktop app | Works inside a GitHub Codespace, which has no webcam, and on any classmate's laptop for the user study |

## 2. The interaction model

```
relaxed hand ──► nothing happens (skeleton shown in grey)
hold a pose  ──► ring fills (amber) ──► mode active (green) ──► relax to stop
two open palms held 1 s ──► reset view            hand at edge / too far ──► paused (red), view frozen
```

| Pose | Mode | Control | Arm time |
|---|---|---|---|
| ✊ fist | Rotate | drag the hand, as if grabbing the model | 0.25 s |
| 🤏 pinch | Zoom | move up to zoom in, down to zoom out | 0.25 s |
| ✋ flat palm, **held still** | Section cut | **the palm plane is the cutting plane**. Turn it to orient the cut; move along the normal (or push/pull) to slide it. Close the hand to lock | 0.8 s |
| ☝️ point | Probe | the fingertip is the cursor. Hold still for 0.8 s to pin a value | 0.3 s |
| ✌️ V | Field | swipe left/right to change the result field | 0.3 s |
| 👍 thumbs-up, held still | Snapshot | saves a PNG | 1.0 s |
| 🙌 both palms | Reset | (undo is always available) | 1.0 s |

The two actions that could be triggered by a passing hand (section, snapshot) also require the hand to be
*still* while arming. Every gesture has a keyboard and mouse equivalent, which doubles as the baseline condition
in the user study.

## 3. Pipeline

```
browser webcam ─► MediaPipe Hand Landmarker (WASM, in the browser) ─► 21×3 landmarks + brightness/blur ─┐
desktop webcam ─► MediaPipe (Python, HandTracker) ──────────────────────────────────────────────────────┤
                                                                                                         ▼
  HandFrame: selfie-view coords, real handedness (calibrated swap)
  ├─ features.py   wrist-centred, roll-normalised, palm-scaled, left hands mirrored → 81-D vector
  ├─ classifier.py rules baseline │ trained sklearn model  ─► PersonalAdapter (k-NN few-shot) ─► PoseSmoother
  ├─ quality.py    frame + hand checks → hints (severity 2 pauses input)
  └─ intent.py     NO_HAND → IDLE → ARMING → ACTIVE → (release grace) → IDLE, with PAUSED
                   One Euro filters on palm centre, palm normal and fingertip; swipe detector
                        ▼
                   Commands ─► ViewState (single source of truth) ─► three.js │ PyVista (pyvista_bridge.py)
```

Rendering never happens in the gesture code. `ViewState` is plain data, so every interaction is unit-tested
without a GPU or a camera (`tests/`).

## 4. Measurements for the report

| Question | Script / source | Metric |
|---|---|---|
| How well are poses recognised for **people the model never saw**? | `scripts/train_static.py` (GroupKFold by subject) | macro-F1, per-class F1, confusion matrix |
| Is learning worth it over rules? | same report, `rules` row | macro-F1 difference |
| Does personalisation help? | compare F1 on a held-out user before and after 30 samples/pose | ΔF1 |
| Is the cut plane steady? | `scripts/evaluate_filter.py` | jitter (deg, still hand) vs lag (ms, moving hand) per One Euro setting |
| Is it real-time? | HUD `ms engine` and `fps` chips | engine latency per frame, detection FPS |
| False activations | record 2 min of "none" (talking, typing, scratching your face), replay | modes armed per minute (target ≈ 0) |
| Is it usable? | 4-task study, gestures vs mouse | task time, errors, NASA-TLX |

Datasets: **HaGRID** for static poses (`scripts/extract_hagrid.py` maps palm/stop→open_palm, fist, one→point,
peace, like→thumbs_up, ok→pinch, no_gesture→none, and keeps `user_id` for subject splits), plus your own clips
recorded in the demo. Training adds an x-mirrored copy of every sample, because none of the poses depend on
which hand is used and datasets disagree on mirroring.

## 5. Honest limitations

- The unit tests and the "Watch demo" tour run on a **synthetic hand model**. They prove the logic, not the
  accuracy. Report numbers only from real recordings or HaGRID.
- The rules classifier thresholds were set on synthetic hands. Expect the trained model and the personal
  adapter to matter on real webcams. That gap is itself a good result to report.
- Dynamic gestures use a velocity-based swipe detector plus a DTW template matcher (few-shot, user-recordable).
  A learned temporal model (1D-CNN/GRU on Jester or IPN Hand landmark sequences) is the next step and should be
  compared against the DTW baseline.
- The beam in the browser demo uses closed-form cantilever results as a stand-in for a FEniCSx result file.
  The desktop path (`pyvista_bridge.py`) works on any PyVista mesh with point or cell arrays.
