# Design: a gesture layer people can actually use

This is the computer-vision side of Archimedes. Hands control how **results** are viewed and, in the browser
workbench, also build simple members: geometry, supports and loads (see [`MODELING.md`](MODELING.md)). Real
projects still bring geometry from CAD/BIM and solve in FEniCSx. The gesture modeller covers single prismatic
members, for teaching and quick checks. This document explains how the earlier plan was changed
to make it easier to use, and why each change makes sense from a computer-vision point of view.

## 1. What changed from the first plan

| First plan | Now | Why (CV / usability) |
|---|---|---|
| Two-hand split: the left hand picks the mode, the right hand controls | **One hand points, two hands manipulate.** One hand does the precise work (probe, section, field); the big moves (zoom, pan, turn, reset) need the *same* pose on both hands, as in VR | Showing a pose with both hands is a far stronger signal than one hand's pose, so the big moves are never set off by accident, and a lone pinch can mean nothing (no pinch/point confusion). Precise work stays one-handed, so arms do not tire |
| Separate clutch hand | **Relaxing the hand is the clutch.** A mode starts only after its pose has been held briefly | Removes accidental input ("Midas touch") without asking for a second hand. Repositioning works like lifting a mouse |
| Modes chosen by finger count (two vs three fingers) | **Poses that are easy to tell apart** for modes: fist, flat palm, point, V, thumbs-up, pinch. Finger counts only answer menus, held still, and must change between answers | Counting fingers fails under self-occlusion and side views. The chosen poses differ in finger *shape*, not just count. Counting is read with a thumb-span feature (thumb tip to little-finger knuckle over palm width), which separates a tucked thumb from a raised one far better than the thumb's joint angles |
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
both thumbs up held 0.8 s ──► reset view          hand at edge / too far ──► paused (red), view frozen
```

| Pose | Mode | Control | Arm time |
|---|---|---|---|
| ✊ fist | Rotate | drag the hand, as if grabbing the model | 0.2 s |
| ☝️ point | Probe | a ring cursor snapped onto the nearest visible face (it never leaves the model). **Pinch with the other hand to pin**, or hold still 0.8 s | 0.2 s |
| ✋ one flat palm, **held still** | Section cut | always a **straight** cut along the world axis the palm faces (chosen from the first 6 frames, then locked); sliding the palm moves it in 2 % steps with hysteresis | 0.6 s |
| ✌️ V | Field | a dial: every 0.7 palm widths right / left is one field; a strip shows the list and where you are | 0.25 s |
| 👍 thumbs-up, held still | Snapshot (Workbench: solve) | | 1.0 s (1.2 s) |
| 🤏🤏 pinch with **both** hands | Zoom / pan | the distance between the hands *is* the zoom, exactly (spread = in), counted from the moment both pinches appear; moving both pans | 0.15 s |
| ✊✊ two fists | Turn | move them to rotate; twist them like a steering wheel to spin | 0.15 s |
| ☝️☝️ point with both hands | Compare | two cursors and the difference B − A | 0.2 s |
| 👍👍 both thumbs up, held still | Reset | (undo is always available) | 0.8 s |

Rules that keep modes apart:

- a two-hand mode needs the *same* pose on both hands; a still one-hand mode (section, snapshot) is refused while the
  other hand shows anything but a relaxed hand or a fist, so two raised hands are never read as one of them;
- an active one-hand mode is sticky: a second hand appearing does not cancel it, unless it makes the two-hand
  version of the same pose (☝️ → ☝️☝️, ✊ → ✊✊);
- a still mode that is arming drains when the hand moves instead of starting over, so a shaky frame does not reset
  the hold but a passing hand never arms it;
- one-shot modes (snapshot, reset, solve) fire once and wait for the pose to go away.

Building the model uses wizards with one confirmed step at a time (see [`MODELING.md`](MODELING.md)).

Typing numbers by hand: hold up fingers for a digit (0.55 s, still hand); a question with fixed choices (the
increments) is answered by the count alone; 🤏 pinch and hold deletes the last digit and repeats while held; pinch and
move up / down nudges by the increment; both open palms accept; two fists cancel. The keyboard (digits, Backspace,
Enter) always works alongside.

Every gesture has a keyboard and mouse equivalent, which doubles as the baseline condition in the user study.
Confirmations also play a short tick, so you hear that a gesture worked without looking away from the model.

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
