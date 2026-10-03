# Gesture vocabulary

Gestures in the workbench are for **looking at results**. Pipeline stages
(geometry → mesh) stay on menus and keys. Each gesture is a real command: the
engine's output goes through `applyGestureCommand()`, which makes the same state
changes the mouse, menus and `▶` buttons make.

| | Gesture | Hold… | then… |
| --- | --- | --- | --- |
| ⟳ | Orbit | ✊ fist, 0.25 s | drag, like grabbing the model |
| ⇔ | Dolly zoom | 🤏 pinch, 0.25 s | move up to zoom in, down to zoom out |
| ▬ | Section cut | ✋ flat palm, **still**, 0.8 s | **your palm is the cutting plane**, at any angle; turn it to orient the cut, move along it to slide; close the hand to lock |
| ↑ | Probe | ☝️ index point, 0.3 s | the fingertip is the cursor; hold still 0.8 s to pin a node |
| ◐ | Field | ✌️ V sign, 0.3 s | swipe left / right to change the result field |
| ▲ | Solve | 👍 thumbs-up, **still**, 1.2 s | runs the solve once |
| ⌂ | Reset | 🙌 both open palms, 1 s | resets the camera and removes the cut |

Relax your hand and the mode stops. A relaxed or passing hand never moves the
model.

## Why this vocabulary (and not the first draft's)

| First draft | Now | Why |
| --- | --- | --- |
| open palm + drag = orbit | fist + drag | An open palm is the resting hand: every passing hand would orbit the model |
| fist hold = next stage, swipes = prev/next stage | menus / keys | Stages are pre-processing; gestures are for results |
| thumbs-up = solve, on recognition | thumbs-up held **still** 1.2 s, with a ring | A solve costs seconds to minutes; it must never fire by accident |
| blade sweep along a fixed axis | palm plane at any angle | The one thing a mouse can't do in one motion (`cutAxis: 'free'`) |
| two palms, vary separation = warp scale | warp slider | Two raised hands tire quickly, and MediaPipe loses overlapping hands |

## How it runs

```
browser: webcam → MediaPipe Hand Landmarker (WASM) → 21×3 landmarks ─┐
                                                       WebSocket /ws?profile=workbench
python:  archimedes_gestures.GesturePipeline (classifier, smoothing,  ◄┘
         hold-to-arm state machine, One Euro filters, quality hints)
         → commands {orbit, zoom, section, probe, field, solve, reset}
browser: applyGestureCommand() → same state changes as mouse / menus
```

From the repo root:

```bash
make demo
# then open  http://localhost:8000/workbench/   (in a Codespace: the forwarded port 8000, + /workbench/)
# press G (or the Gesture button) and allow the camera
```

- `/workbench/?tour` plays a scripted hand tour through the engine, with no
  camera needed. Use it for presentations, or to check the wiring.
- With no camera or no engine, the `G` toggle falls back to the old scripted
  driver, and the log says why.
- The bottom-left box shows what the camera sees: hand skeleton, current mode,
  and the hold ring.
- The `▶` buttons still test-fire each gesture by hand.

The solver bridge (`ws://127.0.0.1:8791`, DOLFINx in Docker) is independent of
the gesture engine. Without it, solves run on the in-browser hex8 core as before.
