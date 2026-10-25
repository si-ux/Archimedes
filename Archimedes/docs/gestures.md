# Gesture vocabulary

Gestures in the workbench are for **looking at results**. Pipeline stages
(geometry → mesh) stay on menus and keys. Each gesture is a real command: the
engine's output goes through `applyGestureCommand()`, which makes the same state
changes the mouse, menus and `▶` buttons make.

| | Gesture | Hold… | then… |
| --- | --- | --- | --- |
| ⟳ | Orbit | ✊ fist, 0.12 s | drag, like grabbing the model |
| ⇔ | Zoom (one hand) | 🤏 pinch, 0.12 s | move up to zoom in, down to zoom out |
| ⇱ | **Two-hand zoom and rotate** | 🤏🤏 pinch with both hands, 0.1 s | **spread the hands to zoom in, bring them together to zoom out** (1:1 with the distance between them); **move them together to rotate** |
| ▬ | Section cut | ✋ flat palm, **still**, 0.5 s | a **straight** cut along the model axis the palm faces; the axis is picked in the first 6 frames and locked, so tremor never tilts it. Slide the palm to move the cut. **Show your other open palm to tilt the cut freely** with the hand |
| ↑ | Probe | ☝️ index point, 0.12 s | the fingertip is the cursor. **Pinch with your other hand to pin the value** (or hold still 0.6 s) |
| ◐ | Field | ✌️ V sign, 0.2 s | swipe left / right to change the result field |
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
| blade sweep along a fixed axis | straight cut along an axis; tilt with the other palm | A hand always wobbles, so a free plane looked staggered; the free plane is still there, but asked for |
| two palms, vary separation = warp scale | two *pinches*, vary separation = zoom | The distance between hands is the most direct zoom there is; pinching keeps the hands apart and tells the camera where each one is |

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
