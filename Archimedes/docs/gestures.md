# Gesture vocabulary

Gestures in the workbench are for **looking at results**. Pipeline stages
(geometry → mesh) stay on menus and keys. Each gesture is a real command: the
engine's output goes through `applyGestureCommand()`, which makes the same state
changes the mouse, menus and `▶` buttons make.

One hand does the precise work; both hands showing the same pose do the big moves,
as in VR.

| | Gesture | Hold… | then… |
| --- | --- | --- | --- |
| ⟳ | Orbit | ✊ fist, 0.2 s | drag, like grabbing the model |
| ↑ | Probe | ☝️ index point, 0.2 s | a **ring cursor snapped onto the model** (a dashed line shows where you point when it is off the part). **Pinch with your other hand to pin the value** (or hold still 0.8 s) |
| ▬ | Section cut | ✋ one flat palm, **still**, 0.6 s | a **straight** cut (X, Y or Z plane) across the axis the palm faces; the axis is picked in the first 6 frames and locked. Slide the palm to move the cut in steps. The cut is evaluated on the fitted node positions, so it is flat on body-fitted meshes too |
| ◐ | Field | ✌️ V sign, 0.25 s | move right / left: one field per 0.7 palm widths; a strip shows the list |
| ▲ | Solve | 👍 thumbs-up, **still**, 1.2 s | runs the solve once |
| ⇱ | **Zoom / pan** | 🤏🤏 pinch with both hands, 0.15 s | the zoom follows the distance between the hands exactly; move both to pan |
| ◎ | **Turn** | ✊✊ two fists, 0.15 s | move them to rotate, twist them like a steering wheel to spin |
| ⇄ | **Compare** | ☝️☝️ point with both hands | cursors A and B and the difference B − A |
| ⌂ | Reset | 👍👍 both thumbs up, still, 0.8 s | resets the camera and removes the cut |

Relax your hand and the mode stops. A relaxed or passing hand never moves the
model.

## Why this vocabulary (and not the first draft's)

| First draft | Now | Why |
| --- | --- | --- |
| open palm + drag = orbit | fist + drag | An open palm is the resting hand: every passing hand would orbit the model |
| fist hold = next stage, swipes = prev/next stage | menus / keys | Stages are pre-processing; gestures are for results |
| thumbs-up = solve, on recognition | thumbs-up held **still** 1.2 s, with a ring | A solve costs seconds to minutes; it must never fire by accident |
| blade sweep along a fixed axis | straight cut along an axis, always | A hand always wobbles, so a free plane looked staggered |
| two palms, vary separation = warp scale | two *pinches*, vary separation = zoom | The distance between hands is the most direct zoom there is; pinching keeps the hands apart and tells the camera where each one is |
| one-hand pinch = zoom | a lone pinch does nothing | Pinch and point look alike to a camera mid-transition; with no one-hand pinch mode they can never be confused |
| both open palms = reset | both thumbs up | Two open palms are also how people start a section cut with the other hand raised |

## How it runs

```
browser: webcam → MediaPipe Hand Landmarker (WASM) → 21×3 landmarks ─┐
                                                       WebSocket /ws?profile=workbench
python:  archimedes_gestures.GesturePipeline (classifier, smoothing,  ◄┘
         hold-to-arm state machine, One Euro filters, quality hints)
         → commands {orbit, zoom, pan, section, probe, probe2, field, solve, reset}
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
