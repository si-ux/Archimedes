# Gesture vocabulary

Ten gestures. Each one is a **real command** — `act()` in the `GESTURES` table
(top of the workbench script) calls the same code path the mouse and menus use,
so a recognised gesture and a click are indistinguishable downstream.

The panel on the left of the workbench lists all ten live. The `▶` button on
each row fires that gesture by hand, which is how you verify what it does
without a camera attached.

## Continuous — hold and modulate

These four track a value while the pose is held. The scripted driver (toggled
with the **Gesture** button, `Ctrl+G`, or `G`) cycles through exactly these
four, running their real actions — that is what you see moving on screen.

| | Gesture | Pose | Action |
| --- | --- | --- | --- |
| ⟳ | Orbit | open palm, drag | Orbits the camera — yaw follows the hand |
| ⇔ | Dolly zoom | thumb + index, vary the gap | Moves the camera in and out |
| ▬ | Section plane | flat blade, sweep across | Drags the section-cut plane along its axis |
| ↑ | Probe | index point, hold 400 ms | Reads out the peak value of the active field |

## Discrete — fire once on recognition

These are one-shot commands. The scripted driver deliberately does **not**
cycle through them (advancing stages or kicking off solves on a timer would be
hostile); they fire on recognition, or from the `▶` button.

| | Gesture | Pose | Action |
| --- | --- | --- | --- |
| ● | Confirm | closed fist, hold 400 ms | Advance to the next pipeline stage |
| ◀ | Back | quick palm sweep left | Previous stage |
| ▶ | Forward | quick palm sweep right | Next stage |
| ▲ | Solve | thumb up, fingers curled | Runs the solve (whichever engine is selected) |
| ⇿ | Warp scale | two palms, vary separation | Scales the displacement warp |
| ▼ | Section toggle | palm down, press | Toggles the section cut on/off |

## What is and isn't wired

The **actions are real**. The **recognition is not** — there is no camera
input. `tickGesture()` is a scripted driver that stands in for a hand tracker,
so the action layer can be built and tested independently of it.

To attach real tracking, replace the driver with a landmark source and emit
gesture ids into the same entry point:

```js
// the only integration point — everything downstream already works
this.fireGesture('pinch')          // discrete
G.act(this, 'hold')                // continuous, once per frame while held
```

A workable source is MediaPipe Hands (21 landmarks per hand, ~30 fps on CPU).
Classification from landmarks is straightforward geometry:

- **finger extended** — tip further from the wrist than the PIP joint
- **pinch** — thumb-tip to index-tip distance, normalised by palm width
- **fist** — no fingers extended
- **blade** — all fingers extended and adjacent, palm normal roughly horizontal
- **swipe** — palm centroid velocity over a 150 ms window past a threshold
- **two-hand** — both hands present, track the inter-wrist distance

Two things matter more than the classifier: **debounce** discrete gestures
(a 400 ms dwell before firing, then a refractory period, or the solve fires
repeatedly), and **smooth** continuous ones (the camera jitters; the existing
`tgt`/`cam` interpolation already damps it).
