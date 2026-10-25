# ruff: noqa: E702, E731
"""Latency and accuracy of the gesture controls, measured on synthetic hands.

    python scripts/gesture_bench.py

Every scenario drives the real GesturePipeline (rule classifier, pose smoother, intent engine) with a scripted
hand at 30 fps and reports what the user would feel:

* zoom: how long until the view starts to move, and how closely the zoom follows the hand
* probe: lag of the cursor behind the fingertip, and its jitter when the hand is still
* digits: time to read a finger count, and whether a backspace swipe is recognised
* section: how much the cut plane wobbles while the palm is held "still"

The synthetic hand is NOT a camera: use the numbers to compare versions of the code, not as accuracy claims.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from archimedes_gestures.landmarks import FrameInput, HandFrame  # noqa: E402
from archimedes_gestures.pipeline import GesturePipeline  # noqa: E402
from archimedes_gestures.synthetic import COUNT_FINGERS, canonical_fingers, canonical_hand, to_image  # noqa: E402

FPS = 30


def make_pipeline(phase: str = "results", profile: str = "viewer") -> GesturePipeline:
    p = GesturePipeline(model_path=None, profile_path="/nonexistent/p.json", personal_path="/nonexistent/s.npz",
                        clips_root="/nonexistent/clips", profile=profile)
    p.set_phase(phase)
    return p


def hand(pose=None, fingers=None, center=(0.5, 0.5), size=0.12, handedness="Right", t=0.0, rng=None, noise=0.003,
         yaw=0.0, pitch=0.0, roll=0.0) -> HandFrame:
    rng = rng or np.random.default_rng(0)
    canon = canonical_fingers(fingers, rng) if fingers is not None else canonical_hand(pose, rng)
    lm = to_image(canon, center=center, palm_size=size, yaw=yaw, pitch=pitch, roll=roll, handedness=handedness,
                  noise=noise, rng=rng)
    return HandFrame(lm, handedness, 0.97, t)


def run(pipe: GesturePipeline, frames):
    """frames: iterable of (t, [HandFrame...]). Yields (t, result) for each processed frame."""
    for t, hands in frames:
        yield t, pipe.process(FrameInput(t, hands))


def first_time(results, pred, t0=0.0):
    for t, r in results:
        if t >= t0 and pred(r):
            return t - t0
    return None


# ---------------------------------------------------------------- scenarios
def bench_zoom(seed=0) -> dict:
    """Pinch with one hand and move it up by 0.6 palm widths in 0.6 s. Also: two pinches, spread by 40 %."""
    rng = np.random.default_rng(seed)
    out = {}
    pipe = make_pipeline()
    frames, t = [], 0.0
    for _ in range(int(0.5 * FPS)):                      # relaxed hand first
        frames.append((t, [hand("none", center=(0.5, 0.6), t=t, rng=rng)])); t += 1 / FPS
    t_start = t
    n = int(0.6 * FPS)
    for i in range(n + int(0.8 * FPS)):
        a = min(i / n, 1.0)
        frames.append((t, [hand("pinch", center=(0.5, 0.6 - 0.6 * 0.12 * a), t=t, rng=rng)])); t += 1 / FPS
    res = list(run(pipe, frames))
    zoom = [(tt, c) for tt, r in res for c in r["commands"] if c["kind"] == "zoom"]
    out["one_hand_first_response_ms"] = None if not zoom else round(1000 * (zoom[0][0] - t_start))
    out["one_hand_total_zoom"] = round(float(np.prod([c["data"]["factor"] for _, c in zoom])) if zoom else 1.0, 3)

    pipe = make_pipeline()
    frames, t = [], 0.0
    for _ in range(int(0.5 * FPS)):
        frames.append((t, [hand("none", center=(0.35, 0.55), handedness="Left", t=t, rng=rng),
                           hand("none", center=(0.65, 0.55), t=t, rng=rng)])); t += 1 / FPS
    t_start = t
    n = int(0.6 * FPS)
    for i in range(n + int(0.8 * FPS)):
        a = min(i / n, 1.0)
        dx = 0.15 + 0.06 * a                              # half-distance 0.15 -> 0.21: spread by 40 %
        frames.append((t, [hand("pinch", center=(0.5 - dx, 0.55), handedness="Left", t=t, rng=rng),
                           hand("pinch", center=(0.5 + dx, 0.55), t=t, rng=rng)])); t += 1 / FPS
    res = list(run(pipe, frames))
    zoom = [(tt, c) for tt, r in res for c in r["commands"] if c["kind"] == "zoom"]
    out["two_hand_first_response_ms"] = None if not zoom else round(1000 * (zoom[0][0] - t_start))
    out["two_hand_total_zoom"] = round(float(np.prod([c["data"]["factor"] for _, c in zoom])) if zoom else 1.0, 3)
    out["two_hand_expected_zoom"] = round(1 / 1.4, 3)
    return out


def bench_probe(seed=0) -> dict:
    """Point and sweep the fingertip across the interaction box; compare the cursor with where it should be."""
    rng = np.random.default_rng(seed)
    pipe = make_pipeline()
    box = pipe.engine.cfg.box
    frames, truth, t = [], [], 0.0
    for _ in range(int(0.4 * FPS)):
        frames.append((t, [hand("none", center=(0.4, 0.5), t=t, rng=rng)])); truth.append(None); t += 1 / FPS
    n = int(1.0 * FPS)
    x0, x1 = 0.35, 0.65
    for i in range(n):
        x = x0 + (x1 - x0) * i / (n - 1)
        h = hand("point", center=(x, 0.5), t=t, rng=rng)
        frames.append((t, [h])); truth.append((h.landmarks[8, 0] - box[0]) / (box[2] - box[0])); t += 1 / FPS
    for _ in range(int(1.0 * FPS)):                       # then hold still
        h = hand("point", center=(x1, 0.5), t=t, rng=rng)
        frames.append((t, [h])); truth.append((h.landmarks[8, 0] - box[0]) / (box[2] - box[0])); t += 1 / FPS
    res = list(run(pipe, frames))
    cur = []
    for (tt, r), tr in zip(res, truth):
        pc = r["view"]["probe_cursor"]
        cur.append(None if pc is None or tr is None else (tt, pc[0], float(np.clip(tr, 0, 1))))
    first = next((c for c in cur if c is not None), None)
    sweep = [c for c in cur[int(0.4 * FPS):int(1.4 * FPS)] if c is not None]
    err = [abs(c[1] - c[2]) for c in sweep]
    speed = (1.0 / 1.0) * 1.0 * (x1 - x0) / (box[2] - box[0])          # cursor units per second
    still = [c[1] for c in cur[-int(0.7 * FPS):] if c is not None]
    pins = [c for _, r in res for c in r["commands"] if c["kind"] == "probe_pin"]
    return {"first_response_ms": None if first is None else round(1000 * (first[0] - 0.4)),
            "mean_error_pct": round(100 * float(np.mean(err)), 1) if err else None,
            "lag_ms": round(1000 * float(np.mean(err)) / speed) if err else None,
            "still_jitter_pct": round(100 * float(np.std(still)), 2) if still else None,
            "auto_pins": len(pins)}


def bench_digits(seed=0) -> dict:
    """Prompt open; hold up 1..5 fingers (one hand) and see which digit arrives and how soon."""
    out = {}
    for n in (1, 3, 5):
        rng = np.random.default_rng(seed + n)
        pipe = make_pipeline("model")
        pipe.set_prompt({"action": "x", "key": "k", "step": 1, "label": "Load", "unit": "kN", "default": None,
                         "choices": None, "increment": None})
        t, res = 0.0, []
        for _ in range(int(2.0 * FPS)):
            res.append((t, pipe.process(FrameInput(t, [hand(fingers=COUNT_FINGERS[n], center=(0.5, 0.55), t=t, rng=rng)]))))
            t += 1 / FPS
        digs = [(tt, c["data"]["digit"]) for tt, r in res for c in r["commands"] if c["kind"] == "number_digit"]
        out[f"hold_{n}"] = (digs[0][1], round(1000 * digs[0][0])) if digs else None
    return out


def bench_backspace(seed=0) -> dict:
    """Type '30', then swipe an open hand to the left at a normal speed. Count backspaces."""
    out = {}
    for label, width, dur in (("quick flick", 0.30, 0.25), ("relaxed swipe", 0.22, 0.45), ("small wave", 0.12, 0.3)):
        rng = np.random.default_rng(seed)
        pipe = make_pipeline("model")
        pipe.set_prompt({"action": "x", "key": "k", "step": 1, "label": "Load", "unit": "kN", "default": None,
                         "choices": None, "increment": None})
        pipe.numbers.digits = "30"
        pipe.numbers._armed = False
        t, res = 0.0, []
        for _ in range(int(0.6 * FPS)):
            res.append((t, pipe.process(FrameInput(t, [hand(fingers="timrp", center=(0.62, 0.55), t=t, rng=rng)]))))
            t += 1 / FPS
        n = max(int(dur * FPS), 2)
        for i in range(n):
            res.append((t, pipe.process(FrameInput(t, [hand(fingers="timrp", center=(0.62 - width * i / (n - 1), 0.55), t=t, rng=rng)]))))
            t += 1 / FPS
        for _ in range(int(0.6 * FPS)):
            res.append((t, pipe.process(FrameInput(t, [hand(fingers="timrp", center=(0.62 - width, 0.55), t=t, rng=rng)]))))
            t += 1 / FPS
        out[label] = sum(1 for _, r in res for c in r["commands"] if c["kind"] == "number_backspace")
    return out


def bench_pinch_delete(seed=0) -> dict:
    """Type '305', then pinch and hold still for 1.2 s: one delete after ~0.45 s, then repeats."""
    rng = np.random.default_rng(seed)
    pipe = make_pipeline("model")
    pipe.set_prompt({"action": "x", "key": "k", "step": 1, "label": "Load", "unit": "kN", "default": None,
                     "choices": None, "increment": None})
    pipe.numbers.digits = "305"
    t, res = 0.0, []
    for _ in range(int(0.4 * FPS)):
        res.append((t, pipe.process(FrameInput(t, [hand(fingers="timrp", center=(0.5, 0.55), t=t, rng=rng)])))); t += 1 / FPS
    t_pinch = t
    for _ in range(int(1.2 * FPS)):
        res.append((t, pipe.process(FrameInput(t, [hand("pinch", center=(0.5, 0.55), t=t, rng=rng)])))); t += 1 / FPS
    dels = [tt - t_pinch for tt, r in res for c in r["commands"] if c["kind"] == "number_backspace"]
    return {"deletes": len(dels), "first_delete_ms": round(1000 * dels[0]) if dels else None, "left": pipe.numbers.digits}


def bench_section(seed=0) -> dict:
    """Hold a flat palm 'still' with a little hand tremor (yaw/pitch/roll wobble). How much does the plane move?"""
    rng = np.random.default_rng(seed)
    pipe = make_pipeline()
    t, res = 0.0, []
    for _ in range(int(0.5 * FPS)):
        res.append((t, pipe.process(FrameInput(t, [hand("none", center=(0.5, 0.55), t=t, rng=rng)])))); t += 1 / FPS
    for i in range(int(3.0 * FPS)):
        wob = lambda a: a + 0.10 * np.sin(2 * np.pi * rng.uniform(1.0, 3.0) * t) + rng.normal(0, 0.05)  # noqa: E731
        res.append((t, pipe.process(FrameInput(t, [hand("open_palm", center=(0.5 + rng.normal(0, 0.002), 0.55), t=t,
                                                          rng=rng, yaw=wob(0.15), pitch=wob(-0.1), roll=wob(0.0))]))))
        t += 1 / FPS
    secs = [r["view"] for _, r in res if r["view"]["section_on"]]
    if not secs:
        return {"section_frames": 0}
    S = np.array([v["plane_normal"] for v in secs[len(secs) // 3:]])
    mean = S.mean(axis=0); mean /= np.linalg.norm(mean)
    ang = np.degrees(np.arccos(np.clip(S @ mean, -1, 1)))
    axes = np.abs(S).argmax(axis=1)
    off_axis = np.degrees(np.arccos(np.clip(np.abs(S).max(axis=1), 0, 1)))
    origins = np.array([v["plane_origin"] for v in secs[len(secs) // 3:]])
    along = np.einsum("ij,ij->i", origins, S)
    return {"section_frames": len(secs), "plane_wobble_deg_p95": round(float(np.percentile(ang, 95)), 2),
            "axis_flips": int(np.sum(np.diff(axes) != 0)), "off_axis_deg_p95": round(float(np.percentile(off_axis, 95)), 2),
            "offset_wobble": round(float(np.std(along)), 4)}


def main() -> None:
    for name, fn in (("zoom", bench_zoom), ("probe", bench_probe), ("digits", bench_digits),
                     ("backspace (backspaces fired)", bench_backspace), ("pinch delete", bench_pinch_delete),
                     ("section", bench_section)):
        print(f"{name:30s}", fn())


if __name__ == "__main__":
    main()
