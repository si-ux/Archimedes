"""Jitter vs lag of the One Euro filter on the section-plane normal.

Record a clip of a *still* open palm (label "still_palm") and one of a palm
moving steadily (label "moving_palm") in the web demo, then:

  python scripts/evaluate_filter.py --clips data/clips

For each (min_cutoff, beta) setting it reports:
  jitter  - std-dev of the plane-normal angle while the hand is still (degrees)
  lag     - mean delay of the filtered normal behind the raw one when moving (ms),
            estimated by cross-correlation
With --synthetic it runs on a generated hand so you can try it in a Codespace.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from archimedes_gestures import synthetic  # noqa: E402
from archimedes_gestures.filters import OneEuroFilter  # noqa: E402
from archimedes_gestures.geometry import palm_normal  # noqa: E402
from archimedes_gestures.landmarks import HandFrame  # noqa: E402

SETTINGS = [(None, None), (0.3, 0.0), (0.8, 0.4), (1.5, 0.4), (0.8, 2.0), (3.0, 1.0)]


def angle_series(normals: np.ndarray) -> np.ndarray:
    ref = normals.mean(0)
    ref /= np.linalg.norm(ref)
    return np.degrees(np.arccos(np.clip(normals @ ref, -1, 1)))


def run_filter(normals, t, mc, beta):
    if mc is None:
        return normals
    f = OneEuroFilter(mc, beta)
    out = np.array([f(n, ti) for n, ti in zip(normals, t)])
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def lag_ms(raw, filt, t):
    a, b = raw[:, 0] - raw[:, 0].mean(), filt[:, 0] - filt[:, 0].mean()
    dt = np.median(np.diff(t))
    lags = range(0, 15)
    corr = [np.dot(a[: len(a) - k], b[k:]) for k in lags]
    return lags[int(np.argmax(corr))] * dt * 1000


def synthetic_clips():
    rng = np.random.default_rng(0)
    canon = synthetic.canonical_hand("open_palm", rng)
    t = np.arange(0, 4, 1 / 30)
    still = [HandFrame(synthetic.to_image(canon, noise=0.02, rng=rng), "Right", 1, ti) for ti in t]
    moving = [HandFrame(synthetic.to_image(canon, yaw=0.8 * np.sin(ti * 2), noise=0.02, rng=rng), "Right", 1, ti)
              for ti in t]
    return (still, t), (moving, t)


def load(clips: Path, label: str):
    for p in sorted(clips.rglob(f"{label}/*.npz")):
        d = np.load(p)
        return [HandFrame(lm, str(h), 1, ti) for lm, h, ti in zip(d["landmarks"], d["handedness"], d["t"])], d["t"]
    sys.exit(f"no clip labelled '{label}' under {clips}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, default=Path("data/clips"))
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    (still, ts), (moving, tm) = synthetic_clips() if args.synthetic else (load(args.clips, "still_palm"),
                                                                         load(args.clips, "moving_palm"))
    ns = np.array([palm_normal(h) for h in still])
    nm = np.array([palm_normal(h) for h in moving])
    print(f"{'setting':>18s} | jitter (deg) | lag (ms)")
    print("-" * 44)
    for mc, beta in SETTINGS:
        j = angle_series(run_filter(ns, ts, mc, beta)).std()
        lag = lag_ms(nm, run_filter(nm, tm, mc, beta), tm)
        name = "raw" if mc is None else f"mc={mc}, beta={beta}"
        print(f"{name:>18s} | {j:12.2f} | {lag:8.0f}")


if __name__ == "__main__":
    main()
