"""Synthetic hand landmarks: a tiny kinematic hand model.

This exists so the whole pipeline can be unit-tested and demoed in a
Codespace, which has no webcam. It is NOT a substitute for real data: never
report classifier accuracy measured on synthetic hands.
"""
from __future__ import annotations

import numpy as np

from .landmarks import HandFrame

# Right hand, selfie view, palm facing the camera, palm size (wrist->middle MCP) = 1.
# x: towards the pinky, y: up (we flip to image "down" at the end), z: towards the camera.
_MCP = {
    "index": np.array([-0.32, 0.95, 0.0]),
    "middle": np.array([0.0, 1.0, 0.0]),
    "ring": np.array([0.27, 0.93, 0.0]),
    "pinky": np.array([0.50, 0.82, 0.0]),
}
_SEG = {
    "index": (0.45, 0.27, 0.22),
    "middle": (0.50, 0.30, 0.24),
    "ring": (0.46, 0.28, 0.22),
    "pinky": (0.36, 0.22, 0.20),
}
_SPLAY = {"index": -0.10, "middle": 0.0, "ring": 0.08, "pinky": 0.18}
_THUMB_CMC = np.array([-0.30, 0.25, 0.05])
_THUMB_SEG = (0.35, 0.30, 0.25)

# curl per joint (MCP, PIP, DIP) in degrees
STRAIGHT = (5, 5, 3)
CURLED = (85, 100, 60)
HALF = (40, 50, 30)

POSES = ("open_palm", "fist", "point", "peace", "thumbs_up", "pinch")


def _finger_chain(name: str, curl_deg) -> np.ndarray:
    base = _MCP[name]
    d = np.array([np.sin(_SPLAY[name]), np.cos(_SPLAY[name]), 0.0])
    pts = [base]
    p = base.copy()
    total = 0.0
    for seg, c in zip(_SEG[name], curl_deg):
        total += np.radians(c)
        # bend the finger towards the palm side (+z = towards camera)
        dirv = d * np.cos(total) + np.array([0, 0, 1.0]) * np.sin(total)
        p = p + seg * dirv
        pts.append(p)
    return np.array(pts)  # MCP, PIP, DIP, TIP


def _thumb_chain(out: bool, target: np.ndarray | None = None) -> np.ndarray:
    cmc = _THUMB_CMC
    if target is not None:  # pinch: bend the thumb so its tip meets target
        mid = (cmc + target) / 2 + np.array([-0.15, 0.0, 0.10])
        return np.array([cmc, (cmc + mid) / 2, (mid + target) / 2, target])
    if out:
        d = np.array([-0.75, 0.66, 0.05])
        d /= np.linalg.norm(d)
        segs = [d, d, d]
    else:  # tucked across the curled fingers
        d1 = np.array([-0.2, 0.8, 0.55])
        d2 = np.array([0.6, 0.35, 0.7])
        d3 = np.array([0.9, -0.1, 0.4])
        segs = [v / np.linalg.norm(v) for v in (d1, d2, d3)]
    pts = [cmc]
    p = cmc.copy()
    for seg, v in zip(_THUMB_SEG, segs):
        p = p + seg * v
        pts.append(p)
    return np.array(pts)


def canonical_hand(pose: str, rng: np.random.Generator | None = None) -> np.ndarray:
    """21x3 landmarks in the canonical hand frame for a named pose (or 'none')."""
    rng = rng or np.random.default_rng()
    jitter = lambda c: tuple(np.clip(np.array(c) + rng.normal(0, 6, 3), 0, 120))  # noqa: E731

    if pose == "none":
        curls = {n: tuple(rng.uniform(25, 65, 3)) for n in _MCP}
        thumb = _thumb_chain(out=bool(rng.random() < 0.5))
    else:
        ext = {
            "open_palm": "imrp",
            "fist": "",
            "point": "i",
            "peace": "im",
            "thumbs_up": "",
            "pinch": "mrp",
        }[pose]
        curls = {n: jitter(STRAIGHT if n[0] in ext else CURLED) for n in _MCP}
        if pose == "pinch":
            curls["index"] = jitter(HALF)
        thumb = None
        if pose in ("open_palm", "thumbs_up", "point", "peace"):
            thumb = _thumb_chain(out=pose in ("open_palm", "thumbs_up"))
    fingers = {n: _finger_chain(n, curls[n]) for n in _MCP}
    if thumb is None:
        if pose == "pinch":
            thumb = _thumb_chain(False, target=fingers["index"][3] + rng.normal(0, 0.02, 3))
        else:
            thumb = _thumb_chain(out=False)
    lm = np.zeros((21, 3))
    lm[1:5] = thumb
    lm[5:9] = fingers["index"]
    lm[9:13] = fingers["middle"]
    lm[13:17] = fingers["ring"]
    lm[17:21] = fingers["pinky"]
    return lm



# finger-counting hands: which fingers are up for 0..5 (thumb last, as most people count)
COUNT_FINGERS = {0: "", 1: "i", 2: "im", 3: "imr", 4: "imrp", 5: "timrp"}


def canonical_fingers(ext: str, rng: np.random.Generator | None = None) -> np.ndarray:
    """Canonical hand with exactly the fingers in ``ext`` extended (t, i, m, r, p)."""
    rng = rng or np.random.default_rng()
    jitter = lambda c: tuple(np.clip(np.array(c) + rng.normal(0, 5, 3), 0, 120))  # noqa: E731
    curls = {n: jitter(STRAIGHT if n[0] in ext else CURLED) for n in _MCP}
    fingers = {n: _finger_chain(n, curls[n]) for n in _MCP}
    lm = np.zeros((21, 3))
    lm[1:5] = _thumb_chain(out="t" in ext)
    lm[5:9] = fingers["index"]
    lm[9:13] = fingers["middle"]
    lm[13:17] = fingers["ring"]
    lm[17:21] = fingers["pinky"]
    return lm


def _rot(yaw, pitch, roll) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ rx @ ry


def to_image(
    canon: np.ndarray,
    center=(0.5, 0.55),
    palm_size: float = 0.12,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    handedness: str = "Right",
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Pose the canonical hand in selfie-view image coordinates."""
    rng = rng or np.random.default_rng()
    p = canon @ _rot(yaw, pitch, roll).T
    if noise:
        p = p + rng.normal(0, noise, p.shape)
    if handedness == "Left":
        p[:, 0] *= -1
    img = np.empty_like(p)
    img[:, 0] = center[0] + p[:, 0] * palm_size
    img[:, 1] = center[1] - p[:, 1] * palm_size  # image y points down
    img[:, 2] = -p[:, 2] * palm_size  # MediaPipe z: negative = closer
    return img


def random_hand(pose: str, rng: np.random.Generator, handedness: str | None = None) -> HandFrame:
    hand = handedness or ("Right" if rng.random() < 0.7 else "Left")
    lm = to_image(
        canonical_hand(pose, rng),
        center=(rng.uniform(0.3, 0.7), rng.uniform(0.35, 0.7)),
        palm_size=rng.uniform(0.08, 0.18),
        yaw=rng.uniform(-0.5, 0.5),
        pitch=rng.uniform(-0.4, 0.4),
        roll=rng.uniform(-0.6, 0.6),
        handedness=hand,
        noise=0.025,
        rng=rng,
    )
    return HandFrame(lm, hand)


def dataset(n_per_class: int = 200, seed: int = 0, classes=POSES + ("none",)):
    """Return (landmarks[N,21,3], handedness[N], labels[N], subject[N]) of synthetic hands."""
    rng = np.random.default_rng(seed)
    lms, hands, labels, subjects = [], [], [], []
    for label in classes:
        for i in range(n_per_class):
            h = random_hand(label, rng)
            lms.append(h.landmarks)
            hands.append(h.handedness)
            labels.append(label)
            subjects.append(f"synth{i % 8}")
    return np.array(lms), hands, np.array(labels), np.array(subjects)


# ---------------------------------------------------------------------------
# Scripted tour: a camera-free demo of every interaction (used by the web demo's
# "Demo (no camera)" button and by tests).
TOUR = [
    # pose, seconds, start centre, end centre, extra pose kwargs, caption
    ("none", 1.0, (0.5, 0.55), (0.55, 0.5), {}, "A relaxed hand does nothing - no accidental input"),
    ("fist", 0.5, (0.5, 0.5), (0.5, 0.5), {}, "Hold a fist to grab the model…"),
    ("fist", 1.2, (0.5, 0.5), (0.58, 0.47), {}, "…and drag to rotate"),
    ("open_palm", 0.3, (0.58, 0.47), (0.58, 0.47), {}, "Open the hand to let go"),
    ("pinch", 0.4, (0.55, 0.55), (0.55, 0.55), {}, "Pinch…"),
    ("pinch", 1.0, (0.55, 0.55), (0.55, 0.42), {}, "…and move up to zoom in"),
    ("open_palm", 1.2, (0.5, 0.5), (0.5, 0.5), {}, "Hold a flat palm still to start a section cut"),
    ("open_palm", 1.5, (0.5, 0.5), (0.5, 0.5), {"yaw": 1.2}, "Turn the palm - it becomes the cutting plane"),
    ("fist", 0.4, (0.5, 0.5), (0.5, 0.5), {"yaw": 1.2}, "Close the hand to lock the cut"),
    ("point", 0.5, (0.50, 0.68), (0.50, 0.68), {}, "Point to probe stress values"),
    ("point", 1.5, (0.50, 0.68), (0.58, 0.66), {}, "Move the finger over the model"),
    ("point", 1.2, (0.58, 0.66), (0.58, 0.66), {}, "Hold still to pin a value"),
    ("peace", 0.5, (0.4, 0.5), (0.4, 0.5), {}, "Show a V for the field menu"),
    ("peace", 0.25, (0.4, 0.5), (0.7, 0.5), {}, "Swipe to change the result field"),
    ("peace", 0.8, (0.7, 0.5), (0.7, 0.5), {}, "Swipe to change the result field"),
    ("none", 1.0, (0.6, 0.55), (0.6, 0.55), {}, "Relax - the view stays where you left it"),
]


def scripted_tour(fps: int = 30, seed: int = 0, tour=TOUR):
    """Yield (t, HandFrame | None, caption) at ``fps`` for the scripted tour."""
    rng = np.random.default_rng(seed)
    t = 0.0
    for pose, dur, p0, p1, kw, caption in tour:
        canon = canonical_hand(pose, rng)
        n = max(int(dur * fps), 1)
        for i in range(n):
            a = i / max(n - 1, 1)
            c = (p0[0] + (p1[0] - p0[0]) * a, p0[1] + (p1[1] - p0[1]) * a)
            lm = to_image(canon, center=c, palm_size=0.12, noise=0.004, rng=rng, **kw)
            yield t, HandFrame(lm, "Right", 0.95, t), caption
            t += 1.0 / fps
