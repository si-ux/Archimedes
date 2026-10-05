"""Hand landmark containers and normalisation.

Coordinate convention used everywhere in this package
-----------------------------------------------------
All landmarks are stored in **selfie view**: the image as the user sees it in
a mirror. x grows to the user's right, y grows downwards, z is MediaPipe's
relative depth (smaller = closer to the camera). If a source delivers
un-mirrored frames (e.g. the browser demo), call ``HandFrame.from_raw`` with
``mirrored=False`` and the x axis is flipped for you.

Handedness is stored as the user's *real* hand ("Left" / "Right").
MediaPipe's label convention depends on whether the input was mirrored, so
``swap_handedness`` exists, and the calibration step detects the right value
automatically by asking the user to raise their right hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# MediaPipe hand landmark indices
WRIST = 0
THUMB = (1, 2, 3, 4)  # CMC, MCP, IP, TIP
INDEX = (5, 6, 7, 8)  # MCP, PIP, DIP, TIP
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
FINGERS = (THUMB, INDEX, MIDDLE, RING, PINKY)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
TIPS = tuple(f[3] for f in FINGERS)
MCPS = tuple(f[0] for f in FINGERS[1:])

# Wrist + the four finger MCPs barely move when the fingers curl, so the
# palm centre and palm size built from them stay steady through pose changes
# (fist <-> open hand). Anchoring continuous control on these points instead
# of fingertips stops the view jumping whenever the user changes pose.
RIGID_PALM = (WRIST, *MCPS)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


@dataclass
class HandFrame:
    """One detected hand in one camera frame (selfie-view coordinates)."""

    landmarks: np.ndarray  # (21, 3)
    handedness: str = "Right"  # user's real hand
    score: float = 1.0  # detector handedness / presence confidence
    timestamp: float = 0.0  # seconds

    def __post_init__(self) -> None:
        self.landmarks = np.asarray(self.landmarks, dtype=float).reshape(21, 3)

    @classmethod
    def from_raw(
        cls,
        landmarks,
        label: str,
        score: float = 1.0,
        timestamp: float = 0.0,
        mirrored: bool = True,
        swap_handedness: bool = False,
    ) -> "HandFrame":
        lm = np.asarray(landmarks, dtype=float).reshape(21, 3).copy()
        if not mirrored:
            lm[:, 0] = 1.0 - lm[:, 0]
        hand = label.capitalize()
        if swap_handedness:
            hand = "Left" if hand == "Right" else "Right"
        return cls(lm, hand, float(score), float(timestamp))

    # ---- cheap geometric summaries -------------------------------------
    @property
    def palm_center(self) -> np.ndarray:
        return self.landmarks[list(RIGID_PALM)].mean(axis=0)

    @property
    def palm_size(self) -> float:
        """Wrist to middle-finger MCP distance in image units.

        Measured in 3D (MediaPipe's z is on the same scale as x), so a palm tilted
        towards the floor, as in a load sweep, doesn't read as "far away".
        """
        d = self.landmarks[MIDDLE[0]] - self.landmarks[WRIST]
        return float(np.linalg.norm(d)) + 1e-9

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xy = self.landmarks[:, :2]
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)


@dataclass
class FrameInput:
    """Everything the pipeline receives for one camera frame."""

    timestamp: float
    hands: list[HandFrame] = field(default_factory=list)
    brightness: float | None = None  # mean luma 0..255, if the source measured it
    sharpness: float | None = None  # variance of Laplacian, if measured

    def hand(self, which: str) -> HandFrame | None:
        for h in self.hands:
            if h.handedness == which:
                return h
        return None


def normalize(lm: np.ndarray, handedness: str = "Right", rotate: bool = True) -> np.ndarray:
    """Make landmarks independent of position, scale, image-plane rotation and handedness.

    1. translate so the wrist is at the origin
    2. rotate (in the image plane) so wrist -> middle MCP points up
    3. scale so that distance is 1
    4. mirror left hands so one classifier serves both hands
    """
    p = np.asarray(lm, dtype=float).reshape(21, 3) - lm[WRIST]
    p = p.copy()
    if handedness == "Left":
        p[:, 0] *= -1.0
    if rotate:
        v = p[MIDDLE[0], :2]
        ang = np.arctan2(v[0], -v[1])  # angle from "up" (-y in image coords)
        c, s = np.cos(ang), np.sin(ang)
        rot = np.array([[c, s], [-s, c]])  # rotate by -ang
        p[:, :2] = p[:, :2] @ rot.T
    scale = np.linalg.norm(p[MIDDLE[0]]) + 1e-9
    return p / scale
