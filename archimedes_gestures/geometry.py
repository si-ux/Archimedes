"""Continuous control signals extracted from one hand."""
from __future__ import annotations

import numpy as np

from .landmarks import INDEX, PINKY, THUMB, WRIST, HandFrame


def palm_normal(hand: HandFrame) -> np.ndarray:
    """Unit vector pointing out of the palm, in camera coords (x right, y down, z away).

    Built from three rigid landmarks (wrist, index MCP, pinky MCP), so it does
    not depend on what the fingers are doing. The sign is flipped for left
    hands so the vector always leaves the palm side.
    """
    lm = hand.landmarks
    a = lm[INDEX[0]] - lm[WRIST]
    b = lm[PINKY[0]] - lm[WRIST]
    n = np.cross(a, b)
    if hand.handedness == "Right":
        n = -n
    return n / (np.linalg.norm(n) + 1e-9)


def camera_to_view(v: np.ndarray) -> np.ndarray:
    """Camera coords (x right, y down, z away) -> view coords (x right, y up, z towards viewer)."""
    return np.array([v[0], -v[1], -v[2]])


def pinch_distance(hand: HandFrame) -> float:
    """Thumb tip to index tip, in palm units (2D, robust to depth noise)."""
    lm = hand.landmarks
    return float(np.linalg.norm(lm[THUMB[3], :2] - lm[INDEX[3], :2]) / hand.palm_size)


def fingertip(hand: HandFrame) -> np.ndarray:
    return hand.landmarks[INDEX[3], :2].copy()
