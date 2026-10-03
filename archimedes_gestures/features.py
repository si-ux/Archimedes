"""Hand-pose features: scale-, translation- and rotation-invariant descriptors.

The same feature vector feeds every classifier (rules baseline, sklearn
models, the per-user few-shot adapter), so results stay comparable.
"""
from __future__ import annotations

import itertools

import numpy as np

from .landmarks import FINGERS, INDEX, MIDDLE, PINKY, THUMB, TIPS, WRIST, normalize

TIP_PAIRS = list(itertools.combinations(TIPS, 2))


def finger_extension_ratios(p: np.ndarray) -> np.ndarray:
    """Per finger: |wrist->tip| / |wrist->PIP| (thumb: |tip->pinky MCP| / |IP->pinky MCP|).

    Straight fingers give ~1.4-1.9, curled fingers < 1.1. Ratios are used
    instead of absolute lengths so hand size and camera distance drop out.
    """
    out = np.empty(5)
    for i, f in enumerate(FINGERS):
        if f is THUMB:
            ref = p[PINKY[0]]
            out[i] = np.linalg.norm(p[f[3]] - ref) / (np.linalg.norm(p[f[2]] - ref) + 1e-9)
        else:
            out[i] = np.linalg.norm(p[f[3]] - p[WRIST]) / (np.linalg.norm(p[f[1]] - p[WRIST]) + 1e-9)
    return out


def thumb_out(p: np.ndarray) -> float:
    """Thumb tip to index MCP, in palm units (p must be normalised). ~0.3 tucked, ~0.9 out."""
    return float(np.linalg.norm(p[THUMB[3]] - p[INDEX[0]]))


def pinch_distance(p: np.ndarray) -> float:
    """Thumb tip to index tip, in palm units (p must be normalised)."""
    return float(np.linalg.norm(p[THUMB[3]] - p[INDEX[3]]))


def index_reach(p: np.ndarray) -> float:
    """Index tip to wrist in palm units; tells a pinch (index out) from a fist (index curled)."""
    return float(np.linalg.norm(p[INDEX[3]] - p[WRIST]))


def extract(lm: np.ndarray, handedness: str = "Right") -> np.ndarray:
    """Return the full feature vector (1-D float array) for one hand."""
    p = normalize(lm, handedness)
    tip_d = [np.linalg.norm(p[a] - p[b]) for a, b in TIP_PAIRS]
    return np.concatenate(
        [
            p.ravel(),  # 63 normalised coordinates
            finger_extension_ratios(p),  # 5
            [thumb_out(p), pinch_distance(p), index_reach(p)],  # 3
            tip_d,  # 10 fingertip pairwise distances
        ]
    )


FEATURE_DIM = 63 + 5 + 3 + len(TIP_PAIRS)


def extract_batch(landmarks: np.ndarray, handedness: list[str] | None = None) -> np.ndarray:
    landmarks = np.asarray(landmarks).reshape(-1, 21, 3)
    if handedness is None:
        handedness = ["Right"] * len(landmarks)
    return np.stack([extract(lm, h) for lm, h in zip(landmarks, handedness)])


# Convenience used by the rules classifier and the HUD skeleton colouring
def finger_states(lm: np.ndarray, handedness: str = "Right") -> dict[str, float]:
    p = normalize(lm, handedness)
    r = finger_extension_ratios(p)
    return {
        "ratios": r,
        "thumb_out": thumb_out(p),
        "pinch": pinch_distance(p),
        "index_reach": index_reach(p),
        "middle_reach": float(np.linalg.norm(p[MIDDLE[3]] - p[WRIST])),
    }
