"""Dynamic (motion) gestures.

``SwipeDetector`` is the streaming baseline used at runtime: velocity peak +
displacement + direction test, with a refractory period so one swipe fires
exactly one command.

``DTWTemplateMatcher`` lets a user record their *own* motion gestures from a
few examples (few-shot, no training run), and is the baseline the learned
temporal model in Phase 3 has to beat.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SwipeDetector:
    window_s: float = 0.35
    min_distance: float = 1.5  # palm sizes travelled within the window
    min_speed: float = 5.0  # palm sizes / second at the peak
    max_off_axis: float = 0.6  # |dy| / |dx| allowed
    refractory_s: float = 0.6
    _buf: deque = field(default_factory=lambda: deque(maxlen=64))
    _last_fire: float = -1e9

    def reset(self) -> None:
        self._buf.clear()

    def update(self, t: float, xy: np.ndarray, palm_size: float) -> int:
        """Feed palm-centre position; returns +1 (swipe right), -1 (left) or 0."""
        self._buf.append((t, np.asarray(xy, float) / palm_size))
        while self._buf and t - self._buf[0][0] > self.window_s:
            self._buf.popleft()
        if len(self._buf) < 4 or t - self._last_fire < self.refractory_s:
            return 0
        ts = np.array([b[0] for b in self._buf])
        ps = np.stack([b[1] for b in self._buf])
        d = ps[-1] - ps[0]
        v = np.linalg.norm(np.diff(ps, axis=0), axis=1) / np.maximum(np.diff(ts), 1e-3)
        if abs(d[0]) < self.min_distance or v.max() < self.min_speed:
            return 0
        if abs(d[1]) > self.max_off_axis * abs(d[0]):
            return 0
        self._last_fire = t
        self._buf.clear()
        return 1 if d[0] > 0 else -1


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic O(nm) DTW on (T, D) sequences, normalised by path length."""
    n, m = len(a), len(b)
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[n, m] / (n + m))


def trajectory_features(xy: np.ndarray, palm_size: float | np.ndarray) -> np.ndarray:
    """Centre a palm trajectory and express it in palm units so position/scale drop out."""
    xy = np.asarray(xy, float)
    ps = np.reshape(np.asarray(palm_size, float), (-1, 1))
    return (xy - xy[0]) / ps


@dataclass
class DTWTemplateMatcher:
    threshold: float = 0.6
    templates: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def add(self, label: str, seq: np.ndarray) -> None:
        self.templates.setdefault(label, []).append(np.asarray(seq, float))

    def classify(self, seq: np.ndarray) -> tuple[str | None, float]:
        best, best_d = None, np.inf
        for label, temps in self.templates.items():
            for tpl in temps:
                d = dtw_distance(np.asarray(seq, float), tpl)
                if d < best_d:
                    best, best_d = label, d
        if best_d > self.threshold:
            return None, best_d
        return best, best_d
