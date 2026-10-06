"""One Euro filter (Casiez, Roussel & Vogel, CHI 2012).

Low cutoff when the hand is slow (kills jitter), higher cutoff when it moves
fast (keeps lag low). Two knobs: ``min_cutoff`` (jitter) and ``beta`` (lag).
Works on scalars or numpy vectors.
"""
from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.05, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, x, t: float):
        x = np.asarray(x, dtype=float)
        if self._x is None:
            self._x, self._dx, self._t = x, np.zeros_like(x), t
            return x.copy()
        dt = max(t - self._t, 1e-4)
        self._t = t
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        speed = float(np.linalg.norm(self._dx))
        a = _alpha(self.min_cutoff + self.beta * speed, dt)
        self._x = a * x + (1 - a) * self._x
        return self._x.copy()

    @property
    def value(self):
        return None if self._x is None else self._x.copy()
