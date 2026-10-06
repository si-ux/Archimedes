"""Static hand-pose classifiers and temporal smoothing.

Three layers, each usable on its own:

* ``RulePoseClassifier``  - geometric baseline, works with zero training data,
  so the app is usable on day one and gives the report its baseline row.
* ``SklearnPoseClassifier`` - any trained scikit-learn pipeline saved with
  ``scripts/train_static.py``.
* ``PersonalAdapter`` - few-shot k-NN on top of either one, trained from
  ~30 frames per pose that the user records in the calibration screen. This
  is the main "works for *my* hand" feature.

``PoseSmoother`` then turns noisy per-frame probabilities into a stable pose
with hysteresis, so a single bad frame never flips the mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import features as F
from .vocab import NONE, POSES

LABELS = POSES + (NONE,)


def _sig(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class PoseClassifier:
    labels: tuple[str, ...] = LABELS

    def predict_proba(self, lm: np.ndarray, handedness: str = "Right") -> dict[str, float]:
        raise NotImplementedError

    def predict(self, lm: np.ndarray, handedness: str = "Right") -> tuple[str, float]:
        p = self.predict_proba(lm, handedness)
        k = max(p, key=p.get)
        return k, p[k]


class RulePoseClassifier(PoseClassifier):
    """Soft finger-state templates. Thresholds are in palm-normalised units."""

    # which fingers are extended (thumb, index, middle, ring, pinky)
    TEMPLATES = {
        "open_palm": (1, 1, 1, 1, 1),
        "fist": (0, 0, 0, 0, 0),
        "point": (None, 1, 0, 0, 0),  # thumb may be either
        "peace": (None, 1, 1, 0, 0),
        "thumbs_up": (1, 0, 0, 0, 0),
    }

    def __init__(self, finger_thr=1.05, thumb_thr=1.02, temp=0.06, pinch_thr=0.35, none_prior=0.15):
        self.finger_thr, self.thumb_thr, self.temp = finger_thr, thumb_thr, temp
        self.pinch_thr, self.none_prior = pinch_thr, none_prior

    def predict_proba(self, lm, handedness="Right"):
        st = F.finger_states(lm, handedness)
        r = st["ratios"]
        ext = np.empty(5)
        ext[0] = _sig((r[0] - self.thumb_thr) / self.temp)
        ext[1:] = _sig((r[1:] - self.finger_thr) / self.temp)
        pinch = _sig((self.pinch_thr - st["pinch"]) / 0.05) * _sig((st["index_reach"] - 1.1) / 0.08)
        scores = {}
        for name, tpl in self.TEMPLATES.items():
            s = 1.0
            for e, t in zip(ext, tpl):
                if t is not None:
                    s *= e if t else (1 - e)
            scores[name] = s * (1 - pinch)
        scores["pinch"] = pinch
        scores[NONE] = self.none_prior
        z = sum(scores.values())
        return {k: float(v / z) for k, v in scores.items()}


class SklearnPoseClassifier(PoseClassifier):
    def __init__(self, model):
        self.model = model
        self.labels = tuple(model.classes_)

    @classmethod
    def load(cls, path: str | Path) -> "SklearnPoseClassifier":
        import joblib

        return cls(joblib.load(path))

    def predict_proba(self, lm, handedness="Right"):
        x = F.extract(lm, handedness)[None]
        p = self.model.predict_proba(x)[0]
        return {k: float(v) for k, v in zip(self.labels, p)}


class PersonalAdapter(PoseClassifier):
    """Blend a base classifier with a k-NN fitted on the user's own samples.

    The blend weight grows with how many personal samples exist for the
    predicted class, so an empty adapter behaves exactly like the base model.
    """

    def __init__(self, base: PoseClassifier, k: int = 7, max_weight: float = 0.7):
        self.base, self.k, self.max_weight = base, k, max_weight
        self.X: list[np.ndarray] = []
        self.y: list[str] = []
        self.labels = base.labels

    def add_sample(self, lm, label: str, handedness="Right") -> None:
        self.X.append(F.extract(lm, handedness))
        self.y.append(label)

    def clear(self, label: str | None = None) -> None:
        keep = [i for i, y in enumerate(self.y) if label is not None and y != label]
        self.X = [self.X[i] for i in keep]
        self.y = [self.y[i] for i in keep]

    def counts(self) -> dict[str, int]:
        return {lab: self.y.count(lab) for lab in set(self.y)}

    def predict_proba(self, lm, handedness="Right"):
        base = self.base.predict_proba(lm, handedness)
        if len(self.y) < self.k:
            return base
        X = np.stack(self.X)
        x = F.extract(lm, handedness)
        d = np.linalg.norm(X - x, axis=1)
        idx = np.argsort(d)[: self.k]
        # distance-weighted vote; far-away neighbours mean "not one of my poses"
        scale = np.median(d) + 1e-9
        w = np.exp(-((d[idx] / scale) ** 2) * 4)
        knn = {lab: 0.0 for lab in self.labels}
        for i, wi in zip(idx, w):
            knn[self.y[i]] = knn.get(self.y[i], 0.0) + wi
        tot = sum(knn.values())
        if tot < 1e-6:
            return base
        knn = {k: v / tot for k, v in knn.items()}
        alpha = self.max_weight * min(1.0, w.sum() / self.k * 2)
        out = {k: (1 - alpha) * base.get(k, 0.0) + alpha * knn.get(k, 0.0) for k in self.labels}
        z = sum(out.values())
        return {k: v / z for k, v in out.items()}

    def save(self, path: str | Path) -> None:
        np.savez(path, X=np.stack(self.X) if self.X else np.zeros((0, F.FEATURE_DIM)), y=np.array(self.y))

    def load(self, path: str | Path) -> None:
        d = np.load(path, allow_pickle=False)
        self.X = list(d["X"])
        self.y = [str(v) for v in d["y"]]


@dataclass
class PoseSmoother:
    """EMA over class probabilities + enter/exit hysteresis.

    A new pose must beat ``enter`` for ``min_frames`` consecutive frames to
    become current; the current pose is kept until it falls below ``exit``.
    With enter > exit the label cannot chatter at the decision boundary.
    """

    alpha: float = 0.45
    enter: float = 0.6
    exit: float = 0.35
    min_frames: int = 3
    probs: dict[str, float] = field(default_factory=dict)
    current: str = NONE
    _candidate: str = NONE
    _count: int = 0

    def reset(self) -> None:
        self.probs.clear()
        self.current, self._candidate, self._count = NONE, NONE, 0

    def update(self, p: dict[str, float]) -> tuple[str, float]:
        if not self.probs:
            self.probs.update(p)
        for k in set(p) | set(self.probs):
            self.probs[k] = self.alpha * p.get(k, 0.0) + (1 - self.alpha) * self.probs.get(k, 0.0)
        if self.current != NONE and self.probs.get(self.current, 0.0) < self.exit:
            self.current = NONE
        best = max(self.probs, key=self.probs.get)
        if best != self.current and self.probs[best] >= self.enter:
            self._count = self._count + 1 if best == self._candidate else 1
            self._candidate = best
            if self._count >= self.min_frames:
                self.current, self._count = best, 0
        else:
            self._candidate, self._count = NONE, 0
        return self.current, self.probs.get(self.current, 0.0)
