"""Save labelled landmark clips to .npz, the format every training script reads.

Each file holds one clip:
    landmarks  (T, 21, 3)  selfie-view landmarks of the recorded hand
    handedness (T,)        "Left" / "Right"
    t          (T,)        timestamps in seconds
    label      ()          gesture label
    subject    ()          person id - lets evaluation split by person
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .landmarks import FrameInput


class ClipRecorder:
    def __init__(self, root: str | Path = "data/clips"):
        self.root = Path(root)
        self.active = False
        self._buf: list = []

    def start(self, label: str, subject: str, hand: str = "Right") -> None:
        self.label, self.subject, self.hand = label, subject, hand
        self._buf = []
        self.active = True

    def add(self, f: FrameInput) -> None:
        if not self.active:
            return
        h = f.hand(self.hand) or (f.hands[0] if f.hands else None)
        if h is not None:
            self._buf.append((h.landmarks, h.handedness, f.timestamp, h.world))

    def stop(self) -> Path | None:
        self.active = False
        if len(self._buf) < 5:
            return None
        out = self.root / self.subject / self.label
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{int(time.time() * 1000)}.npz"
        extra = {}
        if all(b[3] is not None for b in self._buf):
            extra["world"] = np.stack([b[3] for b in self._buf])  # metric hand shape
        np.savez_compressed(
            path,
            **extra,
            landmarks=np.stack([b[0] for b in self._buf]),
            handedness=np.array([b[1] for b in self._buf]),
            t=np.array([b[2] for b in self._buf]),
            label=np.array(self.label),
            subject=np.array(self.subject),
        )
        return path


def load_clips(root: str | Path, skip_s: float = 0.3):
    """Flatten every clip under ``root`` into per-frame samples.

    The first ``skip_s`` seconds of each clip are dropped: that is the user
    still moving into the pose, and those frames would teach the model noise.
    Returns landmarks (N,21,3), handedness (N,), labels (N,), subjects (N,), clip_ids (N,).
    """
    lms, hands, labels, subjects, clips = [], [], [], [], []
    for i, p in enumerate(sorted(Path(root).rglob("*.npz"))):
        d = np.load(p, allow_pickle=False)
        keep = d["t"] - d["t"][0] >= skip_s
        n = int(keep.sum())
        lms.append((d["world"] if "world" in d.files else d["landmarks"])[keep])  # shape, as used live
        hands += list(d["handedness"][keep])
        labels += [str(d["label"])] * n
        subjects += [str(d["subject"])] * n
        clips += [i] * n
    if not lms:
        raise FileNotFoundError(f"no .npz clips under {root}")
    return np.concatenate(lms), np.array(hands), np.array(labels), np.array(subjects), np.array(clips)
