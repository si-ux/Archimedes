"""Guided 30-second calibration.

Steps
1. ``right_hand`` - "raise your RIGHT hand". Detects whether the camera
   path needs MediaPipe's left/right labels swapped (they depend on whether
   the frame was mirrored), and measures the user's palm size at a
   comfortable distance.
2. ``comfort``    - "move your hand around the area that feels
   comfortable". The palm-centre range becomes the interaction box: the
   probe cursor maps that box to the whole screen, so nobody has to stretch.
3. ``pose:<name>`` (optional) - show each pose for ~1.5 s. Frames after a
   short settle time become personal samples for the few-shot adapter.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .landmarks import FrameInput
from .vocab import MODES, POSES

POSE_NAMES = {
    "open_palm": "a flat open palm ✋",
    "fist": "a fist ✊",
    "point": "pointing with your index finger ☝️",
    "peace": "a V / peace sign ✌️",
    "thumbs_up": "a thumbs-up 👍",
    "pinch": "a pinch (thumb touches index) 🤏",
}


@dataclass
class Profile:
    swap_handedness: bool = False
    dominant: str = "Right"
    palm_size: float = 0.12
    box: tuple = (0.2, 0.15, 0.8, 0.75)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        d = json.loads(Path(path).read_text())
        d["box"] = tuple(d["box"])
        return cls(**d)


@dataclass
class Calibrator:
    with_poses: bool = True
    hold_s: float = 1.5
    settle_s: float = 0.5
    comfort_s: float = 4.0
    steps: list = field(default_factory=list)
    step_i: int = 0
    t_step: float | None = None
    labels_seen: list = field(default_factory=list)
    sizes: list = field(default_factory=list)
    centers: list = field(default_factory=list)
    samples: list = field(default_factory=list)  # (label, landmarks, handedness)
    profile: Profile = field(default_factory=Profile)
    done: bool = False

    def __post_init__(self) -> None:
        self.steps = ["right_hand", "comfort"] + ([f"pose:{p}" for p in POSES] if self.with_poses else [])

    @property
    def step(self) -> str | None:
        return None if self.done else self.steps[self.step_i]

    def instruction(self) -> str:
        s = self.step
        if s is None:
            return "Calibration complete - you're ready to go!"
        if s == "right_hand":
            return "Raise your RIGHT hand only, palm facing the screen, at a comfortable distance"
        if s == "comfort":
            return "Slowly move your hand around the area that feels comfortable - no stretching"
        pose = s.split(":", 1)[1]
        mode = next((m for m in MODES.values() if m.pose == pose), None)
        use = f" (used for {mode.label})" if mode else ""
        return f"Show {POSE_NAMES[pose]}{use} and hold it"

    def _duration(self) -> float:
        return self.comfort_s if self.step == "comfort" else self.hold_s

    def update(self, f: FrameInput) -> dict:
        if self.done:
            return self.status(1.0)
        if not f.hands:
            self.t_step = None  # timer only runs while a hand is visible
            return self.status(0.0, "No hand visible")
        if self.t_step is None:
            self.t_step = f.timestamp
        el = f.timestamp - self.t_step
        h = f.hands[0]
        s = self.step
        if s == "right_hand":
            if len(f.hands) == 1 and el > self.settle_s:
                self.labels_seen.append(h.handedness)
                self.sizes.append(h.palm_size)
        elif s == "comfort":
            h = f.hand(self.profile.dominant) or h
            self.centers.append(h.palm_center[:2])
        elif el > self.settle_s:
            h = f.hand(self.profile.dominant) or h
            self.samples.append((s.split(":", 1)[1], h.shape.copy(), h.handedness))
        prog = min(1.0, el / self._duration())
        if prog >= 1.0:
            self._finish_step()
        return self.status(prog)

    def _finish_step(self) -> None:
        s = self.step
        if s == "right_hand" and self.labels_seen:
            # the hand we were told is "Right" is labelled "Left" -> labels need swapping
            left = sum(1 for x in self.labels_seen if x == "Left")
            mismatch = left > len(self.labels_seen) / 2
            # frames were already labelled with the old setting, so flip relative to it
            self.profile.swap_handedness = self.profile.swap_handedness ^ mismatch
            self.profile.palm_size = float(np.median(self.sizes))
        elif s == "comfort" and len(self.centers) > 5:
            c = np.array(self.centers)
            lo, hi = np.percentile(c, 5, axis=0), np.percentile(c, 95, axis=0)
            span = np.maximum(hi - lo, 0.25)  # never smaller than a quarter of the frame
            mid = (lo + hi) / 2
            lo, hi = np.clip(mid - span / 2, 0, 1), np.clip(mid + span / 2, 0, 1)
            self.profile.box = (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))
        self.step_i += 1
        self.t_step = None
        if self.step_i >= len(self.steps):
            self.done = True

    def status(self, progress: float, warning: str | None = None) -> dict:
        return {
            "step": self.step,
            "step_index": self.step_i,
            "step_count": len(self.steps),
            "instruction": self.instruction(),
            "progress": round(progress, 3),
            "warning": warning,
            "done": self.done,
        }
