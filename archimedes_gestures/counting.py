"""Reading numbers from hands: finger counting and a digit-entry state machine.

How a user enters "300":
    hold up 3 fingers  → ring fills (0.8 s) → "3"
    make a fist        → "30"
    open, then fist    → "300"   (a digit is taken once; change the count to enter it again)
    both open palms    → confirm ("enter"); with no digits this accepts the default
    swipe left         → delete the last digit
    two fists (1 s)    → cancel

Digits above 5 use both hands (5 + 3 = 8). Ten fingers means "confirm", not 10.
The count is smoothed over a few frames, so a finger that flickers while the
hand moves doesn't change the digit.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .dynamic import SwipeDetector
from .features import finger_extension_ratios
from .landmarks import HandFrame, normalize


def count_fingers(hand: HandFrame, finger_thr: float = 1.05, thumb_thr: float = 1.02) -> int:
    r = finger_extension_ratios(normalize(hand.landmarks, hand.handedness))
    return int(r[0] > thumb_thr) + int(np.sum(r[1:] > finger_thr))


@dataclass
class NumberEntry:
    hold_s: float = 0.8
    confirm_s: float = 0.8
    cancel_s: float = 1.0
    still_speed: float = 0.9  # palm widths / s
    digits: str = ""
    max_digits: int = 6
    _hist: dict = field(default_factory=dict)
    _cand: int | None = None
    _cand_t: float = 0.0
    _armed: bool = True
    _confirm_t: float = 0.0
    _cancel_t: float = 0.0
    _last_t: float | None = None
    _nhands: int = 0
    _swipe: SwipeDetector = field(default_factory=lambda: SwipeDetector(min_distance=1.6, min_speed=5.0))

    def reset(self) -> None:
        self.digits = ""
        self._hist.clear()
        self._cand, self._cand_t, self._armed = None, 0.0, True
        self._confirm_t = self._cancel_t = 0.0
        self._last_t = None
        self._swipe.reset()

    @property
    def value(self) -> int | None:
        return int(self.digits) if self.digits else None

    def _smoothed(self, hand: HandFrame) -> int:
        h = self._hist.setdefault(hand.handedness, deque(maxlen=5))
        h.append(count_fingers(hand))
        return int(np.bincount(list(h)).argmax())

    def update(self, t: float, hands: list[HandFrame], speed: float = 0.0,
               control: HandFrame | None = None) -> tuple[list[tuple], dict]:
        """Returns (events, status). Events: ("digit", d), ("backspace",), ("enter", value|None), ("cancel",)."""
        dt = 0.0 if self._last_t is None else max(0.0, min(t - self._last_t, 0.2))
        self._last_t = t
        events: list[tuple] = []
        for k in list(self._hist):
            if not any(h.handedness == k for h in hands):
                del self._hist[k]
        if not hands:
            self._cand, self._armed = None, True
            self._confirm_t = self._cancel_t = 0.0
            return events, self.status(None, 0.0, None)

        counts = [self._smoothed(h) for h in hands]
        total = sum(counts)
        ctrl = control or hands[0]
        # delete = one open hand swiping left; the swipe history restarts whenever the
        # number of hands changes, so a hand entering the frame never reads as a swipe
        if len(hands) != self._nhands or len(hands) != 1 or counts[0] != 5:
            self._swipe.reset()
        self._nhands = len(hands)
        if len(hands) == 1 and counts[0] == 5 and self._swipe.update(t, ctrl.palm_center[:2], ctrl.palm_size) == -1:
            if self.digits:
                self.digits = self.digits[:-1]
                events.append(("backspace",))
            self._armed = False

        two = len(hands) >= 2
        if two and counts[0] == 5 and counts[1] == 5:
            self._confirm_t += dt
            self._cancel_t = 0.0
            self._cand = None
            if self._confirm_t >= self.confirm_s:
                events.append(("enter", self.value))
                self._confirm_t = 0.0
                self._armed = False
            return events, self.status(10, self._confirm_t / self.confirm_s, "confirm")
        if two and total == 0:
            self._cancel_t += dt
            self._confirm_t = 0.0
            self._cand = None
            if self._cancel_t >= self.cancel_s:
                events.append(("cancel",))
                self._cancel_t = 0.0
                self._armed = False
            return events, self.status(0, self._cancel_t / self.cancel_s, "cancel")
        self._confirm_t = self._cancel_t = 0.0

        if total != self._cand:
            self._cand, self._cand_t, self._armed = total, 0.0, True
        elif self._armed and speed < self.still_speed:
            self._cand_t += dt
            if self._cand_t >= self.hold_s and total <= 9:
                if len(self.digits) < self.max_digits:
                    self.digits = (self.digits + str(total)).lstrip("0") or "0"
                    events.append(("digit", total))
                self._armed = False
        prog = self._cand_t / self.hold_s if self._armed else 0.0
        return events, self.status(total, prog, "digit" if self._armed else "taken")

    def status(self, live: int | None, progress: float, kind: str | None) -> dict:
        return {"digits": self.digits, "live": live, "progress": round(min(1.0, progress), 3), "kind": kind}
