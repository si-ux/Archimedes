"""Intent engine: turns (hand, pose) streams into ViewState commands.

State machine
-------------
    NO_HAND --hand seen--> IDLE --mode pose held arm_s--> ARMING --> ACTIVE
       ^                    ^                                         |
       |                    +------ pose released (grace period) -----+
       +-- hand lost for > lost_grace_s (active mode is ended cleanly) -+
    any state --blocking quality problem--> PAUSED (view frozen, hint shown)

Why this is easy on the user
* Nothing moves unless a mode pose has been *held* for a moment: a relaxed
  or passing hand never touches the model (no "Midas touch").
* Releasing the pose is the clutch. Rotate, open hand, move back, fist again
  and keep rotating, like lifting a mouse.
* Motion is read from the rigid palm (wrist + knuckles), not fingertips, so
  the act of opening the hand does not kick the view.
* When a section cut ends, the plane is taken from ~150 ms *before* the
  release, because the frames during a pose change are the noisiest.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .dynamic import SwipeDetector
from .filters import OneEuroFilter
from .geometry import camera_to_view, fingertip, palm_normal
from .landmarks import FrameInput, HandFrame
from .quality import Hint, QualityConfig, check_frame, check_hand
from .view_state import Command
from .vocab import MODES, NONE, POSE_TO_MODE


@dataclass
class EngineConfig:
    dominant: str = "Right"
    orbit_gain: float = 70.0  # degrees per palm-width of hand travel
    zoom_gain: float = 0.9  # log-zoom per palm-width of vertical travel
    section_gain: float = 0.35  # model units per palm-width
    deadzone: float = 0.01  # palm-widths per frame ignored as tremor
    still_speed: float = 0.6  # palm-widths / s counted as "still"
    lost_grace_s: float = 0.3
    release_grace_s: float = 0.15
    reset_hold_s: float = 1.0
    pin_hold_s: float = 0.8
    lock_lookback_s: float = 0.15
    # interaction box (selfie-view image coords): the probe cursor maps this box
    # to the full viewport, so a small comfortable hand motion covers the screen
    box: tuple = (0.2, 0.15, 0.8, 0.75)


@dataclass
class _HandTrack:
    pos: OneEuroFilter = field(default_factory=lambda: OneEuroFilter(1.2, 0.3))
    size: OneEuroFilter = field(default_factory=lambda: OneEuroFilter(0.8, 0.0))
    last_pos: np.ndarray | None = None
    last_t: float | None = None
    speed: float = 0.0

    def update(self, h: HandFrame, t: float) -> tuple[np.ndarray, float, np.ndarray]:
        size = float(self.size(h.palm_size, t))
        pos = self.pos(h.palm_center[:2], t)
        delta = np.zeros(2) if self.last_pos is None else (pos - self.last_pos) / size
        if self.last_t is not None and t > self.last_t:
            self.speed = 0.7 * self.speed + 0.3 * float(np.linalg.norm(delta)) / (t - self.last_t)
        self.last_pos, self.last_t = pos, t
        return pos, size, delta


class IntentEngine:
    def __init__(self, cfg: EngineConfig | None = None, quality: QualityConfig | None = None):
        self.cfg = cfg or EngineConfig()
        self.qcfg = quality or QualityConfig()
        self.reset()

    def reset(self) -> None:
        self.state = "no_hand"
        self.mode: str | None = None
        self.arming: str | None = None
        self.arm_t = 0.0
        self.release_t = 0.0
        self.lost_t = 0.0
        self.reset_t = 0.0
        self.reset_cooldown_until = -1.0
        self.t_prev: float | None = None
        self.tracks: dict[str, _HandTrack] = {}
        self.toast: str | None = None
        self._mode_reset()

    def _mode_reset(self) -> None:
        self.normal_f = OneEuroFilter(0.8, 0.4)
        self.tip_f = OneEuroFilter(1.0, 0.8)
        self.swipe = SwipeDetector()
        self.plane_hist: deque = deque(maxlen=30)
        self.start_pos: np.ndarray | None = None
        self.start_size = 1.0
        self.still_t = 0.0
        self.pinned = False
        self.prev_pos: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _control_hand(self, f: FrameInput) -> HandFrame | None:
        if not f.hands:
            return None
        return f.hand(self.cfg.dominant) or f.hands[0]

    def _end_mode(self, t: float, out: list[Command]) -> None:
        if self.mode == "section":
            # lock from slightly before the release: the release frames are the noisiest
            lock = None
            for ts, n, off in self.plane_hist:
                if ts <= t - self.cfg.lock_lookback_s:
                    lock = (n, off)
            if lock is None and self.plane_hist:
                lock = self.plane_hist[0][1:]
            if lock is not None:
                out.append(Command("section_lock", {"normal_view": lock[0].tolist(), "offset": lock[1]}))
        elif self.mode == "probe":
            out.append(Command("probe_clear"))
        self.mode = None
        self.state = "idle"
        self._mode_reset()

    def update(self, f: FrameInput, poses: dict[str, tuple[str, float]]) -> tuple[list[Command], dict]:
        t = f.timestamp
        dt = 0.0 if self.t_prev is None else max(0.0, min(t - self.t_prev, 0.2))
        self.t_prev = t
        out: list[Command] = []
        self.toast = None
        hints: list[Hint] = check_frame(f, self.qcfg)

        # forget tracks of hands that left
        for k in list(self.tracks):
            if f.hand(k) is None:
                del self.tracks[k]
        for h in f.hands:
            self.tracks.setdefault(h.handedness, _HandTrack()).update(h, t)

        ctrl = self._control_hand(f)
        if ctrl is None:
            self.lost_t += dt
            if self.mode and self.lost_t > self.cfg.lost_grace_s:
                self._end_mode(t, out)
            if not self.mode:
                self.state = "no_hand"
                self.arming = None
            return out, self._hud(None, NONE, 0.0, hints, poses)
        self.lost_t = 0.0
        track = self.tracks[ctrl.handedness]
        pose, conf = poses.get(ctrl.handedness, (NONE, 0.0))
        hints += check_hand(ctrl, self.qcfg, track.speed)

        if any(h.severity >= 2 for h in hints):
            if self.mode:
                self._end_mode(t, out)
            self.state, self.arming = "paused", None
            return out, self._hud(ctrl, pose, conf, hints, poses)

        # ---- two open palms = reset (only when no mode is running) ----
        both_open = len(f.hands) >= 2 and all(poses.get(h.handedness, (NONE,))[0] == "open_palm" for h in f.hands)
        if both_open and not self.mode and t > self.reset_cooldown_until:
            self.arming = None
            self.reset_t += dt
            if self.reset_t >= self.cfg.reset_hold_s:
                out.append(Command("reset"))
                self.toast = "View reset"
                self.reset_t = 0.0
                self.reset_cooldown_until = t + 1.5
            self.state = "idle"
            return out, self._hud(ctrl, pose, conf, hints, poses)
        self.reset_t = 0.0

        # ---- active mode ----
        if self.mode:
            if pose != MODES[self.mode].pose:
                self.release_t += dt
                if self.release_t > self.cfg.release_grace_s:
                    self._end_mode(t, out)
            else:
                self.release_t = 0.0
            if self.mode:
                self._run_mode(ctrl, track, t, dt, out)
                return out, self._hud(ctrl, pose, conf, hints, poses)

        # ---- arming ----
        mode = POSE_TO_MODE.get(pose)
        if mode is None:
            self.state, self.arming, self.arm_t = "idle", None, 0.0
            return out, self._hud(ctrl, pose, conf, hints, poses)
        if mode != self.arming:
            self.arming, self.arm_t = mode, 0.0
        m = MODES[mode]
        if m.still and track.speed > self.cfg.still_speed:
            self.arm_t = 0.0  # moving: start over, so passing hands never arm a hold
        else:
            self.arm_t += dt
        self.state = "arming"
        if self.arm_t >= m.arm_s:
            self.mode, self.arming, self.arm_t, self.release_t = mode, None, 0.0, 0.0
            self.state = "active"
            self._mode_reset()
            self.start_pos, self.start_size = track.last_pos.copy(), track.size.value
            out.append(Command("begin", {"mode": mode}))
            if mode == "snapshot":
                out.append(Command("snapshot"))
                self.toast = "Snapshot saved"
            self._run_mode(ctrl, track, t, dt, out)
        return out, self._hud(ctrl, pose, conf, hints, poses)

    # ------------------------------------------------------------------
    def _run_mode(self, h: HandFrame, track: _HandTrack, t: float, dt: float, out: list[Command]) -> None:
        cfg = self.cfg
        size = track.size.value
        cur = track.last_pos
        # per-frame displacement in palm widths (from the filtered rigid-palm centre)
        delta = np.zeros(2) if self.prev_pos is None else (cur - self.prev_pos) / size
        self.prev_pos = cur.copy()
        if np.linalg.norm(delta) < cfg.deadzone:
            delta = np.zeros(2)

        if self.mode == "orbit":
            # mild acceleration: slow hands are precise, fast hands cover ground
            gain = cfg.orbit_gain * (0.6 + 0.4 * min(track.speed / 2.0, 2.0))
            if delta.any():
                out.append(Command("orbit", {"dx": float(delta[0] * gain), "dy": float(delta[1] * gain)}))
        elif self.mode == "zoom":
            if delta[1]:
                out.append(Command("zoom", {"factor": float(np.exp(delta[1] * cfg.zoom_gain))}))
        elif self.mode == "section":
            n = camera_to_view(self.normal_f(palm_normal(h), t))
            n /= np.linalg.norm(n) + 1e-9
            disp = (cur - self.start_pos) / size
            # slide the plane along its own normal: screen-plane motion for the
            # in-plane part of the normal, push/pull (hand scale) for the depth part
            push = np.log(size / self.start_size)
            offset = cfg.section_gain * (disp[0] * n[0] - disp[1] * n[1] + 2.0 * push * n[2])
            self.plane_hist.append((t, n.copy(), float(offset)))
            out.append(Command("section", {"normal_view": n.tolist(), "offset": float(offset)}))
        elif self.mode == "probe":
            x0, y0, x1, y1 = cfg.box
            tip = self.tip_f(fingertip(h), t)
            xy = np.clip([(tip[0] - x0) / (x1 - x0), (tip[1] - y0) / (y1 - y0)], 0.0, 1.0)
            out.append(Command("probe", {"xy": xy.tolist()}))
            if track.speed < cfg.still_speed:
                self.still_t += dt
                if self.still_t >= cfg.pin_hold_s and not self.pinned:
                    out.append(Command("probe_pin", {"xy": xy.tolist()}))
                    self.toast = "Value pinned"
                    self.pinned = True
            else:
                self.still_t, self.pinned = 0.0, False
        elif self.mode == "field":
            s = self.swipe.update(t, cur, size)
            if s:
                out.append(Command("field", {"step": s}))
                self.toast = "Next field" if s > 0 else "Previous field"

    # ------------------------------------------------------------------
    def progress(self) -> tuple[str | None, float]:
        if self.reset_t > 0:
            return "Reset", self.reset_t / self.cfg.reset_hold_s
        if self.arming:
            m = MODES[self.arming]
            return m.label, min(1.0, self.arm_t / m.arm_s)
        if self.mode == "probe" and self.still_t > 0 and not self.pinned:
            return "Pin value", min(1.0, self.still_t / self.cfg.pin_hold_s)
        return None, 0.0

    def _hud(self, h: HandFrame | None, pose: str, conf: float, hints: list[Hint], poses) -> dict:
        label, prog = self.progress()
        mode = MODES.get(self.mode) if self.mode else None
        hints = sorted(hints, key=lambda x: -x.severity)
        if self.state == "no_hand":
            tip = "Raise your hand so the camera can see it"
        elif mode:
            tip = mode.hint + ". Relax your hand to stop."
        elif self.arming:
            tip = f"Keep holding for {MODES[self.arming].label}…"
        else:
            tip = "Hold a pose to start: ✊ rotate · 🤏 zoom · ✋ cut · ☝️ probe · ✌️ field · 👍 snapshot"
        return {
            "state": self.state,
            "mode": self.mode,
            "mode_label": mode.label if mode else None,
            "pose": pose,
            "confidence": round(float(conf), 3),
            "progress_label": label,
            "progress": round(float(prog), 3),
            "hints": [x.text for x in hints],
            "tip": tip,
            "toast": self.toast,
            "hands": [
                {"handedness": k, "pose": v[0], "confidence": round(float(v[1]), 3)} for k, v in poses.items()
            ],
        }
