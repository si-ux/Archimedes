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
* One hand does the precise work (probe, section, field); the big moves (zoom,
  pan, turn, reset) need the same pose on BOTH hands, as in VR. A lone pinch
  means nothing, so pinch and point can never be confused.
* Releasing the pose is the clutch. Rotate, relax, move back, fist again and
  keep rotating, like lifting a mouse.
* Motion is read from the rigid palm (wrist + knuckles), not fingertips, so
  the act of opening the hand does not kick the view.
* When a section cut ends, the plane is taken from ~150 ms *before* the
  release, because the frames during a pose change are the noisiest.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .filters import OneEuroFilter
from .geometry import camera_to_view, fingertip, palm_normal
from .landmarks import FrameInput, HandFrame
from .quality import Hint, QualityConfig, check_frame, check_hand
from .view_state import Command
from .vocab import MODES, NONE, pose_to_mode, two_hand_modes


@dataclass
class EngineConfig:
    dominant: str = "Right"
    orbit_gain: float = 90.0  # degrees per palm-width of hand travel
    zoom_gain: float = 1.8  # log-zoom per palm-width (kept for the modelling engine's callers)
    grab_zoom_gain: float = 1.0  # exponent on the change of distance between two pinching hands (1 = the model scales exactly with the hands, as in VR)
    pan_gain: float = 2.2  # model half-sizes per image-width of travel of two pinching hands
    turn_gain: float = 260.0  # degrees per image-width of travel of two fists
    twist_gain: float = 1.0  # degrees of spin per degree of twist of the line between two fists
    section_gain: float = 0.5  # model units per palm-width
    section_step: float = 0.02  # the cut moves in steps of this (model units), with hysteresis
    field_step: float = 0.7  # palm widths of sideways travel per field
    deadzone: float = 0.0025  # palm-widths per frame ignored as tremor
    still_speed: float = 0.6  # palm-widths / s counted as "still"
    lost_grace_s: float = 0.3
    release_grace_s: float = 0.25
    pin_hold_s: float = 0.8
    lock_lookback_s: float = 0.15
    # interaction box (selfie-view image coords): the probe cursor maps this box
    # to the full viewport, so a small comfortable hand motion covers the screen
    box: tuple = (0.2, 0.15, 0.8, 0.75)


@dataclass
class _HandTrack:
    pos: OneEuroFilter = field(default_factory=lambda: OneEuroFilter(1.6, 9.0))
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
    def __init__(self, cfg: EngineConfig | None = None, quality: QualityConfig | None = None,
                 modes: dict | None = None):
        self.cfg = cfg or EngineConfig()
        self.qcfg = quality or QualityConfig()
        self.modes = modes or MODES
        self.pose_to_mode = pose_to_mode(self.modes)
        self.two_to_mode = two_hand_modes(self.modes)
        self.reset()

    def reset(self) -> None:
        self.state = "no_hand"
        self.mode: str | None = None
        self.arming: str | None = None
        self.arm_t = 0.0
        self.arm_two: tuple | None = None
        self.release_t = 0.0
        self.lost_t = 0.0
        self.t_prev: float | None = None
        self.tracks: dict[str, _HandTrack] = {}
        self.toast: str | None = None
        self.fired: str | None = None  # a one-shot mode that fired; ignored until its pose goes away
        self._f: FrameInput | None = None
        self._poses: dict = {}
        self._mode_reset()

    def _mode_reset(self) -> None:
        self.normal_f = OneEuroFilter(1.2, 0.8)
        self.offset_f = OneEuroFilter(0.8, 1.0)
        self.tip_f = OneEuroFilter(1.8, 14.0)
        self.tip2_f = OneEuroFilter(1.8, 14.0)
        self.plane_hist: deque = deque(maxlen=30)
        self.start_pos: np.ndarray | None = None
        self.start_size = 1.0
        self.still_t = 0.0
        self.pinned = False
        self.prev_pos: np.ndarray | None = None
        self.click_down = False
        self.two_prev: tuple | None = None
        self.sec_offset = 0.0
        self.field_anchor: float | None = None

    # ------------------------------------------------------------------
    def _control_hand(self, f: FrameInput) -> HandFrame | None:
        if not f.hands:
            return None
        return f.hand(self.cfg.dominant) or f.hands[0]

    def _pose(self, h: HandFrame | None) -> str:
        return NONE if h is None else self._poses.get(h.handedness, (NONE, 0.0))[0]

    def _end_mode(self, t: float, out: list[Command]) -> None:
        if self.mode == "section":
            # lock from slightly before the release: the release frames are the noisiest
            lock = None
            for ts, data in self.plane_hist:
                if ts <= t - self.cfg.lock_lookback_s:
                    lock = data
            if lock is None and self.plane_hist:
                lock = self.plane_hist[0][1]
            if lock is not None:
                out.append(Command("section_lock", dict(lock)))
        elif self.mode in ("probe", "dual"):
            out.append(Command("probe_clear"))
        m = self.modes.get(self.mode) if self.mode else None
        if m is not None and m.fire:
            self.fired = self.mode
        self.mode = None
        self.state = "idle"
        self._mode_reset()

    def _wanted(self, f: FrameInput, ctrl: HandFrame) -> str | None:
        """The mode the hands are asking for right now (before any arming)."""
        if len(f.hands) >= 2:
            p = [self._pose(h) for h in f.hands[:2]]
            if p[0] == p[1] and p[0] in self.two_to_mode:
                return self.two_to_mode[p[0]]
        pose = self._pose(ctrl)
        mode = self.pose_to_mode.get(pose)
        if mode is None:
            return None
        if len(f.hands) >= 2 and self.modes[mode].still:
            # still one-hand modes (section, snapshot/solve) need the other hand away or relaxed, so two raised
            # hands are never read as one of them
            other = [h for h in f.hands if h is not ctrl]
            if any(self._pose(o) not in (NONE, "fist") for o in other):
                return None
        return mode

    def update(self, f: FrameInput, poses: dict[str, tuple[str, float]]) -> tuple[list[Command], dict]:
        t = f.timestamp
        dt = 0.0 if self.t_prev is None else max(0.0, min(t - self.t_prev, 0.2))
        self.t_prev = t
        out: list[Command] = []
        self.toast = None
        hints: list[Hint] = check_frame(f, self.qcfg)
        self._f, self._poses = f, poses

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
                self.state, self.arming, self.fired = "no_hand", None, None
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

        wanted = self._wanted(f, ctrl)
        if self.fired is not None:              # a one-shot mode fired: wait until its pose goes away
            if wanted == self.fired:
                self.state = "idle"
                return out, self._hud(ctrl, pose, conf, hints, poses)
            self.fired = None

        # ---- active mode: sticky, ends only when its gesture has been gone for a moment ----
        if self.mode:
            m = self.modes[self.mode]
            keep = wanted == self.mode or (m.two and len(f.hands) >= 2 and
                                           sum(self._pose(h) == m.pose for h in f.hands[:2]) >= 1 and wanted is None)
            if not m.two and self._pose(ctrl) == m.pose:
                keep = True                     # a second hand appearing never cancels a one-hand mode...
            if not m.two and wanted is not None and self.modes[wanted].two:
                keep = False                    # ...unless it makes the two-hand version (☝️ -> ☝️☝️, ✊ -> ✊✊)
                self.release_t = self.cfg.release_grace_s
            if keep:
                self.release_t = 0.0
            else:
                self.release_t += dt
                if self.release_t > self.cfg.release_grace_s:
                    self._end_mode(t, out)
            if self.mode:
                if keep:
                    self._run_mode(ctrl, track, t, dt, out)
                else:
                    self.two_prev = None        # frozen while the pose flickers: no jump when it comes back
                return out, self._hud(ctrl, pose, conf, hints, poses)

        # ---- arming ----
        if wanted is None:
            self.state, self.arming, self.arm_t = "idle", None, 0.0
            return out, self._hud(ctrl, pose, conf, hints, poses)
        if wanted != self.arming:
            self.arming, self.arm_t = wanted, 0.0
            # two-hand moves count from the moment the hands took the pose, so the arming time loses no motion
            self.arm_two = self._two(f) if self.modes[wanted].two and len(f.hands) >= 2 else None
        m = self.modes[wanted]
        moving = max(self.tracks[h.handedness].speed for h in (f.hands[:2] if m.two else [ctrl]))
        if m.still and moving > self.cfg.still_speed:
            # moving: the hold drains (twice as fast as it fills), so passing hands never arm it but one shaky
            # frame of a held hand does not start it over
            self.arm_t = max(0.0, self.arm_t - 2 * dt)
        else:
            self.arm_t += dt
        self.state = "arming"
        if self.arm_t >= m.arm_s:
            self.mode, self.arming, self.arm_t, self.release_t = wanted, None, 0.0, 0.0
            self.state = "active"
            self._mode_reset()
            self.start_pos, self.start_size = track.last_pos.copy(), track.size.value
            if m.two and self.arm_two is not None:
                pa, pb, dist, ang = self.arm_two
                self.two_prev = (dist, (pa + pb) / 2, ang)
            out.append(Command("begin", {"mode": wanted}))
            if m.fire:
                out.append(Command(m.fire))
                self.toast = m.toast
                self.fired = wanted
                self.mode, self.state = None, "idle"
                return out, self._hud(ctrl, pose, conf, hints, poses)
            self._run_mode(ctrl, track, t, dt, out)
        return out, self._hud(ctrl, pose, conf, hints, poses)

    # ------------------------------------------------------------------
    def _cursor(self, h: HandFrame, filt: OneEuroFilter, t: float) -> np.ndarray:
        x0, y0, x1, y1 = self.cfg.box
        tip = filt(fingertip(h), t)
        return np.clip([(tip[0] - x0) / (x1 - x0), (tip[1] - y0) / (y1 - y0)], 0.0, 1.0)

    def _two(self, f: FrameInput) -> tuple[np.ndarray, np.ndarray, float, float]:
        """(left-to-right) palm positions, their distance and the angle of the line between them."""
        a, b = sorted(f.hands[:2], key=lambda h: h.palm_center[0])
        pa, pb = self.tracks[a.handedness].last_pos, self.tracks[b.handedness].last_pos
        d = pb - pa
        return pa, pb, float(np.linalg.norm(d)) + 1e-6, float(np.degrees(np.arctan2(d[1], d[0])))

    def _run_mode(self, h: HandFrame, track: _HandTrack, t: float, dt: float, out: list[Command]) -> None:
        cfg = self.cfg
        size = track.size.value
        cur = track.last_pos
        # per-frame displacement in palm widths (from the filtered rigid-palm centre)
        delta = np.zeros(2) if self.prev_pos is None else (cur - self.prev_pos) / size
        self.prev_pos = cur.copy()
        if np.linalg.norm(delta) < cfg.deadzone:
            delta = np.zeros(2)
        f = self._f

        if self.mode == "orbit":
            # mild acceleration: slow hands are precise, fast hands cover ground
            gain = cfg.orbit_gain * (0.6 + 0.4 * min(track.speed / 2.0, 2.0))
            if delta.any():
                out.append(Command("orbit", {"dx": float(delta[0] * gain), "dy": float(delta[1] * gain)}))

        elif self.mode in ("grab", "turn"):
            if len(f.hands) < 2:
                self.two_prev = None
                return
            pa, pb, dist, ang = self._two(f)
            mid = (pa + pb) / 2
            if self.two_prev is not None:
                d0, m0, a0 = self.two_prev
                dm = mid - m0
                if self.mode == "grab":
                    dl = np.log(dist / d0)
                    if abs(dl) > 0.002:   # spread (dist grows) -> zoom in -> smaller camera distance
                        out.append(Command("zoom", {"factor": float(np.exp(-dl * cfg.grab_zoom_gain))}))
                    if np.linalg.norm(dm) > 0.0012:
                        out.append(Command("pan", {"dx": float(dm[0] * cfg.pan_gain), "dy": float(dm[1] * cfg.pan_gain)}))
                else:
                    da = (ang - a0 + 180) % 360 - 180
                    dx = dm[0] * cfg.turn_gain + (da * cfg.twist_gain if abs(da) > 0.15 else 0.0)
                    dy = dm[1] * cfg.turn_gain
                    if abs(dx) + abs(dy) > 0.05:
                        out.append(Command("orbit", {"dx": float(dx), "dy": float(dy)}))
            self.two_prev = (dist, mid, ang)

        elif self.mode == "section":
            n = camera_to_view(self.normal_f(palm_normal(h), t))
            n /= np.linalg.norm(n) + 1e-9
            disp = (cur - self.start_pos) / size
            # slide the plane along its axis: screen-plane motion for the in-plane part, push/pull (hand scale) for
            # the depth part; the viewer snaps the axis and locks it for the whole gesture
            push = np.log(size / self.start_size)
            move = np.array([disp[0], -disp[1], 2.0 * push])            # view coords: x right, y up, z to viewer
            raw = float(self.offset_f(cfg.section_gain * float(np.dot(move, n)), t))
            step = cfg.section_step
            if abs(raw - self.sec_offset) > 0.6 * step:                  # steps with hysteresis: no flicker
                self.sec_offset = round(raw / step) * step
            scale = self.sec_offset / raw if abs(raw) > 1e-9 else 0.0
            data = {"normal_view": n.tolist(), "offset": float(self.sec_offset), "disp": (move * scale).tolist(),
                    "gain": cfg.section_gain}
            self.plane_hist.append((t, data))
            out.append(Command("section", data))

        elif self.mode == "probe":
            xy = self._cursor(h, self.tip_f, t)
            out.append(Command("probe", {"xy": xy.tolist()}))
            # the other hand pinching = click: pin the value under the cursor right now
            other = [o for o in f.hands if o is not h]
            pinching = any(self._pose(o) == "pinch" for o in other)
            if pinching and not self.click_down:
                out.append(Command("probe_pin", {"xy": xy.tolist()}))
                self.toast = "Value pinned"
                self.pinned = True
            self.click_down = pinching
            if track.speed < cfg.still_speed and not pinching:
                self.still_t += dt
                if self.still_t >= cfg.pin_hold_s and not self.pinned:
                    out.append(Command("probe_pin", {"xy": xy.tolist()}))
                    self.toast = "Value pinned"
                    self.pinned = True
            else:
                self.still_t, self.pinned = 0.0, False

        elif self.mode == "dual":
            # two cursors: the control hand is A, the other hand is B
            other = next((o for o in f.hands if o is not h), None)
            out.append(Command("probe", {"xy": self._cursor(h, self.tip_f, t).tolist()}))
            if other is not None:
                out.append(Command("probe2", {"xy": self._cursor(other, self.tip2_f, t).tolist()}))

        elif self.mode == "field":
            # a dial, not a swipe: every field_step palm widths of sideways travel is one field
            x = (cur[0] - self.start_pos[0]) / size
            if self.field_anchor is None:
                self.field_anchor = 0.0
            k = int((x - self.field_anchor) / cfg.field_step)          # a full step each way, no half-step flicker
            if k != 0:
                out.append(Command("field", {"step": int(np.sign(k))}))
                self.field_anchor += np.sign(k) * cfg.field_step
                self.toast = "Next field" if k > 0 else "Previous field"

    # ------------------------------------------------------------------
    def progress(self) -> tuple[str | None, float]:
        if self.arming:
            m = self.modes[self.arming]
            return m.label, min(1.0, self.arm_t / m.arm_s)
        if self.mode == "probe" and self.still_t > 0 and not self.pinned:
            return "Pin value", min(1.0, self.still_t / self.cfg.pin_hold_s)
        return None, 0.0

    def _hud(self, h: HandFrame | None, pose: str, conf: float, hints: list[Hint], poses) -> dict:
        label, prog = self.progress()
        mode = self.modes.get(self.mode) if self.mode else None
        hints = sorted(hints, key=lambda x: -x.severity)
        if self.state == "no_hand":
            tip = "Raise your hand so the camera can see it"
        elif mode:
            tip = mode.hint + ". Relax your hand to stop."
        elif self.arming:
            tip = f"Keep holding for {self.modes[self.arming].label}…"
        else:
            one = [m for m in self.modes.values() if not m.two]
            two = [m for m in self.modes.values() if m.two]
            tip = ("One hand: " + " · ".join(f"{m.icon} {m.label.lower()}" for m in one)
                   + "  |  Two hands: " + " · ".join(f"{m.icon} {m.label.lower()}" for m in two))
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
