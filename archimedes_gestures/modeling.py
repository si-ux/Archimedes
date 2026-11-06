"""Gestures for building the model: step-by-step wizards with a confirmation at every step.

Every choice is made in ONE clear way and confirmed before the next step starts, so nothing happens by accident:

    pick from a menu   hold up that many fingers (two hands for 6+), keep them still ~0.7 s
    pick a point       point at it (the cursor snaps to the chosen increment), then hold still ~0.9 s,
                       or pinch with your other hand to confirm at once
    size a shape       pinch with both hands and pull them apart: the size follows the distance between them,
                       snapped to the increment; hold both hands still ~1 s to lock it
    accept             👍 thumbs-up, held ~0.8 s
    go back a step     ✊✊ two fists, held ~0.6 s
    look around        ✊ one fist dragged = rotate · 🤏🤏 both hands pinched (in pointing / menu steps) = zoom

MODEL wizard
    1 plane      1 = plan (XY): draw a column's cross-section · 2 = side (YZ): draw a beam's cross-section
                 3 = front (XZ): draw the member's centre line
    2 shape      1 = rectangle · 2 = circle                      (plane 1 / 2)
    3 size       two-hand pull: rectangle = width × height, circle = diameter (section increment)
    4 extrude    two-hand pull: length (length increment)
    (plane 3)    start point -> end point on the grid (length increment; horizontal = beam, vertical = column),
                 then shape and size of the section
    5 confirm    👍 creates the member

LOADS wizard
    1 what       1 fixed · 2 pinned · 3 roller · 4 point load · 5 uniform load · 6 (5 + 1) trapezoidal load
                 (👍 held still = solve)
    2 start      point at the member, confirm
    3 end        (uniform / trapezoidal) point at the other end, confirm
    4 direction  point your finger in the load direction (snaps to ±x, ±y, ±z on screen), confirm
    5 value      the number card (fingers or keyboard), as before

Positions are emitted as viewport coordinates; the renderer maps them onto the member or the drawing plane (it
knows the camera), snaps them to the increments and keeps the confirmed picks; directions are emitted in screen
space and snapped to a world axis there.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .counting import FingerTracker
from .filters import OneEuroFilter
from .geometry import fingertip
from .intent import EngineConfig, _HandTrack
from .landmarks import INDEX, THUMB, FrameInput, HandFrame
from .quality import QualityConfig, check_frame, check_hand
from .view_state import Command
from .vocab import NONE

# How every number prompt (dimensions, loads, increments) is answered by hand
NUMBER_ROWS = [
    {"icon": "🖐", "label": "Type a digit", "hint": "Hold up 0–5 fingers (two hands for 6–9) still ~0.5 s; change the count to repeat a digit. A question with fixed choices is answered by the count alone"},
    {"icon": "🙌", "label": "Accept the value", "hint": "Both open palms, hold ~1 s (or Enter). Before typing anything this takes the default"},
    {"icon": "🤏↕", "label": "Nudge by the increment", "hint": "Pinch and move up / down — one increment per step"},
    {"icon": "🤏", "label": "Delete a digit", "hint": "Pinch and hold still ~0.5 s; keep holding to delete more (or Backspace)"},
    {"icon": "✊✊", "label": "Cancel", "hint": "Two fists, hold 1 s (or Esc)"},
]

COMMON_ROWS = [
    {"icon": "👍", "label": "Accept", "hint": "Thumbs-up held ~0.8 s"},
    {"icon": "✊✊", "label": "Back one step", "hint": "Two fists held ~0.6 s"},
    {"icon": "✊", "label": "Rotate the view", "hint": "One fist, drag"},
    {"icon": "🤏🤏", "label": "Zoom the view", "hint": "Both hands pinched, spread / together (in menu and pointing steps)"},
]

CHEAT = {
    "model": [
        {"icon": "1️⃣2️⃣3️⃣", "label": "Choose the plane", "hint": "1 plan (column section) · 2 side (beam section) · 3 front (draw the member line)"},
        {"icon": "1️⃣2️⃣", "label": "Choose the shape", "hint": "1 rectangle · 2 circle"},
        {"icon": "🤏↔🤏", "label": "Size it", "hint": "Pinch with both hands and pull apart; the size snaps to your increment. Hold still to lock"},
        {"icon": "🤏↔🤏", "label": "Extrude", "hint": "Pinch and pull apart again for the length. Hold still to lock"},
        {"icon": "☝️", "label": "Draw a line (front plane)", "hint": "Point at the start, hold still (or pinch the other hand), then the end"},
        *COMMON_ROWS,
        *NUMBER_ROWS,
    ],
    "loads": [
        {"icon": "1️⃣…6️⃣", "label": "Choose what to add", "hint": "1 fixed · 2 pinned · 3 roller · 4 point load · 5 uniform · 6 trapezoidal"},
        {"icon": "☝️", "label": "Start / end point", "hint": "Point at the member; hold still ~0.9 s or pinch with the other hand to confirm"},
        {"icon": "☝️➡", "label": "Direction", "hint": "Point the finger the way the load acts; it snaps to an axis. Confirm the same way"},
        {"icon": "🔢", "label": "Value", "hint": "The number card: fingers, or type it"},
        {"icon": "👍", "label": "Solve", "hint": "Thumbs-up held still ~1.2 s on the first step"},
        *COMMON_ROWS,
        *NUMBER_ROWS,
    ],
}

PLANES = {1: ("plan", "Plan (XY) · column section"), 2: ("side", "Side (YZ) · beam section"),
          3: ("front", "Front (XZ) · member line")}
SHAPES = {1: ("rect", "Rectangle"), 2: ("circle", "Circle")}
LOAD_MENU = {1: ("support_fixed", "Fixed support"), 2: ("support_pinned", "Pinned support"),
             3: ("support_roller", "Roller support"), 4: ("point_load", "Point load"),
             5: ("uniform_load", "Uniform load"), 6: ("trapezoidal_load", "Trapezoidal load")}

STEP_TITLES = {
    "plane": "Choose the drawing plane", "shape": "Choose the shape", "draw_rect": "Size the rectangle",
    "draw_circle": "Size the circle", "extrude": "Extrude: pull for the length", "line_start": "Line: start point",
    "line_end": "Line: end point", "confirm": "Create this member?",
    "menu": "What do you want to add?", "start": "Pick the position", "end": "Pick the end of the load",
    "dir": "Pick the load direction",
}
STEP_HELP = {
    "plane": "Hold up 1, 2 or 3 fingers, keep them still",
    "shape": "Hold up 1 (rectangle) or 2 (circle) fingers",
    "draw_rect": "🤏🤏 Pinch with both hands, pull apart: left-right = width, up-down = height. Hold still to lock",
    "draw_circle": "🤏🤏 Pinch with both hands, pull apart: the distance is the diameter. Hold still to lock",
    "extrude": "🤏🤏 Pinch with both hands and pull apart for the length. Hold still to lock",
    "line_start": "☝️ Point at the start, then hold still or pinch with the other hand",
    "line_end": "☝️ Point at the end (it snaps horizontal or vertical), then hold still or pinch",
    "confirm": "👍 Thumbs-up to create it · ✊✊ two fists to go back",
    "menu": "Hold up fingers: 1 fixed · 2 pinned · 3 roller · 4 point · 5 uniform · 6 (5+1) trapezoidal · 👍 solve",
    "start": "☝️ Point at the member, then hold still or pinch with the other hand",
    "end": "☝️ Point at the other end of the load, then hold still or pinch",
    "dir": "☝️ Point your finger the way the load pushes, then hold still or pinch",
}


@dataclass
class ModelingConfig:
    menu_hold_s: float = 0.7
    dwell_s: float = 0.9  # pointer held still this long = confirm
    dwell_speed: float = 0.35  # palm widths / s counted as still for a dwell
    size_hold_s: float = 1.0  # both hands still this long = lock the size
    back_hold_s: float = 0.6
    accept_hold_s: float = 0.8
    solve_hold_s: float = 1.2
    orbit_arm_s: float = 0.2
    section_mm_per_palm: float = 100.0  # hand separation -> section size
    vertical_gain: float = 2.0  # the camera sees less height than width: up-down separation counts double
    length_mm_per_palm: float = 1000.0  # hand separation -> length
    section_step: float = 25.0  # the increments (set from the session's setup)
    length_step: float = 100.0
    min_section: float = 50.0
    min_length: float = 500.0


@dataclass
class CountPicker:
    """A finger-count menu: the count must be a valid choice, held still, and must change after each pick."""
    hold_s: float = 0.7
    fingers: FingerTracker = field(default_factory=FingerTracker)
    hist: deque = field(default_factory=lambda: deque(maxlen=7))
    t: float = 0.0
    cand: int | None = None
    blocked: int | None = -1  # -1 = whatever is shown first is blocked if a hand is already up

    def reset(self, hands_up: bool = True) -> None:
        self.hist.clear()
        self.t, self.cand = 0.0, None
        self.blocked = -1 if hands_up else None

    def update(self, hands: list[HandFrame], dt: float, still: bool, choices) -> tuple[int | None, int | None, float]:
        """Returns (picked, live count, progress)."""
        if not hands:
            self.hist.clear()
            self.t, self.cand, self.blocked = 0.0, None, None
            return None, None, 0.0
        up = self.fingers.update(hands)
        n = int(sum(sum(v) for v in up.values()))
        self.hist.append(n)
        live = int(np.bincount(list(self.hist)).argmax())
        if self.blocked == -1:
            self.blocked = live
        if self.blocked is not None:
            if live == self.blocked:
                return None, live, 0.0
            self.blocked = None
        if live != self.cand:
            self.cand, self.t = live, 0.0
        if live in choices and still:
            self.t += dt
        else:
            self.t = 0.0
        if live in choices and self.t >= self.hold_s:
            self.blocked, self.t = live, 0.0
            return live, live, 1.0
        return None, live, min(1.0, self.t / self.hold_s) if live in choices else 0.0


class ModelingEngine:
    def __init__(self, cfg: EngineConfig | None = None, mcfg: ModelingConfig | None = None,
                 quality: QualityConfig | None = None):
        self.cfg = cfg or EngineConfig()
        self.m = mcfg or ModelingConfig()
        self.qcfg = quality or QualityConfig()
        self.phase = "loads"
        self.reset()

    # ---- state --------------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.t_prev: float | None = None
        self.tracks: dict[str, _HandTrack] = {}
        self.tip_f = OneEuroFilter(1.8, 14.0)
        self.dir_f = OneEuroFilter(1.5, 4.0)
        self.pinch_f: dict[str, OneEuroFilter] = {}
        self.picker = CountPicker(self.m.menu_hold_s)
        self.step = "plane" if self.phase == "model" else "menu"
        self.data: dict = {}
        self.history: list[tuple[str, dict]] = []
        self._step_reset()
        self.toast: str | None = None
        self.label: str | None = None
        self.progress = 0.0
        self.orbit = False
        self.orbit_t = 0.0
        self.prev_pos: np.ndarray | None = None
        self.grab_prev: tuple | None = None
        self.back_t = 0.0
        self.back_latched = False

    def _step_reset(self) -> None:
        self.dwell_t = 0.0
        self.click_down = True  # a pinch already held when the step starts is not a click
        self.size_t = 0.0
        self.size_val: dict = {}
        self.size_hist: list = []
        self.size_anchor: np.ndarray | None = None
        self.need_release = True  # a sizing step starts with a fresh pinch, never with the one that locked the last
        self.accept_t = 0.0
        self.dir_vec: np.ndarray | None = None
        self.dir_still_t = 0.0
        self.picker.reset()

    def set_phase(self, phase: str) -> None:
        if phase != self.phase:
            self.phase = phase
            self.reset()

    def set_increments(self, section: float | None = None, length: float | None = None) -> None:
        if section:
            self.m.section_step = float(section)
        if length:
            self.m.length_step = float(length)

    def set_value(self, key: str, value) -> None:
        """Values the renderer measured (the line length and its orientation)."""
        self.data[key] = value

    def _go(self, step: str, out: list[Command]) -> None:
        self.history.append((self.step, dict(self.data)))
        self.step = step
        self._step_reset()
        out.append(Command("wiz_step", {"phase": self.phase, "step": step}))

    def _back(self, out: list[Command]) -> None:
        if self.history:
            self.step, self.data = self.history.pop()
            self.toast = "Back"
        else:
            self.data = {}
            self.toast = "Cancelled"
        self._step_reset()
        out.append(Command("wiz_step", {"phase": self.phase, "step": self.step, "back": True}))

    def _restart(self, out: list[Command]) -> None:
        self.step = "plane" if self.phase == "model" else "menu"
        self.data, self.history = {}, []
        self._step_reset()
        out.append(Command("wiz_step", {"phase": self.phase, "step": self.step, "done": True}))

    # ---- helpers --------------------------------------------------------------------------------------------
    def _to_view(self, xy) -> list[float]:
        x0, y0, x1, y1 = self.cfg.box
        return np.clip([(xy[0] - x0) / (x1 - x0), (xy[1] - y0) / (y1 - y0)], 0.0, 1.0).round(4).tolist()

    def _pinch_point(self, h: HandFrame) -> np.ndarray:
        """Where thumb and index meet, filtered: the size follows the hands without the jitter of single frames."""
        f = self.pinch_f.setdefault(h.handedness, OneEuroFilter(0.8, 0.8))
        return f((h.landmarks[THUMB[3], :2] + h.landmarks[INDEX[3], :2]) / 2, h.timestamp)

    def _pose(self, poses, h) -> str:
        return poses.get(h.handedness, (NONE, 0.0))[0] if h is not None else NONE

    @staticmethod
    def _snap(raw: float, step: float, prev: float | None, lo: float) -> float:
        """Snap to the increment, with hysteresis so the value does not flicker between two steps."""
        v = max(lo, round(raw / step) * step)
        if prev is not None and abs(raw - prev) < 0.75 * step:
            return prev
        return v

    # ---- main --------------------------------------------------------------------------------------------
    def update(self, f: FrameInput, poses: dict[str, tuple[str, float]]) -> tuple[list[Command], dict]:
        t = f.timestamp
        dt = 0.0 if self.t_prev is None else max(0.0, min(t - self.t_prev, 0.2))
        self.t_prev = t
        out: list[Command] = []
        self.toast = self.label = None
        self.progress = 0.0
        hints = check_frame(f, self.qcfg)
        for k in list(self.tracks):
            if f.hand(k) is None:
                del self.tracks[k]
        for h in f.hands:
            self.tracks.setdefault(h.handedness, _HandTrack()).update(h, t)
        if not f.hands:
            self.orbit, self.prev_pos, self.grab_prev = False, None, None
            self.back_t, self.back_latched = 0.0, False
            self.picker.update([], dt, True, ())
            return out, self._hud("no_hand", NONE, 0.0, hints, poses)

        ctrl = f.hand(self.cfg.dominant) or f.hands[0]
        other = next((h for h in f.hands if h is not ctrl), None)
        track = self.tracks[ctrl.handedness]
        pose, conf = poses.get(ctrl.handedness, (NONE, 0.0))
        hints += check_hand(ctrl, self.qcfg, track.speed)
        if any(h.severity >= 2 for h in hints):
            self.orbit = False
            return out, self._hud("paused", pose, conf, hints, poses)
        p2 = [self._pose(poses, h) for h in f.hands[:2]]
        speed = max(self.tracks[h.handedness].speed for h in f.hands)

        # ---- back: two fists held ----
        if len(f.hands) >= 2 and p2 == ["fist", "fist"]:
            if not self.back_latched:
                self.back_t += dt
                self.label, self.progress = "Back", min(1.0, self.back_t / self.m.back_hold_s)
                if self.back_t >= self.m.back_hold_s:
                    self._back(out)
                    self.back_latched = True
            return out, self._hud("arming", pose, conf, hints, poses)
        self.back_t, self.back_latched = 0.0, False

        sizing = self.step in ("draw_rect", "draw_circle", "extrude")
        # ---- look around: one fist dragged = orbit; two pinches (outside the sizing steps) = zoom ----
        if self._view_controls(f, ctrl, track, pose, p2, dt, out, allow_grab=not sizing):
            return out, self._hud("active", pose, conf, hints, poses)

        step = self.step
        if step in ("plane", "shape", "menu"):
            self._menu_step(f, ctrl, pose, speed, dt, out)
        elif step in ("line_start", "line_end", "start", "end"):
            self._point_step(ctrl, other, track, pose, poses, t, dt, out)
        elif step == "dir":
            self._dir_step(ctrl, other, track, pose, poses, t, dt, out)
        elif sizing:
            self._size_step(f, p2, dt, out)
        elif step == "confirm":
            self._confirm_step(pose, speed, dt, out)
        return out, self._hud("active" if self.progress >= 1 else "arming" if self.progress > 0 else "idle",
                              pose, conf, hints, poses)

    # ---- view controls inside the wizard ----------------------------------------------------------------
    def _view_controls(self, f, ctrl, track, pose, p2, dt, out, allow_grab: bool) -> bool:
        if allow_grab and len(f.hands) >= 2 and p2 == ["pinch", "pinch"]:
            a, b = sorted(f.hands[:2], key=lambda h: h.palm_center[0])
            pa, pb = self.tracks[a.handedness].last_pos, self.tracks[b.handedness].last_pos
            dist = float(np.linalg.norm(pb - pa)) + 1e-6
            if self.grab_prev is not None:
                dl = np.log(dist / self.grab_prev)
                if abs(dl) > 0.002:
                    out.append(Command("zoom", {"factor": float(np.exp(-dl * self.cfg.grab_zoom_gain))}))
            self.grab_prev = dist
            self.label = "Zoom"
            return True
        self.grab_prev = None
        if len(f.hands) == 1 and pose == "fist" and self.step not in ("plane", "shape", "menu") or \
                (len(f.hands) == 1 and pose == "fist" and track.speed > 0.6):
            # a dragged fist rotates; in menus a still fist is "0 fingers" and is simply ignored
            if not self.orbit:
                self.orbit_t += dt
                if self.orbit_t >= self.m.orbit_arm_s:
                    self.orbit, self.prev_pos = True, track.last_pos.copy()
                    out.append(Command("begin", {"mode": "orbit"}))
            if self.orbit:
                cur = track.last_pos
                delta = (cur - self.prev_pos) / track.size.value
                self.prev_pos = cur.copy()
                if np.linalg.norm(delta) > self.cfg.deadzone:
                    g = self.cfg.orbit_gain
                    out.append(Command("orbit", {"dx": float(delta[0] * g), "dy": float(delta[1] * g)}))
                self.label = "Rotate"
                return True
            return False
        self.orbit, self.orbit_t, self.prev_pos = False, 0.0, None
        return False

    # ---- menus ------------------------------------------------------------------------------------------
    def _menu_step(self, f, ctrl, pose, speed, dt, out) -> None:
        if self.step == "menu" and pose == "thumbs_up" and len(f.hands) == 1:
            self.accept_t = self.accept_t + dt if speed < self.cfg.still_speed else 0.0
            self.label, self.progress = "Solve", min(1.0, self.accept_t / self.m.solve_hold_s)
            if self.accept_t >= self.m.solve_hold_s:
                out.append(Command("solve"))
                self.toast = "Solving…"
                self.accept_t = -1e9
            return
        self.accept_t = 0.0
        choices = {"plane": PLANES, "shape": SHAPES, "menu": LOAD_MENU}[self.step]
        picked, live, prog = self.picker.update(f.hands, dt, speed < self.cfg.still_speed * 1.5, choices)
        self.live_count = live
        if live in choices:
            self.label, self.progress = choices[live][1], prog
        if picked is None:
            return
        name, label = choices[picked]
        self.toast = label
        if self.step == "plane":
            self.data["plane"] = name
            out.append(Command("wiz_plane", {"plane": name}))
            self._go("line_start" if name == "front" else "shape", out)
        elif self.step == "shape":
            self.data["shape"] = name
            self._go("draw_rect" if name == "rect" else "draw_circle", out)
        else:
            self.data["op"] = name
            self._go("start", out)

    # ---- pointing at a spot -------------------------------------------------------------------------------
    def _point_step(self, ctrl, other, track, pose, poses, t, dt, out) -> None:
        if pose != "point":
            self.dwell_t = 0.0
            self.label = "Point with your index finger"
            return
        xy = self._to_view(self.tip_f(fingertip(ctrl), t))
        slot = {"line_start": "p0", "line_end": "p1", "start": "t0", "end": "t1"}[self.step]
        out.append(Command("wiz_cursor", {"xy": xy, "slot": slot, "target": "plane" if self.step.startswith("line") else "member"}))
        clicking = self._pose(poses, other) == "pinch"
        click = clicking and not self.click_down
        self.click_down = clicking
        self.dwell_t = self.dwell_t + dt if track.speed < self.m.dwell_speed else 0.0
        self.label, self.progress = "Hold still to confirm", min(1.0, self.dwell_t / self.m.dwell_s)
        if not (click or self.dwell_t >= self.m.dwell_s):
            return
        out.append(Command("wiz_pick", {"slot": slot, "xy": xy}))
        self.toast = {"p0": "Start point", "p1": "End point", "t0": "Position", "t1": "End"}[slot]
        if self.step == "line_start":
            self._go("line_end", out)
        elif self.step == "line_end":
            self._go("shape", out)   # the renderer measured the line and sent L / orientation
        else:
            op = self.data.get("op", "")
            if self.step == "start" and op in ("uniform_load", "trapezoidal_load"):
                self._go("end", out)
            elif op.startswith("support_"):
                out.append(Command("wiz_place", {"op": op}))
                self._restart(out)
            else:
                self._go("dir", out)

    def _dir_step(self, ctrl, other, track, pose, poses, t, dt, out) -> None:
        if pose != "point":
            self.dwell_t, self.dir_vec = 0.0, None
            self.label = "Point your finger the way the load acts"
            return
        lm = ctrl.landmarks
        v = lm[INDEX[3], :2] - lm[INDEX[0], :2]
        v = self.dir_f(v / (np.linalg.norm(v) + 1e-9), t)
        v = v / (np.linalg.norm(v) + 1e-9)
        turned = self.dir_vec is not None and float(np.dot(v, self.dir_vec)) < 0.995
        self.dir_vec = v
        out.append(Command("wiz_dir", {"vec": v.round(4).tolist()}))
        clicking = self._pose(poses, other) == "pinch"
        click = clicking and not self.click_down
        self.click_down = clicking
        self.dwell_t = 0.0 if turned or track.speed > self.m.dwell_speed * 2 else self.dwell_t + dt
        self.label, self.progress = "Hold the direction to confirm", min(1.0, self.dwell_t / self.m.dwell_s)
        if click or self.dwell_t >= self.m.dwell_s:
            out.append(Command("wiz_pick", {"slot": "dir", "vec": v.round(4).tolist()}))
            out.append(Command("wiz_place", {"op": self.data.get("op")}))
            self.toast = "Direction set — now the value"
            self._restart(out)

    # ---- two-hand sizing --------------------------------------------------------------------------------
    def _size_step(self, f, p2, dt, out) -> None:
        m = self.m
        if len(f.hands) < 2 or p2 != ["pinch", "pinch"]:
            self.size_t = 0.0
            self.pinch_f.clear()
            self.need_release = False
            self.label = "Pinch with both hands"
            if self.size_val:
                out.append(Command("wiz_size", {**self.size_val, "locked": False}))
            return
        if self.need_release:
            self.label = "Open your fingers, then pinch again"
            return
        a, b = sorted(f.hands[:2], key=lambda h: h.palm_center[0])
        pa, pb = self._pinch_point(a), self._pinch_point(b)
        palm = (self.tracks[a.handedness].size.value + self.tracks[b.handedness].size.value) / 2   # filtered
        d = np.abs(pb - pa) / palm                    # palm widths, x and y
        prev = self.size_val
        if self.step == "draw_rect":
            val = {"kind": "rect",
                   "b": self._snap(d[0] * m.section_mm_per_palm, m.section_step, prev.get("b"), m.min_section),
                   "h": self._snap(max(d[1], 0.3) * m.section_mm_per_palm * m.vertical_gain, m.section_step, prev.get("h"), m.min_section)}
        elif self.step == "draw_circle":
            val = {"kind": "circle", "d": self._snap(float(np.hypot(*d)) * m.section_mm_per_palm, m.section_step,
                                                      prev.get("d"), m.min_section)}
        else:
            val = {"kind": "extrude", "L": self._snap(float(np.hypot(*d)) * m.length_mm_per_palm, m.length_step,
                                                       prev.get("L"), m.min_length)}
        speed = max(self.tracks[h.handedness].speed for h in (a, b))
        # The hold counts while the HANDS are still, not while the number is: a value flickering between two steps
        # does not restart it, and the value locked is the one shown most during the hold. A slow, steady pull is not
        # still: the hold restarts whenever the raw size has drifted a whole step from where the hold began.
        raw = d * np.array([1.0, m.vertical_gain]) * m.section_mm_per_palm if self.step == "draw_rect" else \
            np.array([float(np.hypot(*d)) * (m.section_mm_per_palm if self.step == "draw_circle" else m.length_mm_per_palm)])
        step = m.length_step if self.step == "extrude" else m.section_step
        drift = self.size_anchor is None or len(self.size_anchor) != len(raw) or \
            bool(np.any(np.abs(raw - self.size_anchor) > step))
        if speed > self.cfg.still_speed or drift:
            self.size_t, self.size_hist, self.size_anchor = 0.0, [], raw
        else:
            self.size_t += dt
            self.size_hist.append(tuple(sorted(val.items())))
        if self.size_hist:
            val = dict(max(set(self.size_hist), key=self.size_hist.count))
        self.size_val = val
        self.label, self.progress = "Hold still to lock", min(1.0, self.size_t / m.size_hold_s)
        out.append(Command("wiz_size", {**val, "locked": False}))
        if self.size_t >= m.size_hold_s:
            out.append(Command("wiz_size", {**val, "locked": True}))
            self.data.update({k: v for k, v in val.items() if k != "kind"})
            self.toast = {"rect": f"{val.get('b', 0):g} × {val.get('h', 0):g} mm", "circle": f"Ø{val.get('d', 0):g} mm",
                          "extrude": f"L = {val.get('L', 0) / 1000:g} m"}[val["kind"]]
            if self.step == "extrude" or self.data.get("plane") == "front":
                self._go("confirm", out)
            else:
                self._go("extrude", out)

    def _confirm_step(self, pose, speed, dt, out) -> None:
        if pose != "thumbs_up":
            self.accept_t = 0.0
            return
        self.accept_t = self.accept_t + dt if speed < self.cfg.still_speed else 0.0
        self.label, self.progress = "Create", min(1.0, self.accept_t / self.m.accept_hold_s)
        if self.accept_t >= self.m.accept_hold_s:
            out.append(Command("create_member", self.member_spec()))
            self.toast = "Member created"
            self._restart(out)

    def member_spec(self) -> dict:
        d = self.data
        plane = d.get("plane", "side")
        orient = {"plan": "vertical", "side": "horizontal"}.get(plane, d.get("orientation", "horizontal"))
        spec = {"kind": d.get("shape", "rect"), "orientation": orient, "L": float(d.get("L", 4000.0))}
        if spec["kind"] == "rect":
            spec.update(b=float(d.get("b", 300.0)), h=float(d.get("h", 500.0)))
        else:
            spec.update(d=float(d.get("d", 300.0)))
        return spec

    # ---- HUD ------------------------------------------------------------------------------------------
    def _steps(self) -> list[str]:
        if self.phase == "model":
            if self.data.get("plane") == "front":
                return ["plane", "line_start", "line_end", "shape", "draw", "confirm"]
            return ["plane", "shape", "draw", "extrude", "confirm"]
        op = self.data.get("op", "")
        if op.startswith("support_"):
            return ["menu", "start"]
        if op in ("uniform_load", "trapezoidal_load"):
            return ["menu", "start", "end", "dir", "value"]
        return ["menu", "start", "dir", "value"]

    def wizard(self) -> dict:
        step = self.step
        key = "draw" if step.startswith("draw_") else step
        steps = self._steps()
        choices = {"plane": PLANES, "shape": SHAPES, "menu": LOAD_MENU}.get(step)
        return {
            "phase": self.phase, "step": step, "title": STEP_TITLES.get(step, step), "help": STEP_HELP.get(step, ""),
            "steps": steps, "index": steps.index(key) if key in steps else 0,
            "choices": [{"n": n, "label": lab} for n, (_, lab) in choices.items()] if choices else None,
            "live": getattr(self, "live_count", None) if choices else None,
            "data": {k: v for k, v in self.data.items()}, "size": self.size_val or None,
            "increments": {"section": self.m.section_step, "length": self.m.length_step},
        }

    def _hud(self, state, pose, conf, hints, poses, tip: str | None = None) -> dict:
        hints = sorted(hints, key=lambda x: -x.severity)
        if tip is None:
            tip = "Raise your hand so the camera can see it" if state == "no_hand" else STEP_HELP.get(self.step, "")
        return {
            "state": state, "phase": self.phase, "mode": None,
            "mode_label": self.label if state == "active" else None,
            "pose": pose, "confidence": round(float(conf), 3),
            "progress_label": self.label if state in ("arming", "idle") and self.progress > 0 else None,
            "progress": round(float(self.progress), 3),
            "hints": [x.text for x in hints], "tip": tip, "toast": self.toast, "wizard": self.wizard(),
            "hands": [{"handedness": k, "pose": v[0], "confidence": round(float(v[1]), 3)} for k, v in poses.items()],
        }

