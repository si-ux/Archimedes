"""Gestures for building the model: geometry, supports and loads.

MODEL phase
    🤏🤏 both hands pinch, pull apart, hold    extrude a rectangular member (horizontal pull = beam,
                                              vertical pull = column); then enter b, h, L by fingers
    ☝️  draw a circle with the index finger   circular section; then column/beam, diameter, depth
    ✊ fist drag / 🤏 one-hand pinch           orbit / zoom, as in the viewer

LOADS phase ("hold still" = keep the pose still ~0.8 s on the member; moving instead orbits/zooms)
    ✊ fist, hold still                         fixed support ("clamp")
    🤏 pinch, hold still                        pinned support ("pin it")
    ✌️ V, hold still                            roller support ("two wheels")
    ☝️ point at a spot, hold, then flick         point load in the flick direction
                                              (no flick within 1.5 s = straight down, gravity)
    ✋ flat palm, sweep along the member         uniform load over the swept span; the palm faces
                                              the load direction (palm down = −z)
    🙌 both flat palms at two spots, hold        trapezoidal load between the hands
    👍 thumbs-up, hold still 1.2 s              solve

Positions are emitted as viewport coordinates; the renderer maps them onto
the member (it knows the camera) and sends back t in [0, 1]. Directions are
emitted in view space and snapped to the nearest world axis in the same way.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .dynamic import detect_circle
from .filters import OneEuroFilter
from .geometry import camera_to_view, fingertip, palm_normal
from .intent import EngineConfig, _HandTrack
from .landmarks import INDEX, THUMB, FrameInput, HandFrame
from .quality import QualityConfig, check_frame, check_hand
from .view_state import Command
from .vocab import NONE

TOOLS_LOADS = {
    "fist": ("support_fixed", "Fixed support", "✊"),
    "pinch": ("support_pinned", "Pinned support", "🤏"),
    "peace": ("support_roller", "Roller support", "✌️"),
}

# How every number prompt (dimensions, loads, increments) is answered by hand
NUMBER_ROWS = [
    {"icon": "🙌", "label": "Accept the value", "hint": "Both open palms, hold ~1 s — the green bar fills, then the next prompt opens"},
    {"icon": "🙌", "label": "Accept the default", "hint": "Same gesture before typing anything takes the default shown on the card"},
    {"icon": "🖐", "label": "Type a digit", "hint": "Hold up 0–5 fingers (two hands for 6–9) still ~1 s; change the count to repeat a digit"},
    {"icon": "🤏↕", "label": "Nudge by the increment", "hint": "Pinch and move up / down — one increment per step"},
    {"icon": "👋", "label": "Delete a digit", "hint": "Swipe one open hand to the left"},
    {"icon": "✊✊", "label": "Cancel", "hint": "Two fists, hold 1 s (or Esc)"},
]

CHEAT = {
    "model": [
        {"icon": "⚙️", "label": "Increments first", "hint": "Entering Model asks for length / section / load steps (1–5 fingers); values snap to them"},
        {"icon": "🤏🤏", "label": "Rectangular member", "hint": "Pinch with both hands, pull apart, hold — horizontal = beam, vertical = column"},
        {"icon": "☝️○", "label": "Circular member", "hint": "Draw a circle in the air with your index finger"},
        *NUMBER_ROWS,
        {"icon": "✊", "label": "Rotate", "hint": "Fist and drag"},
        {"icon": "🤏", "label": "Zoom", "hint": "One-hand pinch, move up/down"},
    ],
    "loads": [
        {"icon": "✊", "label": "Fixed support", "hint": "Fist held still on the member (drag instead to rotate)"},
        {"icon": "🤏", "label": "Pinned support", "hint": "Pinch held still on the member (move instead to zoom)"},
        {"icon": "✌️", "label": "Roller support", "hint": "V held still on the member"},
        {"icon": "☝️", "label": "Point load", "hint": "Point at a spot, hold, then flick in the load direction"},
        {"icon": "✋", "label": "Uniform load", "hint": "Sweep a flat palm along the span — palm faces the load direction"},
        {"icon": "🙌", "label": "Trapezoidal load", "hint": "Both flat palms at the two ends of the load, hold still"},
        {"icon": "👍", "label": "Solve", "hint": "Thumbs-up held still for 1.2 s"},
        *NUMBER_ROWS,
    ],
}


@dataclass
class ModelingConfig:
    support_hold_s: float = 0.8
    move_to_orbit: float = 1.2  # palm widths / s while arming: moving means rotate/zoom instead
    point_hold_s: float = 0.6
    flick_dist: float = 1.2  # palm widths away from the anchor
    flick_timeout_s: float = 1.5
    sweep_speed: float = 1.0
    sweep_stop_s: float = 0.3
    sweep_min: float = 0.08  # viewport fraction
    trap_hold_s: float = 0.8
    extrude_min_gain: float = 1.5  # palm widths of extra separation
    extrude_hold_s: float = 0.6
    mm_per_palm: float = 800.0  # length estimate shown while pulling
    length_step: float = 100.0  # the length increment; the estimate snaps to it
    solve_hold_s: float = 1.2
    orbit_arm_s: float = 0.25
    circle_window_s: float = 3.0


class ModelingEngine:
    def __init__(self, cfg: EngineConfig | None = None, mcfg: ModelingConfig | None = None,
                 quality: QualityConfig | None = None):
        self.cfg = cfg or EngineConfig()
        self.m = mcfg or ModelingConfig()
        self.qcfg = quality or QualityConfig()
        self.phase = "loads"
        self.reset()

    def reset(self) -> None:
        self.t_prev: float | None = None
        self.tracks: dict[str, _HandTrack] = {}
        self.mode: str | None = None  # orbit | zoom
        self.arm_pose: str | None = None
        self.arm_t = 0.0
        self.latched: str | None = None  # pose that already fired; ignored until it changes
        self.prev_pos: np.ndarray | None = None
        self.tip_f = OneEuroFilter(1.0, 0.8)
        self.normal_f = OneEuroFilter(0.8, 0.4)
        self.anchor: np.ndarray | None = None
        self.anchor_t = 0.0
        self.anchor_tip: np.ndarray | None = None
        self.still_t = 0.0
        self.sweep: dict | None = None
        self.path: deque = deque(maxlen=200)
        self.two: dict | None = None
        self.two_latched = False
        self.toast: str | None = None
        self.preview: dict | None = None
        self.label: str | None = None
        self.progress = 0.0

    def set_phase(self, phase: str) -> None:
        if phase != self.phase:
            self.phase = phase
            self.reset()

    # ---- helpers ---------------------------------------------------------
    def _to_view(self, xy) -> list[float]:
        x0, y0, x1, y1 = self.cfg.box
        return np.clip([(xy[0] - x0) / (x1 - x0), (xy[1] - y0) / (y1 - y0)], 0.0, 1.0).round(4).tolist()

    @staticmethod
    def _pinch_point(h: HandFrame) -> np.ndarray:
        return (h.landmarks[THUMB[3], :2] + h.landmarks[INDEX[3], :2]) / 2

    def _end_mode(self) -> None:
        self.mode = None
        self.prev_pos = None

    def _drive(self, track: _HandTrack, out: list[Command]) -> None:
        """Continue an orbit/zoom from the rigid-palm motion (same feel as the viewer)."""
        size = track.size.value
        cur = track.last_pos
        delta = np.zeros(2) if self.prev_pos is None else (cur - self.prev_pos) / size
        self.prev_pos = cur.copy()
        if np.linalg.norm(delta) < self.cfg.deadzone:
            return
        if self.mode == "orbit":
            gain = self.cfg.orbit_gain * (0.6 + 0.4 * min(track.speed / 2.0, 2.0))
            out.append(Command("orbit", {"dx": float(delta[0] * gain), "dy": float(delta[1] * gain)}))
        elif self.mode == "zoom" and delta[1]:
            out.append(Command("zoom", {"factor": float(np.exp(delta[1] * self.cfg.zoom_gain))}))

    # ---- main --------------------------------------------------------------
    def update(self, f: FrameInput, poses: dict[str, tuple[str, float]]) -> tuple[list[Command], dict]:
        t = f.timestamp
        dt = 0.0 if self.t_prev is None else max(0.0, min(t - self.t_prev, 0.2))
        self.t_prev = t
        out: list[Command] = []
        self.toast = self.preview = self.label = None
        self.progress = 0.0
        hints = check_frame(f, self.qcfg)
        for k in list(self.tracks):
            if f.hand(k) is None:
                del self.tracks[k]
        for h in f.hands:
            self.tracks.setdefault(h.handedness, _HandTrack()).update(h, t)
        if not f.hands:
            self._end_mode()
            self.latched = None
            self.two, self.two_latched = None, False
            self.sweep = self.anchor = None
            self.path.clear()
            return out, self._hud("no_hand", NONE, 0.0, hints, poses)

        ctrl = f.hand(self.cfg.dominant) or f.hands[0]
        track = self.tracks[ctrl.handedness]
        pose, conf = poses.get(ctrl.handedness, (NONE, 0.0))
        hints += check_hand(ctrl, self.qcfg, track.speed)
        if any(h.severity >= 2 for h in hints):
            self._end_mode()
            return out, self._hud("paused", pose, conf, hints, poses)

        # ---- two-hand gestures -------------------------------------------------
        if len(f.hands) >= 2 and self.mode is None:
            p = [poses.get(h.handedness, (NONE, 0.0))[0] for h in f.hands[:2]]
            if self.phase == "model" and p == ["pinch", "pinch"]:
                self._extrude(f.hands[:2], t, dt, out)
                return out, self._hud("active", "pinch", conf, hints, poses)
            if self.phase == "loads" and p == ["open_palm", "open_palm"]:
                self._trapezoid(f.hands[:2], t, dt, out)
                return out, self._hud("arming" if not self.two_latched else "idle", "open_palm", conf, hints, poses)
        if len(f.hands) < 2 or self.mode is not None:
            self.two, self.two_latched = None, False

        # ---- one hand ------------------------------------------------------------
        if self.latched is not None:
            if pose == self.latched:
                return out, self._hud("idle", pose, conf, hints, poses, tip="Done — relax or change your hand for the next one")
            self.latched = None
        if self.mode == "orbit" and pose != "fist" or self.mode == "zoom" and pose != "pinch":
            self._end_mode()
        if self.mode:
            self._drive(track, out)
            self.label = "Rotate" if self.mode == "orbit" else "Zoom"
            return out, self._hud("active", pose, conf, hints, poses)

        if pose != self.arm_pose:
            self.arm_pose, self.arm_t = pose, 0.0
            self.anchor = self.sweep = None
            self.still_t = 0.0
            if pose != "point":
                self.path.clear()

        cursor = self._to_view(fingertip(ctrl) if pose == "point" else ctrl.palm_center[:2])
        if self.phase == "loads":
            state = self._loads(ctrl, track, pose, cursor, t, dt, out)
        else:
            state = self._model(ctrl, track, pose, t, dt, out)
        if self.phase == "loads" and pose in ("fist", "pinch", "peace", "point", "open_palm") and self.mode is None:
            out.append(Command("cursor", {"xy": cursor, "pose": pose}))
        return out, self._hud(state, pose, conf, hints, poses)

    # ---- MODEL phase ------------------------------------------------------------
    def _model(self, h, track, pose, t, dt, out) -> str:
        m = self.m
        if pose in ("fist", "pinch"):
            self.arm_t += dt
            self.label = "Rotate" if pose == "fist" else "Zoom"
            self.progress = min(1.0, self.arm_t / m.orbit_arm_s)
            if self.arm_t >= m.orbit_arm_s:
                self.mode = "orbit" if pose == "fist" else "zoom"
                self.prev_pos = track.last_pos.copy()
                out.append(Command("begin", {"mode": self.mode}))
                return "active"
            return "arming"
        if pose == "point":
            tip = self.tip_f(fingertip(h), t)
            self.path.append((t, tip.copy(), h.palm_size))
            while self.path and t - self.path[0][0] > m.circle_window_s:
                self.path.popleft()
            pts = np.array([q[1] for q in self.path])
            palm = float(np.median([q[2] for q in self.path]))
            best = 0.0
            for s in range(0, max(1, len(pts) - 12), 3):
                found, cover = detect_circle(pts[s:], min_radius=0.6 * palm)
                best = max(best, cover)
                if found:
                    out.append(Command("circle", {"radius": float(np.linalg.norm(pts[s:] - pts[s:].mean(0), axis=1).mean() / palm)}))
                    self.toast = "Circle — circular member"
                    self.latched = "point"
                    self.path.clear()
                    return "active"
            self.label = "Drawing a circle"
            self.progress = best
            self.preview = {"kind": "trace", "points": [self._to_view(q) for q in pts[-60:]]}
            return "arming"
        return "idle"

    def _extrude(self, hands, t, dt, out) -> None:
        m = self.m
        a, b = sorted(hands, key=lambda h: h.palm_center[0])
        pa, pb = self._pinch_point(a), self._pinch_point(b)
        palm = (a.palm_size + b.palm_size) / 2
        d = pb - pa
        sep = float(np.linalg.norm(d) / palm)
        orient = "horizontal" if abs(d[0]) >= abs(d[1]) else "vertical"
        if self.two is None or self.two.get("kind") != "extrude":
            self.two = {"kind": "extrude", "sep0": sep, "still": 0.0}
        if self.two_latched:
            return
        gain = sep - self.two["sep0"]
        step = m.length_step or 100.0
        est = float(np.clip(round(sep * m.mm_per_palm / step) * step, 500, 20000))
        speed = max(self.tracks[a.handedness].speed, self.tracks[b.handedness].speed)
        self.two["still"] = self.two["still"] + dt if (gain > m.extrude_min_gain and speed < self.cfg.still_speed * 1.5) else 0.0
        self.label = f"Extrude {'beam' if orient == 'horizontal' else 'column'} ≈ {est / 1000:.1f} m"
        self.progress = min(1.0, self.two["still"] / m.extrude_hold_s) if gain > m.extrude_min_gain else 0.0
        self.preview = {"kind": "extrude", "a": self._to_view(pa), "b": self._to_view(pb), "orientation": orient,
                        "length_est": est}
        out.append(Command("extrude_preview", {"orientation": orient, "length_est": est, "sep": sep}))
        if self.two["still"] >= m.extrude_hold_s:
            out.append(Command("extrude", {"orientation": orient, "length_est": est}))
            self.toast = "Extruded — now enter the dimensions"
            self.two_latched = True

    # ---- LOADS phase --------------------------------------------------------------
    def _loads(self, h, track, pose, cursor, t, dt, out) -> str:
        m = self.m
        still = track.speed < self.cfg.still_speed
        if pose in TOOLS_LOADS:
            op, label, _ = TOOLS_LOADS[pose]
            if track.speed > m.move_to_orbit and self.arm_t < m.support_hold_s and pose in ("fist", "pinch"):
                self.mode = "orbit" if pose == "fist" else "zoom"  # grab-and-move means rotate / zoom
                self.prev_pos = track.last_pos.copy()
                out.append(Command("begin", {"mode": self.mode}))
                return "active"
            self.arm_t = self.arm_t + dt if still else 0.0
            self.label, self.progress = label, min(1.0, self.arm_t / m.support_hold_s)
            if self.arm_t >= m.support_hold_s:
                out.append(Command("support", {"op": op, "xy": cursor}))
                self.toast = label
                self.latched = pose
                return "active"
            return "arming"
        if pose == "point":
            tip = fingertip(h)
            if self.anchor is None:
                self.still_t = self.still_t + dt if still else 0.0
                self.label, self.progress = "Point load — hold on the spot", min(1.0, self.still_t / m.point_hold_s)
                if self.still_t >= m.point_hold_s:
                    self.anchor, self.anchor_t, self.anchor_tip = np.array(cursor), t, tip.copy()
                return "arming"
            disp = (tip - self.anchor_tip) / h.palm_size
            self.label = "Flick the finger in the load direction"
            self.progress = min(1.0, (t - self.anchor_t) / m.flick_timeout_s)
            self.preview = {"kind": "point", "xy": self.anchor.tolist()}
            if np.linalg.norm(disp) > m.flick_dist:
                out.append(Command("point_load", {"xy": self.anchor.tolist(), "flick": disp.round(3).tolist()}))
            elif t - self.anchor_t > m.flick_timeout_s:
                out.append(Command("point_load", {"xy": self.anchor.tolist(), "flick": None}))
            else:
                return "arming"
            self.toast = "Point load"
            self.latched = pose
            self.anchor = None
            return "active"
        if pose == "open_palm":
            n = camera_to_view(self.normal_f(palm_normal(h), t))
            if self.sweep is None:
                if track.speed > m.sweep_speed:
                    self.sweep = {"xy0": cursor, "xy1": cursor, "n": [n], "stop": 0.0}
                else:
                    self.label = "Uniform load — sweep along the span"
                    return "idle"
            sw = self.sweep
            sw["xy1"] = cursor
            sw["n"].append(n)
            sw["stop"] = sw["stop"] + dt if track.speed < self.cfg.still_speed else 0.0
            self.preview = {"kind": "sweep", "xy0": sw["xy0"], "xy1": sw["xy1"]}
            self.label, self.progress = "Uniform load — stop to finish", min(1.0, sw["stop"] / m.sweep_stop_s)
            if sw["stop"] >= m.sweep_stop_s:
                span = float(np.linalg.norm(np.subtract(sw["xy1"], sw["xy0"])))
                if span >= m.sweep_min:
                    nv = np.mean(sw["n"], axis=0)
                    out.append(Command("udl", {"xy0": sw["xy0"], "xy1": sw["xy1"],
                                               "normal_view": (nv / (np.linalg.norm(nv) or 1)).round(3).tolist()}))
                    self.toast = "Uniform load"
                    self.latched = pose
                self.sweep = None
                return "active" if span >= m.sweep_min else "idle"
            return "arming"
        if pose == "thumbs_up":
            self.arm_t = self.arm_t + dt if still else 0.0
            self.label, self.progress = "Solve", min(1.0, self.arm_t / m.solve_hold_s)
            if self.arm_t >= m.solve_hold_s:
                out.append(Command("solve"))
                self.toast = "Solving…"
                self.latched = pose
                return "active"
            return "arming"
        return "idle"

    def _trapezoid(self, hands, t, dt, out) -> None:
        m = self.m
        if self.two is None or self.two.get("kind") != "trap":
            self.two = {"kind": "trap", "still": 0.0}
        if self.two_latched:
            return
        a, b = sorted(hands, key=lambda h: h.palm_center[0])
        speed = max(self.tracks[h.handedness].speed for h in hands)
        self.two["still"] = self.two["still"] + dt if speed < self.cfg.still_speed else 0.0
        xa, xb = self._to_view(a.palm_center[:2]), self._to_view(b.palm_center[:2])
        self.label, self.progress = "Trapezoidal load", min(1.0, self.two["still"] / m.trap_hold_s)
        self.preview = {"kind": "sweep", "xy0": xa, "xy1": xb}
        if self.two["still"] >= m.trap_hold_s:
            nv = camera_to_view(palm_normal(a)) + camera_to_view(palm_normal(b))
            out.append(Command("trap", {"xy0": xa, "xy1": xb, "normal_view": (nv / (np.linalg.norm(nv) or 1)).round(3).tolist()}))
            self.toast = "Trapezoidal load"
            self.two_latched = True

    # ---- HUD ------------------------------------------------------------------
    def _hud(self, state, pose, conf, hints, poses, tip: str | None = None) -> dict:
        hints = sorted(hints, key=lambda x: -x.severity)
        if tip is None:
            if state == "no_hand":
                tip = "Raise your hand so the camera can see it"
            elif self.phase == "model":
                tip = "🤏🤏 pinch both hands and pull apart for a beam/column · ☝️ draw a circle for a round member"
            else:
                tip = "✊ fixed · 🤏 pinned · ✌️ roller (hold still) · ☝️ point load · ✋ sweep = uniform · 🙌 trapezoidal · 👍 solve"
        return {
            "state": state, "phase": self.phase, "mode": self.mode,
            "mode_label": self.label if state == "active" else None,
            "pose": pose, "confidence": round(float(conf), 3),
            "progress_label": self.label if state in ("arming", "idle") and self.progress > 0 else None,
            "progress": round(float(self.progress), 3),
            "hints": [x.text for x in hints], "tip": tip, "toast": self.toast, "preview": self.preview,
            "hands": [{"handedness": k, "pose": v[0], "confidence": round(float(v[1]), 3)} for k, v in poses.items()],
        }
