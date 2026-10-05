"""ModelSession: the model being built, plus the questions still to ask.

Gestures (or mouse tools) start *actions*: "new rectangular beam", "uniform
load from t0 to t1 along −z", and so on. Most actions need numbers, so an
action owns a queue of prompts. The gesture pipeline shows the current prompt
and reads digits from finger counts. The keyboard can answer too. When the
last prompt is answered, the action is applied to the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .demos import DEMOS
from .model import AXES, CONCRETE, STEEL, Load, Member, Model, Section, Support
from .solver import Results, SolveError, solve

PHASES = ("model", "loads", "results")


@dataclass
class Prompt:
    key: str
    label: str
    unit: str
    default: float | None = None
    choices: dict[int, str] | None = None  # finger count -> option label
    minimum: float = 0.0

    def to_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "unit": self.unit, "default": self.default,
                "choices": self.choices}


@dataclass
class Action:
    op: str
    data: dict = field(default_factory=dict)
    prompts: list[Prompt] = field(default_factory=list)
    answers: dict = field(default_factory=dict)

    @property
    def prompt(self) -> Prompt | None:
        for p in self.prompts:
            if p.key not in self.answers:
                return p
        return None


def snap_t(t: float, ends: float = 0.06) -> float:
    """Snap positions near the ends onto them; supports and tip loads usually belong there."""
    t = min(1.0, max(0.0, float(t)))
    if t < ends:
        return 0.0
    if t > 1 - ends:
        return 1.0
    return round(t, 3)


class ModelSession:
    def __init__(self, demo: str = "cantilever"):
        self.model: Model = DEMOS[demo]()
        self.phase = "loads"
        self.action: Action | None = None
        self.results: Results | None = None
        self.error: str | None = None
        self.message: str | None = None
        self.history: list[Model] = []
        self.version = 0

    # ---- state -----------------------------------------------------------
    @property
    def prompt(self) -> dict | None:
        if self.action and self.action.prompt:
            d = self.action.prompt.to_dict()
            d["action"] = self.action.op
            d["axis"] = self.action.data.get("axis")
            d["sign"] = self.action.data.get("sign")
            d["step"] = len(self.action.answers) + 1
            d["steps"] = len(self.action.prompts)
            return d
        return None

    def state(self) -> dict:
        return {"phase": self.phase, "prompt": self.prompt, "model": self.model.to_dict(),
                "solved": self.results is not None, "error": self.error, "message": self.message,
                "version": self.version}

    def _changed(self, msg: str | None = None) -> None:
        self.version += 1
        self.results = None
        self.message = msg
        self.error = None

    def _push(self) -> None:
        self.history.append(self.model.copy())
        del self.history[:-30]

    def set_phase(self, phase: str) -> None:
        if phase in PHASES and (phase != "results" or self.results is not None):
            self.phase = phase

    # ---- starting actions ------------------------------------------------------
    def begin(self, op: str, **data) -> None:
        """Start an action. Positions (t) and directions must already be resolved."""
        self.error = None
        P = Prompt
        if op == "new_rect":
            est = data.get("length_est")
            vertical = data.get("orientation") == "vertical"
            prompts = [P("b", "Width b", "mm", 300.0, minimum=20), P("h", "Height h", "mm", 300.0 if vertical else 500.0, minimum=20),
                       P("L", "Length L" if not vertical else "Height of column L", "mm", est or (3000.0 if vertical else 4000.0),
                         minimum=200)]
        elif op == "new_circle":
            prompts = [P("orientation", "Column or beam?", "", 1, choices={1: "column (vertical)", 2: "beam (horizontal)"}),
                       P("d", "Diameter", "mm", 300.0, minimum=20), P("L", "Depth (length)", "mm", 3000.0, minimum=200)]
        elif op in ("support_fixed", "support_pinned", "support_roller"):
            self._apply(Action(op, data))
            return
        elif op == "point_load":
            prompts = [P("w0", "Point load", "kN", 10.0)]
        elif op == "uniform_load":
            prompts = [P("w0", "Uniform load", "kN/m", 10.0)]
        elif op == "trapezoidal_load":
            prompts = [P("w0", "Load at start", "kN/m", 0.0), P("w1", "Load at end", "kN/m", 10.0)]
        elif op == "material":
            prompts = [P("m", "Material", "", 1, choices={1: "concrete C30", 2: "steel S355"})]
        else:
            raise ValueError(op)
        self.action = Action(op, data, prompts)

    def set_direction(self, axis: str, sign: int) -> None:
        if self.action and axis in AXES:
            self.action.data.update(axis=axis, sign=1 if sign > 0 else -1)

    def answer(self, value: float | None) -> None:
        """Answer the current prompt; None accepts its default."""
        a = self.action
        if not a or not a.prompt:
            return
        p = a.prompt
        v = p.default if value is None else float(value)
        if v is None:
            self.error = f"{p.label}: enter a number"
            return
        if p.choices is not None and int(v) not in p.choices:
            self.error = f"{p.label}: show {' or '.join(str(k) for k in p.choices)} fingers"
            return
        if p.choices is None and v < p.minimum:
            self.error = f"{p.label} must be at least {p.minimum:g} {p.unit}"
            return
        a.answers[p.key] = v
        self.error = None
        if a.prompt is None:
            self.action = None
            self._apply(a)

    def cancel(self) -> None:
        self.action = None
        self.message = "Cancelled"

    # ---- applying ------------------------------------------------------------
    def _apply(self, a: Action) -> None:
        d, ans = a.data, a.answers
        self._push()
        m = self.model
        if a.op == "new_rect":
            vertical = d.get("orientation") == "vertical"
            m = Model("Column (sketched)" if vertical else "Beam (sketched)",
                      Member(Section("rect", ans["b"], ans["h"]), ans["L"], "vertical" if vertical else "horizontal"),
                      m.material)
            self.model = m
            self.phase = "loads"
            self._changed(f"Created {m.member.section.label()} × {ans['L']:g} mm. Now add supports and loads.")
            return
        if a.op == "new_circle":
            vertical = int(ans["orientation"]) == 1
            self.model = Model("Round column (sketched)" if vertical else "Round beam (sketched)",
                               Member(Section("circle", d=ans["d"]), ans["L"], "vertical" if vertical else "horizontal"),
                               m.material)
            self.phase = "loads"
            self._changed(f"Created Ø{ans['d']:g} × {ans['L']:g} mm. Now add supports and loads.")
            return
        if a.op == "material":
            m.material = CONCRETE if int(ans["m"]) == 1 else STEEL
            self._changed(f"Material: {m.material.name}")
            return
        if a.op.startswith("support_"):
            kind = a.op.split("_", 1)[1]
            t = snap_t(d["t"])
            m.supports = [s for s in m.supports if abs(s.t - t) > 0.02]  # replace a support at the same spot
            m.supports.append(Support(kind, t))
            m.supports.sort(key=lambda s: s.t)
            self._changed(f"{kind.capitalize()} support at t = {t:g}")
            return
        axis, sign = d.get("axis", "z"), int(d.get("sign", -1))
        if a.op == "point_load":
            m.loads.append(Load("point", axis, sign, snap_t(d["t"]), ans["w0"]))
        elif a.op == "uniform_load":
            t0, t1 = sorted((snap_t(d["t0"]), snap_t(d["t1"])))
            m.loads.append(Load("uniform", axis, sign, t0, ans["w0"], t1, ans["w0"]))
        elif a.op == "trapezoidal_load":
            t0, t1 = snap_t(d["t0"]), snap_t(d["t1"])
            w0, w1 = ans["w0"], ans["w1"]
            if t1 < t0:
                t0, t1, w0, w1 = t1, t0, w1, w0
            m.loads.append(Load("trapezoidal", axis, sign, t0, w0, t1, w1))
        self._changed(f"Added {m.loads[-1].label()}")

    # ---- other edits -------------------------------------------------------------
    def load_demo(self, name: str) -> None:
        self._push()
        self.model = DEMOS[name]()
        self.action = None
        self._changed(f"Loaded demo: {self.model.name}")

    def clear(self, what: str) -> None:
        self._push()
        if what in ("loads", "all"):
            self.model.loads = []
        if what in ("supports", "all"):
            self.model.supports = []
        self._changed(f"Cleared {what}")

    def undo(self) -> None:
        if self.history:
            self.model = self.history.pop()
            self.action = None
            self._changed("Undone")

    def solve(self) -> Results | None:
        try:
            self.results = solve(self.model)
        except SolveError as exc:
            self.error = str(exc)
            self.results = None
            return None
        self.error = None
        self.phase = "results"
        self.message = f"Solved in {sum(self.results.stats[k] for k in ('t_assemble_s', 't_solve_s', 't_modes_s')):.2f} s"
        return self.results
