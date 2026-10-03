"""ViewState: the single source of truth the renderer reads.

Gesture code never touches VTK / three.js directly. It emits ``Command``s;
``ViewState.apply`` turns them into numbers, and the renderer (PyVista on the
desktop, three.js in the browser demo) just draws whatever ViewState says.
That keeps rendering on the main thread and makes every gesture testable
without a GPU.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

import numpy as np

FIELDS = ("von_mises", "displacement", "max_principal", "strain")


@dataclass
class Command:
    kind: str  # orbit | zoom | section | section_lock | section_off | probe | probe_clear | probe_pin
    #            field | snapshot | begin | reset | undo
    data: dict = field(default_factory=dict)


@dataclass
class ViewState:
    azimuth: float = 30.0  # degrees
    elevation: float = 20.0
    distance: float = 1.0  # multiplier on the default camera distance
    focal_point: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    section_on: bool = False
    section_locked: bool = False
    plane_origin: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    plane_normal: list = field(default_factory=lambda: [1.0, 0.0, 0.0])
    field_index: int = 0
    probe_cursor: list | None = None  # [x, y] in 0..1 viewport coords
    probe_pins: list = field(default_factory=list)
    snapshots: int = 0
    _history: list = field(default_factory=list, repr=False)

    @property
    def field_name(self) -> str:
        return FIELDS[self.field_index % len(FIELDS)]

    def camera_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(right, up, back) unit vectors of the camera in world coords (z-up world)."""
        az, el = np.radians(self.azimuth), np.radians(self.elevation)
        back = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, back)
        right /= np.linalg.norm(right) + 1e-9
        up = np.cross(back, right)
        return right, up, back

    def view_to_world(self, v_view: np.ndarray) -> np.ndarray:
        r, u, b = self.camera_basis()
        return v_view[0] * r + v_view[1] * u + v_view[2] * b

    def _push(self) -> None:
        snap = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k != "_history"}
        self._history.append(snap)
        del self._history[:-50]

    def apply(self, cmd: Command) -> None:
        d = cmd.data
        k = cmd.kind
        if k == "orbit":
            self.azimuth = (self.azimuth - d["dx"]) % 360
            self.elevation = float(np.clip(self.elevation + d["dy"], -89, 89))
        elif k == "zoom":
            self.distance = float(np.clip(self.distance * d["factor"], 0.2, 5.0))
        elif k == "section":
            n = self.view_to_world(np.asarray(d["normal_view"], float))
            n /= np.linalg.norm(n) + 1e-9
            self.section_on, self.section_locked = True, False
            self.plane_normal = n.tolist()
            self.plane_origin = (np.asarray(self.focal_point) + d.get("offset", 0.0) * n).tolist()
        elif k == "section_off":
            self._push()
            self.section_on, self.section_locked = False, False
        elif k == "section_lock":
            self.section_locked = True
            if "normal_view" in d:
                self.apply(Command("section", d))
                self.section_locked = True
        elif k == "probe":
            self.probe_cursor = list(d["xy"])
        elif k == "probe_clear":
            self.probe_cursor = None
        elif k == "probe_pin":
            self._push()
            self.probe_pins.append(list(d["xy"]))
            del self.probe_pins[:-5]
        elif k == "field":
            self._push()
            self.field_index = (self.field_index + d["step"]) % len(FIELDS)
        elif k == "snapshot":
            self.snapshots += 1
        elif k == "begin":  # start of a continuous gesture: one undo step per gesture
            self._push()
        elif k == "reset":
            self._push()
            hist = self._history
            self.__dict__.update(asdict(ViewState()))
            self._history = hist
        elif k == "undo" and self._history:
            snap = self._history.pop()
            hist = self._history
            self.__dict__.update(snap)
            self._history = hist

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        d["field_name"] = self.field_name
        return d
