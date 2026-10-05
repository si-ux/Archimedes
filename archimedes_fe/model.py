"""Member model: geometry, material, supports and loads.

Units throughout: mm, N, MPa, t/mm³. Loads are entered in kN (point) and kN/m
(line loads); 1 kN = 1000 N and 1 kN/m = 1 N/mm.

World axes are z-up. A *horizontal* member (beam) runs along +x from 0 to L,
width b along y, height h along z, centred on the x axis. A *vertical* member
(column) runs along +z from 0 to L, width b along x, height h along y.
Positions along a member are given as t in [0, 1].
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field

AXES = ("x", "y", "z")


@dataclass
class Section:
    kind: str = "rect"  # rect | circle
    b: float = 300.0  # rect width (mm)
    h: float = 500.0  # rect height (mm)
    d: float = 300.0  # circle diameter (mm)

    @property
    def area(self) -> float:
        return self.b * self.h if self.kind == "rect" else math.pi * self.d ** 2 / 4

    @property
    def i_min(self) -> float:
        if self.kind == "circle":
            return math.pi * self.d ** 4 / 64
        return min(self.b * self.h ** 3, self.h * self.b ** 3) / 12

    @property
    def i_strong(self) -> float:
        if self.kind == "circle":
            return self.i_min
        return self.b * self.h ** 3 / 12

    def label(self) -> str:
        return f"{self.b:g} × {self.h:g} mm" if self.kind == "rect" else f"Ø{self.d:g} mm"


@dataclass
class Material:
    name: str = "Concrete C30"
    E: float = 30000.0  # MPa
    nu: float = 0.2
    rho: float = 2.5e-9  # t/mm³
    strength: float = 30.0  # MPa: fc for concrete, fy for steel

    @property
    def G(self) -> float:
        return self.E / (2 * (1 + self.nu))


STEEL = Material("Steel S355", 200000.0, 0.3, 7.85e-9, 355.0)
CONCRETE = Material()


@dataclass
class Support:
    """fixed: every node of the cross-section held in x, y, z.
    pinned: a bearing line held in x, y, z (beam: bottom edge; column base: centre line);
            above a column's base a pin holds it sideways only, so it can still shorten.
    roller: a bearing line free to slide along the member, held in the other two directions.
    """

    kind: str  # fixed | pinned | roller
    t: float  # 0..1 along the member


@dataclass
class Load:
    kind: str  # point | uniform | trapezoidal
    axis: str  # x | y | z (world)
    sign: int  # +1 / -1 along the axis
    t0: float  # point: position; line loads: start
    w0: float  # point: kN; line loads: kN/m at t0
    t1: float | None = None  # line loads: end
    w1: float | None = None  # trapezoidal: kN/m at t1 (uniform: same as w0)

    def label(self) -> str:
        d = ("+" if self.sign > 0 else "−") + self.axis.upper()
        if self.kind == "point":
            return f"Point {self.w0:g} kN {d}"
        if self.kind == "uniform":
            return f"Uniform {self.w0:g} kN/m {d}"
        return f"Trapezoidal {self.w0:g}→{self.w1:g} kN/m {d}"


@dataclass
class Member:
    section: Section = field(default_factory=Section)
    length: float = 4000.0  # mm
    orientation: str = "horizontal"  # horizontal (beam) | vertical (column)

    @property
    def axis(self) -> int:
        return 0 if self.orientation == "horizontal" else 2


@dataclass
class Model:
    name: str = "Untitled"
    member: Member = field(default_factory=Member)
    material: Material = field(default_factory=Material)
    supports: list[Support] = field(default_factory=list)
    loads: list[Load] = field(default_factory=list)

    # ---- serialisation ----------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["summary"] = self.summary()
        d["issues"] = self.issues()
        d["load_labels"] = [ld.label() for ld in self.loads]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        m = d["member"]
        return cls(
            name=d.get("name", "Untitled"),
            member=Member(Section(**m["section"]), m["length"], m["orientation"]),
            material=Material(**d["material"]),
            supports=[Support(**s) for s in d.get("supports", [])],
            loads=[Load(**ld) for ld in d.get("loads", [])],
        )

    def copy(self) -> "Model":
        return copy.deepcopy(self)

    # ---- checks -----------------------------------------------------------
    def issues(self) -> list[str]:
        """Plain-language reasons the model cannot be solved yet (empty = solvable)."""
        out = []
        kinds = [s.kind for s in self.supports]
        if not self.supports:
            out.append("Add supports: ✊ fixed, 🤏 pinned or ✌️ roller")
        elif self.member.orientation == "horizontal":
            if "fixed" not in kinds and "pinned" not in kinds:
                out.append("Rollers alone let the beam slide: add a pinned or fixed support")
            elif "fixed" not in kinds and len(self.supports) < 2:
                out.append("One pin lets the beam rotate: add a roller, a second pin or make it fixed")
        else:
            base = [s for s in self.supports if s.t <= 0.05]
            if not any(s.kind in ("fixed", "pinned") for s in base):
                out.append("A column needs a pinned or fixed support at its base (t = 0)")
            elif "fixed" not in kinds and len(self.supports) < 2:
                out.append("A pinned base alone lets the column topple: fix the base or restrain the top")
        if not self.loads:
            out.append("Add a load: ☝️ point, ✋ uniform or 🙌 trapezoidal")
        return out

    # ---- engineering summary -------------------------------------------------
    def effective_length_factor(self) -> float:
        """K for a column from its end supports (braced/unbraced textbook cases)."""
        ends = {0: None, 1: None}
        for s in self.supports:
            if s.t <= 0.05:
                ends[0] = s.kind
            elif s.t >= 0.95:
                ends[1] = s.kind
        a, b = ends[0], ends[1]
        if a == "fixed" and b is None:
            return 2.0
        if a == "fixed" and b == "fixed":
            return 0.5
        if "fixed" in (a, b) and ("pinned" in (a, b) or "roller" in (a, b)):
            return 0.7
        return 1.0

    def summary(self) -> dict:
        s, m, L = self.member.section, self.material, self.member.length
        A, Imin = s.area, s.i_min
        r = math.sqrt(Imin / A)
        out = {
            "kind": "Column" if self.member.orientation == "vertical" else "Beam",
            "section": s.label(),
            "length_mm": L,
            "area_mm2": A,
            "I_min_mm4": Imin,
            "r_mm": r,
            "material": m.name,
            "E_MPa": m.E,
            "nu": m.nu,
            "rho_t_mm3": m.rho,
            "self_weight_kN_m": m.rho * 9.81e3 * A,  # t/mm³ * mm/s² ... -> N/mm = kN/m
        }
        if self.member.orientation == "vertical":
            K = self.effective_length_factor()
            lam = K * L / r
            pcr = math.pi ** 2 * m.E * Imin / (K * L) ** 2 / 1e3  # kN
            psq = A * m.strength / 1e3  # kN
            lam_t = math.pi * math.sqrt(m.E / m.strength)  # Euler meets squash load
            cls = "short" if lam < 0.5 * lam_t else "long" if lam > lam_t else "intermediate"
            out.update(K=K, slenderness=lam, slenderness_transition=lam_t, P_cr_kN=pcr, P_squash_kN=psq,
                       column_class=cls,
                       governs="buckling (Euler)" if pcr < psq else "crushing (material strength)")
        return out
