"""Study specification — the contract shared by the UI and the solver.

Working units are millimetre / newton / megapascal / tonne-per-mm^3, which is
the self-consistent set the workbench uses everywhere:

    length   mm
    force    N
    stress   MPa  ( = N/mm^2 )
    density  t/mm^3   ( kg/m^3 * 1e-12 )
    mass     tonne
    time     s        ( so omega comes out in rad/s )
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

GRAVITY = 9810.0  # mm/s^2


@dataclass
class Geometry:
    """Implicit L-bracket: a vertical leg and a top arm of equal wall
    thickness, filleted at the re-entrant corner, extruded through depth."""

    W: float = 120.0   # bounding width  (x)
    H: float = 180.0   # bounding height (y)
    D: float = 60.0    # depth           (z)
    t: float = 30.0    # wall thickness
    r: float = 10.0    # re-entrant fillet radius

    def normalised(self) -> "Geometry":
        t = max(1.0, min(self.t, min(self.W, self.H) * 0.7))
        r = max(0.0, min(self.r, min(self.W - t, self.H - t) * 0.85))
        return Geometry(self.W, self.H, self.D, t, r)

    def contains(self, x: float, y: float) -> bool:
        """Point-in-profile test. The fillet *adds* material into the void
        quadrant: inside the corner square the boundary is the arc of radius r
        centred at (t + r, H - t - r), so material is everything outside it."""
        if x <= self.t:
            return True
        if y >= self.H - self.t:
            return True
        if self.r > 0 and x <= self.t + self.r and y >= self.H - self.t - self.r:
            a = x - (self.t + self.r)
            b = y - (self.H - self.t - self.r)
            return a * a + b * b >= self.r * self.r
        return False


@dataclass
class Material:
    id: str = "al7075"
    name: str = "Al 7075-T6"
    E: float = 71700.0        # MPa
    nu: float = 0.33
    rho: float = 2810.0       # kg/m^3 (converted on use)
    alpha: float = 23.6e-6    # 1/K
    sy: float = 503.0         # MPa, yield

    @property
    def lam(self) -> float:
        return self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def mu(self) -> float:
        return self.E / (2 * (1 + self.nu))

    def rho_t(self, scale: float = 1.0) -> float:
        """Density in tonne/mm^3."""
        return self.rho * scale * 1e-12


LIBRARY = {
    "al7075": Material("al7075", "Al 7075-T6", 71700.0, 0.33, 2810.0, 23.6e-6, 503.0),
    "steel": Material("steel", "Steel 4340", 205000.0, 0.29, 7850.0, 12.3e-6, 470.0),
    "ti64": Material("ti64", "Ti-6Al-4V", 113800.0, 0.34, 4430.0, 8.6e-6, 880.0),
    "cfrp": Material("cfrp", "CFRP [0/90]s", 135000.0, 0.31, 1570.0, 2.1e-6, 600.0),
}


@dataclass
class Mesh:
    mesher: str = "voxel"   # "voxel" (matches the UI 1:1) | "gmsh" (tet, curvature resolved)
    h: float = 6.0          # target element size, mm
    hz: float = 6.0         # through-depth element size, mm
    across: int = 2         # minimum elements across the wall
    budget: int = 24000     # element ceiling
    order: int = 1          # 1 = hex8/tet4, 2 = hex20/tet10 (gmsh path only)


@dataclass
class Loads:
    tip_force: float = 4200.0   # N resultant on the arm tip patch
    theta: float = 0.0          # degrees, rotates the tip force from -Y toward +X
    pressure: float = 0.8       # MPa on the outer web face (x = 0), acting +X
    dT: float = 0.0             # K above the stress-free reference
    gravity: bool = True
    density_scale: float = 1.0


@dataclass
class BCs:
    enc: bool = True      # encastre the base of the leg (y = 0)
    force: bool = True
    press: bool = True
    sym: bool = False     # U3 = 0 on the mid-depth plane


@dataclass
class Solver:
    rtol: float = 1e-6
    maxit: int = 4000
    ksp: str = "cg"
    pc: str = "gamg"      # gamg carries the rigid-body near-nullspace


@dataclass
class Outputs:
    u: bool = True
    s: bool = True
    e: bool = True
    rf: bool = True
    modes: int = 0        # number of eigenpairs; 0 disables the modal pass


@dataclass
class Study:
    name: str = "bracket_L"
    version: int = 1
    geometry: Geometry = field(default_factory=Geometry)
    material: Material = field(default_factory=Material)
    mesh: Mesh = field(default_factory=Mesh)
    loads: Loads = field(default_factory=Loads)
    bcs: BCs = field(default_factory=BCs)
    solver: Solver = field(default_factory=Solver)
    outputs: Outputs = field(default_factory=Outputs)

    # ---------------------------------------------------------------- io --
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Study":
        d = dict(d or {})

        def sub(cls, key):
            raw = dict(d.get(key) or {})
            keep = {f for f in cls.__dataclass_fields__}
            return cls(**{k: v for k, v in raw.items() if k in keep})

        mat_raw = d.get("material") or {}
        if isinstance(mat_raw, str):
            material = LIBRARY.get(mat_raw, LIBRARY["al7075"])
        elif "id" in mat_raw and len(mat_raw) == 1:
            material = LIBRARY.get(mat_raw["id"], LIBRARY["al7075"])
        else:
            base = asdict(LIBRARY.get(mat_raw.get("id", "al7075"), LIBRARY["al7075"]))
            base.update({k: v for k, v in mat_raw.items() if k in base})
            material = Material(**base)

        return Study(
            name=d.get("name", "bracket_L"),
            version=int(d.get("version", 1)),
            geometry=sub(Geometry, "geometry"),
            material=material,
            mesh=sub(Mesh, "mesh"),
            loads=sub(Loads, "loads"),
            bcs=sub(BCs, "bcs"),
            solver=sub(Solver, "solver"),
            outputs=sub(Outputs, "outputs"),
        )

    @staticmethod
    def load(path: str) -> "Study":
        with open(path, "r", encoding="utf-8") as fh:
            return Study.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.to_json())

    # ------------------------------------------------------------ sizing --
    def element_sizes(self) -> tuple[float, float, bool]:
        """Resolve (h, hz, clamped) honouring the wall-span requirement and
        the element budget. Mirrors the UI so both engines agree on the grid."""
        g = self.geometry.normalised()
        h = max(0.4, min(self.mesh.h, g.t / max(1, self.mesh.across)))
        hz = max(0.4, h * (self.mesh.hz / self.mesh.h if self.mesh.h else 1.0))
        fill = (g.t * g.H + g.W * g.t - g.t * g.t) / (g.W * g.H)

        def est(a, b):
            return round(g.W / a) * round(g.H / a) * max(1, round(g.D / b)) * fill

        clamped = False
        guard = 0
        while est(h, hz) > self.mesh.budget and guard < 400:
            h *= 1.06
            hz *= 1.06
            clamped = True
            guard += 1
        return h, hz, clamped
