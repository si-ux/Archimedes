"""Structured hex8 voxel core.

This reproduces the workbench's in-browser discretisation *exactly* — same
grid resolution, same cell occupancy, same node numbering, same local vertex
order.  That matters for two reasons:

  * results come back in the UI's own node order, so the viewport can swap in
    DOLFINx fields without touching the renderer;
  * the browser preview and the authoritative solve are then the same
    discrete problem, so any difference between them is solver quality, not
    mesh quality.

The local vertex order ``n = a + 2b + 4c`` is also basix's reference ordering
for a hexahedron, so the connectivity can be handed to DOLFINx unpermuted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .spec import Geometry, Study

# local vertex (a, b, c) for n = a + 2b + 4c  — basix hexahedron reference order
CORN = np.array([(n & 1, (n >> 1) & 1, (n >> 2) & 1) for n in range(8)], dtype=np.int64)


def _round(v: float) -> int:
    """Round half up, matching JavaScript's Math.round (Python rounds half to even)."""
    return int(math.floor(v + 0.5))


@dataclass
class VoxelMesh:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    solid: np.ndarray      # (nx*ny*nz,) uint8, index (k*ny + j)*nx + i
    nid: np.ndarray        # (NX*NY*NZ,) int64, -1 where unused
    cells: np.ndarray      # (ne, 8) int64 -> compacted node ids
    ecell: np.ndarray      # (ne,) int64 -> grid cell index
    points: np.ndarray     # (nn, 3) float64, mm
    gidx: np.ndarray       # (nn, 3) int64 -> (i, j, k) grid coordinates
    W: float = 0.0         # effective bounding box after the wall snap
    H: float = 0.0
    D: float = 0.0

    @property
    def ne(self) -> int:
        return int(self.cells.shape[0])

    @property
    def nn(self) -> int:
        return int(self.points.shape[0])

    @property
    def ndof(self) -> int:
        return self.nn * 3

    # ---------------------------------------------------------- quality --
    def quality(self, g: Geometry) -> dict:
        e = (self.dx, self.dy, self.dz)
        ar = max(e) / min(e)
        i = self.ecell % self.nx
        j = (self.ecell // self.nx) % self.ny
        k = self.ecell // (self.nx * self.ny)
        occ = self.solid.reshape(self.nz, self.ny, self.nx)
        open_faces = np.zeros(self.ne, dtype=np.int64)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            inside = (ii >= 0) & (jj >= 0) & (ii < self.nx) & (jj < self.ny)
            nbr = np.zeros(self.ne, dtype=bool)
            nbr[inside] = occ[k[inside], jj[inside], ii[inside]] > 0
            open_faces += (~nbr).astype(np.int64)
        return {
            "elements": self.ne,
            "nodes": self.nn,
            "dof": self.ndof,
            "scaled_jacobian": 1.0,       # perfect right hexahedra by construction
            "aspect_ratio": float(ar),
            "across_wall": int(_round(g.t / self.dx)),
            "single_span": int((open_faces >= 3).sum()),
        }


def build(g: Geometry, h: float, hz: float) -> VoxelMesh:
    """Grid the profile.

    The in-plane spacing is snapped so the wall is an exact whole number of
    cells (dx = t/m).  Sampling occupancy at cell centres on an unsnapped grid
    makes the *effective* wall thickness wobble with h — at t=30 a naive grid
    gives 24/32/30/28 mm for h=12/8/6/4 — and since bending stiffness goes as
    t^3 that error swamps any real discretisation error, so refinement stops
    converging.  With the snap, both walls are exact at every h and only the
    fillet staircase refines.
    """
    g = g.normalised()
    m = max(1, _round(g.t / h))               # cells across the wall
    dx = dy = g.t / m
    nx = max(m + 1, _round(g.W / dx))
    ny = max(m + 1, _round(g.H / dy))
    nz = max(1, _round(g.D / hz))
    dz = g.D / nz
    # effective bounding box after the snap (within dx/2 of what was asked for)
    W, H = nx * dx, ny * dy
    NX, NY, NZ = nx + 1, ny + 1, nz + 1

    # cell occupancy from the implicit profile, sampled at cell centres
    cx = (np.arange(nx) + 0.5) * dx
    cy = (np.arange(ny) + 0.5) * dy
    XX, YY = np.meshgrid(cx, cy, indexing="xy")
    inside = np.zeros_like(XX, dtype=bool)
    inside |= XX <= g.t
    inside |= YY >= H - g.t
    if g.r > 0:
        box = (XX <= g.t + g.r) & (YY >= H - g.t - g.r)
        a = XX - (g.t + g.r)
        b = YY - (H - g.t - g.r)
        inside |= box & (a * a + b * b >= g.r * g.r)
    plane = inside.astype(np.uint8)                       # (ny, nx)
    solid = np.repeat(plane[None, :, :], nz, axis=0).reshape(-1)

    # mark used nodes, then number them in (k, j, i) order — i fastest
    used = np.zeros(NX * NY * NZ, dtype=bool)
    kk, jj, ii = np.nonzero(solid.reshape(nz, ny, nx))
    for a, b, c in CORN:
        used[((kk + c) * NY + (jj + b)) * NX + (ii + a)] = True
    nid = np.full(NX * NY * NZ, -1, dtype=np.int64)
    nid[used] = np.arange(int(used.sum()), dtype=np.int64)

    # elements in the same (k, j, i) order the UI uses
    order = np.lexsort((ii, jj, kk))
    ei, ej, ek = ii[order], jj[order], kk[order]
    ecell = (ek * ny + ej) * nx + ei
    cells = np.empty((ecell.size, 8), dtype=np.int64)
    for n, (a, b, c) in enumerate(CORN):
        cells[:, n] = nid[((ek + c) * NY + (ej + b)) * NX + (ei + a)]

    qi = np.nonzero(used)[0]
    gi = qi % NX
    gj = (qi // NX) % NY
    gk = qi // (NX * NY)
    points = np.stack([gi * dx, gj * dy, gk * dz], axis=1).astype(np.float64)
    gidx = np.stack([gi, gj, gk], axis=1).astype(np.int64)

    return VoxelMesh(nx, ny, nz, dx, dy, dz, solid, nid, cells, ecell, points, gidx,
                     W=W, H=H, D=g.D)


def from_study(study: Study) -> VoxelMesh:
    h, hz, _ = study.element_sizes()
    return build(study.geometry, h, hz)


# ------------------------------------------------------------- tagging --
def node_sets(m: VoxelMesh, g: Geometry) -> dict[str, np.ndarray]:
    """Boolean masks over the compacted node list for each named region."""
    g = g.normalised()
    x, y, z = m.points[:, 0], m.points[:, 1], m.points[:, 2]
    tol = min(m.dx, m.dy, m.dz) * 1e-6
    span = max(1, _round(g.t / m.dx))
    return {
        "base": y <= tol,                                    # encastre face
        "tip": (y >= m.H - tol) & (x >= m.W - span * m.dx - tol),
        "web": x <= tol,                                     # pressure face
        "midplane": np.abs(z - m.dz * _round(m.nz / 2)) <= tol,
    }
