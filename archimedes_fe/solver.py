"""Voxel brick FE solver for a single prismatic member (numpy + scipy).

Element: 8-node brick with Wilson-Taylor incompatible modes, condensed out
statically. A plain trilinear brick locks in bending (it comes out far too
stiff when a beam is only a few elements deep). The nine bubble modes remove
that, so a 6-element-deep beam gets the Euler-Bernoulli deflection within a
few per cent. Every cell of the structured grid has the same size, so one
24×24 element matrix serves the whole mesh and assembly is a single vectorised
COO build.

Results per node (Gauss-point values extrapolated to the corners, averaged at shared nodes):
displacements U1..U3, stresses S11..S23 (MPa), strains E11..E23 (tensor
shear), von Mises, Tresca, principal stresses, equivalent strain, strain
energy density. Plus reactions and the first natural modes (lumped mass,
shift-invert Lanczos).
"""
from __future__ import annotations

import base64
import math
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .model import AXES, Model

# local node n = a + 2b + 4c, (a, b, c) in {0, 1}^3 - also basix's hex order
CORNERS = np.array([[a, b, c] for c in (0, 1) for b in (0, 1) for a in (0, 1)])
GAUSS = np.array([-1.0, 1.0]) / math.sqrt(3.0)


class SolveError(RuntimeError):
    pass


def elasticity(E: float, nu: float) -> np.ndarray:
    """Isotropic D for Voigt order [11, 22, 33, 12, 23, 13] with engineering shear."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def _b_matrix(dn: np.ndarray) -> np.ndarray:
    """Strain-displacement matrix from shape-function gradients dn (n, 3)."""
    n = dn.shape[0]
    B = np.zeros((6, 3 * n))
    for i in range(n):
        dx, dy, dz = dn[i]
        c = 3 * i
        B[0, c] = dx
        B[1, c + 1] = dy
        B[2, c + 2] = dz
        B[3, c], B[3, c + 1] = dy, dx
        B[4, c + 1], B[4, c + 2] = dz, dy
        B[5, c], B[5, c + 2] = dz, dx
    return B


def brick_matrices(h: tuple[float, float, float], D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Condensed 24×24 stiffness of a box element with incompatible modes, and corner strain operators.

    The second return value Bc (8, 6, 24) gives the strain at each corner node,
    extrapolated from the 2×2×2 Gauss points (where brick strains are most
    accurate) with the bubble modes recovered from the element displacements.
    """
    h = np.asarray(h, float)
    scale = 2.0 / h  # d(xi)/dx
    detj = np.prod(h) / 8.0
    sgn = CORNERS * 2 - 1  # natural coords of the nodes
    Kuu = np.zeros((24, 24))
    Kua = np.zeros((24, 9))
    Kaa = np.zeros((9, 9))
    gauss_b = []
    for g in sgn:  # Gauss points in corner order
                q = g / math.sqrt(3.0)
                f = 1 + sgn * q  # (8, 3)
                dn = np.empty((8, 3))
                dn[:, 0] = sgn[:, 0] * f[:, 1] * f[:, 2] / 8
                dn[:, 1] = sgn[:, 1] * f[:, 0] * f[:, 2] / 8
                dn[:, 2] = sgn[:, 2] * f[:, 0] * f[:, 1] / 8
                Bu = _b_matrix(dn * scale)
                # bubbles P_k = 1 - q_k^2, one per direction, applied to all three components
                da = np.diag(-2 * q) * scale  # (3 modes, 3 grads)
                Ba = _b_matrix(da)
                Kuu += Bu.T @ D @ Bu * detj
                Kua += Bu.T @ D @ Ba * detj
                Kaa += Ba.T @ D @ Ba * detj
                gauss_b.append((Bu, Ba))
    R = -np.linalg.solve(Kaa, Kua.T)  # bubble amplitudes from nodal displacements (9, 24)
    K = Kuu + Kua @ R
    Bg = np.array([Bu + Ba @ R for Bu, Ba in gauss_b])  # (8 gauss, 6, 24)
    # trilinear extrapolation: corner c sits at natural coord sqrt(3)*sgn_c in "Gauss space"
    E = np.prod(1 + sgn[None, :, :] * math.sqrt(3.0) * sgn[:, None, :], axis=2) / 8.0  # (corner, gauss)
    Bc = np.einsum("cg,gij->cij", E, Bg)
    return 0.5 * (K + K.T), Bc


@dataclass
class Mesh:
    nodes: np.ndarray  # (nn, 3) mm
    elems: np.ndarray  # (ne, 8)
    cells: np.ndarray  # (ne, 3) grid index of each element
    grid: tuple[int, int, int]  # cells per axis
    h: tuple[float, float, float]  # cell size per axis (mm)
    origin: np.ndarray  # world position of grid corner (0, 0, 0)
    axis: int  # member axis (0 = x, 2 = z)
    layer: np.ndarray = field(default=None)  # node layer index along the member axis

    @property
    def nn(self) -> int:
        return len(self.nodes)


def build_mesh(model: Model, max_nodes: int = 6000) -> Mesh:
    m = model.member
    s = m.section
    if s.kind == "circle":
        wa = wb = s.d
        na = nb = 8
    else:
        wa, wb = s.b, s.h
        big = max(wa, wb)
        na = max(2, int(round(6 * wa / big / 2)) * 2)  # even, so a centre node line exists
        nb = max(2, int(round(6 * wb / big / 2)) * 2)
    ha, hb = wa / na, wb / nb
    per_slice = (na + 1) * (nb + 1)
    nl = int(round(m.length / (2 * max(ha, hb))))
    nl = max(10, min(nl, max_nodes // per_slice - 1, 160))
    hl = m.length / nl

    # cross-section occupancy
    ia, ib = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
    if s.kind == "circle":
        ca = (ia + 0.5) * ha - wa / 2
        cb = (ib + 0.5) * hb - wb / 2
        occ = ca ** 2 + cb ** 2 <= (s.d / 2) ** 2 * 1.02
    else:
        occ = np.ones_like(ia, bool)

    if m.orientation == "horizontal":  # x = length, y = b, z = h
        grid = (nl, na, nb)
        h = (hl, ha, hb)
        origin = np.array([0.0, -wa / 2, -wb / 2])
        axis = 0
        occ3 = np.broadcast_to(occ[None, :, :], grid)
    else:  # z = length, x = b, y = h
        grid = (na, nb, nl)
        h = (ha, hb, hl)
        origin = np.array([-wa / 2, -wb / 2, 0.0])
        axis = 2
        occ3 = np.broadcast_to(occ[:, :, None], grid)

    cells = np.argwhere(occ3)
    gn = np.array(grid) + 1
    node_id = lambda ijk: (ijk[..., 0] * gn[1] + ijk[..., 1]) * gn[2] + ijk[..., 2]  # noqa: E731
    raw = node_id(cells[:, None, :] + CORNERS[None, :, :])  # (ne, 8)
    used, inv = np.unique(raw, return_inverse=True)
    ijk = np.stack(np.unravel_index(used, tuple(gn)), axis=1)
    # number nodes slice by slice along the member: keeps the matrix banded
    order = np.lexsort(tuple(ijk[:, a] for a in range(3) if a != axis) + (ijk[:, axis],))
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    elems = rank[inv].reshape(raw.shape)
    ijk = ijk[order]
    nodes = origin + ijk * np.array(h)
    mesh = Mesh(nodes, elems, cells, grid, h, origin, axis)
    mesh.layer = ijk[:, axis]
    return mesh


def _encode(a: np.ndarray, dtype=np.float32) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


FIELD_INFO = {
    # key: (label, unit, group)
    "von_mises": ("von Mises stress", "MPa", "Stress"),
    "tresca": ("Tresca stress", "MPa", "Stress"),
    "s_max": ("Max principal stress S1", "MPa", "Stress"),
    "s_min": ("Min principal stress S3", "MPa", "Stress"),
    "S11": ("Stress S11 (σxx)", "MPa", "Stress"),
    "S22": ("Stress S22 (σyy)", "MPa", "Stress"),
    "S33": ("Stress S33 (σzz)", "MPa", "Stress"),
    "S12": ("Stress S12 (τxy)", "MPa", "Stress"),
    "S13": ("Stress S13 (τxz)", "MPa", "Stress"),
    "S23": ("Stress S23 (τyz)", "MPa", "Stress"),
    "U": ("Displacement |U|", "mm", "Displacement"),
    "U1": ("Displacement U1 (x)", "mm", "Displacement"),
    "U2": ("Displacement U2 (y)", "mm", "Displacement"),
    "U3": ("Displacement U3 (z)", "mm", "Displacement"),
    "E_eq": ("Equivalent strain", "µε", "Strain"),
    "E11": ("Strain E11 (εxx)", "µε", "Strain"),
    "E22": ("Strain E22 (εyy)", "µε", "Strain"),
    "E33": ("Strain E33 (εzz)", "µε", "Strain"),
    "E12": ("Strain E12 (εxy)", "µε", "Strain"),
    "E13": ("Strain E13 (εxz)", "µε", "Strain"),
    "E23": ("Strain E23 (εyz)", "µε", "Strain"),
    "SED": ("Strain energy density", "kJ/m³", "Energy"),
}


@dataclass
class Results:
    model: Model
    mesh: Mesh
    u: np.ndarray  # (nn, 3)
    fields: dict[str, np.ndarray]
    reactions: np.ndarray  # total (3,) N
    support_reactions: list[dict]
    modes: list[dict]  # [{"f": Hz, "shape": (nn, 3)}]
    stats: dict

    def payload(self) -> dict:
        """Compact JSON for the browser: base64 float32 arrays."""
        m = self.mesh
        fields = {}
        for k, v in self.fields.items():
            label, unit, group = FIELD_INFO[k]
            fields[k] = {"label": label, "unit": unit, "group": group, "data": _encode(v),
                         "min": float(v.min()), "max": float(v.max()),
                         "argmax": int(np.argmax(np.abs(v)))}
        for i, md in enumerate(self.modes):
            mag = np.linalg.norm(md["shape"], axis=1)
            fields[f"mode{i + 1}"] = {"label": f"Mode {i + 1} · {md['f']:.2f} Hz", "unit": "norm.",
                                      "group": "Mode shapes", "data": _encode(mag), "min": 0.0,
                                      "max": float(mag.max()), "argmax": int(np.argmax(mag)),
                                      "shape": _encode(md["shape"]), "freq": md["f"]}
        return {
            "nodes": _encode(m.nodes),
            "elems": _encode(m.elems, np.int32),
            "cells": _encode(m.cells, np.int32),
            "grid": list(m.grid),
            "h": list(m.h),
            "origin": m.origin.tolist(),
            "axis": m.axis,
            "disp": _encode(self.u),
            "fields": fields,
            "field_order": list(fields),
            "stats": self.stats,
            "reactions": {"total_kN": (self.reactions / 1e3).round(4).tolist(), "supports": self.support_reactions},
        }


def _slice_nodes(mesh: Mesh, t: float) -> tuple[np.ndarray, int]:
    nl = mesh.grid[mesh.axis]
    k = int(round(np.clip(t, 0, 1) * nl))
    return np.flatnonzero(mesh.layer == k), k


def _bearing_line(mesh: Mesh, idx: np.ndarray) -> np.ndarray:
    """Beam: the bottom row of a slice. Column: the centre row (y = 0) of a slice."""
    p = mesh.nodes[idx]
    if mesh.axis == 0:
        z = p[:, 2]
        return idx[np.isclose(z, z.min())]
    y = np.abs(p[:, 1])
    return idx[np.isclose(y, y.min())]


def restraints(model: Model, mesh: Mesh) -> tuple[np.ndarray, list[tuple[int, np.ndarray]]]:
    """Restrained DOF indices, plus (support index, its dofs) for reaction reporting."""
    fixed = set()
    per_support = []
    for si, s in enumerate(model.supports):
        idx, k = _slice_nodes(mesh, s.t)
        if s.kind == "fixed":
            comps = (0, 1, 2)
            nodes = idx
        else:
            nodes = _bearing_line(mesh, idx)
            if mesh.axis == 0:  # beam: pin holds x, y, z on the bottom line; roller slides along x only
                comps = (0, 1, 2) if s.kind == "pinned" else (1, 2)
            else:  # column: base pin holds x, y, z; above the base pin and roller hold it sideways only
                comps = (0, 1, 2) if (s.kind == "pinned" and k == 0) else (0, 1)
        dofs = (3 * nodes[:, None] + np.array(comps)[None, :]).ravel()
        fixed.update(dofs.tolist())
        per_support.append((si, dofs))
    return np.array(sorted(fixed), dtype=int), per_support


def load_vector(model: Model, mesh: Mesh) -> np.ndarray:
    f = np.zeros(3 * mesh.nn)
    L = model.member.length
    nl = mesh.grid[mesh.axis]
    hl = L / nl
    for ld in model.loads:
        comp = AXES.index(ld.axis)
        if ld.kind == "point":
            idx, _ = _slice_nodes(mesh, ld.t0)
            f[3 * idx + comp] += ld.sign * ld.w0 * 1e3 / len(idx)
            continue
        x0, x1 = sorted((ld.t0 * L, (ld.t1 if ld.t1 is not None else 1.0) * L))
        w0, w1 = ld.w0, ld.w1 if (ld.kind == "trapezoidal" and ld.w1 is not None) else ld.w0
        if ld.t1 is not None and ld.t1 < ld.t0:
            w0, w1 = w1, w0
        for k in range(nl + 1):
            a, b = max(x0, (k - 0.5) * hl), min(x1, (k + 0.5) * hl)
            if b <= a:
                continue
            xm = 0.5 * (a + b)
            w = w0 + (w1 - w0) * ((xm - x0) / (x1 - x0) if x1 > x0 else 0.0)  # kN/m == N/mm
            idx = np.flatnonzero(mesh.layer == k)
            f[3 * idx + comp] += ld.sign * w * (b - a) / len(idx)
    return f


def solve(model: Model, n_modes: int = 6, max_nodes: int = 6000) -> Results:
    if model.issues():
        raise SolveError(model.issues()[0])
    t0 = time.perf_counter()
    mesh = build_mesh(model, max_nodes)
    mat = model.material
    D = elasticity(mat.E, mat.nu)
    Ke, Bc = brick_matrices(mesh.h, D)
    ne, nn = len(mesh.elems), mesh.nn
    edofs = (3 * mesh.elems[:, :, None] + np.arange(3)[None, None, :]).reshape(ne, 24)
    rows = np.repeat(edofs, 24, axis=1).ravel()
    cols = np.tile(edofs, (1, 24)).ravel()
    K = sp.coo_matrix((np.tile(Ke.ravel(), ne), (rows, cols)), shape=(3 * nn, 3 * nn)).tocsr()
    vol = float(np.prod(mesh.h))
    mass = np.zeros(nn)
    np.add.at(mass, mesh.elems.ravel(), mat.rho * vol / 8.0)
    t_asm = time.perf_counter()

    fixed, per_support = restraints(model, mesh)
    free = np.setdiff1d(np.arange(3 * nn), fixed)
    f = load_vector(model, mesh)
    Kff = K[free][:, free].tocsc()
    u = np.zeros(3 * nn)
    try:
        # nodes are numbered slice by slice, so the natural order is already banded
        lu = spla.splu(Kff, permc_spec="NATURAL")
        u[free] = lu.solve(f[free])
    except RuntimeError as exc:  # singular: a mechanism
        raise SolveError("The supports let the member move freely: add or change a support") from exc
    if not np.all(np.isfinite(u)) or np.abs(u).max() > 1e3 * model.member.length:
        raise SolveError("The supports let the member move freely: add or change a support")
    t_sol = time.perf_counter()

    r = K @ u - f
    reactions = np.array([r[fixed][fixed % 3 == c].sum() for c in range(3)])
    sup_reac = []
    for si, dofs in per_support:
        s = model.supports[si]
        sup_reac.append({"kind": s.kind, "t": s.t,
                         "kN": [float(r[dofs[dofs % 3 == c]].sum() / 1e3) for c in range(3)]})

    # Gauss-point strains extrapolated to each element's corners, then averaged at shared nodes
    ue = u[edofs]  # (ne, 24)
    eps_c = np.einsum("cij,ej->eci", Bc, ue).reshape(-1, 6)  # engineering shear
    cnt = np.zeros(nn)
    np.add.at(cnt, mesh.elems.ravel(), 1.0)
    en = np.zeros((nn, 6))
    np.add.at(en, mesh.elems.ravel(), eps_c)
    en /= cnt[:, None]
    sn = en @ D.T
    s11, s22, s33, s12, s23, s13 = sn.T
    e11, e22, e33, g12, g23, g13 = en.T
    tens = np.empty((nn, 3, 3))
    tens[:, 0, 0], tens[:, 1, 1], tens[:, 2, 2] = s11, s22, s33
    tens[:, 0, 1] = tens[:, 1, 0] = s12
    tens[:, 1, 2] = tens[:, 2, 1] = s23
    tens[:, 0, 2] = tens[:, 2, 0] = s13
    pr = np.linalg.eigvalsh(tens)  # ascending
    vm = np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3 * (s12 ** 2 + s23 ** 2 + s13 ** 2))
    e_eq = math.sqrt(2) / 3 * np.sqrt((e11 - e22) ** 2 + (e22 - e33) ** 2 + (e33 - e11) ** 2
                                      + 1.5 * (g12 ** 2 + g23 ** 2 + g13 ** 2))
    sed = 0.5 * np.einsum("ij,ij->i", sn, en)  # MPa = MJ/m³
    U = u.reshape(nn, 3)
    fields = {
        "von_mises": vm, "tresca": pr[:, 2] - pr[:, 0], "s_max": pr[:, 2], "s_min": pr[:, 0],
        "S11": s11, "S22": s22, "S33": s33, "S12": s12, "S13": s13, "S23": s23,
        "U": np.linalg.norm(U, axis=1), "U1": U[:, 0], "U2": U[:, 1], "U3": U[:, 2],
        "E_eq": e_eq * 1e6, "E11": e11 * 1e6, "E22": e22 * 1e6, "E33": e33 * 1e6,
        "E12": g12 / 2 * 1e6, "E13": g13 / 2 * 1e6, "E23": g23 / 2 * 1e6,
        "SED": sed * 1e3,
    }

    modes = []
    if n_modes:
        Mff = sp.diags(np.repeat(mass, 3)[free]).tocsc()
        try:
            # shift-invert about 0 reuses the static factorisation
            opinv = spla.LinearOperator(Kff.shape, matvec=lu.solve, dtype=float)
            vals, vecs = spla.eigsh(Kff, k=n_modes, M=Mff, sigma=0, which="LM", OPinv=opinv)
            order = np.argsort(vals)
            for j in order:
                shape = np.zeros(3 * nn)
                shape[free] = vecs[:, j]
                shape = shape.reshape(nn, 3)
                shape /= np.abs(shape).max() or 1.0
                modes.append({"f": float(math.sqrt(max(vals[j], 0)) / (2 * math.pi)), "shape": shape})
        except Exception:  # noqa: BLE001 - modes are a bonus; statics still stand
            modes = []
    t_end = time.perf_counter()

    stats = {
        "element": "8-node brick, incompatible modes (Wilson-Taylor), 2×2×2 Gauss",
        "elements": int(ne), "nodes": int(nn), "dof": int(3 * nn), "restrained_dof": int(len(fixed)),
        "free_dof": int(len(free)),
        "grid": list(mesh.grid), "cell_mm": [round(v, 2) for v in mesh.h],
        "aspect": round(max(mesh.h) / min(mesh.h), 2),
        "solver": "sparse direct LU (SuperLU)", "eigen": "shift-invert Lanczos (ARPACK), lumped mass",
        "t_assemble_s": round(t_asm - t0, 3), "t_solve_s": round(t_sol - t_asm, 3),
        "t_modes_s": round(t_end - t_sol, 3),
        "max_U_mm": float(fields["U"].max()), "max_von_mises_MPa": float(vm.max()),
        "strain_energy_J": float(0.5 * u @ (K @ u) / 1e3),
        "applied_kN": (np.array([f[c::3].sum() for c in range(3)]) / 1e3).round(4).tolist(),
    }
    return Results(model, mesh, U, fields, reactions, sup_reac, modes, stats)
