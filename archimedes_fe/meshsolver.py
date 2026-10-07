"""General hex8 finite-element solver for unstructured meshes (numpy + scipy, pyamg for big models).

The browser CAD/FEM workbench meshes arbitrary parts into 8-node hexahedra whose
boundary nodes are snapped onto curved faces, so elements are distorted (but must
keep a positive Jacobian). This module solves such a mesh directly: no grid
assumptions, a full isoparametric Jacobian at every Gauss point.

Element
    Trilinear hex8, 2×2×2 Gauss. For the linear analyses (static, modal, the
    buckling prestress) the default adds Wilson–Taylor incompatible modes: nine
    bubble modes P_k = 1 − ξ_k² per element, condensed out statically. Their
    gradients use the Jacobian at the element centre, scaled by det J0 / det J
    (Taylor's correction) so ∫ B_a dV = 0 and the element passes the patch test
    even when distorted. This removes shear locking in bending. Nonlinear
    (plasticity) runs on the plain hex8.

Analyses
    static     linear elastic (+ thermal strain α·dT, + compression-only contact)
    nonlinear  small-strain J2 plasticity, linear isotropic hardening, radial
               return + consistent tangent, Newton–Raphson in load increments
    modal      lowest natural frequencies, lumped mass, shift-invert Lanczos
    buckling   linear buckling K φ = λ (−K_σ) φ from the static stress field

Units: mm, N, MPa, tonne/mm³ (the material dict gives rho in kg/m³), s.
Local node order: n = a + 2b + 4c, reference corner (ξ, η, ζ) = (2a−1, 2b−1, 2c−1).
Voigt order [11, 22, 33, 12, 23, 13], engineering shear strains.
"""
from __future__ import annotations

import base64
import math
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .solver import elasticity

# ---------------------------------------------------------------- reference element
CORNERS = np.array([[a, b, c] for c in (0, 1) for b in (0, 1) for a in (0, 1)])
SGN = (2 * CORNERS - 1).astype(float)  # (8, 3) natural coords of the nodes
GP = SGN / math.sqrt(3.0)  # 2×2×2 Gauss points in corner order, all weights 1


def _shape(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trilinear N (p, 8) and dN/dξ (p, 8, 3) at natural points q (p, 3)."""
    f = 1 + SGN[None, :, :] * q[:, None, :]  # (p, 8, 3)
    N = f.prod(axis=2) / 8
    dN = np.empty(f.shape)
    dN[..., 0] = SGN[:, 0] * f[..., 1] * f[..., 2] / 8
    dN[..., 1] = SGN[:, 1] * f[..., 0] * f[..., 2] / 8
    dN[..., 2] = SGN[:, 2] * f[..., 0] * f[..., 1] / 8
    return N, dN


N_GP, DN_GP = _shape(GP)  # (8 gauss, 8 nodes), (8, 8, 3)
DN_C = _shape(np.zeros((1, 3)))[1][0]  # (8, 3) at the element centre
EXTRAP = np.linalg.inv(N_GP)  # (corner, gauss): Gauss values -> trilinear corner values
VOIGT_I = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
CHUNK = 4000  # elements per vectorised block (bounds temporary memory to ~100 MB)
# solver="auto" switches from SuperLU to AMG-preconditioned CG above these free-dof counts. SuperLU has no
# nested-dissection ordering, so 3-D fill grows fast: measured here 6 s at 19k dofs, 33 s at 40k and 95 s at
# 65k, against 3-12 s for CG + smoothed aggregation. Eigen solves need many shift-invert solves and keep the
# factorisation longer.
DIRECT_MAX_DOF = 20_000
DIRECT_MAX_DOF_EIGEN = 60_000

# Above those counts a slender part (a beam, a shaft) still factorises cheaply: its reverse Cuthill–McKee
# envelope n·bandwidth, which tracks the LU fill, stays small. Measured: a 600 mm cantilever at 26k dofs has an
# envelope of 1.4e7 and factorises in 1.8 s (AMG-CG 11 s); a 19³ cube at 23k dofs has 7e7 and takes 8 s.
FILL_FACTOR = 18.0  # direct while n·bw < 18 · direct_max^1.5: 5e7 for static, 2.6e8 for modal / buckling


def _envelope(A: sp.csr_matrix) -> float:
    """n × bandwidth after reverse Cuthill–McKee: a cheap estimate of the LU fill."""
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    p = reverse_cuthill_mckee(A, symmetric_mode=True)
    B = A[p][:, p].tocoo()
    return float(A.shape[0]) * float(np.abs(B.row.astype(np.int64) - B.col).max(initial=0))


SUPPORT_MSG = ("not enough supports: the model can move as a rigid body or is a mechanism. "
               "Add fixed or contact supports so every rigid-body motion is prevented")


class _Singular(ValueError):
    pass


def _report(progress, stage: str, frac: float) -> None:
    if progress is not None:
        progress(stage, float(min(max(frac, 0.0), 1.0)))


def _bmat(dNx: np.ndarray) -> np.ndarray:
    """Strain-displacement matrices from gradients dNx (..., n, 3) -> (..., 6, 3n)."""
    lead, n = dNx.shape[:-2], dNx.shape[-2]
    B = np.zeros(lead + (6, n, 3))
    dx, dy, dz = dNx[..., 0], dNx[..., 1], dNx[..., 2]
    B[..., 0, :, 0] = dx
    B[..., 1, :, 1] = dy
    B[..., 2, :, 2] = dz
    B[..., 3, :, 0], B[..., 3, :, 1] = dy, dx
    B[..., 4, :, 1], B[..., 4, :, 2] = dz, dy
    B[..., 5, :, 0], B[..., 5, :, 2] = dz, dx
    return B.reshape(lead + (6, 3 * n))


def _geometry(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Physical gradients dN/dx (nc, 8g, 8n, 3) and det J (nc, 8g) for element coords X (nc, 8, 3)."""
    J = np.einsum("gni,enj->egij", DN_GP, X)  # J_ij = dx_j / dξ_i
    det = np.linalg.det(J)
    dNx = np.einsum("egji,gni->egnj", np.linalg.inv(J), DN_GP)
    return dNx, det


def _iso(Xg: np.ndarray, Yg: np.ndarray, w: np.ndarray, lam: float, mu: float) -> np.ndarray:
    """Isotropic stiffness block between gradient sets Xg (nc, 8, p, 3) and Yg (nc, 8, q, 3) -> (nc, 3p, 3q).

    K[a,i,b,j] = Σ_g w (λ ∂_i Na ∂_j Nb + μ ∂_j Na ∂_i Nb + μ δ_ij ∇Na·∇Nb), i.e. ∫ Bᵀ D B dV written
    on the gradients: one batched (3p × 8)(8 × 3q) product per element instead of (24 × 48)(48 × 24).
    """
    nc, _, p, _ = Xg.shape
    q = Yg.shape[2]
    Xw = (Xg * w[..., None, None]).reshape(nc, 8, 3 * p)
    T = (Xw.transpose(0, 2, 1) @ Yg.reshape(nc, 8, 3 * q)).reshape(nc, p, 3, q, 3)
    K = lam * T + mu * T.transpose(0, 1, 4, 3, 2)
    tr = np.einsum("eakbk->eab", T)
    for i in range(3):
        K[:, :, i, :, i] += mu * tr
    return K.reshape(nc, 3 * p, 3 * q)


def _element_ops(X: np.ndarray, lam: float, mu: float, incompatible: bool):
    """Element stiffness (nc, 24, 24) and the Gauss-point geometry needed to recover strains.

    geo holds dN/dx (nc, 8g, 8n, 3) and w = det J (nc, 8g); with incompatible modes also the
    bubble gradients dPx (nc, 8g, 3 modes, 3) and the condensation R (nc, 9, 24): the bubble
    amplitudes are α = R u with R = −Kaa⁻¹ Kau, and K = Kuu + Kua R.
    """
    dNx, w = _geometry(X)
    K = _iso(dNx, dNx, w, lam, mu)
    geo = {"dNx": dNx, "w": w}
    if incompatible:
        J0 = np.einsum("ni,enj->eij", DN_C, X)
        det0 = np.linalg.det(J0)
        # dP_m/dξ_i = −2 ξ_m δ_mi, mapped with the centre Jacobian and scaled by det J0 / det J (Taylor),
        # so Σ_g w dPx = 0: a constant strain field never excites the bubbles (patch test)
        dPx = np.einsum("ejm,gm->egmj", np.linalg.inv(J0), -2.0 * GP) * (det0[:, None] / w)[..., None, None]
        Kua = _iso(dNx, dPx, w, lam, mu)  # (nc, 24, 9)
        Kaa = _iso(dPx, dPx, w, lam, mu)  # (nc, 9, 9)
        R = -np.linalg.solve(Kaa, Kua.transpose(0, 2, 1))  # (nc, 9, 24)
        K = K + Kua @ R
        geo.update(dPx=dPx, R=R)
    return 0.5 * (K + K.transpose(0, 2, 1)), geo


def _voigt(g: np.ndarray) -> np.ndarray:
    """Displacement gradients (..., 3, 3), g_ij = ∂u_i/∂x_j -> strains (..., 6), engineering shear."""
    return np.stack([g[..., 0, 0], g[..., 1, 1], g[..., 2, 2], g[..., 0, 1] + g[..., 1, 0],
                     g[..., 1, 2] + g[..., 2, 1], g[..., 0, 2] + g[..., 2, 0]], axis=-1)


def _strain(geo: dict, ue: np.ndarray) -> np.ndarray:
    """Gauss-point strains (nc, 8, 6) from element displacements ue (nc, 24), bubbles included."""
    g = np.einsum("egnj,eni->egij", geo["dNx"], ue.reshape(-1, 8, 3))
    if "R" in geo:
        a = np.einsum("emk,ek->em", geo["R"], ue).reshape(-1, 3, 3)  # (nc, mode, component)
        g += np.einsum("egmj,emi->egij", geo["dPx"], a)
    return _voigt(g)


def _rigid_modes(xyz: np.ndarray) -> np.ndarray:
    """Six rigid-body modes (3 nn, 6) about the centroid: the AMG near-nullspace."""
    c = xyz - xyz.mean(axis=0)
    nn = len(c)
    B = np.zeros((nn, 3, 6))
    B[:, 0, 0] = B[:, 1, 1] = B[:, 2, 2] = 1.0
    B[:, 1, 3], B[:, 2, 3] = -c[:, 2], c[:, 1]  # about x
    B[:, 0, 4], B[:, 2, 4] = c[:, 2], -c[:, 0]  # about y
    B[:, 0, 5], B[:, 1, 5] = -c[:, 1], c[:, 0]  # about z
    return B.reshape(3 * nn, 6)


# ---------------------------------------------------------------- linear algebra
class _LinearSolver:
    """Solves A_ff x = b on the free dofs.

    direct: SuperLU with minimum-degree ordering on Aᵀ+A (symmetric mode). A
    pivot below 1e-12 × the largest diagonal flags a mechanism (measured:
    mechanisms give ~1e-14, a 2000:1 one-element-deep beam ~7e-11).
    amg: pyamg smoothed aggregation (rigid-body modes as near-nullspace, 3×3
    blocks) preconditioning CG on the full-size system with constrained rows
    replaced by identity; CG + Jacobi if pyamg is missing.
    """

    def __init__(self, A: sp.csr_matrix, free: np.ndarray, xyz: np.ndarray, method: str = "auto",
                 direct_max: int = DIRECT_MAX_DOF):
        self.free = free
        nf = len(free)
        if nf == 0:
            raise ValueError("every dof is constrained: nothing to solve")
        if method not in ("auto", "direct", "amg"):
            raise ValueError(f"unknown solver {method!r} (use 'auto', 'direct' or 'amg')")
        self.Aff = A[free][:, free].tocsr()
        if method == "auto":
            method = "direct" if nf <= direct_max or _envelope(self.Aff) < FILL_FACTOR * direct_max ** 1.5 else "amg"
        dmax = float(np.abs(self.Aff.diagonal()).max())
        if method == "direct":
            try:
                lu = spla.splu(self.Aff.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                               options={"SymmetricMode": True})
            except RuntimeError as exc:  # exactly singular
                raise _Singular(SUPPORT_MSG) from exc
            piv = np.abs(lu.U.diagonal())
            if not np.all(np.isfinite(piv)) or piv.min() < 1e-12 * dmax:
                raise _Singular(SUPPORT_MSG)
            self._lu = lu
            self.name = "SuperLU direct (MMD on AᵀA+A)"
            return
        n = A.shape[0]
        mask = np.zeros(n, bool)
        mask[free] = True
        scale = float(self.Aff.diagonal().mean())
        P = sp.diags(mask.astype(float))
        self._A = (P @ A @ P + sp.diags((~mask) * scale)).tocsr()
        try:
            import pyamg
        except ImportError:
            pyamg = None
        if pyamg is not None and n % 3 == 0:
            ml = pyamg.smoothed_aggregation_solver(self._A.tobsr(blocksize=(3, 3)), B=_rigid_modes(xyz),
                                                   max_coarse=500)
            self._M = ml.aspreconditioner(cycle="V")
            self._maxiter = 500  # SA-preconditioned CG takes ~20-60 iterations on a supported model
            self.name = "CG + pyamg smoothed aggregation (rigid-body near-nullspace), rtol 1e-8"
        else:
            d = self._A.diagonal()
            self._M = sp.diags(1.0 / np.where(d > 0, d, 1.0))
            self._maxiter = max(5000, n // 2)
            self.name = "CG + Jacobi (pyamg not installed), rtol 1e-8"
        self._lu = None

    def solve(self, b: np.ndarray) -> np.ndarray:
        if self._lu is not None:
            x = self._lu.solve(b)
        else:
            full = np.zeros(self._A.shape[0])
            full[self.free] = b
            xf, info = spla.cg(self._A, full, rtol=1e-8, M=self._M, maxiter=self._maxiter)
            if info != 0:
                raise _Singular("iterative solver did not converge: " + SUPPORT_MSG)
            x = xf[self.free]
        if not np.all(np.isfinite(x)):
            raise _Singular(SUPPORT_MSG)
        return x

    def operator(self) -> spla.LinearOperator:
        return spla.LinearOperator(self.Aff.shape, matvec=self.solve, dtype=float)


# ---------------------------------------------------------------- problem setup
class _Model:
    """Mesh, material, constraints and the per-chunk element operators."""

    def __init__(self, nodes, elems, material, fixed, prescribed, dT, incompatible):
        self.nodes = np.ascontiguousarray(nodes, dtype=float)
        self.elems = np.ascontiguousarray(elems, dtype=np.int64)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must be an (nn, 3) array")
        if self.elems.ndim != 2 or self.elems.shape[1] != 8:
            raise ValueError("elems must be an (ne, 8) array")
        self.nn, self.ne = len(self.nodes), len(self.elems)
        if self.ne == 0:
            raise ValueError("the mesh has no elements")
        if self.elems.min() < 0 or self.elems.max() >= self.nn:
            raise ValueError("elems reference node indices outside nodes")
        self.ndof = 3 * self.nn
        self.size = float(np.linalg.norm(np.ptp(self.nodes, axis=0))) or 1.0

        m = material
        self.E, self.nu = float(m["E"]), float(m["nu"])
        self.rho = float(m.get("rho", 0.0)) * 1e-12  # kg/m³ -> t/mm³
        self.alpha = float(m.get("alpha", 0.0))
        self.sy = float(m.get("sy", np.inf))
        self.H = float(m.get("H", self.E / 100.0))
        self.D = elasticity(self.E, self.nu)
        self.C = np.linalg.inv(self.D)  # compliance, for elastic strain energy
        self.G = self.E / (2 * (1 + self.nu))
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.incompatible = incompatible

        dT = np.asarray(dT, dtype=float)
        if dT.ndim and dT.shape != (self.nn,):
            raise ValueError("dT must be a scalar or one value per node")
        self.dT = dT if dT.ndim else float(dT)
        self.thermal = bool(np.any(dT != 0)) and self.alpha != 0.0

        # constraints: fixed (value 0) plus prescribed (value given)
        vals = {int(d): 0.0 for d in (fixed if fixed is not None else [])}
        for d, v in (prescribed or {}).items():
            vals[int(d)] = float(v)
        cons = np.array(sorted(vals), dtype=np.int64)
        if len(cons) and (cons.min() < 0 or cons.max() >= self.ndof):
            raise ValueError("a fixed/prescribed dof is outside 0 .. 3*nn-1")
        self.cons = cons
        self.ucons = np.array([vals[d] for d in cons.tolist()], dtype=float)
        # nodes no element uses carry no stiffness: drop their dofs from the system
        used = np.zeros(self.nn, bool)
        used[self.elems.ravel()] = True
        mask = np.repeat(used, 3)
        mask[cons] = False
        self.free = np.flatnonzero(mask)

        self.edofs = (3 * self.elems[:, :, None] + np.arange(3)).reshape(self.ne, 24)
        self.rows = np.repeat(self.edofs, 24, axis=1).astype(np.int32).ravel()
        self.cols = np.tile(self.edofs, (1, 24)).astype(np.int32).ravel()
        self.chunks = [slice(i, min(i + CHUNK, self.ne)) for i in range(0, self.ne, CHUNK)]
        self._cache = {} if self.ne <= 12000 else None
        self.check_jacobians()

    def check_jacobians(self) -> None:
        for s in self.chunks:
            X = self.nodes[self.elems[s]]
            det = np.linalg.det(np.einsum("gni,enj->egij", DN_GP, X))
            det_c = np.linalg.det(np.einsum("ni,enj->eij", DN_C, X))
            bad = (det.min(axis=1) <= 0) | (det_c <= 0) | ~np.isfinite(det).all(axis=1)
            if bad.any():
                e = int(np.flatnonzero(bad)[0])
                raise ValueError(f"element {s.start + e} has a non-positive Jacobian (min det J = "
                                 f"{det[e].min():.3g} mm³): it is inverted, degenerate, or its nodes are not "
                                 f"in the order n = a + 2b + 4c")

    def ops(self, i: int):
        """(K_e, geo) for chunk i; geo is cached on small meshes for the recovery pass."""
        K, geo = _element_ops(self.nodes[self.elems[self.chunks[i]]], self.lam, self.G, self.incompatible)
        if self._cache is not None:
            self._cache[i] = geo
        return K, geo

    def geo(self, i: int) -> dict:
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        if self.incompatible:
            return self.ops(i)[1]
        dNx, w = _geometry(self.nodes[self.elems[self.chunks[i]]])
        geo = {"dNx": dNx, "w": w}
        if self._cache is not None:
            self._cache[i] = geo
        return geo

    def dT_gauss(self, s: slice) -> np.ndarray:
        if isinstance(self.dT, float):
            return np.full((s.stop - s.start, 8), self.dT)
        return np.einsum("gn,en->eg", N_GP, self.dT[self.elems[s]])

    def matrix(self, data: np.ndarray) -> sp.csr_matrix:
        """Global matrix from element matrices (ne, 24, 24): one COO -> CSR build."""
        return sp.coo_matrix((data.ravel(), (self.rows, self.cols)), shape=(self.ndof, self.ndof)).tocsr()

    def scatter(self, fe: np.ndarray) -> np.ndarray:
        """Global vector from element vectors (ne, 24)."""
        return np.bincount(self.edofs.ravel(), weights=fe.ravel(), minlength=self.ndof)

    def nodal(self, vg: np.ndarray) -> np.ndarray:
        """Gauss values (ne, 8, k) -> extrapolated to the corners, averaged at shared nodes (nn, k)."""
        vc = np.einsum("cg,egk->eck", EXTRAP, vg)
        idx = self.elems.ravel()
        cnt = np.maximum(np.bincount(idx, minlength=self.nn), 1)
        return np.stack([np.bincount(idx, weights=vc[:, :, k].ravel(), minlength=self.nn) / cnt
                         for k in range(vg.shape[2])], axis=1)


def _contact_setup(contact, mdl: _Model):
    if not contact:
        return None
    cn, cv = [], []
    for grp in contact:
        idx = np.asarray(grp["nodes"], dtype=np.int64).ravel()
        n = np.asarray(grp["normal"], dtype=float).reshape(3)
        nrm = np.linalg.norm(n)
        if nrm == 0:
            raise ValueError("contact normal must be non-zero")
        if len(idx) and (idx.min() < 0 or idx.max() >= mdl.nn):
            raise ValueError("contact node index outside nodes")
        cn.append(idx)
        cv.append(np.repeat((n / nrm)[None], len(idx), axis=0))
    cn = np.concatenate(cn)
    if not len(cn):
        return None
    return {"nodes": cn, "normal": np.concatenate(cv), "tol": 1e-10 * mdl.size}


def _contact_matrix(ct, active: np.ndarray, k: float, ndof: int) -> sp.csr_matrix:
    """Penalty springs k n nᵀ at the active contact nodes."""
    idx, nv = ct["nodes"][active], ct["normal"][active]
    d = 3 * idx[:, None] + np.arange(3)
    rows = np.broadcast_to(d[:, :, None], (len(idx), 3, 3)).ravel()
    cols = np.broadcast_to(d[:, None, :], (len(idx), 3, 3)).ravel()
    vals = (k * nv[:, :, None] * nv[:, None, :]).ravel()
    return sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()


def _update_active(ct, u: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Release active nodes pulled off the support (tensile), activate inactive nodes that penetrate."""
    gap = np.einsum("ij,ij->i", u.reshape(-1, 3)[ct["nodes"]], ct["normal"])  # u·n, > 0 = into the support
    new = active.copy()
    new[active & (gap < -ct["tol"])] = False
    new[~active & (gap > ct["tol"])] = True
    return new


# ---------------------------------------------------------------- linear analyses
def _assemble_linear(mdl: _Model, progress):
    """Global K, thermal load and lumped mass (nn,)."""
    data = np.empty((mdl.ne, 24, 24))
    fth = np.zeros((mdl.ne, 24))
    mass = np.zeros((mdl.ne, 8))
    for i, s in enumerate(mdl.chunks):
        Ke, geo = mdl.ops(i)
        data[s] = Ke
        w = geo["w"]
        mass[s] = mdl.rho * w @ N_GP  # row-sum (lumped) mass: ρ ∫ N_n dV
        if mdl.thermal:
            # f = ∫ Bᵀ D ε0 dV with D ε0 = E α dT / (1 − 2ν) I; the bubbles take no share (Σ_g w dPx = 0)
            s0 = mdl.E / (1 - 2 * mdl.nu) * mdl.alpha * mdl.dT_gauss(s)
            fth[s] = np.einsum("egni,eg->eni", geo["dNx"], s0 * w).reshape(-1, 24)
        _report(progress, "assemble", (i + 1) / len(mdl.chunks))
    K = mdl.matrix(data)
    del data
    m = np.bincount(mdl.elems.ravel(), weights=mass.ravel(), minlength=mdl.nn)
    return K, mdl.scatter(fth), m


def _gauss_stress_linear(mdl: _Model, u: np.ndarray):
    """σ = D (B u − ε0) at the Gauss points (ne, 8, 6), and det J (ne, 8)."""
    sig = np.empty((mdl.ne, 8, 6))
    wts = np.empty((mdl.ne, 8))
    for i, s in enumerate(mdl.chunks):
        geo = mdl.geo(i)
        eps = _strain(geo, u[mdl.edofs[s]])
        if mdl.thermal:
            eps = eps - (mdl.alpha * mdl.dT_gauss(s))[..., None] * VOIGT_I
        sig[s] = eps @ mdl.D.T
        wts[s] = geo["w"]
    return sig, wts


def _static(mdl: _Model, K, f, ct, solver, progress, direct_max=DIRECT_MAX_DOF):
    """Linear solve with the constraints, and the contact active-set loop when there is contact."""
    u = np.zeros(mdl.ndof)
    u[mdl.cons] = mdl.ucons
    free, cons = mdl.free, mdl.cons
    if ct is None:
        _report(progress, "solve", 0.0)
        S = _LinearSolver(K, free, mdl.nodes, solver, direct_max)
        u[free] = S.solve(f[free] - K[free][:, cons] @ mdl.ucons)
        _report(progress, "solve", 1.0)
        return u, S, sp.csr_matrix(K.shape), None
    kpen = 1e3 * float(K.diagonal()[free].mean())
    active = np.ones(len(ct["nodes"]), bool)
    seen = set()
    for it in range(1, 41):
        _report(progress, "contact", it / 40)
        Kc = _contact_matrix(ct, active, kpen, mdl.ndof)
        A = (K + Kc).tocsr()
        try:
            S = _LinearSolver(A, free, mdl.nodes, solver, direct_max)
        except _Singular as exc:
            raise ValueError(SUPPORT_MSG + f" (contact iteration {it}: {int(active.sum())} of "
                             f"{len(active)} contact nodes in compression - the load lifts the part off its "
                             f"compression-only supports)") from exc
        u[free] = S.solve(f[free] - A[free][:, cons] @ mdl.ucons)
        new = _update_active(ct, u, active)
        if np.array_equal(new, active):
            break
        key = new.tobytes()
        if key in seen:  # cycling between two sets: stop at the current one
            break
        seen.add(key)
        active = new
    info = {"active": int(active.sum()), "released": int((~active).sum()), "iterations": it}
    return u, S, Kc, info


def _reaction(mdl: _Model, r: np.ndarray, Kc, u: np.ndarray) -> np.ndarray:
    """Total support force on the part: Σ (internal − external) at constrained dofs + contact springs."""
    tot = np.zeros(mdl.ndof)
    tot[mdl.cons] = r[mdl.cons]
    fc = -(Kc @ u)
    fc[mdl.cons] = 0.0  # already inside r at constrained dofs
    tot += fc
    return tot.reshape(-1, 3).sum(axis=0)


def _geometric_stiffness(mdl: _Model, sig: np.ndarray) -> sp.csr_matrix:
    """K_σ = ∫ Gᵀ S G dV, plain hex8 gradients, the 3×3 Gauss stress S repeated for the three components."""
    data = np.zeros((mdl.ne, 8, 3, 8, 3))
    v = [0, 3, 5, 3, 1, 4, 5, 4, 2]  # Voigt index of S_ij
    for i, s in enumerate(mdl.chunks):
        geo = mdl.geo(i)
        dNx, w = geo["dNx"], geo["w"]
        S = sig[s][..., v].reshape(-1, 8, 3, 3)
        kg = np.einsum("egaj,egjk,egbk,eg->eab", dNx, S, dNx, w)
        for k in range(3):
            data[s, :, k, :, k] = kg
    return mdl.matrix(data.reshape(mdl.ne, 24, 24))


# ---------------------------------------------------------------- J2 plasticity
def _radial_return(ee: np.ndarray, alpha_n: np.ndarray, mdl: _Model):
    """Return mapping for von Mises with linear isotropic hardening σ_y = sy + H ε̄p.

    ee (m, 6) trial elastic strain, alpha_n (m,) equivalent plastic strain.
    Returns σ, Δε_p (engineering shear), Δγ (= Δε̄p) and the consistent tangent (m, 6, 6):
    C = D − 6G²Δγ/q_tr · I_dev + 6G² (Δγ/q_tr − 1/(3G+H)) N̂⊗N̂, N̂ = s_tr / ‖s_tr‖.
    """
    D, G, H = mdl.D, mdl.G, mdl.H
    sig = ee @ D.T
    s = sig.copy()
    s[:, :3] -= sig[:, :3].mean(axis=1, keepdims=True)
    nrm = np.sqrt((s[:, :3] ** 2).sum(1) + 2 * (s[:, 3:] ** 2).sum(1))
    q = math.sqrt(1.5) * nrm
    fy = q - (mdl.sy + H * alpha_n)
    yld = fy > 1e-10 * mdl.sy
    m = len(ee)
    C = np.broadcast_to(D, (m, 6, 6)).copy()
    deps = np.zeros_like(ee)
    dg = np.zeros(m)
    if yld.any():
        dg[yld] = fy[yld] / (3 * G + H)
        Nh = s[yld] / nrm[yld, None]  # unit deviatoric direction (tensor components)
        sig[yld] -= math.sqrt(6.0) * G * dg[yld, None] * Nh
        deps[yld] = math.sqrt(1.5) * dg[yld, None] * Nh * np.array([1, 1, 1, 2, 2, 2])
        Idev = np.diag([1, 1, 1, 0.5, 0.5, 0.5]) - np.outer(VOIGT_I, VOIGT_I) / 3
        a = 6 * G * G * dg[yld] / q[yld]
        b = 6 * G * G * (dg[yld] / q[yld] - 1 / (3 * G + H))
        C[yld] += -a[:, None, None] * Idev + b[:, None, None] * Nh[:, :, None] * Nh[:, None, :]
    return sig, deps, dg, C


def _plastic_pass(mdl: _Model, u, lam, eps_p, alpha):
    """Tangent K_t, internal force, and the trial Gauss state for displacement u at load factor lam."""
    data = np.empty((mdl.ne, 24, 24))
    fe = np.empty((mdl.ne, 24))
    sig = np.empty((mdl.ne, 8, 6))
    ep_new = np.empty_like(eps_p)
    al_new = np.empty_like(alpha)
    for i, s in enumerate(mdl.chunks):
        geo = mdl.geo(i)
        w = geo["w"]
        B = _bmat(geo["dNx"])  # (nc, 8, 6, 24)
        nc = s.stop - s.start
        ee = _strain(geo, u[mdl.edofs[s]]) - eps_p[s]
        if mdl.thermal:
            ee -= (lam * mdl.alpha * mdl.dT_gauss(s))[..., None] * VOIGT_I
        sg, deps, dg, C = _radial_return(ee.reshape(-1, 6), alpha[s].ravel(), mdl)
        sg = sg.reshape(nc, 8, 6)
        CB = (C.reshape(nc, 8, 6, 6) @ B).reshape(nc, 48, 24)
        Bw = (B * w[..., None, None]).reshape(nc, 48, 24)
        Ke = Bw.transpose(0, 2, 1) @ CB
        data[s] = 0.5 * (Ke + Ke.transpose(0, 2, 1))
        fe[s] = np.einsum("eki,ek->ei", Bw, sg.reshape(nc, 48))
        sig[s] = sg
        ep_new[s] = eps_p[s] + deps.reshape(nc, 8, 6)
        al_new[s] = alpha[s] + dg.reshape(nc, 8)
    return mdl.matrix(data), mdl.scatter(fe), sig, ep_new, al_new


def _nonlinear(mdl: _Model, f, ct, steps, solver, progress):
    """Load-controlled Newton–Raphson with step halving; contact active set updated per increment."""
    if not math.isfinite(mdl.sy):
        raise ValueError("nonlinear analysis needs the yield stress material['sy']")
    free, cons = mdl.free, mdl.cons
    steps = max(int(steps), 1)
    u = np.zeros(mdl.ndof)
    eps_p = np.zeros((mdl.ne, 8, 6))
    alpha = np.zeros((mdl.ne, 8))
    sig = np.zeros((mdl.ne, 8, 6))
    nc_all = len(ct["nodes"]) if ct else 0
    active = np.ones(nc_all, bool)
    kpen = None
    lam, dlam, total_it, cuts, set_its = 0.0, 1.0 / steps, 0, 0, 0
    history = [(0.0, 0.0)]
    Kc = sp.csr_matrix((mdl.ndof, mdl.ndof))
    first = True
    names = set()
    pre = {"S": None, "key": None}

    def newton(u_try, lt, act):
        nonlocal kpen, first
        its = 0
        for _ in range(30):
            Kt, fint, sg, epn, aln = _plastic_pass(mdl, u_try, lt, eps_p, alpha)
            if ct is not None:
                if kpen is None:
                    kpen = 1e3 * float(Kt.diagonal()[free].mean())
                Kc_ = _contact_matrix(ct, act, kpen, mdl.ndof)
            else:
                Kc_ = sp.csr_matrix(Kt.shape)
            r = fint + Kc_ @ u_try - lt * f
            # converged when the out-of-balance force is 1e-8 of the applied load (or of the reactions, for a
            # purely prescribed / thermal load); the absolute floor covers an unloaded model
            ref = max(np.linalg.norm(lt * f), np.linalg.norm(r[cons]))
            rn = np.linalg.norm(r[free])
            if not np.isfinite(rn):
                return False, its, None
            if rn <= 1e-8 * ref or rn <= 1e-14 * mdl.E * mdl.size ** 2:
                return True, its, (sg, epn, aln, Kc_, r)
            K = (Kt + Kc_).tocsr()
            key = act.tobytes() if ct is not None else b""
            du = None
            if pre["S"] is not None and pre["key"] == key:
                # the factorised tangent of an earlier iteration preconditions CG on this one: the elastic-plastic
                # tangent stays close to it, so a few back-substitutions replace a fresh factorisation
                Kff = K[free][:, free]
                x, info = spla.cg(Kff, r[free], rtol=1e-10, maxiter=40, M=pre["S"].operator())
                if info == 0 and np.all(np.isfinite(x)):
                    du = x
                    names.add("CG preconditioned by the factorised tangent")
            if du is None:
                try:
                    S = _LinearSolver(K, free, mdl.nodes, solver)
                except _Singular:
                    if first:
                        raise
                    return False, its, None
                first = False
                names.add(S.name)
                du = S.solve(r[free])
                if S._lu is not None:  # keep it as the preconditioner (an AMG-CG solve is not one)
                    pre.update(S=S, key=key)
            u_try[free] -= du
            its += 1
        return False, its, None

    while lam < 1.0 - 1e-12:
        dl = min(dlam, 1.0 - lam)
        lt = lam + dl
        act = active.copy()
        u_try = u.copy()
        u_try[cons] = lt * mdl.ucons
        ok = False
        for _ in range(40):
            conv, its, out = newton(u_try, lt, act)
            total_it += its
            if not conv:
                break
            if ct is None:
                ok = True
                break
            new = _update_active(ct, u_try, act)
            set_its += 1
            if np.array_equal(new, act):
                ok = True
                break
            act = new
        if ok:
            sig, eps_p, alpha, Kc, r = out
            u, lam, active = u_try, lt, act
            history.append((lam, float(np.linalg.norm(u.reshape(-1, 3), axis=1).max())))
            dlam = min(1.0 / steps, 2 * dl)
            _report(progress, "nonlinear", lam)
        else:
            cuts += 1
            dlam = dl / 2
            if dlam < 1.0 / steps / 2 ** 8:
                if ct is not None and not act.any():
                    raise ValueError(SUPPORT_MSG + " (all contact nodes released)")
                raise ValueError(f"nonlinear solution did not converge at load factor {lam + dl:.4g}: the load may "
                                 f"exceed the collapse load, or the model has not enough supports")
    info = None
    if ct is not None:
        info = {"active": int(active.sum()), "released": int((~active).sum()), "iterations": set_its}
    react = _reaction(mdl, r - Kc @ u, Kc, u)  # r − Kc u = f_int − f: support + contact force at constrained dofs
    return u, sig, alpha, history, info, react, {"newton_iterations": total_it, "step_cuts": cuts,
                                                 "solver": ", ".join(sorted(names)) or "none"}


# ---------------------------------------------------------------- driver
def solve_mesh(nodes, elems, material, fixed, f, analysis="static", n_modes=3, contact=None, prescribed=None,
               dT=0.0, steps=10, solver="auto", incompatible=None, progress=None) -> dict:
    """Solve a hex8 mesh. See the module docstring for the method; units mm, N, MPa.

    nodes (nn, 3) mm; elems (ne, 8) local order n = a + 2b + 4c; material {E, nu, rho [kg/m³], alpha [1/K],
    sy [MPa], H [MPa, default E/100]}; fixed: constrained dofs (3*node + k), value 0 unless given in
    prescribed {dof: mm}; f (3 nn,) external nodal forces, N; dT scalar or (nn,) temperature change, K.

    contact [{"nodes", "normal"}]: compression-only supports, u·n ≤ 0 with n the outward normal of the
    supported face. Active-set iteration with penalty springs k = 1e3 × mean diag(K): tensile active nodes
    are released, penetrating inactive nodes re-activated, until the set stops changing (≤ 40 solves).
    If the load lifts the part off and nothing else holds it, ValueError("...not enough supports...").
    Modal / buckling run the static step first and keep its final active set (bonded in the normal).

    solver: "auto" = SuperLU up to DIRECT_MAX_DOF free dofs (DIRECT_MAX_DOF_EIGEN for modal / buckling),
    otherwise CG preconditioned by pyamg smoothed aggregation (CG + Jacobi without pyamg); "direct" / "amg".
    A singular (under-supported) system raises ValueError("...not enough supports...").

    Returns {u (nn, 3), umag, vm, p1, sed [N·mm/mm³], stress (nn, 6) [s11 s22 s33 s12 s23 s13] - Gauss values
    extrapolated to the corners and averaged at nodes; energy [N·mm] elastic strain energy; reaction (3,) total
    support force on the part at fixed, prescribed and contact dofs [N]; peq (nn,) | None; history
    [(load factor, max |u|)] | None; modes [{f [Hz], shape (nn, 3)}]; buckling [{factor, shape}];
    contact {active, released, iterations} | None; stats}.
    """
    t0 = time.perf_counter()
    if analysis not in ("static", "nonlinear", "modal", "buckling"):
        raise ValueError(f"unknown analysis {analysis!r}")
    if incompatible is None:
        incompatible = analysis != "nonlinear"
    if analysis == "nonlinear":
        incompatible = False
    mdl = _Model(nodes, elems, material, fixed, prescribed, dT, bool(incompatible))
    f = np.asarray(f, dtype=float).ravel()
    if f.shape != (mdl.ndof,):
        raise ValueError(f"f must have 3*nn = {mdl.ndof} entries")
    ct = _contact_setup(contact, mdl)
    if len(mdl.cons) == 0 and ct is None:
        raise ValueError(SUPPORT_MSG)
    _report(progress, "setup", 1.0)
    stats = {"elements": mdl.ne, "nodes": mdl.nn, "dof": mdl.ndof, "free_dof": int(len(mdl.free)),
             "analysis": analysis,
             "element": "hex8, 2×2×2 Gauss" + (", Wilson–Taylor incompatible modes" if mdl.incompatible else "")}
    res = {"peq": None, "history": None, "modes": [], "buckling": [], "contact": None}

    if analysis == "nonlinear":
        t1 = time.perf_counter()
        u, sig, alpha, hist, cinfo, react, nl = _nonlinear(mdl, f, ct, steps, solver, progress)
        wts = np.concatenate([mdl.geo(i)["w"] for i in range(len(mdl.chunks))])
        res.update(history=hist, contact=cinfo, peq=np.maximum(mdl.nodal(alpha[..., None])[:, 0], 0.0))
        stats.update(nl, steps=steps)
        stats["t_solve_s"] = round(time.perf_counter() - t1, 3)
    else:
        K, fth, mass = _assemble_linear(mdl, progress)
        t1 = time.perf_counter()
        ftot = f + fth
        dmax = DIRECT_MAX_DOF_EIGEN if analysis in ("modal", "buckling") else DIRECT_MAX_DOF
        u, S, Kc, cinfo = _static(mdl, K, ftot, ct, solver, progress, dmax)
        t2 = time.perf_counter()
        stats.update(solver=S.name, t_assemble_s=round(t1 - t0, 3), t_solve_s=round(t2 - t1, 3))
        res["contact"] = cinfo
        react = _reaction(mdl, K @ u - ftot, Kc, u)
        sig, wts = _gauss_stress_linear(mdl, u)
        free = mdl.free
        if analysis == "modal" and n_modes > 0:
            _report(progress, "modes", 0.0)
            Mff = sp.diags(np.repeat(mass, 3)[free]).tocsc()
            if Mff.diagonal().min() <= 0:
                raise ValueError("modal analysis needs a positive density material['rho']")
            k = min(int(n_modes), len(free) - 1)
            # shift-invert about σ = 0 reuses the static factorisation of K
            vals, vecs = spla.eigsh(S.Aff, k=k, M=Mff, sigma=0.0, which="LM", OPinv=S.operator())
            for j in np.argsort(vals):
                res["modes"].append({"f": float(math.sqrt(max(vals[j], 0.0)) / (2 * math.pi)),
                                     "shape": _shape_full(mdl, vecs[:, j])})
            _report(progress, "modes", 1.0)
        if analysis == "buckling" and n_modes > 0:
            _report(progress, "buckling", 0.0)
            Gff = _geometric_stiffness(mdl, sig)[free][:, free]
            k = min(int(n_modes), len(free) - 1)
            # (−K_σ) φ = μ K φ with K positive definite: the largest μ give the smallest load factors λ = 1/μ
            mu, vecs = spla.eigsh(-Gff, k=k, M=S.Aff, Minv=S.operator(), which="LA")
            for j in np.argsort(-mu):
                if mu[j] > 1e-12 * np.abs(mu).max():
                    res["buckling"].append({"factor": float(1.0 / mu[j]), "shape": _shape_full(mdl, vecs[:, j])})
            _report(progress, "buckling", 1.0)

    _report(progress, "recover", 0.0)
    U = u.reshape(-1, 3)
    sed_g = 0.5 * np.einsum("egi,ij,egj->eg", sig, mdl.C, sig)  # ½ σ : ε_elastic, N·mm/mm³
    nod = mdl.nodal(np.concatenate([sig, sed_g[..., None]], axis=2))
    sn = nod[:, :6]
    s11, s22, s33, s12, s23, s13 = sn.T
    vm = np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3 * (s12 ** 2 + s23 ** 2 + s13 ** 2))
    tens = sn[:, [0, 3, 5, 3, 1, 4, 5, 4, 2]].reshape(-1, 3, 3)
    p1 = np.linalg.eigvalsh(tens)[:, 2]
    umag = np.linalg.norm(U, axis=1)
    stats.update(time_s=round(time.perf_counter() - t0, 3), max_u=float(umag.max()), max_vm=float(vm.max()))
    _report(progress, "recover", 1.0)
    res.update(u=U, umag=umag, vm=vm, p1=p1, sed=nod[:, 6], stress=sn, energy=float((sed_g * wts).sum()),
               reaction=react, stats=stats)
    return res


def _shape_full(mdl: _Model, v_free: np.ndarray) -> np.ndarray:
    v = np.zeros(mdl.ndof)
    v[mdl.free] = v_free
    v = v.reshape(-1, 3)
    return v / (np.linalg.norm(v, axis=1).max() or 1.0)


# ---------------------------------------------------------------- helpers
def box_mesh(nx: int, ny: int, nz: int, lx: float, ly: float, lz: float, origin=(0.0, 0.0, 0.0)):
    """Structured hex8 box: nodes ((nx+1)(ny+1)(nz+1), 3) numbered x fastest, elems (nx ny nz, 8) in n = a+2b+4c."""
    xs = np.linspace(0, lx, nx + 1) + origin[0]
    ys = np.linspace(0, ly, ny + 1) + origin[1]
    zs = np.linspace(0, lz, nz + 1) + origin[2]
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    base = (i + (nx + 1) * (j + (ny + 1) * k)).ravel()
    off = CORNERS[:, 0] + (nx + 1) * (CORNERS[:, 1] + (ny + 1) * CORNERS[:, 2])
    return nodes, base[:, None] + off[None, :]


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


def encode_results(res: dict) -> dict:
    """JSON-safe copy of a solve_mesh result: arrays as base64 little-endian float32, the rest as-is.

    Per-node fields keep their row-major layout ("u" and mode shapes are nn × 3, "stress" nn × 6); the
    3-vector "reaction" stays a plain list.
    """
    def enc(v, key=None):
        if isinstance(v, np.ndarray):
            return [float(x) for x in v] if key == "reaction" else _b64(v)
        if isinstance(v, dict):
            return {k: enc(x, k) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [enc(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        return v
    return enc(res)
