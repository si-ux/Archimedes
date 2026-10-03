"""Field recovery, reactions and export.

Stress is discontinuous across element boundaries for a CG1 displacement
field, so nodal values come from an L2 projection onto CG1 rather than from
point interpolation (which would just pick whichever cell was visited last).
The invariants are then evaluated in numpy on the projected components, which
keeps the eigenvalue work out of UFL.
"""

from __future__ import annotations

import numpy as np


def project(expr, V):
    """L2-project a UFL expression onto the space V."""
    import ufl
    from dolfinx.fem.petsc import LinearProblem

    w, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(w, v) * ufl.dx
    L = ufl.inner(expr, v) * ufl.dx
    problem = LinearProblem(a, L, petsc_options_prefix="archimedes_proj_",
                            petsc_options={"ksp_type": "cg", "pc_type": "jacobi",
                                           "ksp_rtol": 1e-10})
    return problem.solve()


def max_principal(sx, sy, sz, txy, tyz, tzx):
    """Largest eigenvalue of a symmetric 3x3 field (Cardano, vectorised)."""
    q = (sx + sy + sz) / 3.0
    p2 = ((sx - q) ** 2 + (sy - q) ** 2 + (sz - q) ** 2
          + 2.0 * (txy ** 2 + tyz ** 2 + tzx ** 2))
    p = np.sqrt(np.maximum(p2, 0.0) / 6.0)
    out = np.array(q, dtype=float, copy=True)
    ok = p > 1e-12
    if not np.any(ok):
        return out
    a = (sx[ok] - q[ok]) / p[ok]
    b = (sy[ok] - q[ok]) / p[ok]
    c = (sz[ok] - q[ok]) / p[ok]
    d = txy[ok] / p[ok]
    e = tyz[ok] / p[ok]
    f = tzx[ok] / p[ok]
    det = a * (b * c - e * e) - d * (d * c - e * f) + f * (d * e - b * f)
    r = np.clip(det / 2.0, -1.0, 1.0)
    out[ok] = q[ok] + 2.0 * p[ok] * np.cos(np.arccos(r) / 3.0)
    return out


def von_mises(sx, sy, sz, txy, tyz, tzx):
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2
                          + 6.0 * (txy ** 2 + tyz ** 2 + tzx ** 2)))


def recover_fields(msh, V, uh, mat, study, log=lambda *a: None) -> dict:
    """Nodal displacement, stress invariants and strain-energy density."""
    import ufl
    from dolfinx import fem

    d = 3
    eps = ufl.sym(ufl.grad(uh))
    eth = mat.alpha * study.loads.dT
    eps_m = eps - eth * ufl.Identity(d)
    sig = 2.0 * mat.mu * eps_m + mat.lam * ufl.tr(eps_m) * ufl.Identity(d)

    u_arr = uh.x.array.reshape(-1, d)
    umag = np.linalg.norm(u_arr, axis=1)
    out = {"u": u_arr, "umag": umag}

    if study.outputs.s or study.outputs.e:
        T = fem.functionspace(msh, ("Lagrange", 1, (d, d)))
        sh = project(sig, T)
        S = sh.x.array.reshape(-1, d, d)
        sx, sy, sz = S[:, 0, 0], S[:, 1, 1], S[:, 2, 2]
        txy, tyz, tzx = S[:, 0, 1], S[:, 1, 2], S[:, 0, 2]
        if study.outputs.s:
            out["vm"] = von_mises(sx, sy, sz, txy, tyz, tzx)
            out["p1"] = max_principal(sx, sy, sz, txy, tyz, tzx)
        if study.outputs.e:
            Es = fem.functionspace(msh, ("Lagrange", 1))
            sed = project(0.5 * ufl.inner(sig, eps_m), Es)
            out["sed"] = sed.x.array.copy()

    # total strain energy — exact for the discretisation, straight from the form
    from mpi4py import MPI
    total = fem.assemble_scalar(fem.form(0.5 * ufl.inner(sig, eps_m) * ufl.dx))
    out["strain_energy"] = float(msh.comm.allreduce(total, op=MPI.SUM))
    out["_fn"] = {"u": uh, "sig": sh if (study.outputs.s or study.outputs.e) else None}
    return out


def sample_at(msh, fn, points: np.ndarray, bs: int) -> np.ndarray:
    """Evaluate a Function at arbitrary points (used to bring tetrahedral
    results onto the UI's voxel nodes). Points that miss the domain — the
    voxel grid staircases the fillet, so a few always do — stay zero."""
    from dolfinx import geometry

    pts = np.ascontiguousarray(points, dtype=np.float64)
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    keep, cells = [], []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            keep.append(i)
            cells.append(links[0])
    out = np.zeros((pts.shape[0], bs), dtype=np.float64)
    if keep:
        out[keep] = np.asarray(fn.eval(pts[keep], cells)).reshape(len(keep), bs)
    return out, len(keep)


def sample_ui_fields(msh, vm, fields: dict, mat, study) -> dict:
    """Resample a non-matching (tetrahedral) solution onto the voxel nodes so
    the workbench viewport can render it."""
    fns = fields.get("_fn") or {}
    pts = vm.points
    u, hit = sample_at(msh, fns["u"], pts, 3)
    out = {"u": u, "umag": np.linalg.norm(u, axis=1),
           "strain_energy": fields["strain_energy"], "_hits": hit,
           "_total": int(pts.shape[0])}
    if fns.get("sig") is not None:
        S, _ = sample_at(msh, fns["sig"], pts, 9)
        S = S.reshape(-1, 3, 3)
        sx, sy, sz = S[:, 0, 0], S[:, 1, 1], S[:, 2, 2]
        txy, tyz, tzx = S[:, 0, 1], S[:, 1, 2], S[:, 0, 2]
        if study.outputs.s:
            out["vm"] = von_mises(sx, sy, sz, txy, tyz, tzx)
            out["p1"] = max_principal(sx, sy, sz, txy, tyz, tzx)
        if study.outputs.e:
            eth = mat.alpha * study.loads.dT
            tr = sx + sy + sz
            # strain energy density recovered from the sampled stress
            c1 = (1.0 + mat.nu) / mat.E
            c2 = mat.nu / mat.E
            ex = c1 * sx - c2 * tr
            ey = c1 * sy - c2 * tr
            ez = c1 * sz - c2 * tr
            g = 2.0 * c1
            out["sed"] = 0.5 * (sx * ex + sy * ey + sz * ez
                                + g * (txy ** 2 + tyz ** 2 + tzx ** 2))
            del eth
    if "reaction" in fields:
        out["reaction"] = fields["reaction"]
    return out


def reactions(a, L, uh, bcs, V, on_base) -> list[float]:
    """Sum of the residual r = K u - f over the constrained nodes."""
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector

    A = assemble_matrix(fem.form(a))
    A.assemble()
    b = assemble_vector(fem.form(L))
    b.assemble()

    x = uh.x.petsc_vec if hasattr(uh.x, "petsc_vec") else uh.vector
    r = A.createVecLeft()
    A.mult(x, r)
    r.axpy(-1.0, b)

    dofs = fem.locate_dofs_geometrical(V, on_base)
    bs = V.dofmap.index_map_bs
    arr = r.getArray(readonly=True).reshape(-1, bs)
    sel = arr[dofs]
    return [float(sel[:, i].sum()) for i in range(bs)]


# ------------------------------------------------------------- exports --
def to_ui_order(values: np.ndarray, perm: np.ndarray | None, nn: int) -> np.ndarray:
    """Scatter a DOLFINx-ordered nodal array back into the UI's node order."""
    if perm is None:
        return values
    shape = (nn,) + values.shape[1:]
    out = np.zeros(shape, dtype=values.dtype)
    out[perm] = values
    return out


def write_csv(path: str, points: np.ndarray, fields: dict) -> None:
    cols = ["node", "x", "y", "z", "ux", "uy", "uz", "umag"]
    extra = [k for k in ("vm", "p1", "sed") if k in fields]
    cols += extra
    n = points.shape[0]
    u = fields["u"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for i in range(n):
            row = [str(i), f"{points[i,0]:.4f}", f"{points[i,1]:.4f}", f"{points[i,2]:.4f}",
                   f"{u[i,0]:.6e}", f"{u[i,1]:.6e}", f"{u[i,2]:.6e}",
                   f"{fields['umag'][i]:.6e}"]
            row += [f"{fields[k][i]:.6e}" for k in extra]
            fh.write(",".join(row) + "\n")


def write_xdmf(path: str, msh, uh, fields_fn: dict) -> None:
    """Full-fidelity result file for ParaView."""
    from dolfinx.io import XDMFFile
    with XDMFFile(msh.comm, path, "w") as xf:
        xf.write_mesh(msh)
        uh.name = "displacement"
        xf.write_function(uh)
        for name, fn in (fields_fn or {}).items():
            fn.name = name
            xf.write_function(fn)
