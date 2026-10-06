"""Natural frequencies via SLEPc.

Generalised Hermitian problem  K phi = lambda M phi  with lambda = omega^2.

Constrained rows are handled the usual way: the stiffness diagonal is set to a
large value and the mass diagonal to a small one, which pushes the spurious
constraint modes to lambda ~ 1e7 and far away from the physical spectrum.
Shift-and-invert about sigma = 0 then converges on the lowest true modes.
"""

from __future__ import annotations

import numpy as np


def eigenmodes(msh, V, a_form, bcs, mat, study, log=lambda *x: None,
               progress=lambda *x: None):
    import ufl
    from dolfinx import fem
    from dolfinx.fem.petsc import assemble_matrix
    from petsc4py import PETSc
    from slepc4py import SLEPc

    n_req = int(study.outputs.modes)
    rho = mat.rho_t(study.loads.density_scale)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    m_form = fem.form(rho * ufl.inner(u, v) * ufl.dx)

    K = assemble_matrix(fem.form(a_form), bcs=bcs, diag=1e8)
    K.assemble()
    M = assemble_matrix(m_form, bcs=bcs, diag=1e-8)
    M.assemble()

    eps = SLEPc.EPS().create(msh.comm)
    eps.setOperators(K, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setDimensions(nev=n_req, ncv=max(2 * n_req + 10, 20))
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTarget(0.0)
    eps.setTolerances(tol=1e-8, max_it=400)

    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    ksp = st.getKSP()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    for cand in ("mumps", "superlu_dist", "umfpack"):
        try:
            pc.setFactorSolverType(cand)
            break
        except Exception:                     # pragma: no cover
            continue

    log("modal", f"SLEPc shift-invert · requesting {n_req} eigenpairs", "info")
    eps.solve()
    nconv = eps.getConverged()
    log("modal", f"{nconv} eigenpairs converged", "ok" if nconv >= n_req else "warn")

    freqs: list[float] = []
    shapes: list[np.ndarray] = []
    vr = K.createVecRight()
    bs = V.dofmap.index_map_bs
    for i in range(min(nconv, n_req)):
        lam = eps.getEigenpair(i, vr).real
        if lam <= 1e-6 or lam > 1e12:         # rigid body / constraint artefact
            continue
        hz = float(np.sqrt(lam) / (2.0 * np.pi))
        phi = vr.getArray(readonly=True).copy().reshape(-1, bs)
        peak = float(np.abs(phi).max()) or 1.0
        freqs.append(hz)
        shapes.append(phi / peak)
        log("modal", f"mode {len(freqs)} — {hz:.1f} Hz", "ok")
        progress(6, 100.0 * len(freqs) / max(1, n_req), f"mode {len(freqs)}")
    return freqs, shapes
