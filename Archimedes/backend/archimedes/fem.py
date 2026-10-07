"""DOLFINx linear elasticity + modal extraction.

Weak form (small strain, isotropic, stress-free reference temperature):

    a(u, v) = integral  sigma(u) : eps(v)  dx
    L(v)    = integral  f_body . v  dx
            + integral  t_n . v      ds(tip) + ds(web)
            + integral  (3*lam + 2*mu) * alpha * dT * div(v) dx

The thermal term is the standard  B^T D eps_0  contribution written in weak
form; with eps_0 = alpha*dT*I the deviatoric part drops out and only the
divergence survives.

Solved with CG + GAMG.  GAMG needs the six rigid-body modes as a near
nullspace to coarsen elasticity properly — without it the coarse grids are
badly conditioned and the iteration count roughly triples.

Every dolfinx/petsc import is function-local so this module can be imported
(for its docstrings and signatures) outside the container.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from .spec import Study, GRAVITY
from . import voxel


Reporter = Callable[[str, str, str], None]     # (channel, message, kind)
Progress = Callable[[int, float, str], None]   # (stage, pct, note)


def _noop(*_a, **_k):
    return None


# --------------------------------------------------------------- meshes --
def _dolfinx_mesh_from_voxel(vm: "voxel.VoxelMesh"):
    """Wrap the structured voxel core as a DOLFINx hexahedral mesh.

    Returns (mesh, perm) where perm maps DOLFINx geometry-node index ->
    original voxel node index, so results can be scattered back into the
    UI's node order.
    """
    from mpi4py import MPI
    import basix.ufl
    import ufl
    from dolfinx.mesh import create_mesh

    elem = basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,))
    domain = ufl.Mesh(elem)
    # dolfinx >= 0.11 takes the coordinate element before the point array
    mesh = create_mesh(MPI.COMM_WORLD, vm.cells, domain, vm.points)
    perm = mesh.geometry.input_global_indices
    return mesh, np.asarray(perm, dtype=np.int64)


def _gmsh_mesh(study: Study, h: float):
    """Curvature-resolved tetrahedral mesh of the same solid, via gmsh."""
    from mpi4py import MPI
    # dolfinx 0.11 renamed dolfinx.io.gmshio -> dolfinx.io.gmsh and returns a
    # MeshData object rather than the old (mesh, cell_tags, facet_tags) tuple
    from dolfinx.io import gmsh as dio_gmsh
    from .geometry import build_gmsh_model

    import gmsh as gmsh_api
    # interruptible=True installs a SIGINT handler, which raises when the
    # bridge runs a solve off the main thread
    gmsh_api.initialize(interruptible=False)
    try:
        build_gmsh_model(study.geometry.normalised(), h, order=study.mesh.order)
        data = dio_gmsh.model_to_mesh(gmsh_api.model, MPI.COMM_WORLD, 0, gdim=3)
    finally:
        gmsh_api.finalize()
    return data.mesh if hasattr(data, "mesh") else data[0]


# ------------------------------------------------------------ machinery --
def _rigid_body_nullspace(V):
    """Six rigid-body modes, orthonormalised — GAMG's near nullspace.

    Elasticity coarsens badly without this: the aggregates cannot represent
    rigid-body motion, so the coarse-grid correction is poor and the iteration
    count rises sharply.
    """
    from petsc4py import PETSc
    import dolfinx.la as la

    bs = V.dofmap.index_map_bs
    length0 = V.dofmap.index_map.size_local
    basis = [la.vector(V.dofmap.index_map, bs=bs) for _ in range(6)]
    b = [x.array for x in basis]
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    for i in range(3):                                  # translations
        b[i][dofs[i]] = 1.0

    x = V.tabulate_dof_coordinates()
    block = V.dofmap.list.flatten()
    x0, x1, x2 = x[block, 0], x[block, 1], x[block, 2]
    b[3][dofs[0]] = -x1                                 # rotation about z
    b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2                                  # rotation about y
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1                                  # rotation about x
    b[5][dofs[1]] = -x2

    la.orthonormalize(basis)
    vecs = [PETSc.Vec().createWithArray(arr[: bs * length0], bsize=bs, comm=V.mesh.comm)
            for arr in b]
    return PETSc.NullSpace().create(vectors=vecs)


def solve(study: Study,
          log: Reporter = _noop,
          progress: Progress = _noop) -> dict:
    """Run the study and return UI-ordered nodal fields plus statistics."""
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc
    import dolfinx
    from dolfinx import fem, mesh as dmesh
    from dolfinx.fem.petsc import LinearProblem

    t_all = time.perf_counter()
    times: dict[str, float] = {}

    def mark(key, t0):
        times[key] = (time.perf_counter() - t0) * 1000.0
        return time.perf_counter()

    g = study.geometry.normalised()
    mat = study.material
    h, hz, clamped = study.element_sizes()
    if clamped:
        log("mesh", f"element size raised to {h:.2f} mm to respect the "
                    f"{study.mesh.budget} element budget", "warn")

    # 1 · geometry + mesh --------------------------------------------------
    t0 = time.perf_counter()
    progress(1, 0.0, "geometry")
    log("geom", f"L-bracket {g.W:g} x {g.H:g} x {g.D:g} mm · wall {g.t:g} · fillet r{g.r:g}", "info")

    vm = voxel.build(g, h, hz)
    if study.mesh.mesher == "gmsh":
        msh = _gmsh_mesh(study, h)
        perm = None
        log("mesh", f"gmsh tetrahedral mesh · "
                    f"{msh.topology.index_map(msh.topology.dim).size_global} cells", "info")
    else:
        msh, perm = _dolfinx_mesh_from_voxel(vm)
        log("mesh", f"voxel core {vm.nx}x{vm.ny}x{vm.nz} — {vm.ne} hex8 · "
                    f"{vm.nn} nodes · {vm.ndof} DOF", "info")
    t0 = mark("mesh", t0)
    progress(1, 100.0, "mesh ready")

    # 2 · function space + material ---------------------------------------
    progress(2, 0.0, "material")
    # The voxel path is pinned to P1 because the UI maps results back through a
    # 1:1 node correspondence.  On the gmsh path P2 is worth having: linear
    # tetrahedra lock in bending and noticeably under-predict deflection.
    degree = 1 if perm is not None else max(1, min(2, study.mesh.order))
    if perm is None and study.mesh.order >= 2:
        log("mesh", "quadratic (P2) tetrahedra — linear tets lock in bending", "info")
    V = fem.functionspace(msh, ("Lagrange", degree, (3,)))
    lam = fem.Constant(msh, PETSc.ScalarType(mat.lam))
    mu = fem.Constant(msh, PETSc.ScalarType(mat.mu))
    log("mat", f"{mat.name} — E {mat.E/1000:.1f} GPa · nu {mat.nu} · "
               f"lambda {mat.lam/1000:.1f} GPa · mu {mat.mu/1000:.1f} GPa", "info")

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(3)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    t0 = mark("material", t0)

    # 3 · loads ------------------------------------------------------------
    progress(3, 0.0, "loads")
    tol = min(vm.dx, vm.dy, vm.dz) * 1e-3
    span = max(1, int(round(g.t / vm.dx)))
    # the wall snap can shift the outer box by up to dx/2 — locate against the
    # mesh that actually exists, not the requested dimensions
    Weff, Heff = (vm.W, vm.H) if study.mesh.mesher == "voxel" else (g.W, g.H)

    def on_base(x):
        return x[1] < tol

    def on_tip(x):
        return (x[1] > Heff - tol) & (x[0] > Weff - span * vm.dx - tol)

    def on_web(x):
        return x[0] < tol

    fdim = msh.topology.dim - 1
    tip_f = dmesh.locate_entities_boundary(msh, fdim, on_tip)
    web_f = dmesh.locate_entities_boundary(msh, fdim, on_web)
    marks = np.hstack([np.full_like(tip_f, 1), np.full_like(web_f, 2)])
    ents = np.hstack([tip_f, web_f])
    srt = np.argsort(ents)
    ft = dmesh.meshtags(msh, fdim, ents[srt], marks[srt])
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)

    L = ufl.inner(fem.Constant(msh, PETSc.ScalarType((0.0, 0.0, 0.0))), v) * ufl.dx

    if study.loads.gravity:
        w = mat.rho_t(study.loads.density_scale) * GRAVITY
        L += ufl.inner(fem.Constant(msh, PETSc.ScalarType((0.0, -w, 0.0))), v) * ufl.dx

    if study.bcs.force and study.loads.tip_force:
        area = fem.assemble_scalar(fem.form(fem.Constant(msh, PETSc.ScalarType(1.0)) * ds(1)))
        area = msh.comm.allreduce(area, op=MPI.SUM)
        if area > 0:
            th = np.deg2rad(study.loads.theta)
            trac = (study.loads.tip_force / area) * np.array([np.sin(th), -np.cos(th), 0.0])
            L += ufl.inner(fem.Constant(msh, PETSc.ScalarType(tuple(trac))), v) * ds(1)
            log("load", f"tip resultant {study.loads.tip_force/1000:.2f} kN over "
                        f"{area:.1f} mm2 at {study.loads.theta:g} deg", "info")

    if study.bcs.press and study.loads.pressure:
        p = PETSc.ScalarType((study.loads.pressure, 0.0, 0.0))
        L += ufl.inner(fem.Constant(msh, p), v) * ds(2)
        log("load", f"web pressure {study.loads.pressure:g} MPa on the x=0 face", "info")

    if study.loads.dT:
        k = (3.0 * mat.lam + 2.0 * mu.value) * mat.alpha * study.loads.dT
        L += fem.Constant(msh, PETSc.ScalarType(k)) * ufl.div(v) * ufl.dx
        log("load", f"uniform dT {study.loads.dT:g} K above the stress-free state", "info")

    bcs = []
    if study.bcs.enc:
        dofs = fem.locate_dofs_geometrical(V, on_base)
        zero = np.zeros(3, dtype=PETSc.ScalarType)
        bcs.append(fem.dirichletbc(zero, dofs, V))
        log("load", f"encastre on {dofs.size} base nodes — U1=U2=U3=0", "info")
    if study.bcs.sym:
        Vz, _ = V.sub(2).collapse()
        zmid = vm.dz * int(round(vm.nz / 2))
        dofs = fem.locate_dofs_geometrical(
            (V.sub(2), Vz), lambda x: np.abs(x[2] - zmid) < tol)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs[0], V.sub(2)))
    if not bcs:
        log("load", "no Dirichlet set — the stiffness matrix is singular", "err")
    t0 = mark("loads", t0)

    # 4 · solve ------------------------------------------------------------
    progress(4, 0.0, "assembling")
    opts = {
        "ksp_type": study.solver.ksp,
        "pc_type": study.solver.pc,
        "ksp_rtol": study.solver.rtol,
        "ksp_max_it": study.solver.maxit,
        "ksp_norm_type": "unpreconditioned",
    }
    problem = LinearProblem(a, L, bcs=bcs, petsc_options_prefix="archimedes_",
                            petsc_options=opts)

    if study.solver.pc == "gamg":
        try:
            problem.A.setNearNullSpace(_rigid_body_nullspace(V))
            log("solve", "GAMG near-nullspace: 6 rigid-body modes attached", "info")
        except Exception as exc:                       # pragma: no cover
            log("solve", f"near-nullspace unavailable ({exc}) — GAMG will be slower", "warn")

    it_seen = {"n": 0}

    def monitor(ksp, it, rnorm):
        it_seen["n"] = it
        if it % 10 == 0:
            b0 = ksp.getRhs().norm() or 1.0
            frac = 0.0
            if rnorm > 0 and b0 > 0:
                import math as _m
                tgt = _m.log(study.solver.rtol)
                cur = _m.log(max(rnorm / b0, 1e-300))
                frac = min(1.0, max(0.0, cur / tgt)) if tgt else 1.0
            progress(4, 100.0 * frac, f"it {it} · |r|/|b| {rnorm/b0:.2e}")

    problem.solver.setMonitor(monitor)
    uh = problem.solve()
    ksp = problem.solver
    reason = ksp.getConvergedReason()
    its = ksp.getIterationNumber()
    res = ksp.getResidualNorm()
    log("solve", f"KSP {study.solver.ksp} · PC {study.solver.pc} — {its} iterations, "
                 f"|r| {res:.3e}" + ("" if reason > 0 else f" — DIVERGED ({reason})"),
        "ok" if reason > 0 else "err")
    t0 = mark("solve", t0)

    # 5 · recover ----------------------------------------------------------
    progress(5, 0.0, "recovering fields")
    from .post import recover_fields, reactions
    fields = recover_fields(msh, V, uh, mat, study, log)
    if study.outputs.rf and study.bcs.enc:
        fields["reaction"] = reactions(a, L, uh, bcs, V, on_base)
        log("post", f"reaction resultant "
                    f"{np.linalg.norm(fields['reaction'])/1000:.3f} kN", "info")
    t0 = mark("recover", t0)

    # 6 · optional modal ---------------------------------------------------
    freqs: list[float] = []
    modes: list[np.ndarray] = []
    if study.outputs.modes:
        progress(6, 0.0, "eigen extraction")
        from .modal import eigenmodes
        freqs, modes = eigenmodes(msh, V, a, bcs, mat, study, log, progress)
    times["modal"] = (time.perf_counter() - t0) * 1000.0

    out = {
        "engine": f"dolfinx {dolfinx.__version__}",
        "mesher": study.mesh.mesher,
        "stats": {
            "elements": vm.ne if perm is not None else int(
                msh.topology.index_map(msh.topology.dim).size_global),
            "nodes": int(V.dofmap.index_map.size_global),
            "dof": int(V.dofmap.index_map.size_global * V.dofmap.index_map_bs),
            "iterations": int(its),
            "residual": float(res),
            "converged": bool(reason > 0),
            "h": float(h),
        },
        "quality": vm.quality(g),
        "times": times,
        "wall": (time.perf_counter() - t_all) * 1000.0,
        "fields": fields,
        "frequencies": freqs,
        "modes": modes,
        "voxel": vm,
        "perm": perm,
        "mesh": msh,
    }
    return out


# ------------------------------------------------- any part: posted mesh --
def mesh_arrays(payload: dict):
    """Decode a posted mesh: nodes (nn, 3) mm, elems (ne, 8), fixed dofs, nodal forces (3 nn,) N.

    Arrays arrive as JSON lists or base64 little-endian float32 / int32 (what the Workbench sends).
    Local node order is n = a + 2b + 4c, which is DOLFINx's hexahedron vertex order.
    """
    import base64 as _b64

    def arr(v, dtype, width=None):
        if isinstance(v, str):
            raw = _b64.b64decode(v)
            a = np.frombuffer(raw, dtype="<i4" if np.issubdtype(dtype, np.integer) else "<f4").astype(dtype)
        else:
            a = np.asarray(v, dtype=dtype)
        return a.reshape(-1, width) if width else a.ravel()

    nodes = arr(payload["nodes"], float, 3)
    elems = arr(payload["elems"], np.int64, 8)
    fixed = arr(payload.get("fixed", []), np.int64)
    f = arr(payload["f"], float)
    if f.size != nodes.size:
        raise ValueError(f"f must have 3*nn = {nodes.size} entries")
    if elems.size and (elems.min() < 0 or elems.max() >= len(nodes)):
        raise ValueError("element node index outside nodes")
    return nodes, elems, fixed, f


def solve_general(payload: dict, log: Reporter = _noop, progress: Progress = _noop) -> dict:
    """Linear static solve of a posted hex8 mesh (any CAD part the Workbench meshed).

    Supports are per-dof (fixed = 3·node + k, value 0); loads are nodal forces (surface loads and
    self weight already lumped by the Workbench); dT adds the thermal strain. CG + GAMG with the
    rigid-body near nullspace. Fields come back in the posted node order.
    """
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc
    import basix.ufl
    import dolfinx
    from dolfinx import fem
    from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
    from dolfinx.mesh import create_mesh

    t_all = time.perf_counter()
    nodes, elems, fixed, f = mesh_arrays(payload)
    m = payload.get("material") or {}
    E, nu = float(m.get("E", 210000.0)), float(m.get("nu", 0.3))
    alpha, dT = float(m.get("alpha", 0.0)), float(payload.get("dT", 0.0))
    lam_v, mu_v = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    progress(1, 0.0, "mesh")
    domain = ufl.Mesh(basix.ufl.element("Lagrange", "hexahedron", 1, shape=(3,)))
    msh = create_mesh(MPI.COMM_WORLD, elems, domain, nodes)
    perm = np.asarray(msh.geometry.input_global_indices, dtype=np.int64)   # geometry node -> posted node
    log("mesh", f"posted mesh — {len(elems)} hex8 · {len(nodes)} nodes", "info")
    V = fem.functionspace(msh, ("Lagrange", 1, (3,)))
    # P1 dofs sit on the geometry nodes: map posted node -> dof block through the cell layouts
    gdm, vdm = msh.geometry.dofmap, V.dofmap.list
    block_of = np.full(len(nodes), -1, dtype=np.int64)
    block_of[perm[gdm.ravel()]] = vdm.ravel()
    lam, mu = fem.Constant(msh, PETSc.ScalarType(lam_v)), fem.Constant(msh, PETSc.ScalarType(mu_v))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2.0 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(3)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = fem.form(ufl.inner(sigma(u), eps(v)) * ufl.dx)
    k_th = (3.0 * lam_v + 2.0 * mu_v) * alpha * dT
    L = fem.form(fem.Constant(msh, PETSc.ScalarType(k_th)) * ufl.div(v) * ufl.dx)
    bcs = []
    for k in range(3):
        nd = fixed[fixed % 3 == k] // 3
        dofs = 3 * block_of[nd] + k
        dofs = np.unique(dofs[dofs >= 0]).astype(np.int32)
        bcs.append(fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V.sub(k)))
    if not len(fixed):
        raise ValueError("no supports: the stiffness matrix is singular")
    progress(3, 100.0, "loads")
    A = assemble_matrix(a, bcs=bcs)
    A.assemble()
    b = assemble_vector(L)
    arr = b.getArray()                                    # nodal forces into their dofs
    for k in range(3):
        ok = block_of >= 0
        np.add.at(arr, 3 * block_of[ok] + k, f[3 * np.nonzero(ok)[0] + k])
    apply_lifting(b, [a], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)
    A.setNearNullSpace(_rigid_body_nullspace(V))
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("cg")
    ksp.getPC().setType("gamg")
    ksp.setTolerances(rtol=float(payload.get("rtol", 1e-8)), max_it=int(payload.get("maxit", 4000)))
    uh = fem.Function(V)
    progress(4, 0.0, "solving")
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    its, reason = ksp.getIterationNumber(), ksp.getConvergedReason()
    log("solve", f"CG + GAMG — {its} iterations" + ("" if reason > 0 else f" — DIVERGED ({reason})"),
        "ok" if reason > 0 else "err")
    # stresses: cellwise-constant projection is enough for display; nodal average
    from .post import project, von_mises, max_principal
    W = fem.functionspace(msh, ("Lagrange", 1, (3, 3)))
    sh = project(sigma(uh) - k_th * ufl.Identity(3), W)
    U = uh.x.array.reshape(-1, 3)[block_of]
    S = sh.x.array.reshape(-1, 9)[block_of]
    sx, sy, sz, txy, tyz, tzx = S[:, 0], S[:, 4], S[:, 8], S[:, 1], S[:, 5], S[:, 2]
    vm = von_mises(sx, sy, sz, txy, tyz, tzx)
    p1 = max_principal(sx, sy, sz, txy, tyz, tzx)
    Au = A.createVecLeft()
    A.mult(uh.x.petsc_vec, Au)
    energy = 0.5 * float(uh.x.petsc_vec.dot(Au))
    progress(5, 100.0, "recovered")
    return {"engine": f"dolfinx {dolfinx.__version__}", "u": U, "umag": np.linalg.norm(U, axis=1), "vm": vm, "p1": p1,
            "energy": energy,
            "stats": {"elements": int(len(elems)), "nodes": int(len(nodes)), "dof": int(nodes.size),
                      "iterations": int(its), "converged": bool(reason > 0),
                      "time_s": round(time.perf_counter() - t_all, 3)}}
