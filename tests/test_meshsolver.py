"""The general hex8 mesh solver against theory: beams, patch test, thermal, modes, buckling, plasticity, contact."""
import base64
import json
import math
import sys
import time

import numpy as np
import pytest

from archimedes_fe.meshsolver import CORNERS, box_mesh, encode_results, solve_mesh
from archimedes_fe.solver import elasticity

STEEL = {"E": 210000.0, "nu": 0.3, "rho": 7850.0, "alpha": 1.2e-5, "sy": 250.0}


def at(nodes, axis, value, tol=1e-6):
    return np.flatnonzero(np.abs(nodes[:, axis] - value) < tol)


def dofs(idx, comps=(0, 1, 2)):
    return [3 * int(i) + c for i in idx for c in comps]


def face_quads(nodes, elems, axis, value, tol=1e-6):
    """Element faces lying on the plane x[axis] = value, as (nq, 4) node lists in cyclic order."""
    out = []
    o = [a for a in range(3) if a != axis]
    for side in (0, 1):
        loc = [int(np.flatnonzero((CORNERS[:, axis] == side) & (CORNERS[:, o[0]] == p) & (CORNERS[:, o[1]] == q))[0])
               for p, q in ((0, 0), (1, 0), (1, 1), (0, 1))]
        quads = elems[:, loc]
        on = np.all(np.abs(nodes[quads, axis] - value) < tol, axis=1)
        out.append(quads[on])
    return np.concatenate(out)


def traction_load(nodes, quads, t):
    """Consistent nodal forces (3 nn,) of a uniform traction t (N/mm², per unit area) on bilinear quads."""
    f = np.zeros((len(nodes), 3))
    g = np.array([-1.0, 1.0]) / math.sqrt(3.0)
    sg = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    area = 0.0
    for s in g:
        for r in g:
            N = (1 + sg[:, 0] * s) * (1 + sg[:, 1] * r) / 4
            dNs = sg[:, 0] * (1 + sg[:, 1] * r) / 4
            dNr = sg[:, 1] * (1 + sg[:, 0] * s) / 4
            X = nodes[quads]  # (nq, 4, 3)
            dA = np.linalg.norm(np.cross(np.einsum("n,qnj->qj", dNs, X), np.einsum("n,qnj->qj", dNr, X)), axis=1)
            area += dA.sum()
            np.add.at(f, quads, N[None, :, None] * dA[:, None, None] * np.asarray(t, float)[None, None, :])
    return f.ravel(), area


def cantilever(nx=20, ny=2, nz=4, L=1000.0, b=50.0, h=100.0, P=1000.0, **kw):
    nodes, elems = box_mesh(nx, ny, nz, L, b, h)
    fixed = dofs(at(nodes, 0, 0.0))
    f, _ = traction_load(nodes, face_quads(nodes, elems, 0, L), (0.0, 0.0, -P / (b * h)))
    return nodes, elems, fixed, f, solve_mesh(nodes, elems, STEEL, fixed, f, **kw)


# 1 --------------------------------------------------------------------------------------------------------
def test_cantilever_tip_deflection_matches_timoshenko_and_plain_hex8_locks():
    L, b, h, P = 1000.0, 50.0, 100.0, 1000.0
    E, nu = STEEL["E"], STEEL["nu"]
    Ib, G = b * h ** 3 / 12, E / (2 * (1 + nu))
    expected = P * L ** 3 / (3 * E * Ib) + P * L / (5 / 6 * G * b * h)
    nodes, _, _, _, r = cantilever()
    tip = at(nodes, 0, L)
    d_inc = -r["u"][tip, 2].mean()
    _, _, _, _, r0 = cantilever(incompatible=False)
    d_plain = -r0["u"][tip, 2].mean()
    print(f"\ncantilever tip: incompatible {d_inc:.5f} mm, plain {d_plain:.5f} mm, Timoshenko {expected:.5f} mm")
    assert d_inc == pytest.approx(expected, rel=0.03)
    assert d_plain < 0.95 * d_inc  # shear locking of the plain trilinear brick
    # bending stress at mid-span, top fibre: M c / I
    mid_top = np.intersect1d(at(nodes, 0, L / 2), at(nodes, 2, h))
    assert r["stress"][mid_top, 0].mean() == pytest.approx(P * L / 2 * (h / 2) / Ib, rel=0.03)


# 2 --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("incompatible", [False, True])
def test_patch_test_on_distorted_mesh(incompatible):
    rng = np.random.default_rng(7)
    nodes, elems = box_mesh(3, 3, 3, 30.0, 30.0, 30.0)
    inner = np.all((nodes > 1e-9) & (nodes < 30 - 1e-9), axis=1)
    nodes[inner] += rng.uniform(-3.5, 3.5, (inner.sum(), 3))
    A = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0], [3.0, 1.0, -2.0]]) * 1e-4
    c = np.array([0.1, -0.2, 0.3])
    exact = nodes @ A.T + c
    pres = {3 * i + k: exact[i, k] for i in np.flatnonzero(~inner) for k in range(3)}
    r = solve_mesh(nodes, elems, STEEL, [], np.zeros(3 * len(nodes)), prescribed=pres, incompatible=incompatible)
    eps = np.array([A[0, 0], A[1, 1], A[2, 2], A[0, 1] + A[1, 0], A[1, 2] + A[2, 1], A[0, 2] + A[2, 0]])
    sig = elasticity(STEEL["E"], STEEL["nu"]) @ eps
    np.testing.assert_allclose(r["u"][inner], exact[inner], rtol=0, atol=1e-8 * np.abs(exact).max())
    np.testing.assert_allclose(r["stress"], np.broadcast_to(sig, r["stress"].shape), rtol=0,
                               atol=1e-8 * np.abs(sig).max())


# 3 --------------------------------------------------------------------------------------------------------
def test_reactions_balance_applied_loads_with_gravity():
    nodes, elems = box_mesh(10, 2, 3, 600.0, 60.0, 90.0)
    nn = len(nodes)
    fixed = dofs(at(nodes, 0, 0.0))
    # gravity: ρ g V/8 to each corner of every element (uniform elements), plus a tip point load
    vol = 60.0 * 6.0 * 30.0 * 30.0
    w = np.bincount(elems.ravel(), minlength=nn) * vol / 8 * STEEL["rho"] * 1e-12 * 9810.0  # t/mm³ · mm/s² -> N
    f = np.zeros((nn, 3))
    f[:, 2] -= w
    f[at(nodes, 0, 600.0)[0]] += (150.0, -80.0, -400.0)
    r = solve_mesh(nodes, elems, STEEL, fixed, f.ravel())
    print("\nreaction", r["reaction"], "applied", f.sum(axis=0))
    np.testing.assert_allclose(r["reaction"] + f.sum(axis=0), 0.0, atol=1e-8 * np.abs(f).sum())
    assert r["energy"] == pytest.approx(0.5 * f.ravel() @ r["u"].ravel(), rel=1e-6)


# 4 --------------------------------------------------------------------------------------------------------
def test_thermal_free_bar_expands_without_stress():
    L, dT = 100.0, 50.0
    nodes, elems = box_mesh(5, 2, 2, L, 20.0, 20.0)
    o = at(nodes, 0, 0.0)
    n0 = o[np.argmin(np.linalg.norm(nodes[o], axis=1))]
    nx = int(np.argmin(np.linalg.norm(nodes - (L, 0, 0), axis=1)))
    ny = int(np.argmin(np.linalg.norm(nodes - (0, 20, 0), axis=1)))
    fixed = [3 * n0, 3 * n0 + 1, 3 * n0 + 2, 3 * nx + 1, 3 * nx + 2, 3 * ny + 2]  # 3-2-1, statically determinate
    r = solve_mesh(nodes, elems, STEEL, fixed, np.zeros(3 * len(nodes)), dT=dT)
    grow = r["u"][at(nodes, 0, L), 0].mean()
    print(f"\nfree bar ΔL {grow:.6f} mm vs αΔT L {STEEL['alpha'] * dT * L:.6f}, max |σ| {np.abs(r['stress']).max():.2e}")
    assert grow == pytest.approx(STEEL["alpha"] * dT * L, rel=1e-8)
    assert np.abs(r["stress"]).max() < 1e-6
    assert np.abs(r["reaction"]).max() < 1e-6


def test_thermal_restrained_bar_and_cube():
    dT, E, nu, a = 50.0, STEEL["E"], STEEL["nu"], STEEL["alpha"]
    nodes, elems = box_mesh(4, 2, 2, 100.0, 20.0, 20.0)
    zero = np.zeros(3 * len(nodes))
    # uniaxial: ends held axially (rollers), free laterally -> σ11 = −E α dT
    n0 = int(np.argmin(np.linalg.norm(nodes, axis=1)))
    ny = int(np.argmin(np.linalg.norm(nodes - (0, 20, 0), axis=1)))
    fixed = dofs(np.concatenate([at(nodes, 0, 0.0), at(nodes, 0, 100.0)]), (0,)) + [3 * n0 + 1, 3 * n0 + 2, 3 * ny + 2]
    r = solve_mesh(nodes, elems, STEEL, fixed, zero, dT=dT)
    print(f"\nrestrained bar σ11 {r['stress'][:, 0].mean():.4f} vs {-E * a * dT:.4f} MPa")
    np.testing.assert_allclose(r["stress"][:, 0], -E * a * dT, rtol=1e-8)
    assert np.abs(r["stress"][:, 1:]).max() < 1e-6
    # every face on rollers: no strain at all -> hydrostatic σ = −E α dT / (1 − 2ν)
    nodes, elems = box_mesh(3, 3, 3, 30.0, 30.0, 30.0)
    fixed = sum((dofs(np.concatenate([at(nodes, k, 0.0), at(nodes, k, 30.0)]), (k,)) for k in range(3)), [])
    r = solve_mesh(nodes, elems, STEEL, fixed, np.zeros(3 * len(nodes)), dT=dT)
    s = -E * a * dT / (1 - 2 * nu)
    print(f"clamped cube σ {r['stress'][:, :3].mean():.4f} vs {s:.4f} MPa")
    np.testing.assert_allclose(r["stress"][:, :3], s, rtol=1e-8)
    assert np.abs(r["u"]).max() < 1e-12


# 5 --------------------------------------------------------------------------------------------------------
def test_cantilever_first_frequency_matches_euler_bernoulli():
    L, b, h = 1000.0, 50.0, 100.0
    nodes, elems = box_mesh(20, 2, 4, L, b, h)
    r = solve_mesh(nodes, elems, STEEL, dofs(at(nodes, 0, 0.0)), np.zeros(3 * len(nodes)), analysis="modal",
                   n_modes=3)
    rho = STEEL["rho"] * 1e-12

    def fb(i_):
        return 1.875 ** 2 / (2 * math.pi) * math.sqrt(STEEL["E"] * i_ / (rho * b * h * L ** 4))

    weak, strong = fb(h * b ** 3 / 12), fb(b * h ** 3 / 12)
    f = [m["f"] for m in r["modes"]]
    print(f"\nmodes {np.round(f, 3)} Hz; Euler-Bernoulli weak {weak:.3f}, strong {strong:.3f}")
    assert f[0] == pytest.approx(weak, rel=0.05)
    assert f[1] == pytest.approx(strong, rel=0.05)
    shape = r["modes"][0]["shape"]
    assert np.abs(shape[:, 1]).max() > 5 * np.abs(shape[:, 2]).max()  # first mode bends in the weak (y) direction


# 6 --------------------------------------------------------------------------------------------------------
def test_column_buckling_matches_euler():
    a, L, P = 20.0, 1000.0, 1000.0
    nodes, elems = box_mesh(2, 2, 40, a, a, L)
    f, _ = traction_load(nodes, face_quads(nodes, elems, 2, L), (0.0, 0.0, -P / a ** 2))
    r = solve_mesh(nodes, elems, STEEL, dofs(at(nodes, 2, 0.0)), f, analysis="buckling", n_modes=3)
    pcr = math.pi ** 2 * STEEL["E"] * a ** 4 / 12 / (4 * L ** 2)
    lam = [b["factor"] for b in r["buckling"]]
    print(f"\nbuckling factors {np.round(lam, 4)}: λP = {lam[0] * P:.1f} N vs Euler {pcr:.1f} N")
    assert lam[0] * P == pytest.approx(pcr, rel=0.08)
    assert lam[1] == pytest.approx(lam[0], rel=1e-3)  # square section: two equal modes
    assert r["stress"][:, 2].mean() == pytest.approx(-P / a ** 2, rel=0.01)


# 7 --------------------------------------------------------------------------------------------------------
def plastic_bar(stress, steps=5):
    L, a = 100.0, 10.0
    mat = {"E": 200000.0, "nu": 0.3, "rho": 7850.0, "alpha": 0.0, "sy": 250.0, "H": 2000.0}
    nodes, elems = box_mesh(10, 2, 2, L, a, a)
    n0 = int(np.argmin(np.linalg.norm(nodes, axis=1)))
    ny = int(np.argmin(np.linalg.norm(nodes - (0, a, 0), axis=1)))
    fixed = dofs(at(nodes, 0, 0.0), (0,)) + [3 * n0 + 1, 3 * n0 + 2, 3 * ny + 2]
    f, area = traction_load(nodes, face_quads(nodes, elems, 0, L), (stress, 0.0, 0.0))
    r = solve_mesh(nodes, elems, mat, fixed, f, analysis="nonlinear", steps=steps)
    return nodes, mat, f.reshape(-1, 3)[:, 0].sum() / area, r


def test_plastic_bar_beyond_yield_matches_linear_hardening():
    nodes, mat, sigma, r = plastic_bar(300.0)
    ep = (sigma - mat["sy"]) / mat["H"]
    tip = r["u"][at(nodes, 0, 100.0), 0].mean()
    print(f"\nplastic bar: σ11 {r['stress'][:, 0].mean():.3f} MPa (F/A {sigma:.3f}), εp {r['peq'].mean():.6f} vs {ep:.6f},"
          f" tip {tip:.5f} mm, Newton iterations {r['stats']['newton_iterations']}")
    np.testing.assert_allclose(r["stress"][:, 0], sigma, rtol=1e-6)
    np.testing.assert_allclose(r["vm"], sigma, rtol=1e-6)
    np.testing.assert_allclose(r["peq"], ep, rtol=0.02)
    assert tip == pytest.approx((sigma / mat["E"] + ep) * 100.0, rel=0.02)
    lam, umax = zip(*r["history"])
    assert lam[0] == 0 and lam[-1] == pytest.approx(1.0) and np.all(np.diff(umax) > 0)
    assert r["reaction"][0] == pytest.approx(-sigma * 100.0, rel=1e-8)


def test_plastic_bar_below_yield_stays_elastic():
    nodes, mat, sigma, r = plastic_bar(200.0, steps=2)
    assert np.abs(r["peq"]).max() == 0.0
    np.testing.assert_allclose(r["stress"][:, 0], sigma, rtol=1e-8)
    assert r["u"][at(nodes, 0, 100.0), 0].mean() == pytest.approx(sigma / mat["E"] * 100.0, rel=1e-8)


# 8 --------------------------------------------------------------------------------------------------------
def block_on_base(load, analysis="static"):
    """100×100×50 block (y up) on a compression-only base; two base nodes held horizontally only."""
    nodes, elems = box_mesh(4, 4, 2, 100.0, 100.0, 50.0)
    base = at(nodes, 1, 0.0)
    n0 = int(np.argmin(np.linalg.norm(nodes, axis=1)))
    n1 = int(np.argmin(np.linalg.norm(nodes - (100, 0, 0), axis=1)))
    fixed = [3 * n0, 3 * n0 + 2, 3 * n1 + 2]
    f, _ = traction_load(nodes, face_quads(nodes, elems, 1, 100.0), np.asarray(load) / (100.0 * 50.0))
    contact = [{"nodes": base, "normal": (0.0, -1.0, 0.0)}]
    r = solve_mesh(nodes, elems, STEEL, fixed, f, contact=contact, analysis=analysis, steps=2)
    return nodes, base, f.reshape(-1, 3), r


def test_contact_uniform_load_all_nodes_active():
    nodes, base, f, r = block_on_base((0.0, -10e3, 0.0))
    print("\ncontact uniform:", r["contact"], r["reaction"])
    assert r["contact"]["active"] == len(base) and r["contact"]["released"] == 0
    np.testing.assert_allclose(r["reaction"], -f.sum(axis=0), atol=1e-6 * 10e3)
    assert r["reaction"][1] == pytest.approx(10e3, rel=1e-8)


@pytest.mark.parametrize("analysis", ["static", "nonlinear"])
def test_contact_overturning_load_lifts_one_edge(analysis):
    nodes, base, f, r = block_on_base((3e3, -10e3, 0.0), analysis)
    print(f"\ncontact eccentric ({analysis}):", r["contact"], r["reaction"])
    assert r["contact"]["released"] > 0 and r["contact"]["active"] > 0
    np.testing.assert_allclose(r["reaction"], -f.sum(axis=0), atol=1e-6 * 10e3)
    lifted = base[nodes[base, 0] < 1e-6]  # the horizontal push (+x) lifts the x = 0 edge
    assert r["u"][lifted, 1].min() > 0


def test_contact_upward_load_cannot_be_held():
    # documented behaviour: every contact node releases, nothing else holds the part vertically -> ValueError
    with pytest.raises(ValueError, match="not enough supports"):
        block_on_base((0.0, 10e3, 0.0))


# 9 --------------------------------------------------------------------------------------------------------
def test_distorted_cylinder_in_tension():
    R, L, F = 50.0, 200.0, 100e3
    nodes, elems = box_mesh(8, 8, 10, 2.0, 2.0, L, origin=(-1.0, -1.0, 0.0))
    x, y = nodes[:, 0].copy(), nodes[:, 1].copy()
    nodes[:, 0] = R * x * np.sqrt(1 - y ** 2 / 2)  # elliptical-grid square -> disc map
    nodes[:, 1] = R * y * np.sqrt(1 - x ** 2 / 2)
    c = int(np.argmin(np.linalg.norm(nodes, axis=1)))
    e = int(np.argmin(np.linalg.norm(nodes - (R, 0, 0), axis=1)))
    fixed = dofs(at(nodes, 2, 0.0), (2,)) + [3 * c, 3 * c + 1, 3 * e + 1]
    quads = face_quads(nodes, elems, 2, L)
    f, area = traction_load(nodes, quads, (0.0, 0.0, 1.0))
    f *= F / area
    r = solve_mesh(nodes, elems, STEEL, fixed, f)
    nominal = F / (math.pi * R ** 2)
    print(f"\ncylinder σzz {r['stress'][:, 2].min():.4f}..{r['stress'][:, 2].max():.4f} MPa vs F/πR² {nominal:.4f}")
    np.testing.assert_allclose(r["stress"][:, 2], nominal, rtol=0.03)
    np.testing.assert_allclose(r["stress"][:, 2], F / area, rtol=1e-6)  # exact for the meshed section


# 10 -------------------------------------------------------------------------------------------------------
def test_speed_20k_elements():
    nodes, elems = box_mesh(40, 25, 20, 2000.0, 500.0, 400.0)
    f = np.zeros(3 * len(nodes))
    tip = at(nodes, 0, 2000.0)
    f[3 * tip + 2] = -1e5 / len(tip)
    t = time.perf_counter()
    r = solve_mesh(nodes, elems, STEEL, dofs(at(nodes, 0, 0.0)), f)
    dt = time.perf_counter() - t
    print(f"\n{len(elems)} elements, {r['stats']['free_dof']} free dof: {dt:.2f} s ({r['stats']['solver']})")
    assert dt < 60
    assert np.abs(r["reaction"][2] - 1e5) < 1e-3 * 1e5


# extras ---------------------------------------------------------------------------------------------------
def test_amg_and_jacobi_match_direct(monkeypatch):
    nodes, elems, fixed, f, r = cantilever(nx=10, ny=2, nz=2)
    ra = solve_mesh(nodes, elems, STEEL, fixed, f, solver="amg")
    assert "pyamg" in ra["stats"]["solver"]
    np.testing.assert_allclose(ra["u"], r["u"], atol=1e-6 * np.abs(r["u"]).max())
    monkeypatch.setitem(sys.modules, "pyamg", None)  # import fails -> CG + Jacobi
    rj = solve_mesh(nodes, elems, STEEL, fixed, f, solver="amg")
    assert "Jacobi" in rj["stats"]["solver"]
    np.testing.assert_allclose(rj["u"], r["u"], atol=1e-5 * np.abs(r["u"]).max())


def test_rejects_mechanisms_and_inverted_elements():
    nodes, elems = box_mesh(4, 2, 2, 100.0, 20.0, 20.0)
    f = np.zeros(3 * len(nodes))
    f[-1] = 1.0
    with pytest.raises(ValueError, match="not enough supports"):
        solve_mesh(nodes, elems, STEEL, dofs([0]), f)  # one pinned node: free to rotate
    with pytest.raises(ValueError, match="not enough supports"):
        solve_mesh(nodes, elems, STEEL, [], f)
    bad = elems.copy()
    bad[5] = bad[5][[1, 0, 3, 2, 5, 4, 7, 6]]  # mirrored: negative Jacobian
    with pytest.raises(ValueError, match="element 5 "):
        solve_mesh(nodes, bad, STEEL, dofs(at(nodes, 0, 0.0)), f)


def test_encode_results_is_json_safe():
    nodes, elems = box_mesh(6, 1, 1, 300.0, 30.0, 30.0)
    f = np.zeros(3 * len(nodes))
    f[3 * at(nodes, 0, 300.0) + 1] = -100.0
    r = solve_mesh(nodes, elems, STEEL, dofs(at(nodes, 0, 0.0)), f, analysis="modal", n_modes=2)
    enc = json.loads(json.dumps(encode_results(r)))
    u = np.frombuffer(base64.b64decode(enc["u"]), dtype="<f4").reshape(-1, 3)
    np.testing.assert_allclose(u, r["u"], rtol=1e-6, atol=1e-9)
    assert len(enc["modes"]) == 2 and isinstance(enc["modes"][0]["f"], float)
    assert isinstance(enc["modes"][0]["shape"], str) and enc["stats"]["elements"] == 6


def test_progress_callback_reports_stages():
    seen = []
    nodes, elems = box_mesh(4, 1, 1, 100.0, 20.0, 20.0)
    f = np.zeros(3 * len(nodes))
    f[-1] = -10.0
    solve_mesh(nodes, elems, STEEL, dofs(at(nodes, 0, 0.0)), f, progress=lambda s, x: seen.append((s, x)))
    stages = [s for s, _ in seen]
    assert {"assemble", "solve", "recover"} <= set(stages) and all(0 <= x <= 1 for _, x in seen)
