"""The Workbench's general-mesh FE endpoint (/api/fe/*) and the DOLFINx bridge's posted-mesh path."""
import base64
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from archimedes_fe.meshsolver import box_mesh  # noqa: E402
from webdemo import server  # noqa: E402

STEEL = {"E": 210000.0, "nu": 0.3, "rho": 7850.0, "alpha": 12e-6, "sy": 250.0}


@pytest.fixture
def client():
    return TestClient(server.app)


def b64(a, dtype):
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def dec(s):
    return np.frombuffer(base64.b64decode(s), dtype="<f4")


def cantilever(n=(20, 2, 4), size=(1000.0, 50.0, 100.0), load=-1000.0):
    nodes, elems = box_mesh(*n, *size)
    root = np.nonzero(nodes[:, 0] == 0)[0]
    tip = np.nonzero(np.isclose(nodes[:, 0], size[0]))[0]
    fixed = (3 * root[:, None] + np.arange(3)).ravel()
    f = np.zeros(nodes.size)
    f[3 * tip + 1] = load / len(tip)
    return nodes, elems, fixed, f


def test_info(client):
    j = client.get("/api/fe/info").json()
    assert j["ok"] and set(j["analyses"]) == {"static", "nonlinear", "modal", "buckling"} and j["contact"]


def test_static_base64_matches_beam_theory(client):
    nodes, elems, fixed, f = cantilever()
    body = {"nodes": b64(nodes, "<f4"), "elems": b64(elems, "<i4"), "fixed": b64(fixed, "<i4"),
            "f": b64(f, "<f4"), "material": STEEL}
    j = client.post("/api/fe/solve", json=body).json()
    E, inertia, A, G = 210000.0, 100 * 50**3 / 12, 50 * 100, 210000.0 / 2.6   # bending about z: depth 50 in y
    theory = 1000 * 1000**3 / (3 * E * inertia) + 1000 * 1000 / (5 / 6 * G * A)
    tip = dec(j["u"]).reshape(-1, 3)[np.isclose(nodes[:, 0], 1000), 1].mean()
    assert abs(-tip - theory) / theory < 0.03
    assert np.allclose(j["reaction"], [0, 1000, 0], atol=1e-3)
    assert j["stats"]["analysis"] == "static" and j["wall_s"] >= 0


def test_modal_and_buckling_through_the_endpoint(client):
    nodes, elems, fixed, f = cantilever()
    base = {"nodes": nodes.ravel().tolist(), "elems": elems.ravel().tolist(), "fixed": fixed.tolist(),
            "f": f.tolist(), "material": STEEL}
    j = client.post("/api/fe/solve", json={**base, "analysis": "modal", "n_modes": 2}).json()
    assert [round(m["f"]) for m in j["modes"]] == [42, 83]
    # column 20 x 20 x 1000, fixed base, 1 N down on the top: Euler λ = π²EI / 4L²
    nodes, elems = box_mesh(2, 40, 2, 20.0, 1000.0, 20.0)
    base_n = np.nonzero(nodes[:, 1] == 0)[0]
    top = np.nonzero(np.isclose(nodes[:, 1], 1000))[0]
    f = np.zeros(nodes.size)
    f[3 * top + 1] = -1.0 / len(top)
    body = {"nodes": nodes.ravel().tolist(), "elems": elems.ravel().tolist(),
            "fixed": (3 * base_n[:, None] + np.arange(3)).ravel().tolist(), "f": f.tolist(),
            "material": STEEL, "analysis": "buckling", "n_modes": 2}
    j = client.post("/api/fe/solve", json=body).json()
    euler = np.pi**2 * 210000 * (20**4 / 12) / (4 * 1000**2)
    assert abs(j["buckling"][0]["factor"] - euler) / euler < 0.02


def test_nonlinear_and_contact(client):
    nodes, elems = box_mesh(10, 1, 1, 100.0, 10.0, 10.0)
    x0 = np.nonzero(nodes[:, 0] == 0)[0]
    x1 = np.nonzero(np.isclose(nodes[:, 0], 100))[0]
    fixed = np.concatenate([3 * x0, [3 * x0[0] + 1, 3 * x0[0] + 2, 3 * x0[1] + 2]])
    # every x = 0 node held axially, plus 3-2-1 against rigid motion: a bar in uniaxial tension
    f = np.zeros(nodes.size)
    f[3 * x1] = 300.0 * 100 / len(x1)                    # 300 MPa over 100 mm²
    body = {"nodes": nodes.ravel().tolist(), "elems": elems.ravel().tolist(), "fixed": fixed.tolist(),
            "f": f.tolist(), "material": {**STEEL, "H": 2000.0}, "analysis": "nonlinear", "steps": 5}
    j = client.post("/api/fe/solve", json=body).json()
    assert abs(dec(j["peq"]).max() - (300 - 250) / 2000) < 1e-3
    # a block on a compression-only support, pushed down: it bears on the support
    nodes, elems = box_mesh(4, 2, 2, 40.0, 20.0, 20.0)
    bot = np.nonzero(nodes[:, 1] == 0)[0]
    top = np.nonzero(np.isclose(nodes[:, 1], 20))[0]
    xs = np.nonzero(nodes[:, 0] == 0)[0]
    zs = np.nonzero(nodes[:, 2] == 0)[0]
    f = np.zeros(nodes.size)
    f[3 * top + 1] = -1000.0 / len(top)
    body = {"nodes": nodes.ravel().tolist(), "elems": elems.ravel().tolist(),
            "fixed": np.concatenate([3 * xs, 3 * zs + 2]).tolist(), "f": f.tolist(), "material": STEEL,
            "contact": [{"nodes": bot.tolist(), "normal": [0, -1, 0]}]}
    j = client.post("/api/fe/solve", json=body).json()
    assert j["contact"]["active"] > 0 and abs(j["reaction"][1] - 1000) < 1e-3


def test_errors_are_400(client):
    nodes, elems, fixed, f = cantilever()
    base = {"nodes": nodes.ravel().tolist(), "elems": elems.ravel().tolist(), "f": f.tolist(), "material": STEEL}
    r = client.post("/api/fe/solve", json={**base, "fixed": []})
    assert r.status_code == 400 and "not enough supports" in r.json()["error"]
    r = client.post("/api/fe/solve", json={**base, "fixed": fixed.tolist(), "analysis": "dance"})
    assert r.status_code == 400 and "unknown analysis" in r.json()["error"]
    r = client.post("/api/fe/solve", json={"nodes": [0, 0, 0]})
    assert r.status_code == 400 and r.json()["error"].startswith("bad request")


def test_dolfinx_bridge_posted_mesh_protocol():
    """The bridge's solve_mesh path, on the mock engine (DOLFINx itself only runs in the container)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Archimedes" / "backend"))
    try:
        from archimedes.bridge import MESH_ENGINES
        from archimedes.fem import mesh_arrays
    finally:
        sys.path.pop(0)
    nodes, elems, fixed, f = cantilever(n=(4, 1, 1))
    payload = {"nodes": b64(nodes, "<f4"), "elems": b64(elems, "<i4"), "fixed": b64(fixed, "<i4"), "f": b64(f, "<f4")}
    n2, e2, fx2, f2 = mesh_arrays(payload)
    assert np.allclose(n2, nodes) and (e2 == elems).all() and (fx2 == fixed).all() and np.allclose(f2, f)
    logs = []
    out = MESH_ENGINES["mock"](payload, lambda *a: logs.append(a), lambda *a: None, lambda: None)
    assert out["stats"]["nodes"] == len(nodes) and len(dec(out["u"])) == nodes.size and logs
    with pytest.raises(ValueError):
        mesh_arrays({**payload, "f": [0.0]})
