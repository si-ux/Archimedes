"""The brick solver against beam theory, equilibrium, modes, and the GLB export."""
import json
import math
import struct

import numpy as np
import pytest

from archimedes_fe.demos import DEMOS
from archimedes_fe.export import surface_quads, to_glb
from archimedes_fe.model import STEEL, Load, Member, Model, Section, Support
from archimedes_fe.solver import SolveError, solve


def nearest_layer(r, axis, value):
    c = r.mesh.nodes[:, axis]
    return np.isclose(c, c[np.argmin(np.abs(c - value))])


def cantilever(**load):
    return Model("c", Member(Section("rect", 150, 300), 3000, "horizontal"), STEEL, [Support("fixed", 0.0)],
                 [Load("point", "z", -1, 1.0, 30.0, **load)])


def test_cantilever_tip_deflection_matches_timoshenko():
    m = cantilever()
    r = solve(m, n_modes=0)
    s, mat = m.member.section, m.material
    Ib, P, L = s.b * s.h ** 3 / 12, 30e3, 3000
    expected = P * L ** 3 / (3 * mat.E * Ib) + P * L / (5 / 6 * mat.G * s.area)
    tip = r.fields["U3"][r.mesh.layer == r.mesh.grid[0]].mean()
    assert tip == pytest.approx(-expected, rel=0.03)


def test_bending_stress_away_from_the_support_is_exact():
    m = cantilever()
    r = solve(m, n_modes=0)
    s = m.member.section
    top_mid = (np.abs(r.mesh.nodes[:, 0] - 1500) < 1) & (np.abs(r.mesh.nodes[:, 2] - s.h / 2) < 1)
    assert r.fields["S11"][top_mid].mean() == pytest.approx(30e3 * 1500 * (s.h / 2) / (s.b * s.h ** 3 / 12), rel=0.02)


def test_reactions_balance_the_applied_loads():
    for name, make in DEMOS.items():
        r = solve(make(), n_modes=0)
        np.testing.assert_allclose(r.reactions / 1e3, -np.array(r.stats["applied_kN"]), atol=1e-6, err_msg=name)


def test_simply_supported_udl_midspan_deflection():
    L, w = 6000, 10.0
    m = Model("ss", Member(Section("rect", 200, 400), L, "horizontal"), STEEL,
              [Support("pinned", 0.0), Support("roller", 1.0)], [Load("uniform", "z", -1, 0.0, w, 1.0, w)])
    r = solve(m, n_modes=0)
    Ib = 200 * 400 ** 3 / 12
    eb = 5 * w * L ** 4 / (384 * STEEL.E * Ib)
    mid = nearest_layer(r, 0, L / 2)
    assert -r.fields["U3"][mid].mean() == pytest.approx(eb, rel=0.06)


def test_first_modes_match_beam_theory():
    m = cantilever()
    r = solve(m, n_modes=2)
    s, mat, L = m.member.section, m.material, 3000

    def f(i_):
        return 1.875 ** 2 / (2 * math.pi) * math.sqrt(mat.E * i_ / (mat.rho * s.area * L ** 4))

    weak, strong = f(s.h * s.b ** 3 / 12), f(s.b * s.h ** 3 / 12)
    assert r.modes[0]["f"] == pytest.approx(weak, rel=0.06)
    assert r.modes[1]["f"] == pytest.approx(strong, rel=0.06)


def test_short_column_axial_stress_is_p_over_a():
    m = DEMOS["short_column"]()
    m.loads = [Load("point", "z", -1, 1.0, 1500.0)]
    r = solve(m, n_modes=0)
    mid = nearest_layer(r, 2, 1000)
    assert r.fields["S33"][mid].mean() == pytest.approx(-1500e3 / (400 * 400), rel=0.02)


def test_column_classification():
    assert DEMOS["long_column"]().summary()["column_class"] == "long"
    assert DEMOS["short_column"]().summary()["column_class"] == "short"


def test_every_requested_field_is_there():
    r = solve(DEMOS["cantilever"]())
    want = {"von_mises", "U", "U1", "U2", "U3", "SED", "S11", "S22", "S33", "S12", "S13", "S23",
            "E11", "E22", "E33", "E12", "E13", "E23"}
    assert want <= set(r.fields)
    pay = r.payload()
    assert {"mode1", "mode6"} <= set(pay["fields"]) and pay["fields"]["mode1"]["freq"] > 0


def test_mechanism_is_reported_not_crashed():
    m = cantilever()
    m.supports = [Support("roller", 0.0), Support("roller", 1.0)]
    with pytest.raises(SolveError):
        solve(m)


def test_glb_is_valid_gltf():
    r = solve(DEMOS["continuous"](), n_modes=1)
    for field in ("von_mises", "mode1"):
        b = to_glb(r, field)
        magic, ver, length = struct.unpack("<III", b[:12])
        assert magic == 0x46546C67 and ver == 2 and length == len(b)
        jlen, jtype = struct.unpack("<II", b[12:20])
        gltf = json.loads(b[20:20 + jlen])
        acc = gltf["accessors"]
        assert acc[0]["count"] == acc[2]["count"] and acc[3]["count"] % 3 == 0
        assert acc[3]["count"] // 3 >= 2 * len(surface_quads(r))
