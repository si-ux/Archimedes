"""The Python console: commands, the journal, scripts and errors."""
import numpy as np
import pytest

from archimedes_fe import scripting
from archimedes_fe.scripting import Console, model_commands
from archimedes_fe.session import ModelSession


def run(con, *lines):
    out = None
    for ln in lines:
        out = con.push(ln)
    return out


def test_commands_build_load_and_solve_a_beam():
    con = Console(ModelSession())
    run(con, "beam(b=200, h=400, L=6000)", "material('steel')", "pinned(at=0); roller(at=1)", "uniform(10)")
    m = con.session.model
    assert (m.member.length, m.member.section.h, m.material.name) == (6000, 400, "Steel S355")
    assert [s.kind for s in m.supports] == ["pinned", "roller"] and m.loads[0].kind == "uniform"
    r = run(con, "r = solve()")
    assert r["solved"] and not r["error"] and con.session.results is not None
    # mid-span deflection against 5wL^4/384EI plus shear, as in tests/test_fe.py
    ei = 200000.0 * 200 * 400 ** 3 / 12
    expected = 5 * 10 * 6000 ** 4 / (384 * ei)
    assert -con.cmd.at(0.5, "U3") == pytest.approx(expected, rel=0.06)
    assert run(con, "peak('U3')['value'] < 0")["output"].strip() == "True"


def test_every_load_direction_and_kind():
    con = Console(ModelSession())
    run(con, "column(b=300, h=300, L=3000)", "fixed(at=0)", "point(5, at=1, dir='+x')",
        "trapezoidal(0, 4, start=0, end=1, dir='-y')", "uniform(2, start=0.2, end=0.8, dir='+z')")
    loads = con.session.model.loads
    assert [(ld.kind, ld.axis, ld.sign) for ld in loads] == [("point", "x", 1), ("trapezoidal", "y", -1), ("uniform", "z", 1)]


def test_mistakes_are_explained_not_crashes():
    con = Console(ModelSession())
    r = run(con, "point(5, at=2)")
    assert r["error"] and "from 0 (start) to 1 (end)" in r["output"]
    r = run(con, "point(5, dir='sideways')")
    assert r["error"] and "+x -x" in r["output"]
    r = run(con, "clear('all')", "solve()")
    assert r["error"] and "Can't solve yet" in r["output"]
    r = run(con, "1/0")
    assert r["error"] and "ZeroDivisionError" in r["output"] and "scripting.py" not in r["output"]


def test_blocks_continue_until_a_blank_line():
    con = Console(ModelSession())
    assert run(con, "for t in (0.25, 0.5):")["more"]
    assert run(con, "    point(1, at=t)")["more"]
    r = run(con, "")
    assert not r["more"] and len(con.session.model.loads) == 2 + 2  # the cantilever demo has two


def test_gesture_edits_are_journaled_and_the_journal_replays():
    s = ModelSession()
    s.begin("new_rect", orientation="horizontal")
    for v in (3, 3, 2, 250, 450, 5000):  # increments, then b, h, L
        s.answer(v)
    s.begin("support_fixed", t=0.0)
    s.begin("point_load", t=1.0, axis="z", sign=-1)
    s.answer(12)
    s.begin("uniform_load", t0=0.0, t1=0.5, axis="y", sign=1)
    s.answer(3)
    script = "\n".join(e["cmd"] for e in s.journal)
    assert "increments(length=100, section=25, load=1)" in script and "beam(b=250, h=450, L=5000)" in script
    replay = Console(ModelSession("continuous"))
    assert not replay.execute(script, "<journal>", interactive=False)["error"]
    a, b = s.model, replay.session.model
    assert (a.member, a.supports, a.loads) == (b.member, b.supports, b.loads)


def test_direct_model_edits_are_noticed_and_journaled_as_commands():
    con = Console(ModelSession())
    r = run(con, "model.loads.append(Load('point', 'x', 1, 0.5, 7.0))")
    assert r["changed"] and con.session.results is None
    assert "point(7, at=0.5, dir='+x')" in con.cmd.journal()
    assert model_commands(con.session.model)[0].startswith("beam(")


def test_scripts_run_from_fe_scripts_and_modules_import(tmp_path, monkeypatch):
    (tmp_path / "helpers.py").write_text(
        "from archimedes_fe.scripting import api\n\ndef two_points(P):\n"
        "    api.point(P, at=0.5)\n    api.point(P, at=1.0)\n")
    (tmp_path / "make.py").write_text("demo('cantilever')\nclear('loads')\nfrom helpers import two_points\ntwo_points(4)\n")
    monkeypatch.setattr(scripting, "SCRIPTS", tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    con = Console(ModelSession())
    r = run(con, "run('make.py')")
    assert not r["error"], r["output"]
    assert [ld.t0 for ld in con.session.model.loads] == [0.5, 1.0]
    r = con.run_script("upload.py", "point(1, at=0.25)\nprint('ok')")
    assert r["output"].strip() == "ok" and run(con, "run('upload.py')")["changed"]


def test_bundled_example_scripts_run():
    con = Console(ModelSession())
    outputs = {}
    for name in ("simply_supported_check.py", "column_study.py"):
        r = con.execute(f"run({name!r})", "<test>", interactive=False)
        assert not r["error"], r["output"]
        outputs[name] = r["output"]
    assert "mid-span deflection" in outputs["simply_supported_check.py"]
    assert "P/P_cr" in outputs["column_study.py"]


def test_view_commands_drive_an_attached_view():
    from archimedes_gestures.view_state import ViewState

    v = ViewState()
    con = Console(ModelSession())
    con.cmd.view = v
    r = run(con, "solve()", "show('S11'); cut('z', 0.5)")
    assert r["view"] and v.field_name == "S11" and v.section_on
    assert np.allclose(v.plane_normal, [0, 0, 1]) and abs(v.plane_origin[2]) < 1e-9
