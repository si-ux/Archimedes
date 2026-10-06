"""Python console and scripting for the modeller, in the spirit of the Abaqus CLI.

Every action the gestures and the mouse can take has a command here, and the
session journals each change as the command that repeats it. So a saved
journal is a script, and a script can do anything the hands can::

    >>> beam(b=300, h=500, L=6000)
    >>> material('steel')
    >>> pinned(at=0); roller(at=1)
    >>> uniform(10, dir='-z')
    >>> r = solve()
    >>> peak('U3')

Scripts run with ``run('file.py')`` (from ``fe_scripts/`` or a path), from
"Import script…" in the browser, or ``import`` (``fe_scripts/`` is on the path).
Modules reach the same commands through ``from archimedes_fe.scripting import api``.

This executes arbitrary Python inside the server process, like any engineering
console. Anyone who can open the page can run code, so keep the port private
(the Codespaces default) or start the server with ``ARCHIMEDES_CONSOLE=0``.
"""
from __future__ import annotations

import ast
import codeop
import contextlib
import io
import math
import os
import sys
import threading
import traceback
from pathlib import Path

import numpy as np

from .demos import DEMOS
from .model import AXES, CONCRETE, STEEL, Load, Material, Member, Model, Section, Support
from .session import INCREMENTS, ModelSession, load_command, snap_t

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "fe_scripts"
ENABLED = os.environ.get("ARCHIMEDES_CONSOLE", "1").lower() not in ("0", "off", "false", "no")

# one script at a time: stdout is redirected process-wide while it runs
_LOCK = threading.Lock()
_current: "Commands | None" = None


class ScriptError(Exception):
    """A command was used wrongly; the console shows just the message."""


def _direction(dir: str) -> tuple[str, int]:
    d = str(dir).strip().lower().replace("−", "-")
    sign = -1 if d.startswith("-") else 1
    axis = d.lstrip("+-")
    if axis not in AXES:
        raise ScriptError(f"dir must be one of +x -x +y -y +z -z, not {dir!r}")
    return axis, sign


def _t(v: float, name: str = "at") -> float:
    v = float(v)
    if not 0.0 <= v <= 1.0:
        raise ScriptError(f"{name} is a position along the member from 0 (start) to 1 (end), not {v:g}")
    return snap_t(v, ends=0.0)


def _positive(v: float, name: str, unit: str) -> float:
    v = float(v)
    if not v > 0:
        raise ScriptError(f"{name} must be positive ({unit}), not {v:g}")
    return v


class Commands:
    """The console's commands. Each one edits the session the way a gesture would."""

    def __init__(self, session: ModelSession, view=None):
        self.s = session
        self.view = view  # the pipeline's ViewState, if a browser is attached
        self.solved = False  # set by solve(); the server then sends the results
        self.view_changed = False

    # ---- helpers -------------------------------------------------------
    def _log(self, cmd: str) -> None:
        self.s._log(cmd)

    def _edit(self, msg: str) -> None:
        self.s._changed(msg)
        self.s.action = None

    def _new(self, model: Model, cmd: str) -> Model:
        self.s._push()
        self.s.model = model
        self.s.phase = "loads"
        self._log(cmd)
        self._edit(f"Created {model.member.section.label()} × {model.member.length:g} mm")
        return model

    def _res(self):
        if self.s.results is None:
            raise ScriptError("No results yet: call solve() first")
        return self.s.results

    # ---- geometry ------------------------------------------------------
    def beam(self, b: float = 300, h: float = 500, L: float = 4000, name: str | None = None) -> Model:
        """Rectangular beam along +x: width b, height h, length L (mm)."""
        b, h, L = _positive(b, "b", "mm"), _positive(h, "h", "mm"), _positive(L, "L", "mm")
        m = Model(name or "Beam (script)", Member(Section("rect", b, h), L, "horizontal"), self.s.model.material)
        return self._new(m, f"beam(b={b:g}, h={h:g}, L={L:g})")

    def column(self, b: float = 300, h: float = 300, L: float = 3000, name: str | None = None) -> Model:
        """Rectangular column along +z: b along x, h along y, height L (mm)."""
        b, h, L = _positive(b, "b", "mm"), _positive(h, "h", "mm"), _positive(L, "L", "mm")
        m = Model(name or "Column (script)", Member(Section("rect", b, h), L, "vertical"), self.s.model.material)
        return self._new(m, f"column(b={b:g}, h={h:g}, L={L:g})")

    def round_beam(self, d: float = 300, L: float = 4000, name: str | None = None) -> Model:
        """Circular beam along +x: diameter d, length L (mm)."""
        d, L = _positive(d, "d", "mm"), _positive(L, "L", "mm")
        m = Model(name or "Round beam (script)", Member(Section("circle", d=d), L, "horizontal"), self.s.model.material)
        return self._new(m, f"round_beam(d={d:g}, L={L:g})")

    def round_column(self, d: float = 300, L: float = 3000, name: str | None = None) -> Model:
        """Circular column along +z: diameter d, height L (mm)."""
        d, L = _positive(d, "d", "mm"), _positive(L, "L", "mm")
        m = Model(name or "Round column (script)", Member(Section("circle", d=d), L, "vertical"), self.s.model.material)
        return self._new(m, f"round_column(d={d:g}, L={L:g})")

    def material(self, name: str | Material = "concrete", E: float | None = None, nu: float | None = None,
                 rho: float | None = None, strength: float | None = None) -> Material:
        """'concrete' (C30), 'steel' (S355), or your own: material('C40', E=35000, nu=0.2, rho=2.5e-9, strength=40)."""
        if isinstance(name, Material):
            mat = name
        else:
            base = {"concrete": CONCRETE, "steel": STEEL}.get(str(name).lower())
            if base is None and E is None:
                raise ScriptError("material(): 'concrete', 'steel', or give E (MPa) for a custom one")
            custom = any(v is not None for v in (E, nu, rho, strength))
            label = str(name) if base is None else base.name + (" (modified)" if custom else "")
            base = base or CONCRETE
            mat = Material(label, base.E if E is None else float(E), base.nu if nu is None else float(nu),
                           base.rho if rho is None else float(rho), base.strength if strength is None else float(strength))
        self.s._push()
        self.s.model.material = mat
        if mat in (CONCRETE, STEEL):
            self._log(f"material({'steel' if mat is STEEL or mat == STEEL else 'concrete'!r})")
        else:
            self._log(f"material({mat.name!r}, E={mat.E:g}, nu={mat.nu:g}, rho={mat.rho:g}, strength={mat.strength:g})")
        self._edit(f"Material: {mat.name}")
        return mat

    def demo(self, name: str = "cantilever") -> Model:
        """Load a demo: continuous, cantilever, long_column, short_column."""
        if name not in DEMOS:
            raise ScriptError(f"demo(): one of {', '.join(DEMOS)}")
        self.s.load_demo(name)
        return self.s.model

    def increments(self, length: float | None = None, section: float | None = None, load: float | None = None) -> dict:
        """Snap steps for values entered by hand (mm, mm, kN). Scripts take values as given."""
        for k, v in (("length", length), ("section", section), ("load", load)):
            if v is not None:
                self.s.settings[k] = _positive(v, k, INCREMENTS[k][1])
        self.s.setup_done = True
        self._log("increments(" + ", ".join(f"{k}={v:g}" for k, v in self.s.settings.items()) + ")")
        return dict(self.s.settings)

    # ---- supports (boundary conditions) ----------------------------------
    def _support(self, kind: str, at: float) -> Support:
        t = _t(at)
        self.s._push()
        m = self.s.model
        m.supports = [s for s in m.supports if abs(s.t - t) > 0.02]
        sup = Support(kind, t)
        m.supports.append(sup)
        m.supports.sort(key=lambda s: s.t)
        self._log(f"{kind}(at={t:g})")
        self._edit(f"{kind.capitalize()} support at t = {t:g}")
        return sup

    def fixed(self, at: float = 0.0) -> Support:
        """Fixed support (encastre) at position at (0 = start, 1 = end)."""
        return self._support("fixed", at)

    def pinned(self, at: float = 0.0) -> Support:
        """Pinned support at position at."""
        return self._support("pinned", at)

    def roller(self, at: float = 1.0) -> Support:
        """Roller at position at: free to slide along the member."""
        return self._support("roller", at)

    # ---- loads ---------------------------------------------------------
    def _load(self, ld: Load) -> Load:
        self.s._push()
        self.s.model.loads.append(ld)
        self._log(load_command(ld))
        self._edit(f"Added {ld.label()}")
        return ld

    def point(self, P: float, at: float = 1.0, dir: str = "-z") -> Load:
        """Point load P (kN) at position at, in direction dir (+x -x +y -y +z -z)."""
        axis, sign = _direction(dir)
        return self._load(Load("point", axis, sign, _t(at), _positive(P, "P", "kN")))

    def uniform(self, w: float, start: float = 0.0, end: float = 1.0, dir: str = "-z") -> Load:
        """Uniform line load w (kN/m) from start to end."""
        axis, sign = _direction(dir)
        t0, t1 = sorted((_t(start, "start"), _t(end, "end")))
        if t1 - t0 < 1e-6:
            raise ScriptError("uniform(): start and end must differ")
        w = _positive(w, "w", "kN/m")
        return self._load(Load("uniform", axis, sign, t0, w, t1, w))

    def trapezoidal(self, w0: float, w1: float, start: float = 0.0, end: float = 1.0, dir: str = "-z") -> Load:
        """Linearly varying line load, w0 at start to w1 at end (kN/m)."""
        axis, sign = _direction(dir)
        t0, t1 = _t(start, "start"), _t(end, "end")
        w0, w1 = float(w0), float(w1)
        if t1 < t0:
            t0, t1, w0, w1 = t1, t0, w1, w0
        if t1 - t0 < 1e-6 or min(w0, w1) < 0 or max(w0, w1) <= 0:
            raise ScriptError("trapezoidal(): needs start != end and intensities >= 0, not both 0")
        return self._load(Load("trapezoidal", axis, sign, t0, w0, t1, w1))

    def clear(self, what: str = "loads") -> None:
        """Remove 'loads', 'supports' or 'all'."""
        if what not in ("loads", "supports", "all"):
            raise ScriptError("clear(): 'loads', 'supports' or 'all'")
        self.s.clear(what)

    def undo(self) -> None:
        """Undo the last model edit."""
        self.s.undo()

    # ---- solving and results ---------------------------------------------
    def solve(self):
        """Solve the model. Returns the results (also available as `results`)."""
        issues = self.s.model.issues()
        if issues:
            raise ScriptError("Can't solve yet: " + "; ".join(issues))
        res = self.s.solve()
        if res is None:
            raise ScriptError(self.s.error or "solve failed")
        self.solved = True
        if self.view is not None:  # so show() / cut() in the same script see the new fields
            self.view.fields = self.fields()
            self.view.field_index = 0
        st = res.stats
        print(f"solved: {st.get('elements', '?')} elements, {st.get('dof', '?')} dof, "
              f"{sum(st[k] for k in ('t_assemble_s', 't_solve_s', 't_modes_s')):.2f} s")
        return res

    def fields(self) -> list[str]:
        """Names of the result fields."""
        r = self._res()
        return list(r.fields) + [f"mode{i + 1}" for i in range(len(r.modes))]

    def field(self, name: str = "von_mises") -> np.ndarray:
        """Nodal values of a field (von_mises, U, U1..U3, S11..S23, E11..E23, SED...)."""
        r = self._res()
        if name not in r.fields:
            raise ScriptError(f"No field {name!r}. Try: {', '.join(r.fields)}")
        return r.fields[name]

    def peak(self, name: str = "von_mises") -> dict:
        """Largest |value| of a field and where it is (mm)."""
        r = self._res()
        v = self.field(name)
        i = int(np.argmax(np.abs(v)))
        x, y, z = (float(c) for c in r.mesh.nodes[i])
        return {"field": name, "value": float(v[i]), "at_mm": (round(x, 1), round(y, 1), round(z, 1)),
                "min": float(v.min()), "max": float(v.max())}

    def at(self, t: float, name: str = "U3") -> float:
        """Mean of a field over the cross-section at position t along the member."""
        r = self._res()
        v, c = self.field(name), r.mesh.nodes[:, self.s.model.member.axis]
        target = _t(t, "t") * self.s.model.member.length
        layer = np.isclose(c, c[np.argmin(np.abs(c - target))])
        return float(v[layer].mean())

    def reactions(self) -> list[dict]:
        """Support reactions (kN), plus the total."""
        r = self._res()
        return list(r.support_reactions) + [{"total_kN": (r.reactions / 1e3).round(4).tolist()}]

    def modes(self) -> list[float]:
        """Natural frequencies (Hz)."""
        return [round(m["f"], 4) for m in self._res().modes]

    def summary(self) -> dict:
        """Section, material and (for columns) buckling summary."""
        return self.s.model.summary()

    # ---- view ----------------------------------------------------------
    def _view(self):
        """The browser's view; None when no page is attached (view commands then do nothing)."""
        if self.view is not None:
            self.view_changed = True
        return self.view

    def show(self, name: str = "von_mises") -> None:
        """Colour the viewport by a field (solve first)."""
        v = self._view()
        if v is None:
            return
        if name not in v.fields:
            raise ScriptError(f"No field {name!r}. Try: {', '.join(v.fields)}")
        v.field_index = v.fields.index(name)

    def cut(self, normal="x", at: float = 0.5) -> None:
        """Section cut. normal: 'x' 'y' 'z' or a vector (1, 0, 1); at: 0..1 across the model along it."""
        r = self._res()
        v = self._view()
        if v is None:
            return
        n = np.array({"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[normal] if isinstance(normal, str) else normal, float)
        if n.shape != (3,) or not np.linalg.norm(n) > 0:
            raise ScriptError("cut(): normal is 'x', 'y', 'z' or three numbers")
        n /= np.linalg.norm(n)
        lo, hi = r.mesh.nodes.min(0), r.mesh.nodes.max(0)
        centre, half = (lo + hi) / 2, float((hi - lo).max()) / 2
        corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
        proj = corners @ n
        d = proj.min() + float(at) * (proj.max() - proj.min())
        origin = centre + (d - centre @ n) * n
        v.section_on, v.section_locked = True, True
        v.plane_normal = n.tolist()
        v.plane_origin = ((origin - centre) / half).tolist()

    def uncut(self) -> None:
        """Remove the section cut."""
        v = self._view()
        if v is None:
            return
        v.section_on = v.section_locked = False

    def view(self, azimuth: float = -60, elevation: float = 22) -> None:
        """Camera direction in degrees (default: iso)."""
        v = self._view()
        if v is None:
            return
        v.azimuth, v.elevation = float(azimuth) % 360, float(np.clip(elevation, -89, 89))

    # ---- scripts -------------------------------------------------------
    def journal(self) -> str:
        """Everything done this session, as a script."""
        return "\n".join(e["cmd"] for e in self.s.journal)


COMMAND_NAMES = ["beam", "column", "round_beam", "round_column", "material", "demo", "increments",
                 "fixed", "pinned", "roller", "point", "uniform", "trapezoidal", "clear", "undo",
                 "solve", "fields", "field", "peak", "at", "reactions", "modes", "summary",
                 "show", "cut", "uncut", "view", "journal"]

GROUPS = [("Geometry", ["beam", "column", "round_beam", "round_column", "material", "demo", "increments"]),
          ("Supports", ["fixed", "pinned", "roller"]),
          ("Loads", ["point", "uniform", "trapezoidal", "clear", "undo"]),
          ("Solve & results", ["solve", "fields", "field", "peak", "at", "reactions", "modes", "summary"]),
          ("View", ["show", "cut", "uncut", "view"]),
          ("Scripts", ["run", "journal", "help"])]


class _Proxy:
    """`api` for imported modules: the commands of whichever console is running them."""

    def __getattr__(self, name):
        if _current is None:
            raise RuntimeError("archimedes_fe.scripting.api is only available while a console script runs")
        if name in COMMAND_NAMES:
            return getattr(_current, name)
        if name in ("model", "results"):
            return getattr(_current.s, name)
        raise AttributeError(name)


api = _Proxy()


class Console:
    """One interactive interpreter per modeller session."""

    def __init__(self, session: ModelSession, view=None):
        self.cmd = Commands(session, view)
        self.buffer: list[str] = []
        self.uploaded: dict[str, str] = {}
        self.ns: dict = {"__name__": "__console__", "np": np, "math": math, "Model": Model, "Member": Member,
                         "Section": Section, "Material": Material, "Load": Load, "Support": Support,
                         "STEEL": STEEL, "CONCRETE": CONCRETE, "api": api}
        for name in COMMAND_NAMES:
            self.ns[name] = getattr(self.cmd, name)
        self.ns["run"] = self.run
        self.ns["help"] = self.help
        if str(SCRIPTS) not in sys.path:
            sys.path.insert(0, str(SCRIPTS))

    @property
    def session(self) -> ModelSession:
        return self.cmd.s

    def help(self, what=None) -> None:
        """help() lists the commands; help(beam) explains one."""
        if what is not None:
            if callable(what) and getattr(what, "__doc__", None):
                import inspect
                try:
                    sig = str(inspect.signature(what, eval_str=True))
                except (TypeError, ValueError):
                    sig = "(...)"
                print(f"{getattr(what, '__name__', '?')}{sig}\n    {what.__doc__.strip()}")
            else:
                import pydoc
                print(pydoc.render_doc(what, renderer=pydoc.plaintext))
            return
        print("Archimedes console: Python with modelling commands. Units: mm, kN, kN/m, MPa.")
        print("Positions along the member: at / start / end from 0 to 1. Directions: '+x' '-x' '+y' '-y' '+z' '-z'.")
        for title, names in GROUPS:
            print(f"  {title:<16}" + "  ".join(names))
        print("Objects: model, results, np, Model, Section, Load, Support, Material")
        print("help(point) for one command · run('file.py') runs a script · gestures and mouse edits are journaled")

    # ---- execution -------------------------------------------------------
    def _refresh(self) -> None:
        self.ns["model"] = self.session.model
        self.ns["results"] = self.session.results

    def _exec(self, source: str, filename: str, interactive: bool) -> None:
        tree = ast.parse(source, filename, "exec")
        last = None
        if interactive and tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, filename, "exec"), self.ns)
        if last is not None:
            value = eval(compile(last, filename, "eval"), self.ns)
            if value is not None:
                self.ns["_"] = value
                print(_pretty(value))

    def execute(self, source: str, filename: str = "<console>", interactive: bool = True) -> dict:
        """Run source code; returns {"output", "error", "solved", "view", "changed"}."""
        global _current
        s = self.session
        before = (s.version, _fingerprint(s.model))
        out = io.StringIO()
        error = False
        self.cmd.solved = self.cmd.view_changed = False
        with _LOCK:
            prev, _current = _current, self.cmd
            s.source = "console"
            self._refresh()
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                    self._exec(source, filename, interactive)
            except ScriptError as exc:
                out.write(f"Error: {exc}\n")
                error = True
            except SystemExit:
                out.write("(exit ignored: the console keeps running)\n")
            except BaseException as exc:  # noqa: BLE001 - show the user's traceback, keep the server
                # drop this module's frames so the trace starts at the user's code
                frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename != __file__]
                out.write("Traceback (most recent call last):\n" if frames else "")
                out.write("".join(traceback.format_list(frames)))
                out.write("".join(traceback.format_exception_only(type(exc), exc)))
                error = True
            finally:
                s.source = "gui"
                _current = prev
                self._refresh()
        changed = s.version != before[0]
        if not changed and _fingerprint(s.model) != before[1]:  # model edited directly, e.g. model.loads.append(...)
            s._changed("Model edited from the console")
            s._log("# model edited directly in the console; as commands:")
            for line in model_commands(s.model):
                s._log(line)
            changed = True
        return {"output": out.getvalue(), "error": error, "solved": self.cmd.solved,
                "view": self.cmd.view_changed, "changed": changed}

    def push(self, line: str) -> dict:
        """One line of interactive input. more=True means a block is still open (prompt '...')."""
        self.buffer.append(line)
        source = "\n".join(self.buffer)
        try:
            code = codeop.compile_command(source, "<console>", "exec")
        except (SyntaxError, OverflowError, ValueError):
            code = True  # complete but wrong: run it so the error shows
        # a block (def/for/if...) stays open until a blank line
        if code is None or (len(self.buffer) > 1 and line.strip() != ""):
            return {"more": True, "output": "", "error": False, "solved": False, "view": False, "changed": False}
        self.buffer = []
        r = self.execute(source)
        r["more"] = False
        return r

    def reset_input(self) -> None:
        self.buffer = []

    def run(self, path: str) -> None:
        """Run a script in this console: an imported file, a name in fe_scripts/, or a path."""
        name = str(path)
        if name in self.uploaded:
            source, filename = self.uploaded[name], name
        else:
            p = Path(name)
            for cand in (p, SCRIPTS / p, ROOT / p):
                if cand.is_file():
                    p = cand
                    break
            else:
                raise ScriptError(f"run(): no script {name!r} (looked in fe_scripts/ and the project folder)")
            source, filename = p.read_text(encoding="utf-8"), str(p)
        exec(compile(source, filename, "exec"), self.ns)

    def run_script(self, name: str, source: str) -> dict:
        """A file imported from the browser: remembered, so run(name) repeats it."""
        self.uploaded[name] = source  # its commands are journaled one by one as they run
        return self.execute(source, name, interactive=False)


def model_commands(m: Model) -> list[str]:
    """Commands that rebuild a model from scratch."""
    mem, sec = m.member, m.member.section
    vertical = mem.orientation == "vertical"
    if sec.kind == "circle":
        out = [f"{'round_column' if vertical else 'round_beam'}(d={sec.d:g}, L={mem.length:g}, name={m.name!r})"]
    else:
        out = [f"{'column' if vertical else 'beam'}(b={sec.b:g}, h={sec.h:g}, L={mem.length:g}, name={m.name!r})"]
    mat = m.material
    if mat == STEEL or mat == CONCRETE:
        out.append(f"material({'steel' if mat == STEEL else 'concrete'!r})")
    else:
        out.append(f"material({mat.name!r}, E={mat.E:g}, nu={mat.nu:g}, rho={mat.rho:g}, strength={mat.strength:g})")
    out += [f"{sp.kind}(at={sp.t:g})" for sp in m.supports]
    out += [load_command(ld) for ld in m.loads]
    return out


def _fingerprint(m: Model) -> str:
    return repr((m.member, m.material, m.supports, m.loads))


def _pretty(v) -> str:
    if isinstance(v, Model):
        n, k = len(v.loads), len(v.supports)
        return (f"<{v.name}: {v.member.section.label()} × {v.member.length:g} mm, {v.material.name}, "
                f"{k} support{'s' * (k != 1)}, {n} load{'s' * (n != 1)}>")
    if isinstance(v, Load):
        return f"<{v.label()}>"
    if isinstance(v, Support):
        return f"<{v.kind} support at {v.t:g}>"
    if isinstance(v, Material):
        return f"<{v.name}: E {v.E:g} MPa, nu {v.nu:g}, strength {v.strength:g} MPa>"
    if type(v).__name__ == "Results" and hasattr(v, "fields"):
        return (f"<Results: {v.stats.get('elements', '?')} elements, {len(v.fields)} fields, {len(v.modes)} modes;"
                " fields() lists them, peak('von_mises') reads one>")
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k!r}: {_pretty(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)) and len(v) < 50 and all(isinstance(x, float) for x in v):
        return ("[" if isinstance(v, list) else "(") + ", ".join(f"{x:.6g}" for x in v) + ("]" if isinstance(v, list) else ")")
    if isinstance(v, np.ndarray):
        return np.array2string(v, precision=4, threshold=20) + f"  shape={v.shape}"
    return repr(v)


def main(argv: list[str] | None = None) -> int:
    """Headless use, like ``abaqus cae noGUI=``: run scripts, or an interactive prompt with no arguments.

    python -m archimedes_fe.scripting fe_scripts/column_study.py
    """
    argv = sys.argv[1:] if argv is None else argv
    con = Console(ModelSession())
    if argv:
        status = 0
        for path in argv:
            r = con.execute(f"run({path!r})", "<command line>", interactive=False)
            sys.stdout.write(r["output"])
            status |= int(r["error"])
        return status
    print("Archimedes console (headless). help() lists the commands; Ctrl+D quits.")
    while True:
        try:
            line = input("... " if con.buffer else ">>> ")
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt")
            con.reset_input()
            continue
        sys.stdout.write(con.push(line)["output"])


if __name__ == "__main__":
    raise SystemExit(main())
