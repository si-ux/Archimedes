"""archimedes — command line interface to the FEniCSx solver core.

    archimedes study new  -o study.json          write a default study
    archimedes study show study.json             resolved parameters + mesh sizing
    archimedes mesh       study.json             mesh only, report quality
    archimedes solve      study.json --csv out/  full DOLFINx solve
    archimedes modal      study.json -n 3        natural frequencies
    archimedes sweep      study.json -p mesh.h -v 8,6,4,3   convergence study
    archimedes serve      --engine dolfinx       bridge for the workbench UI
    archimedes call       study.json             drive a running bridge

Only the commands that actually solve import dolfinx, so `study`, `serve
--engine mock` and `call` work on a bare Python install.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import __version__, PROTOCOL
from .spec import Study, LIBRARY


def _p(*a):
    print(*a, flush=True)


def _logger(quiet: bool):
    tags = {"info": "  ", "ok": "OK", "warn": "! ", "err": "XX"}

    def log(ch, msg, kind="info"):
        if not quiet:
            _p(f"{tags.get(kind, '  ')} [{ch:<6}] {msg}")
    return log


def _progress(quiet: bool):
    state = {"stage": 0}

    def prog(stage, pct, note=""):
        if quiet:
            return
        if stage != state["stage"]:
            state["stage"] = stage
        sys.stdout.write(f"\r   stage {stage}  {pct:5.1f}%  {note:<44}")
        sys.stdout.flush()
        if pct >= 100:
            sys.stdout.write("\r" + " " * 72 + "\r")
    return prog


def _load(path: str) -> Study:
    if path in (None, "-", "default"):
        return Study()
    return Study.load(path)


def _apply_overrides(study: Study, pairs: list[str]) -> Study:
    """--set mesh.h=4 --set loads.tip_force=8000"""
    d = study.to_dict()
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, val = pair.split("=", 1)
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node:
                raise SystemExit(f"unknown key {key!r}")
            node = node[p]
        leaf = parts[-1]
        if leaf not in node:
            raise SystemExit(f"unknown key {key!r}")
        cur = node[leaf]
        try:
            node[leaf] = (val.lower() in ("1", "true", "yes")) if isinstance(cur, bool) \
                else type(cur)(val) if cur is not None else val
        except (TypeError, ValueError):
            node[leaf] = val
    return Study.from_dict(d)


# ------------------------------------------------------------ commands --
def cmd_study(args):
    if args.action == "new":
        s = Study()
        if args.material:
            s.material = LIBRARY[args.material]
        s = _apply_overrides(s, args.set)
        text = s.to_json()
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
            _p(f"wrote {args.out}")
        else:
            _p(text)
        return 0

    s = _apply_overrides(_load(args.path), args.set)
    h, hz, clamped = s.element_sizes()
    g = s.geometry.normalised()
    _p(f"study            {s.name}  (spec v{s.version})")
    _p(f"geometry         {g.W:g} x {g.H:g} x {g.D:g} mm · wall {g.t:g} · fillet r{g.r:g}")
    _p(f"material         {s.material.name}  E {s.material.E/1000:g} GPa  nu {s.material.nu}"
       f"  rho {s.material.rho:g} kg/m3  sy {s.material.sy:g} MPa")
    _p(f"lame             lambda {s.material.lam/1000:.2f} GPa   mu {s.material.mu/1000:.2f} GPa")
    _p(f"mesher           {s.mesh.mesher}   h {h:.3f} mm   hz {hz:.3f} mm"
       f"{'   (clamped to budget)' if clamped else ''}")
    from . import voxel
    vm = voxel.build(g, h, hz)
    q = vm.quality(g)
    _p(f"voxel core       {vm.nx} x {vm.ny} x {vm.nz}  ->  {q['elements']} hex8, "
       f"{q['nodes']} nodes, {q['dof']} DOF")
    _p(f"quality          aspect {q['aspect_ratio']:.2f}   across wall {q['across_wall']}   "
       f"single-span {q['single_span']}")
    _p(f"loads            tip {s.loads.tip_force:g} N @ {s.loads.theta:g} deg   "
       f"web {s.loads.pressure:g} MPa   dT {s.loads.dT:g} K   gravity {s.loads.gravity}")
    _p(f"solver           {s.solver.ksp} + {s.solver.pc}   rtol {s.solver.rtol:g}   "
       f"maxit {s.solver.maxit}")
    return 0


def cmd_mesh(args):
    from . import voxel
    s = _apply_overrides(_load(args.path), args.set)
    g = s.geometry.normalised()
    h, hz, clamped = s.element_sizes()
    t0 = time.perf_counter()
    vm = voxel.build(g, h, hz)
    dt = (time.perf_counter() - t0) * 1000
    q = vm.quality(g)
    _p(json.dumps({"mesher": "voxel", "h": h, "hz": hz, "clamped": clamped,
                   "grid": [vm.nx, vm.ny, vm.nz], "ms": round(dt, 1), **q}, indent=2))
    if args.out:
        import numpy as np
        np.savez_compressed(args.out, points=vm.points, cells=vm.cells)
        _p(f"wrote {args.out}")
    return 0


def cmd_solve(args):
    from . import fem
    from .post import write_csv, to_ui_order
    s = _apply_overrides(_load(args.path), args.set)
    if args.mesher:
        s.mesh.mesher = args.mesher
    if args.modes:
        s.outputs.modes = args.modes

    res = fem.solve(s, _logger(args.quiet), _progress(args.quiet))
    st = res["stats"]
    f = res["fields"]
    mat = s.material
    vmmax = float(f["vm"].max()) if "vm" in f else 0.0

    _p("")
    _p(f"engine           {res['engine']}   mesher {res['mesher']}")
    _p(f"mesh             {st['elements']} cells · {st['nodes']} nodes · {st['dof']} DOF")
    _p(f"krylov           {st['iterations']} iterations · |r| {st['residual']:.3e} · "
       f"{'converged' if st['converged'] else 'NOT CONVERGED'}")
    _p(f"|U| max          {float(f['umag'].max()):.5f} mm")
    if vmmax:
        _p(f"sigma_vm max     {vmmax:.2f} MPa   (yield {mat.sy:g} -> SF {mat.sy/vmmax:.2f})")
    if "p1" in f:
        _p(f"sigma_1 max      {float(f['p1'].max()):.2f} MPa")
    _p(f"strain energy    {f['strain_energy']:.4f} mJ")
    if f.get("reaction"):
        r = f["reaction"]
        _p(f"reaction         [{r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}] N")
    if res["frequencies"]:
        _p("frequencies      " + ", ".join(f"{v:.1f} Hz" for v in res["frequencies"]))
    _p(f"wall             {res['wall']/1000:.2f} s")

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        vm = res["voxel"]
        ui = {k: to_ui_order(v, res["perm"], vm.nn) for k, v in f.items()
              if hasattr(v, "shape")}
        write_csv(args.csv, vm.points, ui)
        _p(f"wrote            {args.csv}")
    if args.json:
        payload = {"stats": st, "quality": res["quality"], "times": res["times"],
                   "frequencies": res["frequencies"],
                   "max": {"umag": float(f["umag"].max()),
                           "vm": vmmax,
                           "strain_energy": f["strain_energy"]}}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        _p(f"wrote            {args.json}")
    return 0 if st["converged"] else 1


def cmd_modal(args):
    from . import fem
    s = _apply_overrides(_load(args.path), args.set)
    s.outputs.modes = args.n
    res = fem.solve(s, _logger(args.quiet), _progress(args.quiet))
    _p("")
    for i, hz in enumerate(res["frequencies"], 1):
        _p(f"mode {i}   {hz:10.2f} Hz")
    return 0


def cmd_sweep(args):
    """Mesh-convergence / parameter study — the reason a CLI earns its keep."""
    from . import fem
    base = _apply_overrides(_load(args.path), args.set)
    values = [v.strip() for v in args.values.split(",") if v.strip()]
    _p(f"{'value':>10}  {'elements':>9}  {'DOF':>9}  {'iters':>6}  "
       f"{'|U|max':>10}  {'vm_max':>10}  {'energy':>12}  {'s':>6}")
    _p("-" * 84)
    rows = []
    for v in values:
        s = _apply_overrides(base, [f"{args.param}={v}"])
        res = fem.solve(s, lambda *a: None, lambda *a: None)
        st, f = res["stats"], res["fields"]
        vmx = float(f["vm"].max()) if "vm" in f else 0.0
        _p(f"{v:>10}  {st['elements']:>9}  {st['dof']:>9}  {st['iterations']:>6}  "
           f"{float(f['umag'].max()):>10.5f}  {vmx:>10.2f}  "
           f"{f['strain_energy']:>12.4f}  {res['wall']/1000:>6.2f}")
        rows.append({"value": v, **st, "umax": float(f["umag"].max()),
                     "vm_max": vmx, "energy": f["strain_energy"]})
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"param": args.param, "rows": rows}, fh, indent=2)
        _p(f"\nwrote {args.json}")
    return 0


def cmd_serve(args):
    from .bridge import serve
    serve(args.host, args.port, args.engine, not args.quiet)
    return 0


def cmd_call(args):
    """Act as a client of a running bridge — same path the UI takes."""
    from .wsclient import Client, unpack
    s = _apply_overrides(_load(args.path), args.set)
    if args.modes:
        s.outputs.modes = args.modes
    with Client(args.url) as c:
        hello = c.recv()
        _p(f"connected        {args.url}  engine {hello.get('engine')}  "
           f"protocol v{hello.get('protocol')}")
        c.send({"op": "solve", "id": "cli", "study": s.to_dict()})
        while True:
            msg = c.recv()
            if msg is None:
                _p("connection closed"); return 1
            op = msg.get("op")
            if op == "log":
                _p(f"   [{msg['ch']:<6}] {msg['msg']}")
            elif op == "progress" and not args.quiet:
                sys.stdout.write(f"\r   stage {msg['stage']} {msg['pct']:5.1f}% "
                                 f"{msg.get('note',''):<44}")
                sys.stdout.flush()
            elif op == "error":
                _p(f"\nerror: {msg['msg']}"); return 1
            elif op == "result":
                sys.stdout.write("\r" + " " * 72 + "\r")
                st = msg["stats"]
                umag = unpack(msg["fields"]["umag"])
                _p(f"engine           {msg['engine']}")
                _p(f"mesh             {st['elements']} cells · {st['dof']} DOF")
                _p(f"krylov           {st['iterations']} it · |r| {st['residual']:.3e}")
                _p(f"|U| max          {float(umag.max()):.5f} mm")
                if "vm" in msg["fields"]:
                    _p(f"sigma_vm max     {float(unpack(msg['fields']['vm']).max()):.2f} MPa")
                if msg.get("frequencies"):
                    _p("frequencies      " + ", ".join(f"{v:.1f} Hz" for v in msg["frequencies"]))
                return 0


# --------------------------------------------------------------- parser --
def build_parser():
    ap = argparse.ArgumentParser(
        prog="archimedes",
        description="FEniCSx/DOLFINx solver core for the Archimedes Workbench.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--version", action="version",
                    version=f"archimedes {__version__} (protocol v{PROTOCOL})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, path=True):
        if path:
            p.add_argument("path", nargs="?", default="default",
                           help="study JSON (omit for built-in defaults)")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                       help="override a study field, e.g. --set mesh.h=4")
        p.add_argument("-q", "--quiet", action="store_true")

    p = sub.add_parser("study", help="create or inspect a study")
    p.add_argument("action", choices=["new", "show"])
    p.add_argument("path", nargs="?", default="default")
    p.add_argument("-o", "--out")
    p.add_argument("-m", "--material", choices=sorted(LIBRARY))
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(fn=cmd_study)

    p = sub.add_parser("mesh", help="mesh and report quality (no solve)")
    common(p)
    p.add_argument("-o", "--out", help="write points/cells to an .npz")
    p.set_defaults(fn=cmd_mesh)

    p = sub.add_parser("solve", help="run the DOLFINx solve")
    common(p)
    p.add_argument("--mesher", choices=["voxel", "gmsh"])
    p.add_argument("-n", "--modes", type=int, default=0)
    p.add_argument("--csv", help="write nodal results")
    p.add_argument("--json", help="write a summary")
    p.set_defaults(fn=cmd_solve)

    p = sub.add_parser("modal", help="natural frequencies only")
    common(p)
    p.add_argument("-n", type=int, default=3)
    p.set_defaults(fn=cmd_modal)

    p = sub.add_parser("sweep", help="parameter / mesh-convergence study")
    common(p)
    p.add_argument("-p", "--param", required=True, help="dotted key, e.g. mesh.h")
    p.add_argument("-v", "--values", required=True, help="comma separated")
    p.add_argument("--json")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("serve", help="WebSocket bridge for the workbench UI")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8791)
    p.add_argument("--engine", default="dolfinx", choices=["dolfinx", "mock"])
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("call", help="solve through a running bridge")
    common(p)
    p.add_argument("--url", default="ws://127.0.0.1:8791")
    p.add_argument("-n", "--modes", type=int, default=0)
    p.set_defaults(fn=cmd_call)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        _p("\ninterrupted")
        return 130
    except ModuleNotFoundError as exc:
        _p(f"\nmissing dependency: {exc.name}")
        _p("this command needs the FEniCSx stack — run it inside the container:")
        _p("    docker compose run --rm archimedes archimedes " + " ".join(sys.argv[1:]))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
