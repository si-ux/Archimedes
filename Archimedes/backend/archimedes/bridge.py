"""WebSocket bridge — ws://127.0.0.1:8791, protocol v1.

The workbench UI connects here; if the socket is up it solves with DOLFINx,
otherwise it falls back to its own in-browser hex8 core.  RFC 6455 is
implemented directly on the stdlib socket module so the bridge has no
third-party dependency and can be exercised on a bare Python install
(``--engine mock``) without the container.

Wire format is JSON text frames.  Nodal fields travel as base64 float32 in
the UI's own node order, so the viewport can adopt them without remeshing.

    client -> server
        {"op":"hello",  "protocol":1}
        {"op":"solve",  "id":"...", "study":{...}}           the parametric bracket
        {"op":"solve_mesh", "id":"...", "mesh":{nodes, elems, fixed, f, material, dT}}
                                                             any part, meshed by the Workbench
        {"op":"cancel", "id":"..."}
        {"op":"ping"}

    server -> client
        {"op":"hello",    "protocol":1, "engine":"dolfinx 0.9.0", "capabilities":[...]}
        {"op":"log",      "id":.., "ch":.., "msg":.., "kind":"info|ok|warn|err", "t":..}
        {"op":"progress", "id":.., "stage":1..6, "pct":.., "note":..}
        {"op":"result",   "id":.., "stats":{..}, "grid":{..}, "fields":{..}}
        {"op":"mesh_result", "id":.., "engine":.., "stats":{..}, "u":b64, "umag":b64, "vm":b64, "p1":b64, "energy":..}
        {"op":"error",    "id":.., "msg":..}
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import socketserver
import struct
import threading
import time
import traceback

import numpy as np

from . import PROTOCOL, __version__
from .spec import Study

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# ----------------------------------------------------------- framing ----
class WSError(Exception):
    pass


def _handshake(rfile, wfile) -> bool:
    line = rfile.readline(65536).decode("latin-1").strip()
    if not line.upper().startswith("GET"):
        return False
    headers = {}
    while True:
        h = rfile.readline(65536).decode("latin-1").strip()
        if not h:
            break
        if ":" in h:
            k, v = h.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    key = headers.get("sec-websocket-key")
    if not key:
        wfile.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
        return False
    accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
    wfile.write(
        ("HTTP/1.1 101 Switching Protocols\r\n"
         "Upgrade: websocket\r\nConnection: Upgrade\r\n"
         f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
    wfile.flush()
    return True


def _read_exact(rfile, n: int) -> bytes:
    buf = rfile.read(n)
    if buf is None or len(buf) < n:
        raise WSError("socket closed")
    return buf


def _recv(rfile):
    """Return (opcode, payload) for one (possibly fragmented) message."""
    op_first = None
    chunks = []
    while True:
        b0, b1 = _read_exact(rfile, 2)
        fin = b0 & 0x80
        opcode = b0 & 0x0F
        masked = b1 & 0x80
        ln = b1 & 0x7F
        if ln == 126:
            ln = struct.unpack(">H", _read_exact(rfile, 2))[0]
        elif ln == 127:
            ln = struct.unpack(">Q", _read_exact(rfile, 8))[0]
        mask = _read_exact(rfile, 4) if masked else None
        data = _read_exact(rfile, ln) if ln else b""
        if mask:
            data = bytes(b ^ mask[i & 3] for i, b in enumerate(data))
        if opcode in (0x8, 0x9, 0xA):
            return opcode, data
        if op_first is None:
            op_first = opcode
        chunks.append(data)
        if fin:
            return op_first, b"".join(chunks)


def _send(wfile, payload: bytes, opcode: int = 0x1) -> None:
    n = len(payload)
    head = bytes([0x80 | opcode])
    if n < 126:
        head += bytes([n])
    elif n < 65536:
        head += bytes([126]) + struct.pack(">H", n)
    else:
        head += bytes([127]) + struct.pack(">Q", n)
    wfile.write(head + payload)
    wfile.flush()


# -------------------------------------------------------- serialisation --
def f32(a) -> dict:
    arr = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    return {"dtype": "f32", "shape": list(arr.shape),
            "b64": base64.b64encode(arr.tobytes()).decode("ascii")}


class Cancelled(Exception):
    pass


# ------------------------------------------------------------- engines --
def run_dolfinx(study: Study, log, progress, cancelled):
    from . import fem
    from .post import to_ui_order

    def _log(ch, msg, kind="info"):
        cancelled()
        log(ch, msg, kind)

    def _prog(stage, pct, note=""):
        cancelled()
        progress(stage, pct, note)

    res = fem.solve(study, _log, _prog)
    vm = res["voxel"]
    perm = res["perm"]
    f = res["fields"]
    nn = vm.nn

    if perm is None:
        # tetrahedral solve: node counts do not line up, so resample the
        # solution onto the voxel nodes the viewport actually draws
        from .post import sample_ui_fields
        f = sample_ui_fields(res["mesh"], vm, f, study.material, study)
        _log("post", f"resampled onto {f['_hits']}/{f['_total']} voxel nodes "
                     f"for display", "info")
        conv = lambda a: a                       # noqa: E731  already UI ordered
    else:
        conv = lambda a: to_ui_order(a, perm, nn)   # noqa: E731

    fields = {"umag": f32(conv(f["umag"])),
              "u": f32(conv(f["u"]).reshape(-1))}
    for key in ("vm", "p1", "sed"):
        if key in f:
            fields[key] = f32(conv(f[key]))

    # mode shapes only travel when the node numbering matches; on the
    # tetrahedral path the frequencies are still exact, the shapes just
    # cannot be drawn on the voxel grid
    modes = []
    if perm is not None:
        for hz, phi in zip(res["frequencies"], res["modes"]):
            modes.append({"hz": hz, "phi": f32(to_ui_order(phi, perm, nn).reshape(-1))})

    return {
        "engine": res["engine"],
        "mesher": res["mesher"],
        "stats": res["stats"],
        "quality": res["quality"],
        "times": res["times"],
        "wall": res["wall"],
        "strain_energy": f["strain_energy"],
        "reaction": f.get("reaction"),
        "grid": {"nx": vm.nx, "ny": vm.ny, "nz": vm.nz,
                 "dx": vm.dx, "dy": vm.dy, "dz": vm.dz,
                 "nn": vm.nn, "ne": vm.ne},
        "fields": fields,
        "frequencies": res["frequencies"],
        "modes": modes,
    }


def run_mock(study: Study, log, progress, cancelled):
    """Protocol exerciser. Meshes for real, solves nothing."""
    from . import voxel
    log("core", "MOCK ENGINE — mesh is real, all field values are zero", "warn")
    vm = voxel.from_study(study)
    for stage in range(1, 6):
        cancelled()
        progress(stage, 100.0, "mock")
        time.sleep(0.05)
    z = np.zeros(vm.nn, dtype=np.float32)
    return {
        "engine": "mock",
        "mesher": study.mesh.mesher,
        "stats": {"elements": vm.ne, "nodes": vm.nn, "dof": vm.ndof,
                  "iterations": 0, "residual": 0.0, "converged": False,
                  "h": vm.dx, "mock": True},
        "quality": vm.quality(study.geometry.normalised()),
        "times": {}, "wall": 0.0, "strain_energy": 0.0, "reaction": None,
        "grid": {"nx": vm.nx, "ny": vm.ny, "nz": vm.nz,
                 "dx": vm.dx, "dy": vm.dy, "dz": vm.dz,
                 "nn": vm.nn, "ne": vm.ne},
        "fields": {"umag": f32(z), "u": f32(np.zeros(vm.nn * 3, dtype=np.float32)),
                   "vm": f32(z), "p1": f32(z), "sed": f32(z)},
        "frequencies": [], "modes": [],
    }


def _b64(a) -> str:
    return base64.b64encode(np.ascontiguousarray(np.asarray(a, dtype="<f4")).tobytes()).decode("ascii")


def run_dolfinx_mesh(payload: dict, log, progress, cancelled):
    """Any part: the Workbench posts its own (body-fitted) hex8 mesh, supports and nodal loads."""
    from . import fem

    def _log(ch, msg, kind="info"):
        cancelled()
        log(ch, msg, kind)

    def _prog(stage, pct, note=""):
        cancelled()
        progress(stage, pct, note)

    res = fem.solve_general(payload, _log, _prog)
    return {"engine": res["engine"], "stats": res["stats"], "energy": res["energy"],
            **{k: _b64(res[k]) for k in ("u", "umag", "vm", "p1")}}


def run_mock_mesh(payload: dict, log, progress, cancelled):
    """Protocol exerciser for posted meshes: decodes and checks the mesh, returns zero fields."""
    from .fem import mesh_arrays
    nodes, elems, fixed, f = mesh_arrays(payload)
    log("core", "MOCK ENGINE — posted mesh decoded, all field values are zero", "warn")
    progress(4, 100.0, "mock")
    z = np.zeros(len(nodes))
    return {"engine": "mock", "energy": 0.0,
            "stats": {"elements": int(len(elems)), "nodes": int(len(nodes)), "dof": int(nodes.size),
                      "fixed": int(len(fixed)), "load": float(np.abs(f).sum()), "mock": True},
            "u": _b64(np.zeros(nodes.size)), "umag": _b64(z), "vm": _b64(z), "p1": _b64(z)}


ENGINES = {"dolfinx": run_dolfinx, "mock": run_mock}
MESH_ENGINES = {"dolfinx": run_dolfinx_mesh, "mock": run_mock_mesh}


# -------------------------------------------------------------- server --
class Handler(socketserver.StreamRequestHandler):
    engine = "dolfinx"
    verbose = True

    def _emit(self, obj):
        with self._lock:
            _send(self.wfile, json.dumps(obj).encode("utf-8"))

    def handle(self):
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        peer = f"{self.client_address[0]}:{self.client_address[1]}"
        try:
            if not _handshake(self.rfile, self.wfile):
                return
        except Exception:
            return
        if self.verbose:
            print(f"[bridge] client connected {peer}", flush=True)
        self._emit({"op": "hello", "protocol": PROTOCOL,
                    "engine": self._engine_banner(), "version": __version__,
                    "capabilities": ["solve", "solve_mesh", "cancel", "modal", "voxel", "gmsh"]})
        try:
            while True:
                opcode, data = _recv(self.rfile)
                if opcode == 0x8:
                    break
                if opcode == 0x9:
                    _send(self.wfile, data, 0xA)
                    continue
                if opcode != 0x1:
                    continue
                try:
                    msg = json.loads(data.decode("utf-8"))
                except Exception:
                    self._emit({"op": "error", "msg": "malformed JSON"})
                    continue
                self._dispatch(msg)
        except (WSError, ConnectionError, OSError):
            pass
        finally:
            if self.verbose:
                print(f"[bridge] client gone {peer}", flush=True)

    def _engine_banner(self) -> str:
        if self.engine == "mock":
            return "mock"
        try:
            import dolfinx
            return f"dolfinx {dolfinx.__version__}"
        except Exception:
            return "dolfinx (unavailable)"

    def _dispatch(self, msg):
        op = msg.get("op")
        if op == "ping":
            self._emit({"op": "pong", "t": time.time()})
        elif op == "hello":
            pass
        elif op == "cancel":
            self._cancel.set()
        elif op == "solve":
            self._cancel.clear()
            threading.Thread(target=self._solve, args=(msg,), daemon=True).start()
        elif op == "solve_mesh":
            self._cancel.clear()
            threading.Thread(target=self._solve, args=(msg, True), daemon=True).start()
        else:
            self._emit({"op": "error", "msg": f"unknown op {op!r}"})

    def _solve(self, msg, posted=False):
        job = msg.get("id") or "job"
        t0 = time.perf_counter()

        def log(ch, m, kind="info"):
            self._emit({"op": "log", "id": job, "ch": ch, "msg": m,
                        "kind": kind, "t": round(time.perf_counter() - t0, 2)})
            if self.verbose:
                print(f"[{ch}] {m}", flush=True)

        def progress(stage, pct, note=""):
            self._emit({"op": "progress", "id": job, "stage": stage,
                        "pct": round(float(pct), 1), "note": note})

        def cancelled():
            if self._cancel.is_set():
                raise Cancelled()

        try:
            if posted:
                out = MESH_ENGINES[self.engine](msg.get("mesh") or {}, log, progress, cancelled)
                out["op"] = "mesh_result"
            else:
                study = Study.from_dict(msg.get("study") or {})
                out = ENGINES[self.engine](study, log, progress, cancelled)
                out["op"] = "result"
            out["id"] = job
            self._emit(out)
        except Cancelled:
            self._emit({"op": "error", "id": job, "msg": "cancelled by client"})
        except Exception as exc:
            traceback.print_exc()
            self._emit({"op": "error", "id": job,
                        "msg": f"{type(exc).__name__}: {exc}"})


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(host: str = "127.0.0.1", port: int = 8791,
          engine: str = "dolfinx", verbose: bool = True) -> None:
    if engine not in ENGINES:
        raise SystemExit(f"unknown engine {engine!r}; choose from {sorted(ENGINES)}")
    handler = type("BoundHandler", (Handler,), {"engine": engine, "verbose": verbose})
    with Server((host, port), handler) as srv:
        banner = handler("", ("", 0), None) if False else engine
        print(f"[bridge] archimedes {__version__} · protocol v{PROTOCOL} · "
              f"engine {banner} · listening on ws://{host}:{port}", flush=True)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n[bridge] shutting down", flush=True)
