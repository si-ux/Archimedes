"""glTF 2.0 binary (GLB) export of a solved member, for VR/AR viewers.

The surface of the voxel mesh is written as triangles with per-vertex
colours from the chosen result field (same blue→red scale as the browser),
optionally on the deformed shape. glTF is y-up and in metres, so world
(x, y, z) in mm becomes (x, z, -y) / 1000. Support and load glyphs are added
as simple meshes, so the file opens correctly in any glTF viewer
(Android Scene Viewer, Quest browser, Blender, three.js).
"""
from __future__ import annotations

import json
import struct

import numpy as np

from .solver import CORNERS, Results

STOPS = np.array([[0.13, 0.25, 0.85], [0.0, 0.75, 0.95], [0.2, 0.8, 0.3], [0.98, 0.85, 0.1], [0.9, 0.15, 0.1]])

# the 6 faces of a hex in local-corner indices, wound outward (n = a + 2b + 4c)
FACES = {
    (-1, 0, 0): (0, 4, 6, 2), (1, 0, 0): (1, 3, 7, 5),
    (0, -1, 0): (0, 1, 5, 4), (0, 1, 0): (2, 6, 7, 3),
    (0, 0, -1): (0, 2, 3, 1), (0, 0, 1): (4, 5, 7, 6),
}
assert len(CORNERS) == 8


def colormap(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1) * (len(STOPS) - 1)
    i = np.minimum(t.astype(int), len(STOPS) - 2)
    f = (t - i)[:, None]
    return STOPS[i] * (1 - f) + STOPS[i + 1] * f


def surface_quads(res: Results) -> np.ndarray:
    """(nq, 4) global node ids of every exterior element face."""
    m = res.mesh
    occ = np.zeros(m.grid, bool)
    occ[tuple(m.cells.T)] = True
    quads = []
    for d, loc in FACES.items():
        nb = m.cells + np.array(d)
        inside = np.all((nb >= 0) & (nb < np.array(m.grid)), axis=1)
        exposed = ~inside
        exposed[inside] = ~occ[tuple(nb[inside].T)]
        quads.append(m.elems[exposed][:, list(loc)])
    return np.concatenate(quads)


def _box(center, size):
    c, s = np.asarray(center, float), np.asarray(size, float) / 2
    v = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)]) * s + c
    q = np.array(list(FACES.values()))
    return v, q


def to_glb(res: Results, field: str = "von_mises", deform: float | None = None) -> bytes:
    m = res.mesh
    if field.startswith("mode") and res.modes:
        i = int(field[4:]) - 1
        disp = res.modes[i]["shape"]
        val = np.linalg.norm(disp, axis=1)
        span = max(np.ptp(m.nodes, axis=0))
        disp = disp * 0.08 * span
    else:
        val = res.fields.get(field, res.fields["von_mises"])
        umax = np.abs(res.u).max() or 1.0
        span = max(np.ptp(m.nodes, axis=0))
        scale = deform if deform is not None else 0.08 * span / umax
        disp = res.u * scale
    lo, hi = float(val.min()), float(val.max())
    col = colormap((val - lo) / ((hi - lo) or 1.0)) ** 2.2  # glTF vertex colours are linear, the scale is sRGB
    pos = m.nodes + disp

    quads = surface_quads(res)
    # flat-shaded: 4 own vertices per quad so each face gets its own normal
    vp = pos[quads].reshape(-1, 3)
    vc = col[quads].reshape(-1, 3)
    meshes = [(vp, vc, quads.shape[0])]

    # glyphs: dark boxes at the supports
    L = res.model.member.length
    ax = m.axis
    lo_b, hi_b = m.nodes.min(axis=0), m.nodes.max(axis=0)
    size = (hi_b - lo_b) * 1.1
    for s in res.model.supports:
        c = (lo_b + hi_b) / 2
        c[ax] = s.t * L
        sz = size.copy()
        sz[ax] = max(0.04 * L, 50.0)
        if s.kind != "fixed":
            down = 2 if ax == 0 else 0
            c[down] = lo_b[down] - 0.25 * size[down]
            sz[down] = 0.3 * size[down]
        bv, bq = _box(c, sz)
        shade = {"fixed": 0.25, "pinned": 0.45, "roller": 0.6}[s.kind] ** 2.2
        meshes.append((bv[bq].reshape(-1, 3), np.full((len(bq) * 4, 3), shade), len(bq)))

    # assemble one primitive
    P = np.concatenate([mm[0] for mm in meshes])
    C = np.concatenate([mm[1] for mm in meshes])
    P = np.stack([P[:, 0], P[:, 2], -P[:, 1]], axis=1) / 1000.0  # z-up mm -> y-up m
    nq = len(P) // 4
    base = (np.arange(nq) * 4)[:, None]
    tri = np.concatenate([base + [0, 1, 2], base + [0, 2, 3]], axis=1).reshape(-1, 3)
    a, b, c = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    fn = np.cross(b - a, c - a)
    fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    N = np.zeros_like(P)
    N[tri[:, 0]] = fn
    N[tri[:, 1]] = fn
    N[tri[:, 2]] = fn
    return _pack(P.astype(np.float32), N.astype(np.float32), C.astype(np.float32), tri.astype(np.uint32).ravel())


def _pack(P, N, C, idx) -> bytes:
    blobs = [P.tobytes(), N.tobytes(), C.tobytes(), idx.tobytes()]
    offs, buf = [], b""
    for bl in blobs:
        offs.append(len(buf))
        buf += bl + b"\0" * ((4 - len(bl) % 4) % 4)
    views = [{"buffer": 0, "byteOffset": o, "byteLength": len(bl), **({"target": 34963} if i == 3 else {"target": 34962})}
             for i, (o, bl) in enumerate(zip(offs, blobs))]
    acc = [
        {"bufferView": 0, "componentType": 5126, "count": len(P), "type": "VEC3",
         "min": P.min(axis=0).tolist(), "max": P.max(axis=0).tolist()},
        {"bufferView": 1, "componentType": 5126, "count": len(N), "type": "VEC3"},
        {"bufferView": 2, "componentType": 5126, "count": len(C), "type": "VEC3"},
        {"bufferView": 3, "componentType": 5125, "count": len(idx), "type": "SCALAR"},
    ]
    gltf = {
        "asset": {"version": "2.0", "generator": "archimedes_fe"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "member"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2},
                                    "indices": 3, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0,
                                                "roughnessFactor": 0.8}, "doubleSided": True}],
        "buffers": [{"byteLength": len(buf)}], "bufferViews": views, "accessors": acc,
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(buf))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js
    out += struct.pack("<II", len(buf), 0x004E4942) + buf
    return out
