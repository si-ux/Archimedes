# Archimedes solver core — FEniCSx / DOLFINx

The authoritative solver behind the workbench UI. Linear elasticity with
DOLFINx + PETSc, natural frequencies with SLEPc, optional curvature-resolved
tetrahedral meshing with gmsh.

## Why a container

FEniCSx is not pip-installable in any dependable cross-platform way — DOLFINx,
PETSc, SLEPc and MPI are compiled C++/Fortran stacks distributed through conda
or Debian packages, and there is no Windows wheel at all. The upstream
`dolfinx/dolfinx:stable` image is the supported path, so the solver lives
there and the UI talks to it over a WebSocket.

Alternatives, and why they are not the default:

| Option | Verdict |
| --- | --- |
| `dolfinx/dolfinx` container (**used here**) | Upstream-supported, reproducible, works on Windows/macOS/Linux |
| conda-forge `fenics-dolfinx` | Works on Linux/macOS; no Windows build |
| WSL2 + apt `python3-dolfinx` | Fine on Windows, but ties the project to WSL |
| Native source build | Hours of PETSc/SLEPc compilation, fragile |
| WASM port | Does not exist; PETSc/MPI do not cross-compile usefully |

## Quick start

```bash
docker compose up --build
```

That publishes the bridge on `127.0.0.1:8791`. Open the workbench and the
`BRIDGE` readout in the status bar turns green — solves now run in DOLFINx.
Detach it (click `BRIDGE`, or stop the container) and the UI falls back to its
own in-browser hex8 core.

One-off runs without the server:

```bash
docker compose run --rm archimedes solve studies/bracket.json --csv out/r.csv
```

## CLI

Everything is under one entry point. Commands that do not solve (`study`,
`mesh`, `serve --engine mock`, `call`) run on a bare Python install with only
numpy; the rest need the container.

```
archimedes study new -o study.json           write a default study
archimedes study show study.json             resolved parameters + mesh sizing
archimedes mesh study.json                   mesh only, report quality
archimedes solve study.json --csv out.csv    full solve
archimedes modal study.json -n 3             natural frequencies
archimedes sweep study.json -p mesh.h -v 8,6,4,3
archimedes serve --engine dolfinx            bridge for the UI
archimedes call study.json                   drive a running bridge
```

Any study field can be overridden inline, which is what makes `sweep` useful:

```bash
archimedes solve study.json --set mesh.h=3 --set loads.tip_force=8000
archimedes sweep study.json -p mesh.h -v 10,8,6,4,3 --json convergence.json
```

`sweep` is the mesh-convergence workflow: same study, decreasing element size,
tabulated peak stress and strain energy so you can see whether the answer has
actually converged.

## Units

Millimetre / newton / megapascal / tonne-per-mm³ throughout — self-consistent,
so `omega` comes out in rad/s with no conversion factors:

| quantity | unit |
| --- | --- |
| length | mm |
| force | N |
| stress, modulus | MPa |
| density | t/mm³ (`kg/m³ × 1e-12`) |
| strain energy | mJ |

## Discretisation

Two meshers, selected by `mesh.mesher`:

- **`voxel`** (default) — a structured hex8 core that reproduces the UI's own
  grid *exactly*: same resolution, same cell occupancy, same node numbering,
  same local vertex order. Results come back in the UI's node order, so the
  viewport adopts DOLFINx fields without remeshing, and browser preview vs.
  DOLFINx is a pure solver-quality comparison. The local vertex order
  `n = a + 2b + 4c` is also basix's hexahedron reference order, so the
  connectivity needs no permutation.
- **`gmsh`** — the real B-Rep solid (two boxes fused, re-entrant edge filleted
  in OpenCASCADE) meshed with curvature refinement. Resolves the fillet
  properly, which is what actually drives the stress concentration; the voxel
  core only approximates it with a staircase.

## Solver

CG + GAMG, with the six rigid-body modes attached as GAMG's near-nullspace.
That is not optional for elasticity — without it the coarse grids are poorly
conditioned and the iteration count roughly triples.

Modal extraction is a SLEPc generalised Hermitian problem, shift-and-inverted
about zero. Constrained rows get a large stiffness diagonal and a small mass
diagonal, which parks the spurious constraint modes near 1e7 and well clear of
the physical spectrum.

## Bridge protocol

JSON text frames over RFC 6455, implemented on the stdlib socket module so the
bridge carries no third-party dependency.

```
client -> server
    {"op":"hello",  "protocol":1}
    {"op":"solve",  "id":"...", "study":{...}}
    {"op":"cancel", "id":"..."}
    {"op":"ping"}

server -> client
    {"op":"hello",    "protocol":1, "engine":"dolfinx 0.10.0", "capabilities":[...]}
    {"op":"log",      "id":.., "ch":.., "msg":.., "kind":"info|ok|warn|err", "t":..}
    {"op":"progress", "id":.., "stage":1..6, "pct":.., "note":..}
    {"op":"result",   "id":.., "stats":{..}, "grid":{..}, "fields":{..}}
    {"op":"error",    "id":.., "msg":..}
```

Nodal fields are base64 float32 in the UI's node order.

`--engine mock` meshes for real but solves nothing and returns zero fields —
it exists to exercise the wire protocol without the container, and says so
loudly in its first log line. Never read numbers off a mock run.

## Security note

The bridge executes whatever study specification it is handed. `docker-compose`
binds it to `127.0.0.1` deliberately. Do not expose port 8791 on a network
interface.
