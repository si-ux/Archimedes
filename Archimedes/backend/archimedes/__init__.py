"""Archimedes Workbench — FEniCSx/DOLFINx solver core.

The workbench UI (``Archimedes Workbench.dc.html``) ships an in-browser hex8
solver used for instant preview.  This package is the authoritative engine:
the same study specification is solved with DOLFINx/PETSc, optionally on a
curvature-resolved gmsh tetrahedral mesh, and streamed back to the UI over
``ws://127.0.0.1:8791``.

Nothing here imports dolfinx at module scope — the CLI, the study schema and
the bridge all run on a bare Python install so the wire protocol can be
exercised without the container.
"""

__version__ = "1.0.0"
PROTOCOL = 1

__all__ = ["__version__", "PROTOCOL"]
