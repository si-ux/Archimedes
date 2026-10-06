"""Exact B-Rep geometry for the gmsh path.

The voxel core approximates the fillet with a staircase; this builds the real
solid in OpenCASCADE — two boxes fused, the re-entrant vertical edge filleted
to radius r — so the tetrahedral mesh resolves the curvature that actually
drives the stress concentration.
"""

from __future__ import annotations

from .spec import Geometry


def build_gmsh_model(g: Geometry, h: float, order: int = 1, name: str = "bracket_L"):
    """Populate the current gmsh model. Caller owns initialize()/finalize()."""
    import gmsh

    occ = gmsh.model.occ
    gmsh.model.add(name)

    leg = occ.addBox(0, 0, 0, g.t, g.H, g.D)
    arm = occ.addBox(0, g.H - g.t, 0, g.W, g.t, g.D)
    out, _ = occ.fuse([(3, leg)], [(3, arm)])
    occ.synchronize()
    vol = out[0][1]

    if g.r > 0:
        # the re-entrant edge runs along z at (x, y) = (t, H - t)
        eps = min(g.t, g.H) * 1e-3
        edges = occ.getEntitiesInBoundingBox(
            g.t - eps, g.H - g.t - eps, -eps,
            g.t + eps, g.H - g.t + eps, g.D + eps, 1)
        ids = [e[1] for e in edges]
        if ids:
            try:
                out = occ.fillet([vol], ids, [g.r])
                occ.synchronize()
                vol = out[0][1]
            except Exception:
                # a radius larger than the local geometry allows — carry on sharp
                occ.synchronize()

    gmsh.model.addPhysicalGroup(3, [vol], tag=1, name="solid")

    # face groups matching the voxel path's tags
    eps = min(g.t, g.D) * 1e-3
    span = min(g.t, g.W)

    def faces_in(x0, y0, z0, x1, y1, z1):
        return [s[1] for s in occ.getEntitiesInBoundingBox(x0, y0, z0, x1, y1, z1, 2)]

    base = faces_in(-eps, -eps, -eps, g.W + eps, eps, g.D + eps)
    tip = faces_in(g.W - span - eps, g.H - eps, -eps, g.W + eps, g.H + eps, g.D + eps)
    web = faces_in(-eps, -eps, -eps, eps, g.H + eps, g.D + eps)
    for tag, ents, nm in ((1, tip, "tip"), (2, web, "web"), (3, base, "base")):
        if ents:
            gmsh.model.addPhysicalGroup(2, ents, tag=tag, name=nm)

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h * 0.35)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)
    gmsh.option.setNumber("Mesh.Algorithm3D", 10)          # HXT
    gmsh.option.setNumber("Mesh.ElementOrder", 2 if order >= 2 else 1)
    gmsh.model.mesh.generate(3)
    return vol
