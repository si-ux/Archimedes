"""Apply a ViewState to a PyVista plotter: the hook into the Archimedes desktop viewer.

Call ``bridge.apply(view)`` on the Qt main thread after each pipeline step
(e.g. in the slot connected to the camera QThread's signal). Section cuts use
mapper clipping planes, so clipping happens on the GPU and the mesh is never
re-cut on the CPU.

    bridge = PyVistaBridge(plotter, result_mesh, fields={"von_mises": "VonMises", "displacement": "U_mag"})
    ...
    out = pipeline.process(frame)        # in the worker
    bridge.apply(pipeline.view)          # in the main thread
"""
from __future__ import annotations

import numpy as np

from .view_state import ViewState


class PyVistaBridge:
    def __init__(self, plotter, mesh, fields: dict[str, str], distance_factor: float = 2.2, cmap: str = "jet"):
        import vtk

        self._vtk = vtk
        self.plotter, self.mesh, self.fields, self.cmap = plotter, mesh, fields, cmap
        self.center = np.array(mesh.center)
        self.size = float(mesh.length)
        self.d0 = distance_factor * self.size
        self._field = None
        self.actor = None
        self._plane = vtk.vtkPlane()
        self._picker = vtk.vtkCellPicker()
        self._picker.SetTolerance(0.0005)

    def _show_field(self, name: str) -> None:
        if name == self._field:
            return
        array = self.fields.get(name) or next(iter(self.fields.values()))
        self.actor = self.plotter.add_mesh(self.mesh, scalars=array, cmap=self.cmap, name="archimedes_result",
                                           scalar_bar_args={"title": name.replace("_", " ")})
        self._field = name
        self._plane_attached = False

    def apply(self, view: ViewState) -> None:
        self._show_field(view.field_name)
        _, up, back = view.camera_basis()
        focal = self.center + np.asarray(view.focal_point) * self.size / 2
        cam = self.plotter.camera
        cam.focal_point = focal.tolist()
        cam.position = (focal + back * self.d0 * view.distance).tolist()
        cam.up = up.tolist()

        mapper = self.actor.GetMapper()
        if view.section_on:
            n = np.asarray(view.plane_normal, float)
            o = self.center + np.asarray(view.plane_origin) * self.size / 2
            self._plane.SetOrigin(*o)
            self._plane.SetNormal(*(-n))  # keep the half the palm faces away from
            if not getattr(self, "_plane_attached", False):
                mapper.AddClippingPlane(self._plane)
                self._plane_attached = True
        elif getattr(self, "_plane_attached", False):
            mapper.RemoveAllClippingPlanes()
            self._plane_attached = False

    def probe(self, xy) -> tuple[np.ndarray, float] | None:
        """Pick the result value under viewport coords (0..1, y down). Returns (point, value) or None."""
        w, h = self.plotter.window_size
        if not self._picker.Pick(xy[0] * w, (1 - xy[1]) * h, 0, self.plotter.renderer):
            return None
        cid = self._picker.GetCellId()
        if cid < 0:
            return None
        array = self.fields.get(self._field) or next(iter(self.fields.values()))
        point = np.array(self._picker.GetPickPosition())
        if array in self.mesh.cell_data:
            return point, float(np.linalg.norm(np.atleast_1d(self.mesh.cell_data[array][cid])))
        pid = self.mesh.find_closest_point(point)
        return point, float(np.linalg.norm(np.atleast_1d(self.mesh.point_data[array][pid])))
