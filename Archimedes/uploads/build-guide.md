ARCHIMEDES
FEniCSx Edition
Gesture-driven post-processing for imported CAD models
What this builds
A solver-agnostic 3D post-processor: import geometry from Fusion, SolidWorks or Revit, mesh it, solve it with FEniCSx, then explore the result field with your hands — clip planes, isosurfaces, thresholds, probes, camera orbit.
The viewer consumes VTK and XDMF, which means it also opens results from Abaqus, Ansys, COMSOL, CalculiX and OpenFOAM. FEniCSx is your own solver, not a dependency of the viewer.
Five projects. Each one produces something that works on its own. The gesture layer is the last thing added, not the thing everything hangs off.
# Architecture
  ┌── INGEST ─────────────────────────────────────────────────┐
  │  Fusion / SolidWorks  →  STEP, IGES                       │
  │  Revit / ArchiCAD     →  IFC                              │
  │  Any mesher           →  MSH, STL                         │
  └───────────────────────────┬───────────────────────────────┘
                              ▼
  ┌── MESH ───────────────────────────────────────────────────┐
  │  Gmsh (OpenCASCADE kernel reads STEP natively)            │
  │  ifcopenshell → tessellation → Gmsh  (BIM path)           │
  │  Physical groups tag faces for boundary conditions        │
  └───────────────────────────┬───────────────────────────────┘
                              ▼
  ┌── SOLVE ──────────────────────────────────────────────────┐
  │  DOLFINx: you write the weak form                         │
  │  Linear elasticity · thermal · modal · coupled            │
  │  PETSc backend, MPI-parallel                              │
  └───────────────────────────┬───────────────────────────────┘
                              ▼
  ┌── RESULT FILES ───────────────────────────────────────────┐
  │  XDMF + HDF5   (mesh + fields, time series)               │
  │  VTX / ADIOS2  (fast, native to DOLFINx)                  │
  │  VTU           (universal, what everyone else writes)     │
  └───────────────────────────┬───────────────────────────────┘
                              ▼
  ┌── VIEWER  (the actual product) ───────────────────────────┐
  │  PyVista / VTK unstructured grid                          │
  │  Filters: clip · threshold · contour · warp · probe       │
  │  Qt shell via pyvistaqt                                   │
  └───────────────────────────┬───────────────────────────────┘
                              ▲
  ┌── GESTURE LAYER ──────────┴───────────────────────────────┐
  │  MediaPipe hands in a QThread → Qt signals → filter calls │
  └───────────────────────────────────────────────────────────┘
## Project map
#
Project
Deliverable
Project 1
FEniCSx foundations
Install, weak forms, linear elasticity cantilever, von Mises, validated against theory
Project 2
CAD import pipeline
STEP and IFC → Gmsh → DOLFINx → result file, with physical-group BC tagging
Project 3
Viewer, mouse-driven
PyVista + Qt app: clip, threshold, contour, warp, probe, field switching
Project 4
Gesture layer
MediaPipe QThread mapped onto the Project 3 controls
Project 5
Depth and reach
Time scrubbing, modal animation, multi-solver ingest, spatial/AR mode
PROJECT 1   ·   3–4 weeks
FEniCSx Foundations
Write the weak form yourself, solve a cantilever, validate it
Milestone:  python cantilever.py produces results.xdmf; tip deflection within 3% of PL³/3EI; ParaView shows the von Mises field.
## Install — do not fight this
DOLFINx is not cleanly pip-installable. Two options that actually work:
# Option A — Docker (most reliable, use this first)
docker run -ti -v $(pwd):/root/shared -w /root/shared \
  --shm-size=512m dolfinx/dolfinx:stable
# Option B — conda (better for day-to-day dev, Linux/macOS)
conda create -n fenicsx -c conda-forge \
  fenics-dolfinx mpich pyvista python=3.12
conda activate fenicsx
# Verify
python -c &quot;import dolfinx; print(dolfinx.__version__)&quot;
# Additional packages you will need across all five projects
pip install gmsh meshio ifcopenshell pyvistaqt PySide6 \
            mediapipe opencv-python
On Windows, use WSL2 with the conda route, or Docker Desktop. Native Windows DOLFINx is not worth the time.
## What you must understand before writing code
FEniCSx will not give you a beam element. You give it a variational statement of the problem and it discretises and solves it. That is the whole point — and it means you need three concepts:
Concept
What it means
Weak form
The PDE multiplied by a test function and integrated by parts. Strong form: −∇·σ = f. Weak form: ∫ σ(u):ε(v) dx = ∫ f·v dx + ∫ T·v ds for all test functions v.
Function space
Where the solution lives. For 3D elasticity: continuous piecewise polynomials, vector-valued. In code: (&apos;Lagrange&apos;, 1, (3,)) means linear Lagrange elements with 3 components.
Trial and test functions
The unknown u is the trial function; v is the arbitrary test function. The bilinear form a(u,v) becomes the stiffness matrix; the linear form L(v) becomes the load vector.
Read the DOLFINx tutorial by Jørgen Dokken, chapters 1 through 3, before writing anything. The older FEniCS tutorials use a different API and will actively mislead you.
## Step-by-step walkthrough
Step 1  Build the mesh and function space
Start with a built-in box mesh so you are debugging physics, not geometry. CAD import comes in Project 2.
# cantilever.py
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
# Geometry in metres
L, Wd, H = 1.0, 0.05, 0.1      # length, width, height
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0, 0.0]), np.array([L, Wd, H])],
    [40, 4, 8],
    cell_type=mesh.CellType.hexahedron,
)
# Vector-valued Lagrange space, degree 1, 3 components
V = fem.functionspace(domain, (&apos;Lagrange&apos;, 1, (domain.geometry.dim,)))
Step 2  Define the material and the constitutive relation
Convert E and nu into the Lamé parameters, then write the strain and stress operators as plain Python functions of UFL expressions.
E   = 210e9        # Pa
nu  = 0.3
rho = 7850.0       # kg/m3
g   = 9.81
mu_     = E / (2.0 * (1.0 + nu))
lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
def epsilon(u):
    return ufl.sym(ufl.grad(u))          # small-strain tensor
def sigma(u):
    return (lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u))
            + 2.0 * mu_ * epsilon(u))    # isotropic linear elastic
Step 3  Write the weak form
This is the line that matters. Everything above is setup; this is the physics.
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
# Body force (self-weight) and surface traction
f = fem.Constant(domain, default_scalar_type((0.0, 0.0, -rho * g)))
T = fem.Constant(domain, default_scalar_type((0.0, 0.0, 0.0)))
ds = ufl.Measure(&apos;ds&apos;, domain=domain)
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L_form = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds
Step 4  Apply the clamped boundary condition
Locate the facets at x = 0 geometrically, find their degrees of freedom, and pin all three components to zero.
def clamped(x):
    return np.isclose(x[0], 0.0)
fdim = domain.topology.dim - 1
facets = mesh.locate_entities_boundary(domain, fdim, clamped)
u_D = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
bc  = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, facets), V)
Step 5  Solve
LinearProblem assembles, applies BCs, and hands off to PETSc. Use a direct LU solve while the problem is small.
problem = LinearProblem(
    a, L_form, bcs=[bc],
    petsc_options={&apos;ksp_type&apos;: &apos;preonly&apos;, &apos;pc_type&apos;: &apos;lu&apos;},
)
uh = problem.solve()
uh.name = &apos;displacement&apos;
Step 6  Post-process von Mises stress
Stress is a derived quantity. Build the deviatoric part, form the von Mises scalar, and interpolate it into a discontinuous space (stress is discontinuous across elements for linear displacement elements).
s = sigma(uh) - (1.0 / 3.0) * ufl.tr(sigma(uh)) * ufl.Identity(len(uh))
von_mises = ufl.sqrt(1.5 * ufl.inner(s, s))
V_vm = fem.functionspace(domain, (&apos;DG&apos;, 0))
expr = fem.Expression(von_mises, V_vm.element.interpolation_points())
vm = fem.Function(V_vm)
vm.name = &apos;von_mises&apos;
vm.interpolate(expr)
Step 7  Write the result file
XDMF plus HDF5 is the format to standardise on: it holds the mesh and multiple fields, supports time series, and every viewer reads it.
with io.XDMFFile(domain.comm, &apos;results.xdmf&apos;, &apos;w&apos;) as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
    xdmf.write_function(vm)
# Quick sanity print
tip = uh.x.array.reshape(-1, 3)[:, 2].min()
print(f&apos;Max downward tip deflection: {abs(tip)*1000:.3f} mm&apos;)
Step 8  Validate against theory
Replace self-weight with a tip point load and check against the closed form. This is the step that tells you whether you have understood the formulation or just copied it.
# Analytical cantilever with tip load P:  delta = P L^3 / (3 E I)
I_yy  = Wd * H**3 / 12.0
P     = 1000.0
delta = P * L**3 / (3.0 * E * I_yy)
print(f&apos;Analytical: {delta*1000:.3f} mm&apos;)
# Apply P as a traction over the end face:
#   traction magnitude = P / (Wd * H), applied on the x = L facets
# Expect agreement within ~3% for a slender beam.
# If the beam is stocky (L/H &lt; 10) shear deformation makes FEM
# legitimately softer than Euler-Bernoulli — that is not a bug.
Completion checklist — all must pass before the next project
☐  DOLFINx imports and prints a version number
☐  cantilever.py runs to completion with no PETSc errors
☐  Tip deflection under a tip load is within 3% of PL³/3EI for L/H ≥ 10
☐  results.xdmf opens in ParaView showing both displacement and von_mises
☐  Refining the mesh from [40,4,8] to [80,8,16] changes the answer by under 1% (convergence)
☐  You can explain, out loud, what the test function v is doing in the weak form
PROJECT 2   ·   3–4 weeks
CAD Import Pipeline
Real geometry from Fusion, SolidWorks and Revit into a solved result
Milestone:  Export a bracket from Fusion as STEP, run one command, get results.xdmf with the load applied to the face you tagged.
## The import routes
Source
Export
Route
Source
Export as
Path into the mesher
Fusion 360
STEP (AP214) or IGES
Gmsh OCC kernel reads STEP directly — no conversion
SolidWorks
STEP AP214 or Parasolid x_t
STEP is the reliable route; Parasolid needs a licensed kernel
Onshape / Inventor / CATIA
STEP
Same as above
Revit / ArchiCAD
IFC
ifcopenshell tessellates to triangles, then Gmsh remeshes
Blender / mesh tools
STL, OBJ
Surface mesh only — needs volume meshing and cleanup
Existing FE meshes
MSH, INP, MED
meshio converts; skip Gmsh entirely
## Step-by-step walkthrough
Step 1  Import STEP into Gmsh and mesh it
Gmsh embeds the OpenCASCADE kernel, so it reads STEP and IGES natively. You do not need pythonOCC for this.
# meshing/step_to_msh.py
import gmsh
def mesh_step(step_path, msh_path, size_max=5.0, size_min=1.0, order=1):
    gmsh.initialize()
    gmsh.option.setNumber(&apos;General.Terminal&apos;, 1)
    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()      # MUST call before meshing
    gmsh.option.setNumber(&apos;Mesh.CharacteristicLengthMax&apos;, size_max)
    gmsh.option.setNumber(&apos;Mesh.CharacteristicLengthMin&apos;, size_min)
    gmsh.option.setNumber(&apos;Mesh.ElementOrder&apos;, order)
    gmsh.option.setNumber(&apos;Mesh.Algorithm3D&apos;, 10)   # HXT, fast tets
    gmsh.model.mesh.generate(3)
    gmsh.write(msh_path)
    gmsh.finalize()
if __name__ == &apos;__main__&apos;:
    mesh_step(&apos;bracket.step&apos;, &apos;bracket.msh&apos;, size_max=4.0)
Step 2  Tag faces with physical groups — this is the critical step
Without physical groups you have a mesh but no way to say &apos;clamp this face&apos; or &apos;load that face&apos;. Tagging is what makes an imported CAD part usable for analysis.
Strategy: find surfaces by their bounding box or centre of mass, then assign an integer tag. Those tags come through into DOLFINx as facet markers.
def tag_faces(tol=1e-6):
    &quot;&quot;&quot;Assign physical groups to surfaces by geometric location.&quot;&quot;&quot;
    surfaces = gmsh.model.occ.getEntities(dim=2)
    volumes  = gmsh.model.occ.getEntities(dim=3)
    # Whole-model bounding box, used to identify extreme faces
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(-1, -1)
    fixed_faces, load_faces = [], []
    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if abs(com[0] - xmin) &lt; 1.0:      # faces at the low-x end
            fixed_faces.append(tag)
        elif abs(com[0] - xmax) &lt; 1.0:    # faces at the high-x end
            load_faces.append(tag)
    # Tag numbers are what you reference in DOLFINx
    gmsh.model.addPhysicalGroup(2, fixed_faces, tag=1, name=&apos;fixed&apos;)
    gmsh.model.addPhysicalGroup(2, load_faces,  tag=2, name=&apos;load&apos;)
    gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], tag=3, name=&apos;body&apos;)
Step 3  Read the tagged mesh into DOLFINx
gmshio returns the mesh plus cell and facet tag objects. The facet tags carry your physical group numbers.
from dolfinx.io import gmshio
from mpi4py import MPI
domain, cell_tags, facet_tags = gmshio.read_from_msh(
    &apos;bracket.msh&apos;, MPI.COMM_WORLD, rank=0, gdim=3
)
# Sanity: what tags actually came through?
print(&apos;facet tag values:&apos;, set(facet_tags.values))
# Expect {1, 2} matching the physical groups you created
Step 4  Apply BCs and loads using the tags, not coordinates
This is the payoff. You now reference geometry by name rather than by hard-coded coordinates, so the same solver script works on any tagged part.
FIXED_TAG, LOAD_TAG = 1, 2
V = fem.functionspace(domain, (&apos;Lagrange&apos;, 1, (3,)))
fdim = domain.topology.dim - 1
# Clamp everything on the &apos;fixed&apos; group
fixed_facets = facet_tags.find(FIXED_TAG)
bc = fem.dirichletbc(
    np.zeros(3, dtype=default_scalar_type),
    fem.locate_dofs_topological(V, fdim, fixed_facets), V,
)
# Traction only on the &apos;load&apos; group — note the subdomain_data argument
ds = ufl.Measure(&apos;ds&apos;, domain=domain, subdomain_data=facet_tags)
traction = fem.Constant(domain, default_scalar_type((0.0, 0.0, -1.0e6)))
L_form = (ufl.dot(f, v) * ufl.dx
          + ufl.dot(traction, v) * ds(LOAD_TAG))   # ← tagged surface only
Step 5  The IFC route for Revit models
IFC carries BIM semantics, not clean B-rep solids. ifcopenshell tessellates each element into triangles; you then either remesh the surface into a volume or work with the tessellation directly.
Be realistic: BIM geometry is frequently dirty — non-manifold edges, self-intersections, missing caps. Expect cleanup work.
# meshing/ifc_to_mesh.py
import ifcopenshell
import ifcopenshell.geom
import numpy as np
def extract_elements(ifc_path, ifc_type=&apos;IfcBeam&apos;):
    model = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    out = []
    for el in model.by_type(ifc_type):
        try:
            shape = ifcopenshell.geom.create_shape(settings, el)
        except RuntimeError:
            continue                       # some elements have no geometry
        verts = np.array(shape.geometry.verts).reshape(-1, 3)
        faces = np.array(shape.geometry.faces).reshape(-1, 3)
        out.append({
            &apos;name&apos;:  el.Name,
            &apos;guid&apos;:  el.GlobalId,
            &apos;verts&apos;: verts,
            &apos;faces&apos;: faces,
        })
    return out
# Then: write each element as STL, import into Gmsh,
# and use gmsh.model.mesh.createGeometry() + createTopology()
# to rebuild a volume from the surface triangulation.
Step 6  Wrap it into one command
Build a single entry point so the whole pipeline is reproducible and scriptable. This is what Project 3 will call.
# pipeline.py
import argparse
from meshing.step_to_msh import mesh_step
from solvers.elasticity  import solve_elasticity
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(&apos;cad&apos;)                     # bracket.step
    ap.add_argument(&apos;--size&apos;, type=float, default=4.0)
    ap.add_argument(&apos;--E&apos;,    type=float, default=210e9)
    ap.add_argument(&apos;--nu&apos;,   type=float, default=0.3)
    ap.add_argument(&apos;--load&apos;, type=float, default=1.0e6)
    ap.add_argument(&apos;--out&apos;,  default=&apos;results.xdmf&apos;)
    args = ap.parse_args()
    msh = args.cad.rsplit(&apos;.&apos;, 1)[0] + &apos;.msh&apos;
    mesh_step(args.cad, msh, size_max=args.size)
    solve_elasticity(msh, args.E, args.nu, args.load, args.out)
# Usage:  python pipeline.py bracket.step --size 3 --load 2e6
Problems you will actually hit
Mesh generation fails on imported STEP — usually slivers or tiny faces from the CAD tool. Fix: gmsh.option.setNumber(&apos;Geometry.OCCFixSmallFaces&apos;, 1) and set a sensible CharacteristicLengthMin so Gmsh does not try to resolve micron features.
Physical group tags do not appear in DOLFINx — you forgot gmsh.model.occ.synchronize() before addPhysicalGroup, or you tagged before meshing. Order matters: import, synchronize, tag, mesh, write.
Units. STEP files from Fusion are usually millimetres; your material constants are in Pascals and metres. Decide on one unit system at the mesh stage and scale there, not in three different places later.
Solve is very slow on a real part. A 500k-element tet mesh with LU factorisation will exhaust memory. Switch to an iterative solver: ksp_type &apos;cg&apos;, pc_type &apos;gamg&apos; — algebraic multigrid is the right preconditioner for elasticity.
Completion checklist — all must pass before the next project
☐  A STEP file exported from Fusion meshes without errors
☐  facet_tags.find(1) returns a non-empty array of facet indices
☐  The traction is applied only to the tagged face (check the deformed shape makes physical sense)
☐  One command takes a STEP file to results.xdmf end to end
☐  A 200k-element mesh solves in under two minutes with CG plus GAMG
☐  An IFC file yields at least one element&apos;s geometry as vertices and faces
PROJECT 3   ·   4–5 weeks
The Viewer — Mouse-Driven First
PyVista and Qt: clip, threshold, contour, warp, probe, field switching
Milestone:  A desktop app that opens any XDMF or VTU file and lets you explore the field with sliders and buttons. Useful on its own, with no gestures.
Why mouse controls first
Every gesture in Project 4 is a remapping of a control built here. If the clip plane cannot be moved by a slider, it cannot be moved by a hand either.
Building the controls first means Project 4 is input remapping — a two-week job — rather than building a visualisation system and an input system simultaneously.
It also means that if you stop after this project, you still have a working, genuinely useful tool.
## Step-by-step walkthrough
Step 1  Load results and inspect what fields exist
PyVista reads XDMF, VTU, VTK, and via meshio almost anything else. Always start by listing the available arrays — field naming differs between solvers.
# viewer/loader.py
import pyvista as pv
def load_result(path):
    &quot;&quot;&quot;Load a result file and return a pyvista dataset.&quot;&quot;&quot;
    data = pv.read(path)
    # XDMF and multiblock files return a MultiBlock; flatten it
    if isinstance(data, pv.MultiBlock):
        data = data.combine()
    print(&apos;point arrays:&apos;, list(data.point_data.keys()))
    print(&apos;cell arrays: &apos;, list(data.cell_data.keys()))
    print(&apos;n_points:&apos;, data.n_points, &apos; n_cells:&apos;, data.n_cells)
    return data
def available_scalars(data):
    &quot;&quot;&quot;All scalar fields the user can colour by.&quot;&quot;&quot;
    names = []
    for k, arr in data.point_data.items():
        names.append(k if arr.ndim == 1 else f&apos;{k} (magnitude)&apos;)
    for k in data.cell_data.keys():
        names.append(k)
    return names
Step 2  Build the Qt application shell
pyvistaqt embeds a full VTK render window inside a Qt widget. Left side is the 3D view; right side is the control dock. In Project 4 the webcam panel goes into that same dock.
# viewer/app.py
from PySide6 import QtWidgets, QtCore
from pyvistaqt import QtInteractor
import pyvista as pv
class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(&apos;Archimedes Viewer&apos;)
        self.resize(1400, 900)
        central = QtWidgets.QWidget()
        layout  = QtWidgets.QHBoxLayout(central)
        # 3D view
        self.plotter = QtInteractor(central)
        self.plotter.set_background(&apos;#0F172A&apos;)
        layout.addWidget(self.plotter.interactor, stretch=4)
        # Control dock
        self.controls = ControlPanel(self)
        layout.addWidget(self.controls, stretch=1)
        self.setCentralWidget(central)
        self.data   = None      # full dataset
        self.actor  = None      # main mesh actor
        self.state  = ViewState()
    def open_file(self, path):
        from .loader import load_result
        self.data = load_result(path)
        self.controls.populate_fields(available_scalars(self.data))
        self.refresh()
Step 3  Hold all view parameters in one state object
Every control writes into this object and calls refresh(). Gestures in Project 4 will write into the exact same object. This single decision is what makes the gesture layer a two-week job instead of a rewrite.
# viewer/state.py
from dataclasses import dataclass, field
@dataclass
class ViewState:
    scalar:        str   = &apos;von_mises&apos;
    cmap:          str   = &apos;viridis&apos;
    clim:          tuple = None          # None = auto range
    clip_enabled:  bool  = False
    clip_origin:   tuple = (0.0, 0.0, 0.0)
    clip_normal:   tuple = (1.0, 0.0, 0.0)
    threshold_on:  bool  = False
    threshold_lo:  float = 0.0
    threshold_hi:  float = 1.0
    contour_on:    bool  = False
    contour_vals:  list  = field(default_factory=list)
    warp_on:       bool  = False
    warp_factor:   float = 1.0
    warp_vector:   str   = &apos;displacement&apos;
    show_edges:    bool  = False
    opacity:       float = 1.0
Step 4  Clip planes — use GPU clipping, not filtering
The obvious approach is mesh.clip(), which builds a new dataset every time the plane moves. On a 500k-element mesh that is far too slow for continuous dragging.
The right approach is a VTK clipping plane attached to the mapper. The full mesh stays on the GPU and clipping happens per-frame in hardware. Moving the plane becomes free.
# viewer/filters.py
import vtk
class ClipController:
    &quot;&quot;&quot;GPU-side clipping — no re-filtering when the plane moves.&quot;&quot;&quot;
    def __init__(self, actor):
        self.plane = vtk.vtkPlane()
        self.plane.SetOrigin(0.0, 0.0, 0.0)
        self.plane.SetNormal(1.0, 0.0, 0.0)
        self.mapper  = actor.GetMapper()
        self.enabled = False
    def enable(self):
        if not self.enabled:
            self.mapper.AddClippingPlane(self.plane)
            self.enabled = True
    def disable(self):
        if self.enabled:
            self.mapper.RemoveClippingPlane(self.plane)
            self.enabled = False
    def set_origin(self, x, y, z):
        self.plane.SetOrigin(x, y, z)      # cheap — just a uniform update
    def set_normal(self, nx, ny, nz):
        self.plane.SetNormal(nx, ny, nz)
# In the window:
#   self.clip = ClipController(self.actor)
#   ...on slider move:
#   self.clip.set_origin(x, 0, 0); self.plotter.render()
Keep mesh.clip() for the one case where GPU clipping is not enough: when you want to see the capped cross-section face with field values on it rather than a hollow cut. For that, filter once when the plane stops moving, not on every frame.
Step 5  Threshold, contour and warp
These do require re-filtering, but they are used discretely rather than dragged continuously. Rebuild the actor when they change.
def build_display_mesh(data, state):
    &quot;&quot;&quot;Apply the non-GPU filters in the correct order.&quot;&quot;&quot;
    m = data
    # Warp first — everything downstream should see deformed coords
    if state.warp_on and state.warp_vector in m.point_data:
        m = m.warp_by_vector(state.warp_vector, factor=state.warp_factor)
    if state.threshold_on:
        m = m.threshold(
            value=(state.threshold_lo, state.threshold_hi),
            scalars=state.scalar,
        )
    if state.contour_on and state.contour_vals:
        m = m.contour(isosurfaces=state.contour_vals, scalars=state.scalar)
    return m
def refresh(self):
    if self.data is None:
        return
    display = build_display_mesh(self.data, self.state)
    if self.actor is not None:
        self.plotter.remove_actor(self.actor)
    self.actor = self.plotter.add_mesh(
        display,
        scalars=self.state.scalar,
        cmap=self.state.cmap,
        clim=self.state.clim,
        show_edges=self.state.show_edges,
        opacity=self.state.opacity,
        scalar_bar_args={&apos;title&apos;: self.state.scalar, &apos;color&apos;: &apos;white&apos;},
    )
    self.clip = ClipController(self.actor)
    if self.state.clip_enabled:
        self.clip.enable()
        self.clip.set_origin(*self.state.clip_origin)
        self.clip.set_normal(*self.state.clip_normal)
    self.plotter.render()
Step 6  Point probe — report the field value under a screen position
Given a screen coordinate, pick the 3D point and read the field there. This becomes the gesture &apos;point at a spot and hold&apos; interaction in Project 4.
def probe_at_screen(self, sx, sy):
    &quot;&quot;&quot;Return (world_point, value) for a screen pixel, or None.&quot;&quot;&quot;
    picker = vtk.vtkPointPicker()
    picker.SetTolerance(0.005)
    ok = picker.Pick(sx, sy, 0, self.plotter.renderer)
    if not ok:
        return None
    pid = picker.GetPointId()
    if pid &lt; 0:
        return None
    world = picker.GetPickPosition()
    arr   = self.data.point_data.get(self.state.scalar)
    if arr is None or pid &gt;= len(arr):
        return None
    return world, float(arr[pid])
def show_probe_label(self, world, value):
    self.plotter.add_point_labels(
        [world], [f&apos;{value:.3g}&apos;],
        name=&apos;probe&apos;, font_size=14,
        point_color=&apos;yellow&apos;, point_size=12,
    )
Step 7  Camera control as explicit methods
Do not rely only on VTK&apos;s built-in mouse interactor. Expose orbit, zoom and pan as methods so that Project 4 can call them directly from gesture deltas.
def orbit(self, d_azimuth, d_elevation):
    cam = self.plotter.camera
    cam.azimuth(d_azimuth)
    cam.elevation(max(-89.0, min(89.0, d_elevation)))
    self.plotter.render()
def zoom(self, factor):
    self.plotter.camera.zoom(factor)
    self.plotter.render()
def pan(self, dx, dy):
    cam = self.plotter.camera
    pos, foc = list(cam.position), list(cam.focal_point)
    for i in range(3):
        pos[i] += dx; foc[i] += dy
    cam.position, cam.focal_point = pos, foc
    self.plotter.render()
def reset_view(self):
    self.plotter.reset_camera()
    self.plotter.view_isometric()
Completion checklist — all must pass before the next project
☐  The app opens any XDMF produced in Projects 1 and 2
☐  The field dropdown lists every scalar in the file and switching recolours the mesh
☐  The clip slider sweeps the plane smoothly at 30+ fps on a 200k-element mesh
☐  Threshold, contour and warp each produce visibly correct geometry
☐  Clicking a point shows a label with the correct field value
☐  Orbit, zoom, pan and reset all work as method calls, not just mouse drags
☐  Every control writes to ViewState and nothing else holds view parameters
PROJECT 4   ·   4–5 weeks
The Gesture Layer
MediaPipe in a background thread, mapped onto the Project 3 controls
Milestone:  Hands-free exploration: orbit the model, sweep a clip plane through it, probe a hotspot and read the value, all without touching the mouse.
## Gesture vocabulary for post-processing
This is a different vocabulary from an authoring interface. Every gesture here is continuous and approximate — there is nothing that needs a precise numeric value, which is exactly why the modality fits.
Gesture
Action
Mapped to
Gesture
Action
Mapped to
Open palm, move
Orbit camera
orbit(d_az, d_el) from palm centre delta
Two hands, spread apart
Zoom in
zoom(1 + k·Δspan)
Two hands, move together
Pan
pan(dx, dy) from midpoint delta
Index point, drag
Sweep clip plane along its normal
clip.set_origin() from fingertip depth
Flat hand, tilt
Set clip plane normal
clip.set_normal() from palm normal vector
Pinch, drag vertically
Threshold lower bound
state.threshold_lo, then refresh()
Two-finger pinch, rotate
Warp scale factor
state.warp_factor
Point and hold 1.0 s
Probe the point under the finger
probe_at_screen() + label
Swipe left / right
Cycle scalar field
next / previous in available_scalars()
Fist
Reset view
reset_view()
Open palm shake
Clear all filters
reset ViewState to defaults
Design rules that make this usable rather than annoying
Split by hand role. Non-dominant hand controls the camera; dominant hand controls filters. Without this split every camera movement drags the clip plane with it.
Require an explicit engage gesture. Nothing should respond until the user makes a deliberate &apos;engage&apos; pose — otherwise scratching your nose reorients the model.
Use relative deltas, never absolute positions. Map the change in hand position to the change in camera angle. Absolute mapping means the model jumps whenever tracking is briefly lost.
Smooth everything. Raw landmarks jitter by 3 to 5 pixels per frame. An exponential moving average at alpha 0.2 costs about 150 ms of lag and removes essentially all of it.
Add a deadband. Ignore movements below a threshold so the view holds still when the user is holding still.
## Step-by-step walkthrough
Step 1  Run MediaPipe in a QThread and emit Qt signals
VTK rendering must happen on the main thread. The CV loop runs in a worker thread and communicates only through Qt signals, which Qt delivers on the receiving object&apos;s thread automatically.
# gesture/worker.py
from PySide6 import QtCore
import cv2, numpy as np
import mediapipe as mp
class HandWorker(QtCore.QThread):
    camera_delta  = QtCore.Signal(float, float)   # d_azimuth, d_elevation
    zoom_delta    = QtCore.Signal(float)
    clip_position = QtCore.Signal(float)          # 0..1 along the axis
    probe_request = QtCore.Signal(float, float)   # screen x, y
    mode_changed  = QtCore.Signal(str)
    frame_ready   = QtCore.Signal(np.ndarray)
    def __init__(self, camera_index=0):
        super().__init__()
        self._running = False
        self.camera_index = camera_index
    def run(self):
        hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        cap = cv2.VideoCapture(self.camera_index)
        self._running = True
        prev = {}
        while self._running and cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                self._dispatch(res, prev)
            self.frame_ready.emit(frame)
        cap.release()
        hands.close()
    def stop(self):
        self._running = False
        self.wait()
Step 2  Classify pose and emit deltas
Keep the classifier simple and rule-based. Landmark 0 is the wrist, 4 is the thumb tip, 8 is the index tip, 5 and 17 are the index and pinky knuckles.
WRIST, THUMB, INDEX, MIDDLE, RING, PINKY = 0, 4, 8, 12, 16, 20
INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP = 6, 10, 14, 18
INDEX_MCP, PINKY_MCP = 5, 17
def extended(lm, tip, pip):
    return lm[tip].y &lt; lm[pip].y
def classify(lm):
    i = extended(lm, INDEX,  INDEX_PIP)
    m = extended(lm, MIDDLE, MIDDLE_PIP)
    r = extended(lm, RING,   RING_PIP)
    p = extended(lm, PINKY,  PINKY_PIP)
    pinch = np.hypot(lm[THUMB].x - lm[INDEX].x,
                     lm[THUMB].y - lm[INDEX].y)
    if pinch &lt; 0.05:                  return &apos;PINCH&apos;
    if i and not (m or r or p):       return &apos;POINT&apos;
    if i and m and r and p:           return &apos;OPEN_PALM&apos;
    if not (i or m or r or p):        return &apos;FIST&apos;
    return &apos;UNKNOWN&apos;
def palm_centre(lm):
    return np.array([lm[WRIST].x, lm[WRIST].y])
Step 3  Smooth, deadband, and convert to deltas
Never emit a raw landmark position. Smooth it, difference it against the previous frame, and drop anything below the deadband.
# gesture/smoothing.py
import numpy as np
class EMA:
    def __init__(self, alpha=0.2):
        self.alpha, self.value = alpha, None
    def update(self, v):
        v = np.asarray(v, dtype=float)
        self.value = v if self.value is None else (
            self.alpha * v + (1.0 - self.alpha) * self.value)
        return self.value
class DeltaTracker:
    &quot;&quot;&quot;Smoothed frame-to-frame delta with a deadband.&quot;&quot;&quot;
    def __init__(self, alpha=0.2, deadband=0.004):
        self.ema      = EMA(alpha)
        self.prev     = None
        self.deadband = deadband
    def update(self, v):
        cur = self.ema.update(v)
        if self.prev is None:
            self.prev = cur
            return np.zeros_like(cur)
        d = cur - self.prev
        self.prev = cur
        d[np.abs(d) &lt; self.deadband] = 0.0
        return d
    def reset(self):
        self.prev = None      # call on gesture change to avoid a jump
Step 4  Dispatch to the right control based on hand role and pose
Left hand drives the camera, right hand drives filters. A gesture change resets the delta tracker so the view does not jump when the user switches pose.
def _dispatch(self, res, prev):
    for lms, handed in zip(res.multi_hand_landmarks,
                           res.multi_handedness):
        lm   = lms.landmark
        side = handed.classification[0].label      # &apos;Left&apos; or &apos;Right&apos;
        pose = classify(lm)
        key = (side, pose)
        if prev.get(side) != pose:
            self.trackers[side].reset()            # pose changed — no jump
            prev[side] = pose
            self.mode_changed.emit(f&apos;{side}: {pose}&apos;)
        centre = palm_centre(lm)
        d = self.trackers[side].update(centre)
        # LEFT HAND — camera
        if side == &apos;Left&apos; and pose == &apos;OPEN_PALM&apos;:
            self.camera_delta.emit(float(d[0] * 180.0),
                                   float(-d[1] * 180.0))
        # RIGHT HAND — filters
        elif side == &apos;Right&apos; and pose == &apos;POINT&apos;:
            self.clip_position.emit(float(lm[INDEX].x))
        elif side == &apos;Right&apos; and pose == &apos;FIST&apos;:
            self.mode_changed.emit(&apos;RESET&apos;)
Step 5  Connect the worker to the viewer
This is the entire integration. Because Project 3 exposed everything as methods on ViewState and the window, wiring gestures is a handful of signal connections.
# In ViewerWindow.start_gestures()
from gesture.worker import HandWorker
def start_gestures(self):
    self.worker = HandWorker(camera_index=0)
    self.worker.camera_delta.connect(self.orbit)
    self.worker.zoom_delta.connect(self.zoom)
    self.worker.clip_position.connect(self.on_gesture_clip)
    self.worker.probe_request.connect(self.on_gesture_probe)
    self.worker.frame_ready.connect(self.hud.show_frame)
    self.worker.mode_changed.connect(self.hud.show_mode)
    self.worker.start()
def on_gesture_clip(self, t):
    &quot;&quot;&quot;t in 0..1 → position along the model&apos;s x extent.&quot;&quot;&quot;
    xmin, xmax = self.data.bounds[0], self.data.bounds[1]
    x = xmin + t * (xmax - xmin)
    self.state.clip_enabled = True
    self.state.clip_origin  = (x, 0.0, 0.0)
    self.clip.enable()
    self.clip.set_origin(x, 0.0, 0.0)
    self.plotter.render()          # cheap — GPU clipping, no refilter
def closeEvent(self, event):
    if hasattr(self, &apos;worker&apos;):
        self.worker.stop()         # never leave the thread running
    event.accept()
Step 6  Add the HUD
A small dock showing the webcam feed with landmarks drawn, the current pose per hand, and a dwell progress bar for the probe gesture. Without visible feedback the interface feels broken even when it works.
# gesture/hud.py
from PySide6 import QtWidgets, QtGui, QtCore
import cv2
class GestureHUD(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        self.lbl_mode = QtWidgets.QLabel(&apos;idle&apos;)
        self.lbl_mode.setStyleSheet(&apos;font-family:monospace;font-size:13px;&apos;)
        self.lbl_cam  = QtWidgets.QLabel()
        self.lbl_cam.setFixedSize(280, 210)
        self.bar = QtWidgets.QProgressBar(); self.bar.setRange(0, 100)
        lay.addWidget(self.lbl_mode); lay.addWidget(self.lbl_cam)
        lay.addWidget(self.bar); lay.addStretch()
    def show_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, w * c, QtGui.QImage.Format_RGB888)
        self.lbl_cam.setPixmap(
            QtGui.QPixmap.fromImage(img).scaled(
                self.lbl_cam.size(), QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation))
    def show_mode(self, text):
        self.lbl_mode.setText(text)
Completion checklist — all must pass before the next project
☐  MediaPipe runs at 25+ fps without dropping the viewer below 30 fps
☐  Left-hand open palm orbits smoothly with no visible jitter
☐  Right-hand point sweeps the clip plane continuously through the model
☐  Switching pose does not cause the camera or clip plane to jump
☐  Point-and-hold for one second probes and labels the correct value
☐  Fist resets the view; palm shake clears all filters
☐  Closing the window stops the thread cleanly, with no zombie process
☐  A two-minute recorded demo: orbit, clip sweep, field switch, probe — hands only
PROJECT 5   ·   6–8 weeks
Depth and Reach
Time series, modal animation, multi-solver ingest, spatial mode
Milestone:  The viewer opens an Abaqus or Ansys export as readily as your own; mode shapes animate; a load history scrubs with a hand sweep.
## 5A — Time and load step scrubbing
XDMF supports time series natively. Write each step with a time value, then scrub through them. A horizontal hand sweep maps to the step index — one of the most natural gesture mappings in the whole interface.
# Writing a time series from DOLFINx
with io.XDMFFile(domain.comm, &apos;transient.xdmf&apos;, &apos;w&apos;) as xdmf:
    xdmf.write_mesh(domain)
    for step, t in enumerate(time_values):
        # ... solve for this step, update uh ...
        xdmf.write_function(uh, t)
# Reading and scrubbing in the viewer
reader = pv.get_reader(&apos;transient.xdmf&apos;)
print(&apos;time values:&apos;, reader.time_values)
def set_time_step(self, index):
    index = max(0, min(index, len(self.reader.time_values) - 1))
    self.reader.set_active_time_point(index)
    self.data = self.reader.read().combine()
    self.refresh()
## 5B — Modal analysis and mode shape animation
Modal analysis is a generalised eigenvalue problem, K φ = ω² M φ. SLEPc ships with the DOLFINx conda package and solves it directly.
# solvers/modal.py
from slepc4py import SLEPc
from dolfinx.fem.petsc import assemble_matrix
import numpy as np
def solve_modes(V, a_form, m_form, bcs, n_modes=6):
    K = assemble_matrix(fem.form(a_form), bcs=bcs); K.assemble()
    M = assemble_matrix(fem.form(m_form), bcs=bcs); M.assemble()
    eps = SLEPc.EPS().create(V.mesh.comm)
    eps.setOperators(K, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)   # generalised Hermitian
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
    eps.setDimensions(nev=n_modes)
    eps.setFromOptions()
    eps.solve()
    modes = []
    for i in range(eps.getConverged()):
        lam = eps.getEigenvalue(i).real
        vec = K.createVecRight()
        eps.getEigenvector(i, vec)
        freq = np.sqrt(abs(lam)) / (2.0 * np.pi)     # Hz
        modes.append((freq, vec.array.copy()))
    return modes
# Animation in the viewer: scale the mode shape by sin(2*pi*f*t)
# and re-warp each frame with a QTimer at 30 fps.
## 5C — Multi-solver ingest — the feature that makes this general
Once the viewer reads other vendors&apos; results, it stops being a companion to your solver and becomes a tool anyone can use. This is the highest-leverage work in Project 5.
Solver
Export
Reader
Source
Export route
Reader
Abaqus
Open the ODB, File → Export → VTK, or use odbAccess scripting
pv.read() on the VTU
Ansys Mechanical
Export result as VTK, or read RST with the pyansys / ansys-dpf toolchain
pv.read(), or ansys-dpf-core
COMSOL
Export → Data → VTU
pv.read()
CalculiX
FRD output → ccx2paraview → VTU
pv.read()
OpenFOAM
Native, or foamToVTK
pv.OpenFOAMReader
LS-DYNA
d3plot via lasso-python
convert to VTU with meshio
# viewer/ingest.py — normalise field names across vendors
FIELD_ALIASES = {
    &apos;von_mises&apos;:   [&apos;von_mises&apos;, &apos;S_Mises&apos;, &apos;SEQV&apos;, &apos;vonMises&apos;,
                    &apos;sigma_vm&apos;, &apos;Stress:Equivalent (von-Mises)&apos;],
    &apos;displacement&apos;:[&apos;displacement&apos;, &apos;U&apos;, &apos;Displacement&apos;, &apos;disp&apos;],
    &apos;temperature&apos;: [&apos;temperature&apos;, &apos;NT11&apos;, &apos;TEMP&apos;, &apos;T&apos;],
}
def normalise_fields(dataset):
    &quot;&quot;&quot;Rename vendor-specific arrays to canonical names.&quot;&quot;&quot;
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias in dataset.point_data and alias != canonical:
                dataset.point_data[canonical] = dataset.point_data[alias]
                break
            if alias in dataset.cell_data and alias != canonical:
                dataset.cell_data[canonical] = dataset.cell_data[alias]
                break
    return dataset
## 5D — Spatial and AR mode
Two credible routes, depending on whether you want a browser or a headset.
Route
Approach
Browser (WebXR)
Export the display mesh to glTF with vertex colours baked from the colour map, load it in Three.js with WebXR enabled. Works on any phone with a browser. Lowest effort, widest reach.
Headset (Unity or Unreal)
Export glTF or FBX, build a Quest or Vision Pro app. Native hand tracking is far better than webcam MediaPipe, and the interaction genuinely improves. Much more work, much better result.
Desktop depth camera
An Intel RealSense gives true 3D hand positions and removes the depth ambiguity that limits webcam gestures. A middle path that improves the existing app rather than replacing it.
# Export the current view for AR
def export_gltf(self, path):
    display = build_display_mesh(self.data, self.state)
    display = display.copy()
    # Bake the colour map into vertex colours
    scalars = display.point_data[self.state.scalar]
    lo, hi  = (self.state.clim if self.state.clim
               else (scalars.min(), scalars.max()))
    import matplotlib.cm as cm
    norm  = (scalars - lo) / max(hi - lo, 1e-12)
    rgba  = (cm.get_cmap(self.state.cmap)(norm) * 255).astype(&apos;uint8&apos;)
    display.point_data[&apos;RGBA&apos;] = rgba
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(display, scalars=&apos;RGBA&apos;, rgba=True)
    pl.export_gltf(path)
Completion checklist — all must pass before the next project
☐  A transient XDMF scrubs through all time steps with a hand sweep
☐  The first six natural frequencies match an analytical or commercial reference within 5%
☐  Mode shapes animate smoothly at 30 fps
☐  The viewer opens a VTU exported from at least one commercial solver
☐  Field name normalisation maps that solver&apos;s stress array to von_mises automatically
☐  glTF export opens in a browser WebXR viewer with correct colours
# Timeline and Study Plan
Weeks
Work
Study
Weeks
Project work
Study alongside
1–2
Install DOLFINx via Docker, run the tutorial demos unmodified
Dokken DOLFINx tutorial ch. 1–2; weak forms
3–4
Write cantilever.py from scratch, validate against PL³/3EI
Dokken ch. 3; linear elasticity theory
5–6
Gmsh STEP import, meshing options, mesh quality
Gmsh Python API tutorials t1–t20
7–8
Physical group tagging, tagged BCs, one-command pipeline
Dokken ch. on subdomains and boundary conditions
9–11
PyVista basics, Qt shell, ViewState, field switching
PyVista examples gallery; PySide6 layouts
12–13
GPU clip planes, threshold, contour, warp
VTK mapper and clipping plane documentation
14
Point probe, camera methods, polish
VTK picking documentation
15–16
MediaPipe worker thread, classifier, smoothing
MediaPipe Hands guide; Qt threading docs
17–18
Signal wiring, HUD, tuning deadbands and gains
Nothing new — this is iteration and feel
19–20
Record the demo, write it up, publish the repo
—
21–26
Project 5 features, chosen by what you want to show
SLEPc docs; glTF and WebXR if going spatial
## Resources that actually matter for this stack
Resource
Why
DOLFINx tutorial — Jørgen Dokken
The only FEniCSx resource you need to start. Current API, well explained. Older FEniCS tutorials will mislead you — ignore them.
FEniCSx demo collection
Ships with the installation. The elasticity, hyperelasticity and eigenvalue demos are directly relevant.
Gmsh Python API tutorials (t1–t20)
In the Gmsh source tree. t16 onwards covers OCC and STEP import specifically.
PyVista examples gallery
Extremely good. Almost every filter you need has a runnable example.
VTK textbook and Doxygen docs
For when PyVista does not expose what you need — clipping planes, pickers, custom mappers.
MediaPipe Hands solution guide
Landmark indices, confidence tuning, chirality handling.
SOFA Framework
Real-time FEM used in soft robotics and surgical simulation. Worth studying if the robotics direction firms up.
Bathe — Finite Element Procedures
Free PDF from MIT. The reference for when you need to know why something behaves the way it does.
The ordering discipline that makes this work
Project 3 before Project 4 is the load-bearing decision. Build every control as a method that writes to ViewState, and the gesture layer becomes signal wiring rather than a system rewrite.
Each project ends with something usable. After Project 2 you have a CAD-to-FEA pipeline. After Project 3 you have a viewer worth using. The gesture layer is added value, not the foundation.
If you stall, stall late. A finished Project 3 is a real contribution. A half-finished Project 1 is nothing.
Archimedes FEniCSx Edition  ·  Five projects  ·  CAD in, hands on the result