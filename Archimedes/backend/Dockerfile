# FEniCSx ships as a conda/apt stack that is painful to install natively on
# Windows and macOS, so the solver core runs in the upstream DOLFINx image.
# The UI stays on the host and talks to this container over ws://127.0.0.1:8791.
FROM dolfinx/dolfinx:stable

LABEL org.opencontainers.image.title="Archimedes Workbench solver core" \
      org.opencontainers.image.description="DOLFINx/PETSc linear elasticity + SLEPc modal, WebSocket bridge on 8791"

WORKDIR /opt/archimedes

# gmsh is only needed for the curvature-resolved tetrahedral path; the voxel
# path (the one the UI renders) needs nothing beyond what the base image has.
RUN pip install --no-cache-dir --upgrade "gmsh>=4.11" || \
    echo "gmsh unavailable - the 'gmsh' mesher will be disabled, voxel still works"

COPY pyproject.toml README.md ./
COPY archimedes ./archimedes
RUN pip install --no-cache-dir -e .

EXPOSE 8791

# 0.0.0.0 so the published port reaches the host; the UI dials 127.0.0.1:8791.
ENTRYPOINT ["archimedes"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8791"]
