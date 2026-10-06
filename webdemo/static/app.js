// Archimedes gesture FE workbench - browser side.
// Sensor (webcam + MediaPipe WASM), renderer (three.js) and panels. Gesture logic,
// the model and the solver live in Python; this file draws and resolves screen
// positions onto the member.
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);
const HAND_EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const STOPS = [[0.13, 0.25, 0.85], [0.0, 0.75, 0.95], [0.2, 0.8, 0.3], [0.98, 0.85, 0.1], [0.9, 0.15, 0.1]];
function cmap(t) {
  t = Math.min(1, Math.max(0, t)) * (STOPS.length - 1);
  const i = Math.min(Math.floor(t), STOPS.length - 2), f = t - i;
  return [0, 1, 2].map((k) => STOPS[i][k] + (STOPS[i + 1][k] - STOPS[i][k]) * f);
}
const b64 = (s, T = Float32Array) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };
const fmt = (v, d = 3) => { if (v === undefined || v === null || !isFinite(v)) return "—"; const a = Math.abs(v);
  return a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(2) : (+v.toPrecision(d)).toLocaleString(); };
$("legend-bar").style.background = `linear-gradient(0deg, ${STOPS.map((c) => `rgb(${c.map((v) => v * 255).join(",")})`).join(",")})`;

// ---------------------------------------------------------------------------
// three.js scene: z is up, scene units are metres (model data is in mm)
const host = $("three");
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x0f1216);
host.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 200);
camera.up.set(0, 0, 1);
scene.add(new THREE.HemisphereLight(0xffffff, 0x3a4350, 0.85));
const sun = new THREE.DirectionalLight(0xffffff, 0.8); sun.position.set(4, -6, 9); scene.add(sun);
const world = new THREE.Group(); scene.add(world);          // member + glyphs
const glyphs = new THREE.Group(); scene.add(glyphs);
let ground = null;

const S = {                                                 // app state
  info: null, view: null, model: null, phase: "results", prompt: null, hud: null, res: null,
  version: -1, solved: false, tool: null, toolT0: null, mouseXY: null, cursor: null, busy: false,
};
const overlay = $("overlay"), octx = overlay.getContext("2d");

function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  overlay.width = w; overlay.height = h;
}
window.addEventListener("resize", resize);

// ---------------------------------------------------------------------------
// member geometry helpers (mm)
function geo(m = S.model) {
  const mem = m.member, s = mem.section, L = mem.length, circ = s.kind === "circle";
  const wa = circ ? s.d : s.b, wb = circ ? s.d : s.h, hor = mem.orientation === "horizontal";
  return {
    L, hor, circ, wa, wb,
    center: hor ? [L / 2, 0, 0] : [0, 0, L / 2],
    half: hor ? [L / 2, wa / 2, wb / 2] : [wa / 2, wb / 2, L / 2],
    at: (t) => (hor ? [t * L, 0, 0] : [0, 0, t * L]),
    span: Math.max(L, wa, wb),
  };
}
const v3 = (p) => new THREE.Vector3(p[0] / 1000, p[1] / 1000, p[2] / 1000);
const AX = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };

// ---------------------------------------------------------------------------
// FE result mesh: element faces whose neighbour is missing or cut away
let fe = null;   // decoded results
let feMesh = null, feLines = null, plainMesh = null, outline = null;
let vNode = null;    // node index of each vertex of the visible faces
const LOCAL = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
const FACES = [[[-1, 0, 0], [0, 4, 6, 2]], [[1, 0, 0], [1, 3, 7, 5]], [[0, -1, 0], [0, 1, 5, 4]],
  [[0, 1, 0], [2, 6, 7, 3]], [[0, 0, -1], [0, 2, 3, 1]], [[0, 0, 1], [4, 5, 7, 6]]];

function decodeResults(r) {
  const nodes = b64(r.nodes), elems = b64(r.elems, Int32Array), cells = b64(r.cells, Int32Array);
  const ne = elems.length / 8, nn = nodes.length / 3, [gx, gy, gz] = r.grid;
  const occ = new Int32Array(gx * gy * gz).fill(-1);
  for (let e = 0; e < ne; e++) occ[(cells[3 * e] * gy + cells[3 * e + 1]) * gz + cells[3 * e + 2]] = e;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nn; i++) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], nodes[3 * i + k]); hi[k] = Math.max(hi[k], nodes[3 * i + k]); }
  const fields = {};
  for (const [k, f] of Object.entries(r.fields)) fields[k] = { ...f, arr: b64(f.data), vec: f.shape ? b64(f.shape) : null };
  let umax = 0; const disp = b64(r.disp);
  for (let i = 0; i < nn; i++) umax = Math.max(umax, Math.hypot(disp[3 * i], disp[3 * i + 1], disp[3 * i + 2]));
  return { ...r, nodes, elems, cells, ne, nn, occ, disp, umax, fields, lo, hi,
    center: lo.map((v, k) => (v + hi[k]) / 2), half: Math.max(...hi.map((v, k) => v - lo[k])) / 2 };
}

function cutPlane() {
  const v = S.view;
  if (!v || !v.section_on || !fe) return null;
  const n = v.plane_normal, o = fe.center.map((c, k) => c + v.plane_origin[k] * fe.half);
  return { n, o };
}

// The cut is a real plane: the GPU clips the surface mesh smoothly, and the cap is the
// exact slice of every element the plane crosses (no voxel staircase).
renderer.localClippingEnabled = true;
const clipPlane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 1e6);   // keeps everything until a cut is set
const HEX_EDGES = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
let capMesh = null, capLines = null, capVals = null;

function rebuildFE() {
  disposeObj(feMesh); disposeObj(feLines); feMesh = feLines = null;
  if (!fe) return;
  const { elems, cells, ne, occ, grid: [gx, gy, gz] } = fe;
  const verts = [], lines = [];
  for (let e = 0; e < ne; e++) {                 // exterior faces only; the cap closes the cut
    const ci = cells[3 * e], cj = cells[3 * e + 1], ck = cells[3 * e + 2];
    for (const [d, q] of FACES) {
      const a = ci + d[0], b = cj + d[1], c = ck + d[2];
      if (a >= 0 && b >= 0 && c >= 0 && a < gx && b < gy && c < gz && occ[(a * gy + b) * gz + c] >= 0) continue;
      const n = q.map((l) => elems[8 * e + l]);
      verts.push(n[0], n[1], n[2], n[0], n[2], n[3]);
      lines.push(n[0], n[1], n[1], n[2], n[2], n[3], n[3], n[0]);
    }
  }
  vNode = Int32Array.from(verts);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(vNode.length * 3), 3));
  g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(vNode.length * 3), 3));
  feMesh = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide,
    clippingPlanes: [clipPlane], polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
  feMesh.userData.lineNodes = Int32Array.from(lines);
  const lg = new THREE.BufferGeometry();
  lg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(lines.length * 3), 3));
  feLines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x0b0d10, transparent: true, opacity: 0.35,
    clippingPlanes: [clipPlane] }));
  world.add(feMesh, feLines);
  updateCut();
  updateColors(); updatePositions(performance.now());
}

function updateCut() {
  const P = cutPlane();
  if (!P) { clipPlane.set(new THREE.Vector3(1, 0, 0), 1e6); disposeObj(capMesh); disposeObj(capLines); capMesh = capLines = null; return; }
  const n = new THREE.Vector3(...P.n).normalize();
  clipPlane.set(n.clone().negate(), n.dot(v3(P.o)));       // remove the side the palm faces
}

function updateCap(vec, s) {
  disposeObj(capMesh); disposeObj(capLines); capMesh = capLines = null;
  const P = cutPlane(), F = currentField();
  if (!P || !fe || !F) return;
  const n = P.n, o = P.o, N = fe.nodes, E = fe.elems, len = Math.hypot(...n) || 1;
  const nx = n[0] / len, ny = n[1] / len, nz = n[2] / len;
  // in-plane axes for ordering each slice polygon
  const ux = Math.abs(nx) < 0.9 ? [0, -nz, ny] : [-nz, 0, nx], ul = Math.hypot(...ux);
  const u = ux.map((x) => x / ul), w = [ny * u[2] - nz * u[1], nz * u[0] - nx * u[2], nx * u[1] - ny * u[0]];
  const pos = [], col = [], vals = [], lin = [], lo = F.min, span = (F.max - F.min) || 1;
  const P8 = new Float64Array(24), D8 = new Float64Array(8);
  for (let e = 0; e < fe.ne; e++) {
    let neg = false, posi = false;
    for (let k = 0; k < 8; k++) {
      const m = E[8 * e + k];
      for (let c = 0; c < 3; c++) P8[3 * k + c] = N[3 * m + c] + s * vec[3 * m + c];
      const d = (P8[3 * k] - o[0]) * nx + (P8[3 * k + 1] - o[1]) * ny + (P8[3 * k + 2] - o[2]) * nz;
      D8[k] = d; if (d < 0) neg = true; else posi = true;
    }
    if (!neg || !posi) continue;
    const pts = [];
    for (const [a, b] of HEX_EDGES) {
      if ((D8[a] < 0) === (D8[b] < 0)) continue;
      const t = D8[a] / (D8[a] - D8[b]), fa = F.arr[E[8 * e + a]], fb = F.arr[E[8 * e + b]];
      pts.push([P8[3 * a] + t * (P8[3 * b] - P8[3 * a]), P8[3 * a + 1] + t * (P8[3 * b + 1] - P8[3 * a + 1]),
        P8[3 * a + 2] + t * (P8[3 * b + 2] - P8[3 * a + 2]), fa + t * (fb - fa)]);
    }
    if (pts.length < 3) continue;
    const cx = pts.reduce((q, p) => q + p[0], 0) / pts.length, cy = pts.reduce((q, p) => q + p[1], 0) / pts.length,
      cz = pts.reduce((q, p) => q + p[2], 0) / pts.length;
    pts.sort((p, q) => Math.atan2((p[0] - cx) * w[0] + (p[1] - cy) * w[1] + (p[2] - cz) * w[2], (p[0] - cx) * u[0] + (p[1] - cy) * u[1] + (p[2] - cz) * u[2])
      - Math.atan2((q[0] - cx) * w[0] + (q[1] - cy) * w[1] + (q[2] - cz) * w[2], (q[0] - cx) * u[0] + (q[1] - cy) * u[1] + (q[2] - cz) * u[2]));
    for (let i = 1; i + 1 < pts.length; i++) for (const p of [pts[0], pts[i], pts[i + 1]]) {
      pos.push(p[0] / 1000, p[1] / 1000, p[2] / 1000); vals.push(p[3]);
      const c = cmap((p[3] - lo) / span); col.push(c[0], c[1], c[2]);
    }
    for (let i = 0; i < pts.length; i++) { const a = pts[i], b = pts[(i + 1) % pts.length]; lin.push(a[0] / 1000, a[1] / 1000, a[2] / 1000, b[0] / 1000, b[1] / 1000, b[2] / 1000); }
  }
  if (!pos.length) return;
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  const nrm = new Float32Array(pos.length); for (let i = 0; i < nrm.length; i += 3) { nrm[i] = nx; nrm[i + 1] = ny; nrm[i + 2] = nz; }
  g.setAttribute("normal", new THREE.BufferAttribute(nrm, 3));
  capMesh = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
  capVals = Float32Array.from(vals);
  const lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.Float32BufferAttribute(lin, 3));
  capLines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x0b0d10, transparent: true, opacity: 0.3 }));
  capMesh.visible = showFE(); capLines.visible = showFE() && $("chk-mesh").checked;
  world.add(capMesh, capLines);
}

function currentField() { return fe && S.view ? fe.fields[S.view.field_name] || fe.fields.von_mises : null; }

function updateColors() {
  const F = currentField();
  if (!F || !feMesh) return;
  const col = feMesh.geometry.attributes.color.array, lo = F.min, span = (F.max - F.min) || 1;
  for (let i = 0; i < vNode.length; i++) { const c = cmap((F.arr[vNode[i]] - lo) / span); col[3 * i] = c[0]; col[3 * i + 1] = c[1]; col[3 * i + 2] = c[2]; }
  feMesh.geometry.attributes.color.needsUpdate = true;
}

function deformation(now) {
  const F = currentField(), warp = +$("rng-warp").value;
  if (F && F.vec) return { vec: F.vec, s: 0.07 * fe.half * 2 * warp * Math.sin(now / 1000 * Math.PI * 1.4) };
  return { vec: fe.disp, s: fe.umax > 0 ? (0.06 * fe.half * 2 / fe.umax) * warp : 0 };
}

function updatePositions(now) {
  if (!feMesh) return;
  const { vec, s } = deformation(now), N = fe.nodes;
  const write = (arr, idx) => {
    for (let i = 0; i < idx.length; i++) { const n = idx[i];
      arr[3 * i] = (N[3 * n] + s * vec[3 * n]) / 1000; arr[3 * i + 1] = (N[3 * n + 1] + s * vec[3 * n + 1]) / 1000;
      arr[3 * i + 2] = (N[3 * n + 2] + s * vec[3 * n + 2]) / 1000; }
  };
  write(feMesh.geometry.attributes.position.array, vNode);
  feMesh.geometry.attributes.position.needsUpdate = true;
  feMesh.geometry.computeVertexNormals();
  feMesh.geometry.computeBoundingSphere();
  write(feLines.geometry.attributes.position.array, feMesh.userData.lineNodes);
  feLines.geometry.attributes.position.needsUpdate = true;
  feLines.visible = $("chk-mesh").checked;
  updateCap(vec, s);
}

function disposeObj(o) { if (!o) return; o.parent && o.parent.remove(o); o.traverse && o.traverse((c) => { c.geometry && c.geometry.dispose(); }); }

// plain member (model / loads phases, or not solved yet) and the undeformed outline
function memberGeometry(G) {
  let g;
  if (G.circ) {
    g = new THREE.CylinderGeometry(G.wa / 2000, G.wa / 2000, G.L / 1000, 40);
    if (G.hor) g.rotateZ(-Math.PI / 2); else g.rotateX(Math.PI / 2);
  } else g = new THREE.BoxGeometry(2 * G.half[0] / 1000, 2 * G.half[1] / 1000, 2 * G.half[2] / 1000);
  g.translate(...G.center.map((v) => v / 1000));
  return g;
}
function rebuildPlain() {
  disposeObj(plainMesh); disposeObj(outline); plainMesh = outline = null;
  if (!S.model) return;
  const G = geo(), g = memberGeometry(G);
  plainMesh = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: 0x6f8fb3, transparent: true, opacity: 0.92 }));
  outline = new THREE.LineSegments(new THREE.EdgesGeometry(memberGeometry(G), 20),
    new THREE.LineBasicMaterial({ color: 0x8b95a3, transparent: true, opacity: 0.5 }));
  world.add(plainMesh, outline);
  // ground grid under the member
  disposeObj(ground);
  const span = G.span / 1000 * 1.6;
  ground = new THREE.GridHelper(span, 16, 0x2a313b, 0x1b2027);
  ground.rotateX(Math.PI / 2);
  const zmin = (G.hor ? -G.half[2] : 0) / 1000 - span * 0.08;
  ground.position.set(G.center[0] / 1000, 0, zmin);
  scene.add(ground);
}
function showFE() { return S.phase === "results" && fe && S.solved; }
function syncVisibility() {
  const r = showFE();
  if (plainMesh) plainMesh.visible = !r;
  if (outline) outline.visible = r ? $("chk-undeformed").checked : true;
  if (feMesh) feMesh.visible = r;
  if (feLines) feLines.visible = r && $("chk-mesh").checked;
  if (capMesh) capMesh.visible = r;
  if (capLines) capLines.visible = r && $("chk-mesh").checked;
}

// ---------------------------------------------------------------------------
// supports and loads as glyphs, with labels
const labels = [];   // {el, pos: Vector3}
function addLabel(text, cls, pos) {
  const el = document.createElement("div"); el.className = "lbl " + cls; el.textContent = text;
  $("labels").appendChild(el); labels.push({ el, pos });
}
function rebuildGlyphs() {
  while (glyphs.children.length) disposeObj(glyphs.children[0]);
  $("labels").innerHTML = ""; labels.length = 0;
  if (!S.model) return;
  const m = S.model, G = geo(m), sz = Math.max(G.wa, G.wb), dark = new THREE.MeshLambertMaterial({ color: 0x59616c });
  const steel = new THREE.MeshLambertMaterial({ color: 0xaab4c0 });
  for (const s of m.supports) {
    const c = G.at(s.t), grp = new THREE.Group();
    if (s.kind === "fixed") {
      const th = Math.max(0.02 * G.L, 40), out = s.t < 0.5 ? -1 : 1;
      const dims = G.hor ? [th, sz * 1.9, sz * 1.9] : [sz * 1.9, sz * 1.9, th];
      const box = new THREE.Mesh(new THREE.BoxGeometry(...dims.map((d) => d / 1000)), dark);
      const p = c.slice(); p[G.hor ? 0 : 2] += out * th / 2;
      box.position.copy(v3(p)); grp.add(box);
      addLabel("fixed", "sup", v3(G.hor ? [p[0], 0, -sz] : [sz, 0, p[2]]));
    } else {
      const hgt = sz * 0.7, cone = new THREE.Mesh(new THREE.ConeGeometry(hgt * 0.6 / 1000, hgt / 1000, 4), steel);
      let apex, label;
      if (G.hor || s.t <= 0.02) {                    // under the member, pointing up
        apex = G.hor ? [c[0], 0, -G.half[2]] : [0, 0, 0];
        cone.rotation.x = Math.PI / 2; cone.position.copy(v3([apex[0], apex[1], apex[2] - hgt / 2]));
        label = [apex[0], 0, apex[2] - hgt * 1.6];
        if (s.kind === "roller") for (const dx of [-0.3, 0.3]) {
          const w = new THREE.Mesh(new THREE.SphereGeometry(hgt * 0.16 / 1000, 12, 8), steel);
          w.position.copy(v3([apex[0] + dx * hgt, 0, apex[2] - hgt * 1.16])); grp.add(w);
        }
      } else {                                        // column above its base: from the side
        apex = [-G.half[0], 0, c[2]];
        cone.rotation.z = -Math.PI / 2; cone.position.copy(v3([apex[0] - hgt / 2, 0, apex[2]]));
        label = [apex[0] - hgt * 1.4, 0, apex[2]];
        if (s.kind === "roller") for (const dz of [-0.3, 0.3]) {
          const w = new THREE.Mesh(new THREE.SphereGeometry(hgt * 0.16 / 1000, 12, 8), steel);
          w.position.copy(v3([apex[0] - hgt * 1.16, 0, apex[2] + dz * hgt])); grp.add(w);
        }
      }
      grp.add(cone);
      addLabel(s.kind, "sup", v3(label));
    }
    glyphs.add(grp);
  }
  // loads: arrows touching the surface the load acts on
  const pmax = Math.max(1e-9, ...m.loads.filter((l) => l.kind === "point").map((l) => Math.abs(l.w0)));
  const wmax = Math.max(1e-9, ...m.loads.filter((l) => l.kind !== "point").flatMap((l) => [Math.abs(l.w0), Math.abs(l.w1 ?? l.w0)]));
  const Lref = Math.max(0.1 * G.L, 2.2 * sz);
  const ext = (axis) => G.half[["x", "y", "z"].indexOf(axis)] * ((G.hor && axis === "x") || (!G.hor && axis === "z") ? 0 : 1);
  m.loads.forEach((ld, i) => {
    const d = AX[ld.axis].map((v) => v * ld.sign), col = ld.kind === "point" ? 0xff9f43 : 0xffc078;
    const arrow = (t, len) => {
      const c = G.at(t), tip = c.map((v, k) => v - d[k] * ext(ld.axis)), tail = tip.map((v, k) => v - d[k] * len);
      if (len > 1) glyphs.add(new THREE.ArrowHelper(new THREE.Vector3(...d), v3(tail), len / 1000, col, Math.min(len * 0.3, sz * 0.5) / 1000, Math.min(len * 0.18, sz * 0.3) / 1000));
      return tail;
    };
    if (ld.kind === "point") {
      const tail = arrow(ld.t0, Lref * (0.4 + 0.6 * Math.abs(ld.w0) / pmax));
      addLabel(m.load_labels[i], "load", v3(tail));
    } else {
      const t1 = ld.t1 ?? 1, w1 = ld.w1 ?? ld.w0, n = Math.max(3, Math.round(Math.abs(t1 - ld.t0) * 14));
      const tails = [];
      for (let j = 0; j <= n; j++) {
        const a = j / n, w = ld.w0 + (w1 - ld.w0) * a;
        tails.push(arrow(ld.t0 + (t1 - ld.t0) * a, Lref * 0.7 * Math.abs(w) / wmax));
      }
      const lg = new THREE.BufferGeometry().setFromPoints(tails.map(v3));
      glyphs.add(new THREE.Line(lg, new THREE.LineBasicMaterial({ color: col })));
      addLabel(m.load_labels[i], "load", v3(tails[Math.floor(tails.length / 2)]));
    }
  });
}

// ---------------------------------------------------------------------------
// camera from ViewState
function frameInfo() {
  const G = S.model ? geo() : null;
  const c = fe && showFE() ? fe.center : G ? G.center : [0, 0, 0];
  const half = fe && showFE() ? fe.half : G ? G.span / 2 : 1000;
  return { c, half };
}
function applyView() {
  const v = S.view; if (!v) return;
  const { c, half } = frameInfo();
  const az = THREE.MathUtils.degToRad(v.azimuth), el = THREE.MathUtils.degToRad(v.elevation);
  const back = new THREE.Vector3(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el));
  const focal = v3(c.map((x, k) => x + v.focal_point[k] * half));
  // fit the bounding sphere into the narrower of the two fields of view, with a margin for the overlays
  const fovV = THREE.MathUtils.degToRad(camera.fov), fovH = 2 * Math.atan(Math.tan(fovV / 2) * camera.aspect);
  const D = (half / 1000) / Math.sin(Math.min(fovV, fovH) / 2) * 1.25 * v.distance;
  camera.position.copy(focal).addScaledVector(back, D);
  camera.near = D / 200; camera.far = D * 30; camera.updateProjectionMatrix();
  camera.lookAt(focal);
}

// ---------------------------------------------------------------------------
// screen <-> member mapping (gesture positions arrive as viewport fractions)
function toScreen(p) { const v = p.clone().project(camera); return [(v.x + 1) / 2, (1 - v.y) / 2]; }
function screenToT(xy) {
  const G = geo(), a = toScreen(v3(G.at(0))), b = toScreen(v3(G.at(1)));
  const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy || 1;
  return Math.min(1, Math.max(0, ((xy[0] - a[0]) * dx + (xy[1] - a[1]) * dy) / L2));
}
function axisFromScreenVector(t, sv) {       // flick (screen, y down) -> best-aligned world axis
  const G = geo(), p = v3(G.at(t)), p0 = toScreen(p), n = Math.hypot(sv[0], sv[1]) || 1;
  let best = null, score = -2;
  for (const ax of ["x", "y", "z"]) for (const sg of [1, -1]) {
    const q = p.clone().add(new THREE.Vector3(...AX[ax]).multiplyScalar(sg * G.span / 4000)), s1 = toScreen(q);
    const ex = s1[0] - p0[0], ey = s1[1] - p0[1], len = Math.hypot(ex, ey);
    if (len < 0.01) continue;
    const sc = (ex * sv[0] + ey * sv[1]) / (len * n);
    if (sc > score) { score = sc; best = { axis: ax, sign: sg }; }
  }
  return best || { axis: "z", sign: -1 };
}
function axisFromViewVector(nv) {             // palm normal (view space) -> nearest world axis
  camera.updateMatrixWorld();
  const e = camera.matrixWorld.elements, r = [e[0], e[1], e[2]], u = [e[4], e[5], e[6]], b = [e[8], e[9], e[10]];
  const w = [0, 1, 2].map((k) => nv[0] * r[k] + nv[1] * u[k] + nv[2] * b[k]);
  const k = [0, 1, 2].reduce((a, i) => (Math.abs(w[i]) > Math.abs(w[a]) ? i : a), 0);
  return { axis: "xyz"[k], sign: Math.sign(w[k]) || -1 };
}
function tLabel(t) { const G = geo(); return `${G.hor ? "x" : "z"} = ${fmt(t * G.L / 1000)} m`; }

// gesture commands from the engine that need the camera to land on the member
const TOOL_NAMES = { support_fixed: "Fixed", support_pinned: "Pinned", support_roller: "Roller" };
function resolveCommand(c) {
  const d = c.data || {};
  switch (c.kind) {
    case "cursor": S.cursor = { xy: d.xy, pose: d.pose, until: performance.now() + 250 }; break;
    case "support": sendAction({ op: d.op, t: screenToT(d.xy) }); break;
    case "point_load": {
      const t = screenToT(d.xy), dir = d.flick ? axisFromScreenVector(t, d.flick) : { axis: "z", sign: -1 };
      sendAction({ op: "point_load", t, ...dir }); break;
    }
    case "udl": sendAction({ op: "uniform_load", t0: screenToT(d.xy0), t1: screenToT(d.xy1), ...axisFromViewVector(d.normal_view) }); break;
    case "trap": sendAction({ op: "trapezoidal_load", t0: screenToT(d.xy0), t1: screenToT(d.xy1), ...axisFromViewVector(d.normal_view) }); break;
  }
}
const sendAction = (action) => send({ type: "model_action", action });

// ---------------------------------------------------------------------------
// probe and pins (results phase)
const ray = new THREE.Raycaster();
function pick(xy) {
  if (!feMesh || !feMesh.visible) return null;
  ray.setFromCamera(new THREE.Vector2(xy[0] * 2 - 1, -(xy[1] * 2 - 1)), camera);
  const hits = ray.intersectObject(feMesh).filter((h) => clipPlane.distanceToPoint(h.point) >= -1e-6);
  if (capMesh) for (const h of ray.intersectObject(capMesh)) { h.cap = true; hits.push(h); }
  hits.sort((a, b) => a.distance - b.distance);
  const hit = hits[0];
  if (!hit) return null;
  if (hit.cap) {                                 // on the cut face: value interpolated across the slice
    const Fc = currentField(), f = hit.face, vv = [f.a, f.b, f.c].map((i) => capVals[i]);
    return { point: hit.point.clone(), text: `${Fc.label}: ${fmt((vv[0] + vv[1] + vv[2]) / 3, 4)} ${Fc.unit}`,
      where: `on the cut · (${fmt(hit.point.x)}, ${fmt(hit.point.y)}, ${fmt(hit.point.z)}) m` };
  }
  const pos = feMesh.geometry.attributes.position, f = hit.face;
  let best = f.a, bd = Infinity;
  for (const v of [f.a, f.b, f.c]) { const d = new THREE.Vector3().fromBufferAttribute(pos, v).distanceTo(hit.point); if (d < bd) { bd = d; best = v; } }
  const n = vNode[best], F = currentField(), N = fe.nodes;
  return { point: hit.point.clone(), text: `${F.label}: ${fmt(F.arr[n], 4)} ${F.unit}`,
    where: `node ${n} · (${fmt(N[3 * n] / 1000)}, ${fmt(N[3 * n + 1] / 1000)}, ${fmt(N[3 * n + 2] / 1000)}) m` };
}
const pinCache = new Map();

// ---------------------------------------------------------------------------
// left panel
function renderFieldSelect() {
  const sel = $("sel-field"); sel.innerHTML = "";
  if (!fe) return;
  const groups = {};
  for (const k of fe.field_order) { const f = fe.fields[k]; (groups[f.group] ||= []).push([k, f.label]); }
  for (const [g, items] of Object.entries(groups)) {
    const og = document.createElement("optgroup"); og.label = g;
    for (const [k, l] of items) og.appendChild(new Option(l, k));
    sel.appendChild(og);
  }
  if (S.view) sel.value = S.view.field_name;
}
function renderLegend() {
  const F = currentField(), ticks = $("legend-ticks"); ticks.innerHTML = "";
  if (!F || !showFE()) { $("legend-meta").textContent = S.solved ? "" : "Not solved yet: add supports and loads, then 👍 or Solve."; return; }
  for (let i = 0; i <= 6; i++) {
    const a = i / 6, sp = document.createElement("span");
    sp.style.top = `${(1 - a) * 100}%`; sp.textContent = `${fmt(F.min + (F.max - F.min) * a, 4)} ${F.unit}`;
    ticks.appendChild(sp);
  }
  const N = fe.nodes, n = F.argmax, { s } = deformation(0);
  $("legend-meta").innerHTML = `<b>${F.label}</b><br>peak ${fmt(F.arr[n], 4)} ${F.unit} at (${fmt(N[3 * n] / 1000)}, ${fmt(N[3 * n + 1] / 1000)}, ${fmt(N[3 * n + 2] / 1000)}) m`
    + (F.freq ? `<br>natural frequency ${fmt(F.freq, 4)} Hz (shape animated)` : `<br>deformation shown ×${fmt(s, 3)}`)
    + `<br>nodal values: Gauss points extrapolated to corners, averaged`;
}
function kv(tbl, rows) {
  $(tbl).innerHTML = rows.filter(Boolean).map(([k, v, em]) => `<tr${em ? ' class="em"' : ""}><td>${k}</td><td>${v}</td></tr>`).join("");
}
function renderPanels() {
  const m = S.model; if (!m) return;
  const sm = m.summary, st = fe && S.solved ? fe.stats : null;
  kv("tbl-mesh", st ? [
    ["Element", "Hex8 + incompatible modes"], ["Integration", "2×2×2 Gauss"],
    ["Elements", st.elements.toLocaleString()], ["Nodes", st.nodes.toLocaleString()],
    ["DOF (free / fixed)", `${st.free_dof.toLocaleString()} / ${st.restrained_dof.toLocaleString()}`],
    ["Grid (cells)", st.grid.join(" × ")], ["Cell size", st.cell_mm.map((v) => fmt(v)).join(" × ") + " mm"],
    ["Max aspect ratio", fmt(st.aspect)], ["Linear solver", "sparse LU (SuperLU)"],
    ["Eigen solver", "Lanczos, shift-invert"], ["Time (asm / solve / modes)", `${st.t_assemble_s} / ${st.t_solve_s} / ${st.t_modes_s} s`],
  ] : [["Status", "not solved — the mesh is built at solve time"]]);
  kv("tbl-model", [
    ["Name", m.name], ["Type", sm.kind],
    S.settings && ["Increments", `L ${fmt(S.settings.length)} mm · section ${fmt(S.settings.section)} mm · load ${fmt(S.settings.load)} kN`], ["Section", sm.section], ["Length", `${fmt(sm.length_mm / 1000)} m`],
    ["Material", sm.material], ["E / ν", `${fmt(sm.E_MPa)} MPa / ${sm.nu}`], ["Density", `${fmt(sm.rho_t_mm3 * 1e12)} kg/m³`],
    ["Area A", `${fmt(sm.area_mm2)} mm²`], ["I (min)", `${fmt(sm.I_min_mm4)} mm⁴`], ["r (min)", `${fmt(sm.r_mm)} mm`],
    ["Self-weight", `${fmt(sm.self_weight_kN_m)} kN/m (not applied)`],
    sm.K !== undefined && ["Effective length K", fmt(sm.K)],
    sm.K !== undefined && ["Slenderness KL/r", `${fmt(sm.slenderness)} (limit ${fmt(sm.slenderness_transition)})`],
    sm.K !== undefined && ["Euler load P_cr", `${fmt(sm.P_cr_kN)} kN`],
    sm.K !== undefined && ["Squash load A·f", `${fmt(sm.P_squash_kN)} kN`],
    sm.K !== undefined && ["Column class", `${sm.column_class} — ${sm.governs}`, true],
  ]);
  $("lst-bc").innerHTML = m.supports.map((s) => `<li>${s.kind} support at ${tLabel(s.t)}</li>`).join("")
    + m.load_labels.map((l, i) => { const ld = m.loads[i]; return `<li style="color:var(--load)">${l} · ${ld.kind === "point" ? tLabel(ld.t0) : tLabel(ld.t0) + " → " + tLabel(ld.t1 ?? 1)}</li>`; }).join("")
    || "<li class='muted'>none yet</li>";
  $("lst-issues").innerHTML = m.issues.map((i) => `<li>${i}</li>`).join("");
  if (st) {
    const modes = fe.field_order.filter((k) => k.startsWith("mode")).map((k) => fmt(fe.fields[k].freq, 3)).join(", ");
    const sup = fe.reactions.supports.map((r) => [`R ${r.kind} @ ${fmt(r.t)}`, r.kN.map((v) => fmt(v, 3)).join(", ") + " kN"]);
    kv("tbl-res", [["Max |U|", `${fmt(st.max_U_mm, 4)} mm`, true], ["Max von Mises", `${fmt(st.max_von_mises_MPa, 4)} MPa`, true],
      ["Strain energy", `${fmt(st.strain_energy_J, 4)} J`], ["Applied (x,y,z)", st.applied_kN.map((v) => fmt(v, 4)).join(", ") + " kN"],
      ["Reactions (x,y,z)", fe.reactions.total_kN.map((v) => fmt(v, 4)).join(", ") + " kN"], ...sup, ["Frequencies", modes + " Hz"]]);
  } else kv("tbl-res", [["Status", "solve to see results"]]);
}

// ---------------------------------------------------------------------------
// phases, toolbar, mouse tools
const TOOLBARS = {
  model: [["⚙️ Increments…", () => sendAction({ op: "setup" })], ["New beam (rect)", () => sendAction({ op: "new_rect", orientation: "horizontal" })],
    ["New column (rect)", () => sendAction({ op: "new_rect", orientation: "vertical" })],
    ["New circular member", () => sendAction({ op: "new_circle" })], ["Material…", () => sendAction({ op: "material" })]],
  loads: [["✊ Fixed", "support_fixed"], ["🤏 Pinned", "support_pinned"], ["✌️ Roller", "support_roller"],
    ["☝️ Point load", "point_load"], ["✋ Uniform load", "uniform_load"], ["🙌 Trapezoidal", "trapezoidal_load"],
    ["👍 Solve", () => send({ type: "solve" })]],
  results: [["von Mises", "f:von_mises"], ["|U|", "f:U"], ["S11", "f:S11"], ["S33", "f:S33"], ["SED", "f:SED"], ["Mode 1", "f:mode1"], ["Mode 2", "f:mode2"]],
};
function renderTabs() {
  for (const b of document.querySelectorAll("#tabs button")) {
    b.classList.toggle("on", b.dataset.phase === S.phase);
    b.disabled = b.dataset.phase === "results" && !S.solved;
  }
  const tb = $("toolbar"); tb.innerHTML = "";
  for (const [label, act] of TOOLBARS[S.phase] || []) {
    const b = document.createElement("button"); b.textContent = label;
    if (typeof act === "function") b.onclick = act;
    else if (act.startsWith("f:")) { b.onclick = () => command("set_field", { name: act.slice(2) }); b.classList.toggle("on", S.view && S.view.field_name === act.slice(2)); }
    else { b.onclick = () => { S.tool = S.tool === act ? null : act; S.toolT0 = null; renderTabs(); }; b.classList.toggle("on", S.tool === act); }
    tb.appendChild(b);
  }
  renderCheat();
}
function placeTool(t) {
  const tool = S.tool;
  if (!tool) return false;
  if (tool.startsWith("support_")) sendAction({ op: tool, t });
  else if (tool === "point_load") sendAction({ op: tool, t, axis: "z", sign: -1 });
  else if (S.toolT0 === null) { S.toolT0 = t; toast("Now click where the load ends"); return true; }
  else { sendAction({ op: tool, t0: S.toolT0, t1: t, axis: "z", sign: -1 }); S.toolT0 = null; }
  S.tool = null; renderTabs();
  return true;
}

// ---------------------------------------------------------------------------
// number prompt card
function renderPrompt() {
  const p = S.prompt, card = $("prompt");
  if (!p) { card.classList.add("hidden"); return; }
  const wasHidden = card.classList.contains("hidden");
  card.classList.remove("hidden");
  $("prompt-step").textContent = `Step ${p.step} of ${p.steps}` + (p.default !== null && p.default !== undefined ? ` · default ${fmt(p.default)} ${p.unit}` : "");
  $("prompt-label").textContent = p.label;
  $("prompt-unit").textContent = p.unit || (p.choices ? "choose by fingers" : "");
  $("prompt-choices").innerHTML = p.choices ? Object.entries(p.choices).map(([k, v]) => `<div data-choice="${k}"><b>${k}</b> finger${k > 1 ? "s" : ""} — ${v}${+k === p.default ? " <span class='muted'>(default)</span>" : ""}</div>`).join("") : "";
  for (const d of $("prompt-choices").children) d.onclick = () => send({ type: "number", value: +d.dataset.choice });
  $("prompt-inc").textContent = p.increment ? `increment ${fmt(p.increment)} ${p.unit}: values snap to it, 🤏 pinch + move up/down = ±${fmt(p.increment)} · ` : "";
  const ax = $("prompt-axes"); ax.innerHTML = "";
  if (p.axis) for (const a of ["x", "y", "z"]) for (const sg of [1, -1]) {
    const b = document.createElement("button"); b.textContent = `${sg > 0 ? "+" : "−"}${a.toUpperCase()}`;
    b.classList.toggle("on", p.axis === a && p.sign === sg);
    b.onclick = () => send({ type: "set_axis", axis: a, sign: sg }); ax.appendChild(b);
  }
  if (wasHidden) { $("prompt-input").value = ""; setTimeout(() => $("prompt-input").focus(), 0); }
}
function submitPrompt() {
  const raw = $("prompt-input").value.trim(), v = raw === "" ? null : Number(raw);
  if (raw !== "" && !isFinite(v)) { toast("Enter a number", true); return; }
  $("prompt-input").value = ""; send({ type: "number", value: v });
}
$("prompt-ok").onclick = submitPrompt;
$("prompt-cancel").onclick = () => send({ type: "prompt_cancel" });
$("prompt-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); submitPrompt(); }
  if (e.key === "Escape") send({ type: "prompt_cancel" });
  e.stopPropagation();
});

// ---------------------------------------------------------------------------
// view cube (mouse): click a face for that view
const gz = { r: new THREE.WebGLRenderer({ antialias: true, alpha: true }), scene: new THREE.Scene(), cam: new THREE.PerspectiveCamera(30, 1, 0.1, 20) };
gz.r.setPixelRatio(window.devicePixelRatio); gz.r.setSize(112, 112); $("gizmo").appendChild(gz.r.domElement);
gz.cam.up.set(0, 0, 1);
const FACE_VIEWS = [["RIGHT", "+X", 0, 0], ["LEFT", "−X", 180, 0], ["BACK", "+Y", 90, 0], ["FRONT", "−Y", -90, 0], ["TOP", "+Z", -90, 89], ["BOTTOM", "−Z", -90, -89]];
const faceMats = FACE_VIEWS.map(([name, ax]) => {
  const c = document.createElement("canvas"); c.width = c.height = 128; const x = c.getContext("2d");
  x.fillStyle = "#222832"; x.fillRect(0, 0, 128, 128); x.strokeStyle = "#4da3ff"; x.lineWidth = 4; x.strokeRect(2, 2, 124, 124);
  x.fillStyle = "#e6e9ee"; x.font = "bold 24px system-ui"; x.textAlign = "center"; x.fillText(name, 64, 62);
  x.fillStyle = "#8b95a3"; x.font = "20px system-ui"; x.fillText(ax, 64, 92);
  return new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c) });
});
const cube = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), faceMats); gz.scene.add(cube);
for (const [d, col] of [[[1, 0, 0], 0xff5555], [[0, 1, 0], 0x55dd55], [[0, 0, 1], 0x5599ff]]) {
  gz.scene.add(new THREE.ArrowHelper(new THREE.Vector3(...d), new THREE.Vector3(-0.75, -0.75, -0.75), 0.9, col, 0.18, 0.1));
}
gz.r.domElement.addEventListener("click", (e) => {
  const r = gz.r.domElement.getBoundingClientRect(), rc = new THREE.Raycaster();
  rc.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), gz.cam);
  const hit = rc.intersectObject(cube)[0];
  if (hit) { const [, , az, el] = FACE_VIEWS[hit.face.materialIndex]; command("set_view", { azimuth: az, elevation: el }); }
});
for (const b of document.querySelectorAll("[data-view]")) b.onclick = () => b.dataset.view === "iso" ? command("set_view", { azimuth: -60, elevation: 22 }) : command("fit");
function renderGizmo() {        // the cube turns with the main camera
  const d = camera.getWorldDirection(new THREE.Vector3()).negate();
  gz.cam.position.copy(d.multiplyScalar(3.4)); gz.cam.lookAt(0, 0, 0);
  gz.r.render(gz.scene, gz.cam);
}

// ---------------------------------------------------------------------------
// WebSocket to the Python engine
let ws;
// one id per browser: the server keeps this modeller session alive while you visit the Workbench
// named tab, so the Workbench's "Gesture modeller" link comes back to this one
if (!window.name) window.name = "archimedes-modeller";
const CID = (() => { try { let c = localStorage.getItem("archimedes-cid"); if (!c) { c = Math.random().toString(36).slice(2); localStorage.setItem("archimedes-cid", c); } return c; } catch (e) { return "anon"; } })();
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?cid=${CID}`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connect, 1000);
}
const send = (o) => ws && ws.readyState === 1 && ws.send(JSON.stringify(o));
const command = (kind, data = {}) => send({ type: "command", kind, data });

let inflight = 0, lastHud = null, demoLandmarks = null, demoRunning = false, toastTimer;
function toast(text, err = false) {
  const t = $("toast"); t.textContent = text; t.classList.remove("hidden"); t.classList.toggle("err", err);
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add("hidden"), err ? 3500 : 1800);
}
function onMessage(m) {
  switch (m.type) {
    case "info": S.info = m; renderCheat(); $("chip-model").textContent = `model: ${m.classifier}`; $("sel-hand").value = m.profile.dominant; fillRecLabels(); return;
    case "recorded": $("rec-status").textContent = m.path ? `Saved ${m.path}` : "Nothing saved - no hand was visible"; return;
    case "demo_done": demoRunning = false; $("demo-caption").classList.add("hidden"); demoLandmarks = null; return;
    case "solving": S.busy = true; $("busy").classList.remove("hidden"); return;
    case "results":
      S.busy = false; $("busy").classList.add("hidden");
      fe = decodeResults(m); S.solved = true; pinCache.clear();
      renderFieldSelect(); rebuildFE(); renderPanels(); renderLegend(); syncVisibility(); return;
    case "model": onModel(m); return;
    case "view": S.view = m.view; onView(); return;
  }
  if (m.type !== "state") return;
  inflight = Math.max(0, inflight - 1);
  if (m.view) { S.view = m.view; onView(); }
  if (m.demo) { $("demo-caption").textContent = m.demo; $("demo-caption").classList.remove("hidden"); demoLandmarks = m.landmarks; }
  renderCalibration(m.calibration);
  if (m.hud) renderHud(m.hud);
  for (const c of m.commands || []) resolveCommand(c);
}
function onModel(m) {
  const changed = m.version !== S.version;
  S.model = m.model; S.phase = m.phase; S.prompt = m.prompt; S.solved = m.solved; S.version = m.version; S.settings = m.settings;
  if (!m.solved) { fe = null; disposeObj(feMesh); disposeObj(feLines); feMesh = feLines = null; }
  if (changed) { rebuildPlain(); rebuildGlyphs(); }
  if (m.error) toast(m.error, true); else if (m.message && changed) toast(m.message);
  renderTabs(); renderPrompt(); renderPanels(); renderLegend(); syncVisibility(); applyView();
}
let lastCutKey = "", lastField = "";
function onView() {
  const v = S.view, cutKey = v.section_on ? v.plane_normal.concat(v.plane_origin).map((x) => x.toFixed(3)).join() : "";
  if (cutKey !== lastCutKey) { lastCutKey = cutKey; updateCut(); updatePositions(performance.now()); syncVisibility(); }
  if (v.field_name !== lastField) { lastField = v.field_name; $("sel-field").value = v.field_name; updateColors(); updatePositions(performance.now()); renderLegend(); renderTabs(); }
  if (v.snapshots > (S.snaps || 0)) saveSnapshot();
  S.snaps = v.snapshots;
}
function renderCheat() {
  if (!S.info) return;
  const sheet = (S.info.cheat_sheets && S.info.cheat_sheets[S.phase]) || S.info.cheat_sheet;
  $("cheat-title").textContent = { model: "Gestures · model", loads: "Gestures · loads & supports", results: "Gestures · results" }[S.phase];
  $("cheat").innerHTML = sheet.map((r) => `<li data-label="${r.label}"><span class="ic">${r.icon}</span><span><span class="lb">${r.label}</span><br><span class="muted small">${r.hint}</span></span></li>`).join("");
}

const ICONS = { orbit: "✊", zoom: "🤏", section: "✋", probe: "☝️", field: "✌️", snapshot: "👍" };
function renderHud(h) {
  lastHud = h;
  const badge = $("mode-badge");
  badge.className = `badge ${h.state === "number" ? "arming" : h.state}`;
  $("mode-icon").textContent = h.state === "number" ? "#" : h.mode ? (ICONS[h.mode] || "●") : h.state === "paused" ? "!" : h.state === "no_hand" ? "·" : "○";
  $("ring-fg").style.strokeDashoffset = String(119.4 * (1 - (h.mode_label ? 1 : h.progress)));
  $("mode-text").textContent = h.mode_label || (h.progress_label ? `${h.progress_label}…` : { no_hand: "No hand", idle: "Ready", paused: "Paused", arming: "…", number: "Show a number", active: "…" }[h.state]);
  $("pose-text").textContent = `pose: ${h.pose.replace("_", " ")} · ${(h.confidence * 100).toFixed(0)}%`;
  const hint = $("hint");
  if (h.hints.length) { hint.textContent = h.hints[0]; hint.classList.remove("hidden"); hint.classList.toggle("soft", h.state !== "paused"); }
  else hint.classList.add("hidden");
  $("tip").textContent = h.tip;
  if (h.toast) toast(h.toast);
  if (h.latency_ms !== undefined) $("chip-lat").textContent = `${h.latency_ms.toFixed(1)} ms engine`;
  if (h.personal_samples !== undefined) $("chip-personal").textContent = `${h.personal_samples} personal samples`;
  if (h.number) {                       // finger-count digits into the prompt card
    $("prompt-live").textContent = h.number.live === null ? "–" : h.number.live === 10 ? "OK" : h.number.live;
    if (h.number.digits) $("prompt-input").value = h.number.digits;
    $("prompt-bar").style.width = `${h.progress * 100}%`;
    $("prompt-bar").style.background = h.number.kind === "cancel" ? "var(--bad)" : h.number.kind === "confirm" ? "var(--ok)" : "var(--arming)";
  }
  for (const li of $("cheat").children) {
    li.classList.toggle("active", !!h.mode_label && li.dataset.label === h.mode_label);
    li.classList.toggle("arming", !h.mode_label && !!h.progress_label && li.dataset.label === h.progress_label);
  }
}
function renderCalibration(c) {
  const box = $("calib");
  if (!c || c.done) { if (!box.classList.contains("hidden") && c && c.done) { send({ type: "info" }); toast("Calibration saved ✓"); } box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  $("calib-step").textContent = `Step ${c.step_index + 1} of ${c.step_count}`;
  $("calib-text").textContent = c.instruction; $("calib-bar").style.width = `${c.progress * 100}%`; $("calib-warn").textContent = c.warning || "";
}
function saveSnapshot() {
  renderer.render(scene, camera);
  const a = document.createElement("a"); a.href = renderer.domElement.toDataURL("image/png"); a.download = `archimedes-${Date.now()}.png`; a.click();
}

// ---------------------------------------------------------------------------
// overlay: probe, pins, labels, gesture previews, tool cursor
function drawOverlay(now) {
  const W = overlay.width, H = overlay.height;
  octx.clearRect(0, 0, W, H);
  for (const L of labels) {                                   // glyph labels follow the camera
    const [x, y] = toScreen(L.pos), w2 = (L.el.offsetWidth || 60) / 2;   // keep labels inside the viewport
    L.el.style.left = `${Math.min(W - w2 - 4, Math.max(w2 + 4, x * W))}px`; L.el.style.top = `${Math.min(H - 4, Math.max(24, y * H))}px`;
  }
  const pv = lastHud && lastHud.preview;
  if (pv && S.model) {
    octx.lineWidth = 3; octx.strokeStyle = "#f5b83d"; octx.fillStyle = "#f5b83d";
    if (pv.kind === "trace" && pv.points.length > 1) {
      octx.beginPath(); pv.points.forEach(([x, y], i) => (i ? octx.lineTo(x * W, y * H) : octx.moveTo(x * W, y * H))); octx.stroke();
    } else if (pv.kind === "sweep") {
      const G = geo(), a = toScreen(v3(G.at(screenToT(pv.xy0)))), b = toScreen(v3(G.at(screenToT(pv.xy1))));
      octx.lineWidth = 8; octx.globalAlpha = 0.6; octx.beginPath(); octx.moveTo(a[0] * W, a[1] * H); octx.lineTo(b[0] * W, b[1] * H); octx.stroke(); octx.globalAlpha = 1;
    } else if (pv.kind === "extrude") {
      octx.setLineDash([8, 6]); octx.beginPath(); octx.moveTo(pv.a[0] * W, pv.a[1] * H); octx.lineTo(pv.b[0] * W, pv.b[1] * H); octx.stroke(); octx.setLineDash([]);
      octx.font = "bold 15px system-ui"; octx.fillText(`${pv.orientation === "horizontal" ? "beam" : "column"} ≈ ${fmt(pv.length_est / 1000)} m`, (pv.a[0] + pv.b[0]) / 2 * W + 8, (pv.a[1] + pv.b[1]) / 2 * H - 10);
    } else if (pv.kind === "point") {
      const G = geo(), p = toScreen(v3(G.at(screenToT(pv.xy))));
      octx.beginPath(); octx.arc(p[0] * W, p[1] * H, 10, 0, 7); octx.stroke();
    }
  }
  // tool cursor: gesture cursor or mouse with a tool selected
  const mk = $("cursor-mark");
  const cur = S.cursor && S.cursor.until > now ? S.cursor.xy : S.tool && S.mouseXY ? S.mouseXY : null;
  if (cur && S.model && S.phase === "loads") {
    const t = screenToT(cur), p = toScreen(v3(geo().at(t)));
    mk.style.left = `${p[0] * W}px`; mk.style.top = `${p[1] * H}px`; mk.classList.remove("hidden");
    const name = S.tool ? (TOOL_NAMES[S.tool] || S.tool.replace("_", " ")) : { fist: "fixed?", pinch: "pinned?", peace: "roller?", point: "point load?", open_palm: "uniform load" }[S.cursor.pose] || "";
    mk.innerHTML = `<span>${name} · ${tLabel(t)}</span>`;
  } else mk.classList.add("hidden");
  // probe + pins (results)
  const tip = $("probe-tip");
  const xy = (S.phase === "results" && (S.mouseXY && !S.tool ? S.mouseXY : S.view && S.view.probe_cursor)) || null;
  const r = xy ? pick(xy) : null;
  if (r) { tip.classList.remove("hidden"); tip.style.left = `${xy[0] * W}px`; tip.style.top = `${xy[1] * H}px`; tip.innerHTML = `${r.text}<br><span class="muted small">${r.where}</span>`; }
  else tip.classList.add("hidden");
  if (S.view && showFE()) for (const p of S.view.probe_pins) {
    const key = p.join() + S.view.field_name;
    if (!pinCache.has(key)) pinCache.set(key, pick(p));
    const hit = pinCache.get(key); if (!hit) continue;
    const [x, y] = toScreen(hit.point);
    octx.fillStyle = "#f5b83d"; octx.beginPath(); octx.arc(x * W, y * H, 4, 0, 7); octx.fill();
    octx.font = "12px system-ui"; octx.fillText(hit.text, x * W + 8, y * H - 6);
  }
}

// ---------------------------------------------------------------------------
// camera + MediaPipe (browser is the sensor)
const video = $("video"), cam = $("cam"), cctx = cam.getContext("2d");
const probeCanvas = document.createElement("canvas"); probeCanvas.width = 96; probeCanvas.height = 72;
const pctx = probeCanvas.getContext("2d", { willReadFrequently: true });
let landmarker = null, lastVideoTime = -1, seq = 0, frameQuality = {}, fpsCount = 0, fpsT = performance.now(), lastRaw = null;
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } });
    video.srcObject = stream; await video.play();
    $("cam-label").textContent = "loading hand model…";
    const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", delegate: "GPU" },
      runningMode: "VIDEO", numHands: 2, minHandDetectionConfidence: 0.6, minHandPresenceConfidence: 0.6, minTrackingConfidence: 0.5,
    });
    $("cam-label").textContent = "camera on"; $("start").classList.add("hidden");
  } catch (err) {
    const e = $("start-error"); e.textContent = `Camera unavailable (${err.name || err}). Allow camera access, or use the mouse.`; e.classList.remove("hidden");
  }
}
function measureQuality() {
  pctx.drawImage(video, 0, 0, 96, 72);
  const d = pctx.getImageData(0, 0, 96, 72).data, g = new Float32Array(96 * 72);
  let sum = 0; for (let i = 0; i < g.length; i++) { g[i] = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]; sum += g[i]; }
  let s = 0, s2 = 0, n = 0;
  for (let y = 1; y < 71; y++) for (let x = 1; x < 95; x++) { const i = y * 96 + x, lap = g[i - 1] + g[i + 1] + g[i - 96] + g[i + 96] - 4 * g[i]; s += lap; s2 += lap * lap; n++; }
  frameQuality = { brightness: sum / g.length, sharpness: s2 / n - (s / n) ** 2 };
}
function detect() {
  if (!landmarker || video.readyState < 2 || video.currentTime === lastVideoTime || demoRunning) return;
  lastVideoTime = video.currentTime;
  const now = performance.now(), res = landmarker.detectForVideo(video, now);
  if (seq % 10 === 0) measureQuality();
  const handed = res.handednesses || res.handedness || [];
  const wl = res.worldLandmarks || [];
  const hands = res.landmarks.map((lm, i) => ({ landmarks: lm.map((p) => [p.x, p.y, p.z]),
    world: wl[i] ? wl[i].map((p) => [p.x, p.y, p.z]) : undefined,   // metric 3D: tilt/distance-proof shape
    handedness: handed[i] && handed[i][0] ? handed[i][0].categoryName : "Right", score: handed[i] && handed[i][0] ? handed[i][0].score : 1 }));
  lastRaw = hands;
  if (inflight < 3) { send({ type: "frame", t: now, seq, hands, mirrored: false, ...frameQuality }); inflight++; }
  seq++; fpsCount++;
  if (now - fpsT > 1000) { $("chip-fps").textContent = `${fpsCount} fps`; fpsCount = 0; fpsT = now; }
}
const STATE_COLORS = { active: "#3ecf8e", arming: "#f5b83d", number: "#f5b83d", paused: "#ff6b6b", idle: "#cfd6df", no_hand: "#cfd6df" };
function drawCamera() {
  const w = cam.clientWidth, h = cam.clientHeight;
  if (cam.width !== w) { cam.width = w; cam.height = h; }
  cctx.fillStyle = "#000"; cctx.fillRect(0, 0, w, h);
  let hands = [];
  if (demoRunning && demoLandmarks) hands = demoLandmarks;
  else if (landmarker && video.readyState >= 2) {
    cctx.save(); cctx.translate(w, 0); cctx.scale(-1, 1); cctx.drawImage(video, 0, 0, w, h); cctx.restore();
    hands = (lastRaw || []).map((hd) => hd.landmarks.map((p) => [1 - p[0], p[1], p[2]]));
  }
  cctx.lineWidth = 3; cctx.strokeStyle = cctx.fillStyle = STATE_COLORS[lastHud ? lastHud.state : "idle"] || "#cfd6df";
  for (const lm of hands) {
    for (const [a, b] of HAND_EDGES) { cctx.beginPath(); cctx.moveTo(lm[a][0] * w, lm[a][1] * h); cctx.lineTo(lm[b][0] * w, lm[b][1] * h); cctx.stroke(); }
    for (const p of lm) { cctx.beginPath(); cctx.arc(p[0] * w, p[1] * h, 3, 0, 7); cctx.fill(); }
  }
  // what the engine thinks each finger is doing - makes a misread obvious at a glance
  const fingers = lastHud && lastHud.fingers ? Object.entries(lastHud.fingers) : [];
  fingers.forEach(([hand, up], i) => {
    const x0 = 8, y0 = 10 + i * 26;
    cctx.fillStyle = "rgba(0,0,0,.65)"; cctx.fillRect(x0 - 4, y0 - 4, 196, 22);
    cctx.font = "bold 12px system-ui"; cctx.fillStyle = "#e6e9ee"; cctx.fillText(hand[0], x0, y0 + 11);
    "TIMRP".split("").forEach((ch, k) => {
      const x = x0 + 18 + k * 22; cctx.fillStyle = up[k] ? "#3ecf8e" : "#3a4350";
      cctx.beginPath(); cctx.arc(x + 6, y0 + 7, 7, 0, 7); cctx.fill();
      cctx.fillStyle = up[k] ? "#04150d" : "#8b95a3"; cctx.font = "bold 9px system-ui"; cctx.fillText(ch, x + 3, y0 + 10);
    });
    const pose = (lastHud.hands || []).find((hh) => hh.handedness === hand);
    cctx.fillStyle = "#e6e9ee"; cctx.font = "11px system-ui";
    cctx.fillText(`${up.filter(Boolean).length} · ${pose ? pose.pose.replace("_", " ") : ""}`, x0 + 132, y0 + 11);
  });
  if (lastHud && lastHud.progress > 0 && !lastHud.mode_label && hands.length) {
    const p = hands[0][0]; cctx.beginPath(); cctx.lineWidth = 5; cctx.strokeStyle = "#f5b83d";
    cctx.arc(p[0] * w, p[1] * h, 26, -Math.PI / 2, -Math.PI / 2 + lastHud.progress * 2 * Math.PI); cctx.stroke();
  }
}

// ---------------------------------------------------------------------------
// mouse + keyboard
let dragging = false, moved = false;
const el = renderer.domElement;
const mouseXY = (e) => { const r = el.getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; };
el.addEventListener("pointerdown", (e) => { dragging = true; moved = false; el.setPointerCapture(e.pointerId); });
el.addEventListener("pointerup", (e) => {
  dragging = false;
  if (moved) return;
  const xy = mouseXY(e);
  if (S.phase === "loads" && S.tool && S.model) placeTool(screenToT(xy));
  else if (S.phase === "results") command("probe_pin", { xy });
});
el.addEventListener("pointermove", (e) => {
  S.mouseXY = mouseXY(e);
  if (dragging && Math.abs(e.movementX) + Math.abs(e.movementY) > 0) { moved = true; command("orbit", { dx: e.movementX * 0.4, dy: e.movementY * 0.4 }); }
});
el.addEventListener("pointerleave", () => { S.mouseXY = null; });
el.addEventListener("wheel", (e) => { e.preventDefault(); command("zoom", { factor: Math.exp(e.deltaY * 0.001) }); }, { passive: false });
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  const k = e.key.toLowerCase();
  if (k === "enter" && (e.ctrlKey || e.metaKey)) send({ type: "solve" });
  else if (k === "m") send({ type: "set_phase", phase: "model" });
  else if (k === "l") send({ type: "set_phase", phase: "loads" });
  else if (k === "v") send({ type: "set_phase", phase: "results" });
  else if (k === "r") command("reset");
  else if (k === "u" || (k === "z" && (e.ctrlKey || e.metaKey))) command("undo");
  else if (k === "x") command("section_off");
  else if (k === "s") command("snapshot");
  else if (k === "c") calibrate();
  else if (k === "d") startTour();
  else if (k === "escape") { S.tool = null; S.toolT0 = null; renderTabs(); send({ type: "calibrate_cancel" }); $("calib").classList.add("hidden"); send({ type: "demo_stop" }); }
});
function calibrate() { if (!landmarker) { toast("Enable the camera first", true); return; } send({ type: "calibrate", with_poses: true }); }
function startTour() { demoRunning = true; $("start").classList.add("hidden"); send({ type: "demo" }); }

for (const b of document.querySelectorAll("[data-demo]")) b.onclick = () => send({ type: "load_demo", name: b.dataset.demo });
for (const b of document.querySelectorAll("#tabs button")) b.onclick = () => send({ type: "set_phase", phase: b.dataset.phase });
$("sel-field").onchange = (e) => command("set_field", { name: e.target.value });
$("rng-warp").oninput = (e) => { $("warp-val").textContent = `×${(+e.target.value).toFixed(2)}`; updatePositions(performance.now()); renderLegend(); };
$("chk-mesh").onchange = syncVisibility; $("chk-undeformed").onchange = syncVisibility;
$("btn-clear-loads").onclick = () => send({ type: "clear", what: "loads" });
$("btn-clear-sup").onclick = () => send({ type: "clear", what: "supports" });
$("btn-undo-model").onclick = () => send({ type: "undo_model" });
$("btn-camera").onclick = startCamera;
$("btn-skip").onclick = () => { $("start").classList.add("hidden"); $("mode-text").textContent = "Mouse & keys"; $("pose-text").textContent = "camera off"; };
$("btn-tour").onclick = startTour;
$("btn-calib").onclick = calibrate;
$("btn-calib-cancel").onclick = () => { send({ type: "calibrate_cancel" }); $("calib").classList.add("hidden"); };
$("btn-undo").onclick = () => command("undo");
$("btn-reset").onclick = () => command("reset");
$("btn-section-off").onclick = () => command("section_off");
$("sel-hand").onchange = (e) => send({ type: "set_dominant", hand: e.target.value });
function fillRecLabels() {
  const sel = $("rec-label"); if (sel.options.length) return;
  for (const l of ["open_palm", "fist", "point", "peace", "thumbs_up", "pinch", "none", "count_1", "count_2", "count_3", "count_4", "count_5",
    "swipe_left", "swipe_right", "circle", "still_palm", "moving_palm"]) sel.add(new Option(l, l));
}
$("btn-rec").onclick = () => {
  const subject = $("rec-subject").value.trim() || "anon", label = $("rec-label").value, s = $("rec-status");
  if (!landmarker) { s.textContent = "Enable the camera first"; return; }
  let n = 3;
  const tick = () => {
    if (n > 0) { s.textContent = `Get ready: ${label} in ${n}…`; n--; setTimeout(tick, 700); return; }
    s.textContent = `Recording ${label}… hold it (vary angle and distance a little)`;
    send({ type: "record_start", label, subject }); setTimeout(() => send({ type: "record_stop" }), 3000);
  };
  tick();
};

// ---------------------------------------------------------------------------
function loop() {
  const now = performance.now();
  detect(); drawCamera();
  applyView();
  if (feMesh && feMesh.visible) { const F = currentField(); if (F && F.vec) updatePositions(now); }
  renderer.render(scene, camera);
  renderGizmo();
  drawOverlay(now);
  requestAnimationFrame(loop);
}
resize();
S.view = { azimuth: -60, elevation: 22, distance: 1, focal_point: [0, 0, 0], section_on: false, plane_origin: [0, 0, 0], plane_normal: [1, 0, 0],
  field_name: "von_mises", probe_cursor: null, probe_pins: [], snapshots: 0 };
window.__archimedes = { S, resolveCommand, screenToT, onView };   // for automated UI tests
connect();
loop();
