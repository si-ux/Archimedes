// Archimedes gesture demo - browser side.
// Sensor (webcam + MediaPipe in WASM) and renderer (three.js). All gesture logic lives in Python.
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);
const HAND_EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

// ---------------------------------------------------------------------------
// Result model: a cantilever beam with closed-form results, standing in for a
// FEniCSx result file. Fixed at x=0, 10 kN tip load at x=L, bending about y.
const L = 4.0, B = 0.3, H = 0.4, P = 10e3, E = 30e9, I = B * H ** 3 / 12, A = B * H;
const DEFORM_SCALE = 40;
const FIELDS = {
  von_mises:     { title: "von Mises stress", unit: "MPa", scale: 1e-6 },
  displacement:  { title: "Displacement", unit: "mm", scale: 1e3 },
  max_principal: { title: "Max principal stress", unit: "MPa", scale: 1e-6 },
  strain:        { title: "Axial strain", unit: "µε", scale: 1e6 },
};
const deflection = (x) => P * x * x * (3 * L - x) / (6 * E * I);
function fieldAt(name, x, z) {           // x along the beam, z height from the neutral axis
  const M = P * (L - x), sig = M * z / I;
  const tau = 1.5 * P / A * (1 - (2 * z / H) ** 2);
  switch (name) {
    case "von_mises": return Math.sqrt(sig * sig + 3 * tau * tau);
    case "displacement": return deflection(x);
    case "max_principal": return sig / 2 + Math.sqrt(sig * sig / 4 + tau * tau);
    case "strain": return sig / E;
  }
}
const RANGES = {};
for (const f of Object.keys(FIELDS)) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i <= 40; i++) for (let j = 0; j <= 20; j++) {
    const v = fieldAt(f, L * i / 40, -H / 2 + H * j / 20); lo = Math.min(lo, v); hi = Math.max(hi, v);
  }
  RANGES[f] = [lo, hi];
}
// blue -> cyan -> green -> yellow -> red, the colour scale FE users expect
const STOPS = [[0.13, 0.25, 0.85], [0.0, 0.75, 0.95], [0.2, 0.8, 0.3], [0.98, 0.85, 0.1], [0.9, 0.15, 0.1]];
function cmap(t) {
  t = Math.min(1, Math.max(0, t)) * (STOPS.length - 1);
  const i = Math.min(Math.floor(t), STOPS.length - 2), f = t - i;
  return STOPS[i].map((a, k) => a + (STOPS[i + 1][k] - a) * f);
}
$("legend-bar").style.background =
  `linear-gradient(90deg, ${STOPS.map((c) => `rgb(${c.map((v) => v * 255).join(",")})`).join(",")})`;

// ---------------------------------------------------------------------------
// three.js scene (z is up, to match ViewState)
const host = $("three");
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.localClippingEnabled = true;
renderer.setClearColor(0x0f1216);
host.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 100);
camera.up.set(0, 0, 1);
scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const sun = new THREE.DirectionalLight(0xffffff, 0.9); sun.position.set(3, -4, 6); scene.add(sun);
const CENTER = new THREE.Vector3(L / 2, 0, 0), D0 = 6.0, MODEL_SCALE = 2.0;

const geo = new THREE.BoxGeometry(L, B, H, 80, 6, 12);
geo.translate(L / 2, 0, 0);
const rest = geo.attributes.position.array.slice();
for (let i = 0; i < rest.length; i += 3) geo.attributes.position.array[i + 2] -= deflection(rest[i]) * DEFORM_SCALE;
geo.computeVertexNormals();
geo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(rest.length), 3));
const clipPlane = new THREE.Plane(new THREE.Vector3(1, 0, 0), 1e3);
const beamMat = new THREE.MeshLambertMaterial({ vertexColors: true, clippingPlanes: [clipPlane], side: THREE.DoubleSide });
const beam = new THREE.Mesh(geo, beamMat);
scene.add(beam);
// undeformed outline + support block for context
scene.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(L, B, H).translate(L / 2, 0, 0)),
  new THREE.LineBasicMaterial({ color: 0x3a4350 })));
const wall = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.8, 1.0), new THREE.MeshLambertMaterial({ color: 0x555d68 }));
wall.position.set(-0.05, 0, 0); scene.add(wall);

// Section cap: a plane whose fragment shader evaluates the same closed-form field,
// so the cut face shows the stress *inside* the beam, not a hollow shell.
const capMat = new THREE.ShaderMaterial({
  side: THREE.DoubleSide,
  uniforms: { field: { value: 0 }, lo: { value: 0 }, hi: { value: 1 } },
  vertexShader: `varying vec3 wp; void main(){ vec4 w = modelMatrix * vec4(position,1.0); wp = w.xyz;
    gl_Position = projectionMatrix * viewMatrix * w; }`,
  fragmentShader: `varying vec3 wp; uniform int field; uniform float lo, hi;
    const float L=${L.toFixed(3)}, B=${B.toFixed(3)}, H=${H.toFixed(3)}, P=${P.toFixed(1)}, E=${E.toExponential(3)};
    const float I=${I.toExponential(6)}, A=${A.toFixed(4)}, S=${DEFORM_SCALE.toFixed(1)};
    vec3 cm(float t){ t=clamp(t,0.,1.)*4.; vec3 c[5];
      c[0]=vec3(.13,.25,.85); c[1]=vec3(0.,.75,.95); c[2]=vec3(.2,.8,.3); c[3]=vec3(.98,.85,.1); c[4]=vec3(.9,.15,.1);
      int i=int(min(floor(t),3.)); float f=t-float(i);
      vec3 a=c[0],b=c[1]; for(int k=0;k<4;k++){ if(k==i){ a=c[k]; b=c[k+1]; } } return mix(a,b,f); }
    void main(){ float x=wp.x; float d=P*x*x*(3.*L-x)/(6.*E*I); float z=wp.z+d*S;
      if(x<0.||x>L||abs(wp.y)>B/2.||abs(z)>H/2.) discard;
      float M=P*(L-x), sig=M*z/I, tau=1.5*P/A*(1.-pow(2.*z/H,2.)), v;
      if(field==0) v=sqrt(sig*sig+3.*tau*tau); else if(field==1) v=d;
      else if(field==2) v=sig/2.+sqrt(sig*sig/4.+tau*tau); else v=sig/E;
      gl_FragColor=vec4(cm((v-lo)/(hi-lo)),1.); }`,
});
const cap = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), capMat);
cap.visible = false; scene.add(cap);
const planeGlow = new THREE.Mesh(new THREE.PlaneGeometry(3.0, 1.2),
  new THREE.MeshBasicMaterial({ color: 0x4da3ff, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }));
planeGlow.visible = false; scene.add(planeGlow);

let currentField = null;
function paintField(name) {
  if (name === currentField) return;
  currentField = name;
  const [lo, hi] = RANGES[name], col = geo.attributes.color.array;
  for (let i = 0; i < rest.length; i += 3) {
    const c = cmap((fieldAt(name, rest[i], rest[i + 2]) - lo) / (hi - lo));
    col[i] = c[0]; col[i + 1] = c[1]; col[i + 2] = c[2];
  }
  geo.attributes.color.needsUpdate = true;
  capMat.uniforms.field.value = Object.keys(FIELDS).indexOf(name);
  capMat.uniforms.lo.value = lo; capMat.uniforms.hi.value = hi;
  const F = FIELDS[name];
  $("legend-title").textContent = F.title;
  $("legend-min").textContent = `${(lo * F.scale).toFixed(2)} ${F.unit}`;
  $("legend-max").textContent = `${(hi * F.scale).toFixed(2)} ${F.unit}`;
}

function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

// ---------------------------------------------------------------------------
// Apply ViewState (the only thing that moves the scene)
let view = null, lastSnapshots = 0;
function applyView(v) {
  view = v;
  const az = THREE.MathUtils.degToRad(v.azimuth), el = THREE.MathUtils.degToRad(v.elevation);
  const back = new THREE.Vector3(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el));
  const focal = CENTER.clone().add(new THREE.Vector3(...v.focal_point).multiplyScalar(MODEL_SCALE));
  camera.position.copy(focal).addScaledVector(back, D0 * v.distance);
  camera.lookAt(focal);
  paintField(v.field_name);
  if (v.section_on) {
    const n = new THREE.Vector3(...v.plane_normal).normalize();
    const o = CENTER.clone().add(new THREE.Vector3(...v.plane_origin).multiplyScalar(MODEL_SCALE));
    // keep the half the palm is facing away from: the cut opens towards the user
    clipPlane.set(n.clone().negate(), n.dot(o));
    for (const m of [cap, planeGlow]) {
      m.visible = true; m.position.copy(o);
      m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
    }
    planeGlow.material.opacity = v.section_locked ? 0.0 : 0.14;
  } else {
    clipPlane.set(new THREE.Vector3(1, 0, 0), 1e3);
    cap.visible = planeGlow.visible = false;
  }
  if (v.snapshots > lastSnapshots && lastSnapshots !== null) saveSnapshot();
  lastSnapshots = v.snapshots;
  renderPins();
}

// ---------------------------------------------------------------------------
// Probe: ray-pick the beam (respecting the cut) or the cap, read the field there
const ray = new THREE.Raycaster();
function pick(xy) {
  ray.setFromCamera(new THREE.Vector2(xy[0] * 2 - 1, -(xy[1] * 2 - 1)), camera);
  const hits = [];
  for (const h of ray.intersectObject(beam)) if (clipPlane.distanceToPoint(h.point) >= -1e-4) { hits.push(h); break; }
  if (cap.visible) for (const h of ray.intersectObject(cap)) {
    const z = h.point.z + deflection(h.point.x) * DEFORM_SCALE;
    if (h.point.x >= 0 && h.point.x <= L && Math.abs(h.point.y) <= B / 2 && Math.abs(z) <= H / 2) hits.push(h);
  }
  if (!hits.length) return null;
  hits.sort((a, b) => a.distance - b.distance);
  const p = hits[0].point, z = p.z + deflection(p.x) * DEFORM_SCALE;
  const F = FIELDS[view.field_name], v = fieldAt(view.field_name, p.x, Math.max(-H / 2, Math.min(H / 2, z)));
  return { point: p.clone(), text: `${F.title}: ${(v * F.scale).toFixed(2)} ${F.unit}  @ x=${p.x.toFixed(2)} m` };
}
function toScreen(p) {
  const v = p.clone().project(camera);
  return [(v.x + 1) / 2 * host.clientWidth, (1 - v.y) / 2 * host.clientHeight];
}
const pinCache = new Map();
function renderPins() {
  const box = $("pins");
  if (!view) return;
  const keys = view.probe_pins.map((p) => `${p[0].toFixed(4)},${p[1].toFixed(4)},${view.field_name}`);
  for (const k of [...pinCache.keys()]) if (!keys.includes(k)) pinCache.delete(k);
  box.innerHTML = "";
  view.probe_pins.forEach((xy, i) => {
    if (!pinCache.has(keys[i])) pinCache.set(keys[i], pick(xy));
    const r = pinCache.get(keys[i]);
    if (!r) return;
    const [sx, sy] = toScreen(r.point), el = document.createElement("div");
    el.className = "pin"; el.style.left = `${sx}px`; el.style.top = `${sy}px`;
    el.textContent = r.text.split("  @")[0]; box.appendChild(el);
  });
}
function updateProbe() {
  const tip = $("probe-tip");
  const xy = mouseProbe || (view && view.probe_cursor);
  if (!xy) { tip.classList.add("hidden"); return; }
  const r = pick(xy);
  tip.classList.remove("hidden");
  tip.style.left = `${xy[0] * host.clientWidth}px`; tip.style.top = `${xy[1] * host.clientHeight}px`;
  tip.textContent = r ? r.text : "— (not on the model)";
}

function saveSnapshot() {
  renderer.render(scene, camera);
  const a = document.createElement("a");
  a.href = renderer.domElement.toDataURL("image/png");
  a.download = `archimedes-${Date.now()}.png`; a.click();
}

// ---------------------------------------------------------------------------
// WebSocket to the Python engine
let ws, info = null;
function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connect, 1000);
}
const send = (o) => ws && ws.readyState === 1 && ws.send(JSON.stringify(o));
const command = (kind, data = {}) => send({ type: "command", kind, data });

let inflight = 0, lastHud = null, demoLandmarks = null, demoRunning = false;
function onMessage(m) {
  if (m.type === "info") { info = m; renderInfo(); return; }
  if (m.type === "recorded") {
    $("rec-status").textContent = m.path ? `Saved ${m.path}` : "Nothing saved - no hand was visible";
    return;
  }
  if (m.type === "demo_done") { demoRunning = false; $("demo-caption").classList.add("hidden"); demoLandmarks = null; return; }
  if (m.view) applyView(m.view);
  if (m.type !== "state") return;
  inflight = Math.max(0, inflight - 1);
  if (m.demo) { $("demo-caption").textContent = m.demo; $("demo-caption").classList.remove("hidden"); demoLandmarks = m.landmarks; }
  renderCalibration(m.calibration);
  if (m.hud) renderHud(m.hud);
}

function renderInfo() {
  const ul = $("cheat"); ul.innerHTML = "";
  for (const row of info.cheat_sheet) {
    const li = document.createElement("li"); li.dataset.label = row.label;
    li.innerHTML = `<span class="ic">${row.icon}</span><span><span class="lb">${row.label}</span><br><span class="muted small">${row.hint}</span></span>`;
    ul.appendChild(li);
  }
  $("chip-model").textContent = `model: ${info.classifier}`;
  const n = Object.values(info.personal_samples).reduce((a, b) => a + b, 0);
  $("chip-personal").textContent = `${n} personal samples`;
  $("sel-hand").value = info.profile.dominant;
  const sel = $("rec-label");
  if (!sel.options.length) for (const l of ["open_palm", "fist", "point", "peace", "thumbs_up", "pinch", "none",
    "swipe_left", "swipe_right", "push", "pull", "still_palm", "moving_palm"]) sel.add(new Option(l, l));
}

const ICONS = { orbit: "✊", zoom: "🤏", section: "✋", probe: "☝️", field: "✌️", snapshot: "👍" };
let toastTimer;
function renderHud(h) {
  lastHud = h;
  const badge = $("mode-badge");
  badge.className = `badge ${h.state}`;
  $("mode-icon").textContent = h.mode ? ICONS[h.mode] : h.state === "paused" ? "!" : h.state === "no_hand" ? "·" : "○";
  $("ring-fg").style.strokeDashoffset = String(119.4 * (1 - (h.mode ? 1 : h.progress)));
  $("mode-text").textContent = h.mode_label ? h.mode_label
    : h.progress_label ? `${h.progress_label}…` : { no_hand: "No hand", idle: "Ready", paused: "Paused", arming: "…" }[h.state];
  $("pose-text").textContent = `pose: ${h.pose.replace("_", " ")} · ${(h.confidence * 100).toFixed(0)}%`;
  const hint = $("hint");
  if (h.hints.length) { hint.textContent = h.hints[0]; hint.classList.remove("hidden"); hint.classList.toggle("soft", h.state !== "paused"); }
  else hint.classList.add("hidden");
  $("tip").textContent = h.tip;
  if (h.toast) {
    const t = $("toast"); t.textContent = h.toast; t.classList.remove("hidden");
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add("hidden"), 1500);
  }
  $("chip-lat").textContent = `${h.latency_ms.toFixed(1)} ms engine`;
  $("chip-personal").textContent = `${h.personal_samples} personal samples`;
  for (const li of $("cheat").children) {
    li.classList.toggle("active", li.dataset.label === h.mode_label);
    li.classList.toggle("arming", !h.mode && li.dataset.label === h.progress_label);
  }
}

function renderCalibration(c) {
  const box = $("calib");
  if (!c || c.done) {
    if (!box.classList.contains("hidden") && c && c.done) { send({ type: "info" }); showToast("Calibration saved ✓"); }
    box.classList.add("hidden"); return;
  }
  box.classList.remove("hidden");
  $("calib-step").textContent = `Step ${c.step_index + 1} of ${c.step_count}`;
  $("calib-text").textContent = c.instruction;
  $("calib-bar").style.width = `${c.progress * 100}%`;
  $("calib-warn").textContent = c.warning || "";
}
function showToast(text) {
  const t = $("toast"); t.textContent = text; t.classList.remove("hidden");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.add("hidden"), 1800);
}

// ---------------------------------------------------------------------------
// Camera + MediaPipe
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
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO", numHands: 2,
      minHandDetectionConfidence: 0.6, minHandPresenceConfidence: 0.6, minTrackingConfidence: 0.5,
    });
    $("cam-label").textContent = "camera on";
    $("start").classList.add("hidden");
  } catch (err) {
    const e = $("start-error");
    e.textContent = `Camera unavailable (${err.name || err}). Allow camera access in the browser, or use the demo.`;
    e.classList.remove("hidden");
  }
}

// cheap image-quality numbers: mean luma and variance of a Laplacian (blur)
function measureQuality() {
  pctx.drawImage(video, 0, 0, 96, 72);
  const d = pctx.getImageData(0, 0, 96, 72).data, g = new Float32Array(96 * 72);
  let sum = 0;
  for (let i = 0; i < g.length; i++) { g[i] = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]; sum += g[i]; }
  let s = 0, s2 = 0, n = 0;
  for (let y = 1; y < 71; y++) for (let x = 1; x < 95; x++) {
    const i = y * 96 + x, lap = g[i - 1] + g[i + 1] + g[i - 96] + g[i + 96] - 4 * g[i];
    s += lap; s2 += lap * lap; n++;
  }
  frameQuality = { brightness: sum / g.length, sharpness: s2 / n - (s / n) ** 2 };
}

function detect() {
  if (!landmarker || video.readyState < 2 || video.currentTime === lastVideoTime || demoRunning) return;
  lastVideoTime = video.currentTime;
  const now = performance.now();
  const res = landmarker.detectForVideo(video, now);
  if (seq % 10 === 0) measureQuality();
  const handed = res.handednesses || res.handedness || [];
  const hands = res.landmarks.map((lm, i) => ({
    landmarks: lm.map((p) => [p.x, p.y, p.z]),
    handedness: handed[i] && handed[i][0] ? handed[i][0].categoryName : "Right",
    score: handed[i] && handed[i][0] ? handed[i][0].score : 1,
  }));
  lastRaw = hands;
  if (inflight < 3) { send({ type: "frame", t: now, seq: seq, hands, mirrored: false, ...frameQuality }); inflight++; }
  seq++; fpsCount++;
  if (now - fpsT > 1000) { $("chip-fps").textContent = `${fpsCount} fps`; fpsCount = 0; fpsT = now; }
}

const STATE_COLORS = { active: "#3ecf8e", arming: "#f5b83d", paused: "#ff6b6b", idle: "#cfd6df", no_hand: "#cfd6df" };
function drawCamera() {
  const w = cam.clientWidth, h = cam.clientHeight;
  if (cam.width !== w) { cam.width = w; cam.height = h; }
  cctx.fillStyle = "#000"; cctx.fillRect(0, 0, w, h);
  let hands = [];
  if (demoRunning && demoLandmarks) hands = demoLandmarks;                    // already selfie view
  else if (landmarker && video.readyState >= 2) {
    cctx.save(); cctx.translate(w, 0); cctx.scale(-1, 1); cctx.drawImage(video, 0, 0, w, h); cctx.restore();
    hands = (lastRaw || []).map((hd) => hd.landmarks.map((p) => [1 - p[0], p[1], p[2]]));   // mirror for display
  }
  cctx.lineWidth = 3; cctx.strokeStyle = cctx.fillStyle = STATE_COLORS[lastHud ? lastHud.state : "idle"];
  for (const lm of hands) {
    for (const [a, b] of HAND_EDGES) {
      cctx.beginPath(); cctx.moveTo(lm[a][0] * w, lm[a][1] * h); cctx.lineTo(lm[b][0] * w, lm[b][1] * h); cctx.stroke();
    }
    for (const p of lm) { cctx.beginPath(); cctx.arc(p[0] * w, p[1] * h, 3, 0, 7); cctx.fill(); }
  }
  if (lastHud && lastHud.progress > 0 && !lastHud.mode && hands.length) {   // dwell ring around the wrist
    const p = hands[0][0];
    cctx.beginPath(); cctx.lineWidth = 5; cctx.strokeStyle = "#f5b83d";
    cctx.arc(p[0] * w, p[1] * h, 26, -Math.PI / 2, -Math.PI / 2 + lastHud.progress * 2 * Math.PI); cctx.stroke();
  }
}

// ---------------------------------------------------------------------------
// Mouse + keyboard fallback (also the baseline condition for the user study)
let mouseProbe = null, dragging = false, moved = false, lastProbeSend = 0;
const el = renderer.domElement;
el.addEventListener("pointerdown", (e) => { dragging = true; moved = false; el.setPointerCapture(e.pointerId); });
el.addEventListener("pointerup", (e) => {
  dragging = false;
  if (!moved && mouseProbe) command("probe_pin", { xy: mouseProbe });
});
el.addEventListener("pointermove", (e) => {
  const r = el.getBoundingClientRect();
  mouseProbe = [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
  if (dragging && (Math.abs(e.movementX) + Math.abs(e.movementY) > 0)) {
    moved = true; mouseProbe = null;
    command("orbit", { dx: e.movementX * 0.4, dy: e.movementY * 0.4 });
  }
});
el.addEventListener("pointerleave", () => { mouseProbe = null; });
el.addEventListener("wheel", (e) => { e.preventDefault(); command("zoom", { factor: Math.exp(e.deltaY * 0.001) }); }, { passive: false });

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  const k = e.key.toLowerCase();
  if ("1234".includes(k) && view) command("field", { step: Number(k) - 1 - view.field_index });
  else if (k === "r") command("reset");
  else if (k === "u" || (k === "z" && (e.ctrlKey || e.metaKey))) command("undo");
  else if (k === "x") command("section_off");
  else if (k === "s") command("snapshot");
  else if (k === "c") calibrate();
  else if (k === "d") startDemo();
  else if (k === "escape") { send({ type: "calibrate_cancel" }); $("calib").classList.add("hidden"); send({ type: "demo_stop" }); }
});

function calibrate() {
  if (!landmarker) { showToast("Enable the camera first"); return; }
  send({ type: "calibrate", with_poses: true });
}
function startDemo() {
  demoRunning = true; $("start").classList.add("hidden"); send({ type: "demo" });
}

$("btn-camera").onclick = startCamera;
$("btn-demo-start").onclick = startDemo;
$("btn-demo").onclick = startDemo;
$("btn-calib").onclick = calibrate;
$("btn-calib-cancel").onclick = () => { send({ type: "calibrate_cancel" }); $("calib").classList.add("hidden"); };
$("btn-undo").onclick = () => command("undo");
$("btn-reset").onclick = () => command("reset");
$("btn-section-off").onclick = () => command("section_off");
$("sel-hand").onchange = (e) => send({ type: "set_dominant", hand: e.target.value });
$("btn-rec").onclick = () => {
  const subject = $("rec-subject").value.trim() || "anon", label = $("rec-label").value;
  if (!landmarker) { $("rec-status").textContent = "Enable the camera first"; return; }
  let n = 3; const s = $("rec-status");
  const tick = () => {
    if (n > 0) { s.textContent = `Get ready: ${label} in ${n}…`; n--; setTimeout(tick, 700); return; }
    s.textContent = `Recording ${label}… hold it (vary angle and distance a little)`;
    send({ type: "record_start", label, subject });
    setTimeout(() => send({ type: "record_stop" }), 3000);
  };
  tick();
};

// ---------------------------------------------------------------------------
function loop() {
  detect();
  drawCamera();
  updateProbe();
  if (view) renderPins();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}
resize();
applyView({ azimuth: 30, elevation: 20, distance: 1, focal_point: [0, 0, 0], section_on: false, section_locked: false,
  plane_origin: [0, 0, 0], plane_normal: [1, 0, 0], field_index: 0, field_name: "von_mises", probe_cursor: null,
  probe_pins: [], snapshots: 0 });
connect();
loop();
