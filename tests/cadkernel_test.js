// Unit tests for Archimedes/cadkernel.js. Run: node tests/cadkernel_test.js (pytest runs it too).
const assert = require('assert');
const K = require('../Archimedes/cadkernel.js');
let n = 0;
const t = (name, fn) => { fn(); n++; console.log('ok', name); };

t('bracket template: inside, fillet, faces', () => {
  const m = K.TEMPLATES.bracket.make(), cm = K.compile(m);
  assert.deepStrictEqual(cm.bbox, [[0, 0, 0], [120, 180, 60]]);
  assert(cm.inside([10, 90, 30]) && cm.inside([80, 170, 30]) && !cm.inside([80, 90, 30]));
  assert(cm.inside([31, 149, 30]), 'concave fillet adds material in the corner');
  assert(!cm.inside([39.5, 140.5, 30]), 'but not past its arc');
  const f = (p, nn) => cm.faceAt(p, nn, 6);
  assert.strictEqual(f([15, 0, 30], [0, -1, 0]), 'extrude1/e0.l0');
  assert.strictEqual(f([110, 180, 30], [0, 1, 0]), 'extrude1/e0.l4');
  assert.strictEqual(f([60, 180, 30], [0, 1, 0]), 'extrude1/e0.l5');
  assert.strictEqual(f([60, 170, 60], [0, 0, 1]), 'extrude1/cap1');
  assert.strictEqual(f([0, 90, 30], [-1, 0, 0]), 'extrude1/e0.l6');
  assert.strictEqual(f([120, 165, 30], [1, 0, 0]), 'extrude1/e0.l3');
  assert.strictEqual(f([33, 147, 30], [1, 0, 0]), 'extrude1/e0.a2');
  assert(cm.faces['extrude1/e0.a2'].name.includes('fillet'));
});

t('voxel mesh matches the old bracket mesher', () => {
  const cm = K.compile(K.TEMPLATES.bracket.make());
  const v = K.voxelize(cm, { h: 6, budget: 1e9, sub: 1 });
  assert.deepStrictEqual([v.nx, v.ny, v.nz], [20, 30, 10]);
  assert.strictEqual(v.ne, 2250);
});

t('sketch holes, rounded rectangles', () => {
  const cm = K.compile(K.TEMPLATES.plate.make());
  assert(!cm.inside([120, 50, 5]) && cm.inside([10, 10, 5]) && !cm.inside([10, 10, 11]));
  assert.strictEqual(cm.faceAt([140, 50, 5], [-1, 0, 0], 2), 'extrude1/e1.c');
  const P = K.compileProfile([{ type: 'rect', x: 0, y: 0, w: 100, h: 50, r: 10 }]);
  assert(!K.inside2D(P, 1, 1) && K.inside2D(P, 10, 10) && K.inside2D(P, 50, 1));
});

t('revolve, hole and circular pattern', () => {
  const cm = K.compile(K.TEMPLATES.shaft.make());
  assert(cm.inside([0, 100, 0]) && !cm.inside([50, 100, 0]) && cm.inside([70, 10, 0]));
  assert(!cm.inside([60, 10, 0]), 'hole');
  const a = Math.PI / 3;
  for (const s of [1, -1]) assert(!cm.inside([60 * Math.cos(a), 10, s * 60 * Math.sin(a)]), 'pattern copy at ±60°');
  assert(cm.inside([60 * Math.cos(a / 2), 10, 60 * Math.sin(a / 2)]), 'solid between the holes');
  assert(!cm.inside([-60, 10, 0]), 'copy at 180°');
  const wall = cm.faceAt([30, 100, 0], [1, 0, 0], 2);
  assert.strictEqual(wall, 'revolve1/e0.l3');
});

t('box, blind hole, linear pattern, mirror', () => {
  const m = { features: [
    { id: 'box1', type: 'box', op: 'boss', x: 0, y: 0, z: 0, sx: 100, sy: 20, sz: 40 },
    { id: 'hole1', type: 'hole', x: 10, y: 20, z: 20, dir: '-y', d: 8, depth: 10, through: false },
    { id: 'lp1', type: 'lpattern', src: 'hole1', dir: '+x', count: 3, spacing: 20 },
    { id: 'cyl1', type: 'cylinder', op: 'boss', x: 90, y: 20, z: 20, axis: '+y', r: 5, h: 10 },
    { id: 'mir1', type: 'mirror', src: 'cyl1', plane: 'YZ', offset: 50 }] };
  const cm = K.compile(m);
  for (const x of [10, 30, 50]) { assert(!cm.inside([x, 15, 20]), 'hole at ' + x); assert(cm.inside([x, 5, 20]), 'blind'); }
  assert(cm.inside([70, 15, 20]));
  assert(cm.inside([90, 25, 20]) && cm.inside([10, 25, 20]), 'mirrored boss');
  assert.deepStrictEqual(cm.bbox[1], [100, 30, 40]);
  assert.strictEqual(cm.faceAt([100, 10, 20], [1, 0, 0], 2), 'box1/+x');
  assert(String(cm.faceAt([30 + 4, 15, 20], [-1, 0, 0], 1)).startsWith('lp1#1/'), 'pattern copies name their faces');
});

t('STL import: parity inside test and facet groups', () => {
  const v = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10]];
  const q = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]];
  let txt = 'solid cube\n';
  for (const f of q) for (const tr of [[f[0], f[1], f[2]], [f[0], f[2], f[3]]])
    txt += 'facet normal 0 0 0\nouter loop\n' + tr.map(i => 'vertex ' + v[i].join(' ')).join('\n') + '\nendloop\nendfacet\n';
  const tris = K.parseSTL(txt);
  assert.strictEqual(tris.length, 12 * 9);
  const cm = K.compile({ features: [{ id: 'stl1', type: 'stl', op: 'boss', data: K.b64(tris), scale: 2, tx: 5, ty: 0, tz: 0 }] });
  assert(cm.inside([15, 10, 10]) && !cm.inside([30, 10, 10]) && !cm.inside([2, 10, 10]));
  assert.strictEqual(Object.keys(cm.faces).length, 6);
  assert.notStrictEqual(cm.faceAt([25, 10, 10], [1, 0, 0], 1), cm.faceAt([15, 20, 10], [0, 1, 0], 1));
});

t('Instant3D edits move sketch geometry', () => {
  const m = K.TEMPLATES.bracket.make(), cm = K.compile(m), sk = m.features[0];
  const seg = cm.steps[0].body.profile.segs.find(s => s.id === 'e0.l3');
  const sk2 = K.moveSegment(sk, seg, 10);
  assert.deepStrictEqual([sk2.entities[0].pts[3][0], sk2.entities[0].pts[4][0]], [130, 130]);
  assert.strictEqual(sk.entities[0].pts[3][0], 120, 'original untouched');
  const arc = cm.steps[0].body.profile.segs.find(s => s.id === 'e0.a2');
  assert.strictEqual(K.moveSegment(sk, arc, -5).entities[0].fillets[2], 15);
  assert.strictEqual(K.segValue(seg).value, 120);
});

t('empty model and defaults', () => {
  const cm = K.compile({ features: [] });
  assert(cm.empty && !cm.inside([1, 1, 1]));
  const v = K.voxelize(cm, { h: 10 });
  assert.strictEqual(v.ne, 0);
  const m = { features: [] }, b = K.featureDefaults('box', m);
  assert.strictEqual(b.id, 'box1'); assert(K.featureFields(b).length > 3);
});
t('edge fillet and chamfer between planes', () => {
  const box = { id: 'box1', type: 'box', op: 'boss', x: 0, y: 0, z: 0, sx: 100, sy: 50, sz: 40 };
  const f = K.compile({ features: [box, { id: 'f1', type: 'fillet', r: 10, edges: [['box1/+x', 'box1/+y']] }] });
  assert(!f.inside([99, 49, 20]) && f.inside([95, 45, 20]) && f.inside([50, 49, 20]) && f.inside([99, 30, 20]));
  assert.strictEqual(f.steps[1].mode, 'remove');
  assert.strictEqual(f.faceAt([97.07, 47.07, 20], [1, 0, 0], 1), 'f1/e0');
  const c = K.compile({ features: [box, { id: 'c1', type: 'chamfer', r: 10, edges: [['box1/+x', 'box1/+y']] }] });
  assert(!c.inside([96, 46, 20]) && c.inside([92, 45, 20]));
});

t('concave fillet adds material in a re-entrant corner', () => {
  const m = K.bracket({ r: 0 });
  m.features.push({ id: 'f1', type: 'fillet', r: 10, edges: [['extrude1/e0.l1', 'extrude1/e0.l2']] });
  const cm = K.compile(m);
  assert.strictEqual(cm.steps[1].mode, 'add');
  assert(cm.inside([31, 149, 30]) && !cm.inside([39.5, 140.5, 30]) && !cm.inside([60, 100, 30]));
});

t('rim fillet on a cylinder, chamfered hole edge', () => {
  const cm = K.compile({ features: [
    { id: 'cyl1', type: 'cylinder', op: 'boss', x: 0, y: 0, z: 0, axis: '+y', r: 20, h: 50 },
    { id: 'f1', type: 'fillet', r: 5, edges: [['cyl1/wall', 'cyl1/cap1']] }] });
  assert(!cm.inside([19.5, 49.5, 0]) && cm.inside([14, 49, 0]) && cm.inside([19.5, 30, 0]));
  const h = K.compile({ features: [
    { id: 'box1', type: 'box', op: 'boss', x: -30, y: 0, z: -30, sx: 60, sy: 20, sz: 60 },
    { id: 'hole1', type: 'hole', x: 0, y: 20, z: 0, dir: '-y', d: 10, through: true },
    { id: 'c1', type: 'chamfer', r: 2, edges: [['box1/+y', 'hole1/wall']] }] });
  assert.strictEqual(h.steps[2].mode, 'remove');
  assert(!h.inside([5.5, 19.5, 0]) && h.inside([5.5, 15, 0]) && h.inside([10, 19.5, 0]));
});

t('shell with an opening', () => {
  const cm = K.compile({ features: [
    { id: 'box1', type: 'box', op: 'boss', x: 0, y: 0, z: 0, sx: 100, sy: 60, sz: 40 },
    { id: 'sh1', type: 'shell', t: 5, open: ['box1/+y'] }] });
  const v = K.voxelize(cm, { h: 2.5, budget: 1e9, sub: 1 });
  const at = (x, y, z) => v.solid[(Math.floor(z / v.dz) * v.ny + Math.floor(y / v.dy)) * v.nx + Math.floor(x / v.dx)];
  assert.strictEqual(at(50, 30, 20), 0, 'hollow');
  assert.strictEqual(at(2, 30, 20), 1, 'side wall');
  assert.strictEqual(at(50, 2, 20), 1, 'floor');
  assert.strictEqual(at(50, 58, 20), 0, 'open top');
  assert.strictEqual(cm.faceAt([50, 5, 20], [0, 1, 0], 2.5), 'sh1/inner');
  const full = 40 * 24 * 16;
  assert(v.ne < 0.5 * full && v.ne > 0.2 * full);
});

t('loft between a square and a circle', () => {
  const cm = K.compile({ features: [
    { id: 's1', type: 'sketch', plane: 'XY', offset: 0, entities: [{ type: 'rect', x: -20, y: -20, w: 40, h: 40 }] },
    { id: 's2', type: 'sketch', plane: 'XY', offset: 100, entities: [{ type: 'circle', cx: 0, cy: 0, r: 10 }] },
    { id: 'l1', type: 'loft', op: 'boss', sketch: 's1', sketch2: 's2' }] });
  assert(cm.inside([19, 19, 1]) && !cm.inside([19, 19, 99]) && cm.inside([5, 5, 99]) && !cm.inside([0, 0, 101]));
  assert(cm.inside([0, 14, 50]) && !cm.inside([0, 16.5, 50]), 'half way: about 15');
});

t('sweep a circle along a filleted path', () => {
  const cm = K.compile({ features: [
    { id: 'pr', type: 'sketch', plane: 'YZ', offset: 0, entities: [{ type: 'circle', cx: 0, cy: 0, r: 5 }] },
    { id: 'pa', type: 'sketch', plane: 'XY', offset: 0, entities: [{ type: 'poly', open: true, pts: [[0, 0], [100, 0], [100, 100]], fillets: [0, 30, 0] }] },
    { id: 'sw', type: 'sweep', op: 'boss', sketch: 'pr', path: 'pa' }] });
  assert(cm.inside([50, 0, 0]) && !cm.inside([50, 6, 0]) && cm.inside([50, 0, 4]) && cm.inside([100, 60, 0]) && !cm.inside([94, 60, 0]));
  const c = [70, 30], a = -Math.PI / 4, q = [c[0] + 30 * Math.cos(a), c[1] + 30 * Math.sin(a)];
  assert(cm.inside([q[0], q[1], 0]) && !cm.inside([100, 0, 0]), 'follows the fillet, not the sharp corner');
  assert(!cm.inside([-2, 0, 0]) && !cm.inside([100, 102, 0]), 'capped ends');
});

t('STL with a missing facet still meshes (three-ray vote)', () => {
  const v = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [0, 0, 10], [10, 0, 10], [10, 10, 10], [0, 10, 10]];
  const q = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]];
  const tris = [];
  for (const f of q) for (const tr of [[f[0], f[1], f[2]], [f[0], f[2], f[3]]]) for (const i of tr) tris.push(...v[i]);
  const holed = Float32Array.from(tris.slice(9));            // drop one facet of the bottom
  const cm = K.compile({ features: [{ id: 'stl1', type: 'stl', op: 'boss', data: K.b64(holed), scale: 1 }] });
  assert(cm.warnings.length === 1 && /open edges/.test(cm.warnings[0]));
  let ins = 0;
  for (let i = 1; i < 10; i += 2) for (let j = 1; j < 10; j += 2) for (let k = 1; k < 10; k += 2) if (cm.inside([i + 0.1, j + 0.2, k + 0.3])) ins++;
  assert(ins >= 120, 'nearly all of the 125 probes inside, got ' + ins);
});
t('sketch constraints: dimensions drive the geometry, DOF count', () => {
  const sk = { entities: [{ type: 'poly', pts: [[0, 0], [98, 3], [101, 52], [-2, 49]], fillets: [0, 0, 0, 0] }], constraints: [] };
  const free = K.solveSketch(sk);
  assert.strictEqual(free.dof, 8);
  sk.constraints = [
    { type: 'fix', a: { e: 0, p: 0 }, value: [0, 0] },
    { type: 'horizontal', a: { e: 0, l: 0 } }, { type: 'horizontal', a: { e: 0, l: 2 } },
    { type: 'vertical', a: { e: 0, l: 1 } }, { type: 'vertical', a: { e: 0, l: 3 } },
    { type: 'length', a: { e: 0, l: 0 }, value: 120 }, { type: 'length', a: { e: 0, l: 1 }, value: 40 }];
  const r = K.solveSketch(sk);
  assert(r.ok, 'converged, residual ' + r.residual);
  assert.strictEqual(r.dof, 0, 'fully defined');
  const P = r.entities[0].pts;
  const near = (a, b) => Math.abs(a - b) < 1e-6;
  assert(near(P[1][0], 120) && near(P[1][1], 0) && near(P[2][0], 120) && near(P[2][1], 40) && near(P[3][0], 0) && near(P[3][1], 40), JSON.stringify(P));
  // change a driving dimension
  sk.entities = r.entities; sk.constraints[5] = { ...sk.constraints[5], value: 150 };
  const r2 = K.solveSketch(sk);
  assert(Math.abs(r2.entities[0].pts[1][0] - 150) < 1e-6 && Math.abs(r2.entities[0].pts[3][1] - 40) < 1e-6);
  // over-constrained is reported, not silently accepted
  sk.constraints.push({ type: 'length', a: { e: 0, l: 2 }, value: 99 });
  assert(K.solveSketch(sk).conflict);
});

t('sketch constraints: circles, tangency, equality, angles, dragging', () => {
  const sk = { entities: [{ type: 'circle', cx: 10, cy: 12, r: 7 }, { type: 'circle', cx: 50, cy: 10, r: 3 },
    { type: 'poly', pts: [[0, 30], [60, 31]], open: true }], constraints: [
    { type: 'equal', a: { e: 0 }, b: { e: 1 } }, { type: 'diameter', a: { e: 0 }, value: 20 },
    { type: 'coincident', a: { e: 0, p: 'c' }, b: { origin: true } }, { type: 'horizontal', a: { e: 2, l: 0 } },
    { type: 'tangent', a: { e: 2, l: 0 }, b: { e: 0 } }] };
  const r = K.solveSketch(sk);
  assert(r.ok);
  const [c0, c1, ln] = r.entities;
  assert(Math.abs(c0.r - 10) < 1e-6 && Math.abs(c1.r - 10) < 1e-6 && Math.hypot(c0.cx, c0.cy) < 1e-6);
  assert(Math.abs(Math.abs(ln.pts[0][1]) - 10) < 1e-6 && Math.abs(ln.pts[0][1] - ln.pts[1][1]) < 1e-9, 'tangent horizontal line at y = ±10');
  const sq = { entities: [{ type: 'poly', pts: [[0, 0], [10, 0], [12, 10]], open: true }], constraints: [{ type: 'angle', a: { e: 0, l: 0 }, b: { e: 0, l: 1 }, value: 60 }] };
  const ra = K.solveSketch(sq), P = ra.entities[0].pts;
  const u = [P[1][0] - P[0][0], P[1][1] - P[0][1]], v = [P[2][0] - P[1][0], P[2][1] - P[1][1]];
  assert(Math.abs(Math.acos((u[0] * v[0] + u[1] * v[1]) / Math.hypot(...u) / Math.hypot(...v)) * 180 / Math.PI - 60) < 1e-4);
  // drag a vertex of a horizontal line: it stays horizontal
  const dl = { entities: [{ type: 'poly', pts: [[0, 0], [10, 0]], open: true }], constraints: [{ type: 'horizontal', a: { e: 0, l: 0 } }] };
  const rd = K.solveSketch(dl, { drag: { ref: { e: 0, p: 1 }, to: [20, 5] } });
  assert(Math.abs(rd.entities[0].pts[0][1] - rd.entities[0].pts[1][1]) < 1e-6 && rd.entities[0].pts[1][0] > 15);
  assert.deepStrictEqual(K.inferHV({ type: 'poly', pts: [[0, 0], [10, 0.2], [10.1, 8]] }, 0).map(c => c.type), ['horizontal', 'vertical']);
  assert.strictEqual(K.dropEntityConstraints([{ type: 'equal', a: { e: 0 }, b: { e: 2 } }, { type: 'radius', a: { e: 2 }, value: 1 }], 1)[1].a.e, 1);
});
t('skin points project onto curved surfaces and settle on edges', () => {
  const cm = K.compile({ features: [{ id: 'cyl1', type: 'cylinder', op: 'boss', x: 0, y: 0, z: 0, axis: '+y', r: 20, h: 50 }] });
  const q = cm.project([18, 25, 4], [1, 0, 0], 4);         // a voxel-skin point inside the wall
  assert(q && Math.abs(Math.hypot(q[0], q[2]) - 20) < 1e-6 && Math.abs(q[1] - 25) < 1e-6, JSON.stringify(q));
  const rim = cm.project([18.5, 50, 5], [0.7, 0.7, 0], 4); // a top corner: onto the rim circle
  assert(rim && Math.abs(Math.hypot(rim[0], rim[2]) - 20) < 1e-3 && Math.abs(rim[1] - 50) < 1e-3, JSON.stringify(rim));
  // a thin plate: the far face must not attract the near one
  const pl = K.compile({ features: [{ id: 'b', type: 'box', op: 'boss', x: 0, y: 0, z: 0, sx: 50, sy: 2, sz: 50 }] });
  assert.strictEqual(pl.project([25, 2, 25], [0, 1, 0], 4), null, 'already on its face: nothing to do');
  const hole = K.compile({ features: [{ id: 'b', type: 'box', op: 'boss', x: -30, y: 0, z: -30, sx: 60, sy: 10, sz: 60 },
    { id: 'h', type: 'hole', x: 0, y: 10, z: 0, dir: '-y', d: 20, through: true }] });
  const qh = hole.project([11.5, 5, 3], [-1, 0, 0], 3);    // skin of the hole wall: the part's outside is towards the axis
  assert(qh && Math.abs(Math.hypot(qh[0], qh[2]) - 10) < 1e-6, JSON.stringify(qh));
});
console.log(n + ' kernel tests passed');
