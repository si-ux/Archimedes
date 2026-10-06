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
console.log(n + ' kernel tests passed');
