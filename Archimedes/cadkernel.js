/* Archimedes CAD kernel: feature-based solid modelling for the Workbench.
 *
 * A part is an ordered feature list, evaluated as constructive solid geometry:
 * each boss adds material, each cut removes it, in tree order, exactly like a
 * feature history in SolidWorks or CATIA Part Design.
 *
 *   sketch       closed profiles on the Front (XY), Top (XZ) or Right (YZ) plane, at an offset:
 *                rectangles (optional corner radius), circles, polylines with per-vertex fillets.
 *                Nested profiles make holes (even-odd).
 *   extrude      a sketch, blind / reversed / mid-plane, boss or cut
 *   revolve      a sketch about its vertical or horizontal axis, any angle, boss or cut
 *   box, cylinder, sphere, hole       primitives (hole = cylindrical cut, optionally through all)
 *   lpattern, cpattern, mirror        copies of an earlier feature
 *   stl          an imported triangle mesh (watertight), boss or cut
 *
 * The kernel also names faces. Every body exposes surface patches (an extrude:
 * one side face per sketch segment plus two caps, and so on), so a point on
 * the voxel skin is mapped to the nearest patch: "Extrude1 · side (line 3)".
 * Loads and supports reference these names, so they survive edits and remeshing.
 *
 * Units: mm. World axes: y up. No DOM; runs in the browser and in Node (tests).
 */
(function (root) {
  'use strict';

  // ---------------------------------------------------------------- planes
  const PLANES = {
    XY: { name: 'Front', u: [1, 0, 0], v: [0, 1, 0], w: [0, 0, 1] },
    XZ: { name: 'Top', u: [1, 0, 0], v: [0, 0, -1], w: [0, 1, 0] },
    YZ: { name: 'Right', u: [0, 0, -1], v: [0, 1, 0], w: [1, 0, 0] }
  };
  const AXES = { x: [1, 0, 0], y: [0, 1, 0], z: [0, 0, 1] };
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = a => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  function dirVec(d) {                                   // '+x', '-z', 'x', or [x, y, z]
    if (Array.isArray(d)) return norm(d);
    const s = String(d || '+z').trim().toLowerCase();
    const a = AXES[s.replace(/^[+-]/, '')] || AXES.z;
    return s[0] === '-' ? mul(a, -1) : a.slice();
  }

  // --------------------------------------------------------------- profiles
  // A profile is a list of loops; a loop is a list of segments
  //   { t: 'line', a: [x, y], b: [x, y], id }  or  { t: 'arc', c, r, a0, sw, id }  (sw: signed sweep)
  // Ids name faces: "e2.l3" = entity 2, edge 3; "e2.a1" = the fillet at vertex 1; "e0.c" = circle.
  function polyLoop(pts, fillets, ent) {
    const n = pts.length, segs = [];
    if (n < 3) return segs;
    const T = [];                                          // per vertex: [tangent in, tangent out, arc]
    for (let i = 0; i < n; i++) {
      const V = pts[i], P = pts[(i + n - 1) % n], N = pts[(i + 1) % n];
      const r = fillets && fillets[i] > 0 ? fillets[i] : 0;
      if (!r) { T.push([V, V, null]); continue; }
      const l1 = Math.hypot(P[0] - V[0], P[1] - V[1]), l2 = Math.hypot(N[0] - V[0], N[1] - V[1]);
      const e1 = [(P[0] - V[0]) / l1, (P[1] - V[1]) / l1], e2 = [(N[0] - V[0]) / l2, (N[1] - V[1]) / l2];
      const al = Math.acos(Math.max(-1, Math.min(1, e1[0] * e2[0] + e1[1] * e2[1])));
      if (al < 1e-3 || al > Math.PI - 1e-3) { T.push([V, V, null]); continue; }
      let rr = r, d = rr / Math.tan(al / 2);
      const dmax = 0.5 * Math.min(l1, l2) * 0.999;
      if (d > dmax) { d = dmax; rr = d * Math.tan(al / 2); }   // radius too large: the biggest that fits
      const bis = [e1[0] + e2[0], e1[1] + e2[1]], bl = Math.hypot(bis[0], bis[1]);
      const c = [V[0] + bis[0] / bl * rr / Math.sin(al / 2), V[1] + bis[1] / bl * rr / Math.sin(al / 2)];
      const t1 = [V[0] + e1[0] * d, V[1] + e1[1] * d], t2 = [V[0] + e2[0] * d, V[1] + e2[1] * d];
      const a0 = Math.atan2(t1[1] - c[1], t1[0] - c[0]);
      let sw = Math.atan2(t2[1] - c[1], t2[0] - c[0]) - a0;
      while (sw > Math.PI) sw -= 2 * Math.PI;
      while (sw < -Math.PI) sw += 2 * Math.PI;
      T.push([t1, t2, { t: 'arc', c, r: rr, a0, sw, id: 'e' + ent + '.a' + i, vertex: i }]);
    }
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      if (T[i][2]) segs.push(T[i][2]);
      const a = T[i][1], b = T[j][0];
      if (Math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-9) segs.push({ t: 'line', a, b, id: 'e' + ent + '.l' + i, edge: i });
    }
    return segs;
  }
  function entityLoop(e, k) {
    if (e.type === 'circle') return [{ t: 'arc', c: [e.cx, e.cy], r: Math.abs(e.r), a0: 0, sw: 2 * Math.PI, id: 'e' + k + '.c' }];
    if (e.type === 'rect') {
      const x0 = Math.min(e.x, e.x + e.w), x1 = Math.max(e.x, e.x + e.w), y0 = Math.min(e.y, e.y + e.h), y1 = Math.max(e.y, e.y + e.h);
      const r = e.r > 0 ? e.r : 0;
      return polyLoop([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], [r, r, r, r], k);
    }
    if (e.type === 'poly' && !e.open) return polyLoop(e.pts, e.fillets, k);
    if (e.type === 'poly') return openPolySegs(e.pts, e.fillets, k);
    return [];
  }
  // an open polyline (a sweep path): interior corners may be filleted, no closing edge
  function openPolySegs(pts, fillets, k) {
    const n = pts.length;
    if (n < 2) return [];
    const fl = (fillets || []).slice(0, n).map((v, i) => (i === 0 || i === n - 1 ? 0 : v || 0));
    if (n === 2) return [{ t: 'line', a: pts[0], b: pts[1], id: 'e' + k + '.l0', edge: 0 }];
    return polyLoop(pts, fl, k).filter(sg => !(sg.t === 'line' && sg.edge === n - 1));
  }
  // discretise for inside tests; each edge remembers its source segment
  function compileProfile(entities) {
    const loops = [], segs = [];
    (entities || []).forEach((e, k) => {
      if (e.type === 'poly' && e.open) return;             // paths are not regions
      const L = entityLoop(e, k);
      if (!L.length) return;
      const X = [], Y = [], S = [];
      for (const s of L) {
        const si = segs.length; segs.push(s);
        if (s.t === 'line') { X.push(s.a[0]); Y.push(s.a[1]); S.push(si); continue; }
        const m = Math.max(3, Math.ceil(Math.abs(s.sw) / (Math.PI / 36)));
        for (let q = 0; q < m; q++) {
          const a = s.a0 + s.sw * q / m;
          X.push(s.c[0] + s.r * Math.cos(a)); Y.push(s.c[1] + s.r * Math.sin(a)); S.push(si);
        }
      }
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (let i = 0; i < X.length; i++) { x0 = Math.min(x0, X[i]); x1 = Math.max(x1, X[i]); y0 = Math.min(y0, Y[i]); y1 = Math.max(y1, Y[i]); }
      loops.push({ X: Float64Array.from(X), Y: Float64Array.from(Y), S, bb: [x0, y0, x1, y1] });
    });
    const P = { loops, segs };
    P.bb = loops.reduce((b, l) => [Math.min(b[0], l.bb[0]), Math.min(b[1], l.bb[1]), Math.max(b[2], l.bb[2]), Math.max(b[3], l.bb[3])],
      [Infinity, Infinity, -Infinity, -Infinity]);
    // outward normal sign of every segment, by probing just beside its midpoint
    for (const s of segs) {
      const [m, n] = segMidNormal(s);
      const eps = 1e-4 * Math.max(1, Math.abs(P.bb[2] - P.bb[0]), Math.abs(P.bb[3] - P.bb[1]));
      s.out = inside2D(P, m[0] + n[0] * eps, m[1] + n[1] * eps) ? -1 : 1;
    }
    return P;
  }
  function segMidNormal(s) {
    if (s.t === 'line') {
      const dx = s.b[0] - s.a[0], dy = s.b[1] - s.a[1], l = Math.hypot(dx, dy) || 1;
      return [[(s.a[0] + s.b[0]) / 2, (s.a[1] + s.b[1]) / 2], [dy / l, -dx / l]];
    }
    const a = s.a0 + s.sw / 2;
    return [[s.c[0] + s.r * Math.cos(a), s.c[1] + s.r * Math.sin(a)], [Math.cos(a), Math.sin(a)]];
  }
  function inside2D(P, x, y) {
    let inside = false;
    for (const L of P.loops) {
      if (x < L.bb[0] || x > L.bb[2] || y < L.bb[1] || y > L.bb[3]) continue;
      const X = L.X, Y = L.Y, n = X.length;
      for (let i = 0, j = n - 1; i < n; j = i++) {
        if ((Y[i] > y) !== (Y[j] > y) && x < (X[j] - X[i]) * (y - Y[i]) / (Y[j] - Y[i]) + X[i]) inside = !inside;
      }
    }
    return inside;
  }
  // distance from (x, y) to segment s, and the outward normal there
  function segDist(s, x, y) {
    if (s.t === 'line') {
      const dx = s.b[0] - s.a[0], dy = s.b[1] - s.a[1], l2 = dx * dx + dy * dy || 1;
      const t = Math.max(0, Math.min(1, ((x - s.a[0]) * dx + (y - s.a[1]) * dy) / l2));
      const l = Math.sqrt(l2);
      return [Math.hypot(x - s.a[0] - dx * t, y - s.a[1] - dy * t), dy / l * s.out, -dx / l * s.out];
    }
    const a = Math.atan2(y - s.c[1], x - s.c[0]);
    let rel = a - s.a0;
    if (s.sw >= 0) { while (rel < 0) rel += 2 * Math.PI; while (rel > 2 * Math.PI) rel -= 2 * Math.PI; }
    else { while (rel > 0) rel -= 2 * Math.PI; while (rel < -2 * Math.PI) rel += 2 * Math.PI; }
    const on = Math.abs(s.sw) >= 2 * Math.PI - 1e-9 || (s.sw >= 0 ? rel <= s.sw : rel >= s.sw);
    if (on) {
      const d = Math.abs(Math.hypot(x - s.c[0], y - s.c[1]) - s.r);
      return [d, Math.cos(a) * s.out, Math.sin(a) * s.out];
    }
    const p0 = [s.c[0] + s.r * Math.cos(s.a0), s.c[1] + s.r * Math.sin(s.a0)];
    const p1 = [s.c[0] + s.r * Math.cos(s.a0 + s.sw), s.c[1] + s.r * Math.sin(s.a0 + s.sw)];
    const d0 = Math.hypot(x - p0[0], y - p0[1]), d1 = Math.hypot(x - p1[0], y - p1[1]);
    const e = d0 < d1 ? s.a0 : s.a0 + s.sw;
    return [Math.min(d0, d1), Math.cos(e) * s.out, Math.sin(e) * s.out];
  }
  function boundaryDist(P, x, y) {
    let best = Infinity;
    for (const s of P.segs) { const d = segDist(s, x, y)[0]; if (d < best) best = d; }
    return best;
  }
  function segLabel(s) {
    if (s.t === 'arc') return Math.abs(s.sw) >= 2 * Math.PI - 1e-9 ? 'circle' : s.vertex !== undefined ? 'fillet ' + (s.vertex + 1) : 'arc';
    return 'line ' + ((s.edge || 0) + 1);
  }

  // ----------------------------------------------------------------- bodies
  // A body: inside(p), bbox [min, max], patches [{ id, name, drag }], and
  // near(p) -> [distance, patch index, normal] for face naming.
  function sketchFrame(sk) {
    const P = PLANES[sk.plane] || PLANES.XY, o = mul(P.w, sk.offset || 0);
    return { P, o, toLocal: p => { const q = sub(p, o); return [dot(q, P.u), dot(q, P.v), dot(q, P.w)]; },
      toWorld: (u, v, w) => add(o, add(mul(P.u, u), add(mul(P.v, v), mul(P.w, w)))) };
  }
  function bboxOfPoints(pts) {
    const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (const p of pts) for (let i = 0; i < 3; i++) { mn[i] = Math.min(mn[i], p[i]); mx[i] = Math.max(mx[i], p[i]); }
    return [mn, mx];
  }
  // a planar patch: unit normal, n·x = d, centre and extent corners (for edge blends)
  function planeOf(normal, corners) {
    const n = norm(normal), c = mul(corners.reduce((a, q) => add(a, q), [0, 0, 0]), 1 / corners.length);
    return { normal: n, d: dot(n, c), center: c, corners };
  }
  function extrudeRange(f) {
    const d = Math.abs(f.depth || 0);
    return f.dir === 'reverse' ? [-d, 0] : f.dir === 'mid' ? [-d / 2, d / 2] : [0, d];
  }
  function extrudeBody(f, sk) {
    const F = sketchFrame(sk), P = compileProfile(sk.entities), [w0, w1] = extrudeRange(f);
    if (!P.loops.length) return null;
    const patches = P.segs.map(s => {
      const pt = { id: f.id + '/' + s.id, name: 'side (' + segLabel(s) + ')', seg: s, drag: { kind: 'seg', sketch: sk.id, seg: s } };
      if (s.t === 'line') {                                // a straight edge sweeps a planar face
        const dx = s.b[0] - s.a[0], dy = s.b[1] - s.a[1], l = Math.hypot(dx, dy) || 1, n2 = [dy / l * s.out, -dx / l * s.out];
        const corners = [F.toWorld(s.a[0], s.a[1], w0), F.toWorld(s.b[0], s.b[1], w0), F.toWorld(s.b[0], s.b[1], w1), F.toWorld(s.a[0], s.a[1], w1)];
        pt.plane = planeOf(add(mul(F.P.u, n2[0]), mul(F.P.v, n2[1])), corners);
      } else if (Math.abs(s.sw) > 1e-9) {                  // an arc sweeps a cylinder
        pt.cyl = { c: F.toWorld(s.c[0], s.c[1], w0), a: F.P.w.slice(), r: s.r, h0: 0, h1: w1 - w0 };
      }
      return pt;
    });
    const capCorners = wc => [F.toWorld(P.bb[0], P.bb[1], wc), F.toWorld(P.bb[2], P.bb[1], wc), F.toWorld(P.bb[2], P.bb[3], wc), F.toWorld(P.bb[0], P.bb[3], wc)];
    const capA = { id: f.id + '/cap0', name: f.dir === 'reverse' ? 'end face' : 'start face', drag: f.dir === 'reverse' ? { kind: 'param', key: 'depth', axis: mul(F.P.w, -1), factor: 1 } : null, plane: planeOf(mul(F.P.w, -1), capCorners(w0)) };
    const capB = { id: f.id + '/cap1', name: f.dir === 'reverse' ? 'start face' : 'end face', drag: f.dir !== 'reverse' ? { kind: 'param', key: 'depth', axis: F.P.w.slice(), factor: f.dir === 'mid' ? 2 : 1 } : null, plane: planeOf(F.P.w.slice(), capCorners(w1)) };
    if (f.dir === 'mid') capA.drag = { kind: 'param', key: 'depth', axis: mul(F.P.w, -1), factor: 2 };
    patches.push(capA, capB);
    const corners = [];
    for (const u of [P.bb[0], P.bb[2]]) for (const v of [P.bb[1], P.bb[3]]) for (const w of [w0, w1]) corners.push(F.toWorld(u, v, w));
    const ns = P.segs.length;
    return {
      bbox: bboxOfPoints(corners), patches,
      inside: p => { const l = F.toLocal(p); return l[2] >= w0 && l[2] <= w1 && inside2D(P, l[0], l[1]); },
      near: p => {
        const l = F.toLocal(p), dw = l[2] < w0 ? w0 - l[2] : l[2] > w1 ? l[2] - w1 : 0;
        let best = Infinity, bi = -1, bn = null, bd2 = Infinity;
        for (let i = 0; i < ns; i++) {
          const r = segDist(P.segs[i], l[0], l[1]);
          if (r[0] < bd2) bd2 = r[0];
          const d = Math.hypot(r[0], dw);
          if (d < best) { best = d; bi = i; bn = add(mul(F.P.u, r[1]), mul(F.P.v, r[2])); }
        }
        const inP = inside2D(P, l[0], l[1]);
        for (const [k, wc, n] of [[ns, w0, mul(F.P.w, -1)], [ns + 1, w1, F.P.w]]) {
          const d = inP ? Math.abs(l[2] - wc) : Math.hypot(l[2] - wc, bd2);
          if (d < best) { best = d; bi = k; bn = n; }
        }
        return [best, bi, bn];
      },
      nearAll: p => {
        const l = F.toLocal(p), dw = l[2] < w0 ? w0 - l[2] : l[2] > w1 ? l[2] - w1 : 0, out = [];
        let bd2 = Infinity;
        for (let i = 0; i < ns; i++) {
          const r = segDist(P.segs[i], l[0], l[1]);
          if (r[0] < bd2) bd2 = r[0];
          out.push([Math.hypot(r[0], dw), i, add(mul(F.P.u, r[1]), mul(F.P.v, r[2]))]);
        }
        const inP = inside2D(P, l[0], l[1]);
        out.push([inP ? Math.abs(l[2] - w0) : Math.hypot(l[2] - w0, bd2), ns, mul(F.P.w, -1)]);
        out.push([inP ? Math.abs(l[2] - w1) : Math.hypot(l[2] - w1, bd2), ns + 1, F.P.w.slice()]);
        return out;
      },
      frame: F, profile: P, range: [w0, w1]
    };
  }
  function revolveBody(f, sk) {
    const F = sketchFrame(sk), P = compileProfile(sk.entities);
    if (!P.loops.length) return null;
    const axisV = f.axis !== 'u', ang = Math.min(360, Math.max(1, f.angle || 360)) * Math.PI / 180, full = ang >= 2 * Math.PI - 1e-6;
    // (rho, h): distance from the axis and position along it, both in the sketch's own coordinates
    const rh = l => axisV ? [Math.hypot(l[0], l[2]), l[1], Math.atan2(l[2], l[0])] : [Math.hypot(l[1], l[2]), l[0], Math.atan2(l[2], l[1])];
    const inProf = (r, h) => (axisV ? inside2D(P, r, h) || inside2D(P, -r, h) : inside2D(P, h, r) || inside2D(P, h, -r));
    const axDir = axisV ? F.P.v : F.P.u;
    const patches = P.segs.map(s => {
      const pt = { id: f.id + '/' + s.id, name: 'surface (' + segLabel(s) + ')', drag: { kind: 'seg', sketch: sk.id, seg: s } };
      if (s.t === 'line' && full) {
        const ra = axisV ? s.a[0] : s.a[1], rb = axisV ? s.b[0] : s.b[1], ha = axisV ? s.a[1] : s.a[0], hb = axisV ? s.b[1] : s.b[0];
        if (Math.abs(ra - rb) < 1e-9 && Math.abs(ra) > 1e-9) pt.cyl = { c: F.toWorld(0, 0, 0), a: axDir.slice(), r: Math.abs(ra), h0: Math.min(ha, hb), h1: Math.max(ha, hb) };
        else if (Math.abs(ha - hb) < 1e-9) {
          const nh = axisV ? segDist(s, (s.a[0] + s.b[0]) / 2, (s.a[1] + s.b[1]) / 2)[2] : segDist(s, (s.a[0] + s.b[0]) / 2, (s.a[1] + s.b[1]) / 2)[1];
          const R1 = Math.max(Math.abs(ra), Math.abs(rb)), cc = F.toWorld(axisV ? 0 : ha, axisV ? ha : 0, 0), e1 = axisV ? F.P.u : F.P.v;
          pt.plane = planeOf(mul(axDir, Math.sign(nh) || 1), [add(cc, mul(e1, R1)), add(cc, mul(F.P.w, R1)), add(cc, mul(e1, -R1)), add(cc, mul(F.P.w, -R1))]);
          pt.plane.center = cc; pt.plane.rho = [Math.min(Math.abs(ra), Math.abs(rb)), R1];
        }
      }
      return pt;
    });
    if (!full) patches.push({ id: f.id + '/end0', name: 'end face 1' }, { id: f.id + '/end1', name: 'end face 2' });
    const R = Math.max(Math.abs(P.bb[0]), Math.abs(P.bb[2]), axisV ? 0 : 0), H0 = axisV ? P.bb[1] : P.bb[0], H1 = axisV ? P.bb[3] : P.bb[2];
    const Rr = axisV ? Math.max(Math.abs(P.bb[0]), Math.abs(P.bb[2])) : Math.max(Math.abs(P.bb[1]), Math.abs(P.bb[3]));
    const corners = [];
    for (const a of [-Rr, Rr]) for (const b of [-Rr, Rr]) for (const h of [H0, H1])
      corners.push(axisV ? F.toWorld(a, h, b) : F.toWorld(h, a, b));
    void R;
    const angOk = phi => { if (full) return true; let a = phi; while (a < 0) a += 2 * Math.PI; return a <= ang; };
    const ns = P.segs.length;
    return {
      bbox: bboxOfPoints(corners), patches,
      inside: p => { const l = F.toLocal(p), [r, h, phi] = rh(l); return angOk(phi) && inProf(r, h); },
      near: p => {
        const l = F.toLocal(p), [r, h, phi] = rh(l);
        let best = Infinity, bi = -1, bn = null;
        for (let i = 0; i < ns; i++) {
          const q = axisV ? segDist(P.segs[i], r, h) : segDist(P.segs[i], h, r);
          const q2 = axisV ? segDist(P.segs[i], -r, h) : segDist(P.segs[i], h, -r);
          const d = Math.min(q[0], q2[0]);
          if (d < best) {
            best = d; bi = i;
            const nr = axisV ? q[1] : q[2], nh = axisV ? q[2] : q[1];
            const radial = axisV ? norm(add(mul(F.P.u, Math.cos(phi)), mul(F.P.w, Math.sin(phi)))) : norm(add(mul(F.P.v, Math.cos(phi)), mul(F.P.w, Math.sin(phi))));
            bn = norm(add(mul(radial, nr), mul(axisV ? F.P.v : F.P.u, nh)));
          }
        }
        if (!full) {
          let a = phi; while (a < 0) a += 2 * Math.PI;
          const gaps = [[ns, Math.abs(a) * r], [ns + 1, Math.abs(ang - a) * r]];
          for (const [k, d] of gaps) if (d < best) { best = d; bi = k; bn = null; }
        }
        return [best, bi, bn];
      },
      nearAll: p => {
        const l = F.toLocal(p), [r, h, phi] = rh(l), out = [];
        const radial = axisV ? norm(add(mul(F.P.u, Math.cos(phi)), mul(F.P.w, Math.sin(phi)))) : norm(add(mul(F.P.v, Math.cos(phi)), mul(F.P.w, Math.sin(phi))));
        for (let i = 0; i < ns; i++) {
          const q = axisV ? segDist(P.segs[i], r, h) : segDist(P.segs[i], h, r);
          const nr = axisV ? q[1] : q[2], nh = axisV ? q[2] : q[1];
          out.push([q[0], i, norm(add(mul(radial, nr), mul(axisV ? F.P.v : F.P.u, nh)))]);
        }
        return out;
      }
    };
  }
  function boxBody(f) {
    const mn = [f.x || 0, f.y || 0, f.z || 0], s = [Math.abs(f.sx || 1), Math.abs(f.sy || 1), Math.abs(f.sz || 1)];
    const mx = add(mn, s);
    const names = ['−X face', '+X face', '−Y face', '+Y face', '−Z face', '+Z face'], keys = ['x', 'y', 'z'], sz = ['sx', 'sy', 'sz'];
    const patches = names.map((n, i) => {
      const ax = Math.floor(i / 2), hi = i % 2 === 1, axis = [0, 0, 0]; axis[ax] = hi ? 1 : -1;
      const cs = [];
      const o1 = (ax + 1) % 3, o2 = (ax + 2) % 3;
      for (const [x1, x2] of [[0, 0], [1, 0], [1, 1], [0, 1]]) { const q = mn.slice(); q[ax] = hi ? mx[ax] : mn[ax]; q[o1] = x1 ? mx[o1] : mn[o1]; q[o2] = x2 ? mx[o2] : mn[o2]; cs.push(q); }
      return { id: f.id + '/' + (hi ? '+' : '-') + 'xyz'[ax], name: n, plane: planeOf(axis, cs),
        drag: hi ? { kind: 'param', key: sz[ax], axis, factor: 1 } : { kind: 'param', key: sz[ax], axis, factor: 1, move: keys[ax] } };
    });
    return {
      bbox: [mn, mx], patches,
      inside: p => p[0] >= mn[0] && p[0] <= mx[0] && p[1] >= mn[1] && p[1] <= mx[1] && p[2] >= mn[2] && p[2] <= mx[2],
      near: p => {
        let best = Infinity, bi = 0;
        for (let i = 0; i < 6; i++) {
          const ax = Math.floor(i / 2), c = i % 2 ? mx[ax] : mn[ax];
          let d2 = (p[ax] - c) ** 2;
          for (let k = 0; k < 3; k++) if (k !== ax) { const o = p[k] < mn[k] ? mn[k] - p[k] : p[k] > mx[k] ? p[k] - mx[k] : 0; d2 += o * o; }
          if (d2 < best) { best = d2; bi = i; }
        }
        const n = [0, 0, 0]; n[Math.floor(bi / 2)] = bi % 2 ? 1 : -1;
        return [Math.sqrt(best), bi, n];
      },
      nearAll: p => {
        const out = [];
        for (let i = 0; i < 6; i++) {
          const ax = Math.floor(i / 2), c = i % 2 ? mx[ax] : mn[ax];
          let d2 = (p[ax] - c) ** 2;
          for (let k = 0; k < 3; k++) if (k !== ax) { const o = p[k] < mn[k] ? mn[k] - p[k] : p[k] > mx[k] ? p[k] - mx[k] : 0; d2 += o * o; }
          const n = [0, 0, 0]; n[ax] = i % 2 ? 1 : -1;
          out.push([Math.sqrt(d2), i, n]);
        }
        return out;
      }
    };
  }
  // cylinder from base centre c along unit axis a, radius r, length h
  function cylBody(id, c, a, r, h, labels, drags) {
    const ref = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0], e1 = norm(cross(a, ref)), e2 = cross(a, e1);
    const top = add(c, mul(a, h));
    const corners = [];
    for (const q of [c, top]) for (const s1 of [-r, r]) for (const s2 of [-r, r]) corners.push(add(q, add(mul(e1, s1), mul(e2, s2))));
    const sq = q => [add(q, add(mul(e1, r), mul(e2, r))), add(q, add(mul(e1, -r), mul(e2, r))), add(q, add(mul(e1, -r), mul(e2, -r))), add(q, add(mul(e1, r), mul(e2, -r)))];
    const cap0 = planeOf(mul(a, -1), sq(c)), cap1 = planeOf(a.slice(), sq(top));
    cap0.rho = [0, r]; cap1.rho = [0, r];
    const patches = [{ id: id + '/wall', name: labels[0], drag: drags[0], cyl: { c, a, r, h0: 0, h1: h } }, { id: id + '/cap0', name: labels[1], drag: drags[1], plane: cap0 },
      { id: id + '/cap1', name: labels[2], drag: drags[2], plane: cap1 }];
    return {
      bbox: bboxOfPoints(corners), patches,
      inside: p => { const q = sub(p, c), t = dot(q, a); if (t < 0 || t > h) return false; const rr = sub(q, mul(a, t)); return dot(rr, rr) <= r * r; },
      near: p => {
        const q = sub(p, c), t = dot(q, a), radv = sub(q, mul(a, t)), rho = Math.hypot(radv[0], radv[1], radv[2]);
        const dt = t < 0 ? -t : t > h ? t - h : 0, dr = rho > r ? rho - r : 0;
        const dw = Math.hypot(rho - r, dt), d0 = Math.hypot(t, dr), d1 = Math.hypot(t - h, dr);
        if (dw <= d0 && dw <= d1) return [dw, 0, rho > 1e-9 ? mul(radv, 1 / rho) : e1];
        return d0 < d1 ? [d0, 1, mul(a, -1)] : [d1, 2, a.slice()];
      },
      nearAll: p => {
        const q = sub(p, c), t = dot(q, a), radv = sub(q, mul(a, t)), rho = Math.hypot(radv[0], radv[1], radv[2]);
        const dt = t < 0 ? -t : t > h ? t - h : 0, dr = rho > r ? rho - r : 0;
        return [[Math.hypot(rho - r, dt), 0, rho > 1e-9 ? mul(radv, 1 / rho) : e1], [Math.hypot(t, dr), 1, mul(a, -1)], [Math.hypot(t - h, dr), 2, a.slice()]];
      }
    };
  }
  function cylinderBody(f) {
    const a = dirVec(f.axis || '+y'), c = [f.x || 0, f.y || 0, f.z || 0];
    return cylBody(f.id, c, a, Math.abs(f.r || 1), Math.abs(f.h || 1), ['wall', 'base', 'top'],
      [{ kind: 'radial', key: 'r', factor: 1 }, null, { kind: 'param', key: 'h', axis: a, factor: 1 }]);
  }
  function holeBody(f, model) {
    const a = dirVec(f.dir || '-y'), c = [f.x || 0, f.y || 0, f.z || 0];
    const depth = f.through ? 4 * modelExtent(model) + 1 : Math.abs(f.depth || 10);
    // starts a little above the entry face so the hole always breaks through it
    const lead = 0.01 * Math.max(1, Math.abs(f.d || 10));
    return cylBody(f.id, sub(c, mul(a, lead)), a, Math.abs(f.d || 10) / 2, depth + lead, ['hole wall', 'entry', 'hole bottom'],
      [{ kind: 'radial', key: 'd', factor: 2 }, null, f.through ? null : { kind: 'param', key: 'depth', axis: a, factor: 1 }]);
  }
  function sphereBody(f) {
    const c = [f.x || 0, f.y || 0, f.z || 0], r = Math.abs(f.r || 1);
    return {
      bbox: [sub(c, [r, r, r]), add(c, [r, r, r])],
      patches: [{ id: f.id + '/surface', name: 'surface', drag: { kind: 'radial', key: 'r', factor: 1 } }],
      inside: p => { const q = sub(p, c); return dot(q, q) <= r * r; },
      near: p => { const q = sub(p, c), l = Math.hypot(q[0], q[1], q[2]); return [Math.abs(l - r), 0, l > 1e-9 ? mul(q, 1 / l) : [0, 1, 0]]; }
    };
  }

  // ---- loft: between two profiles on parallel planes ---------------------------
  // The section at height w is the zero set of the signed 2D distance fields of
  // the two profiles blended linearly: a smooth transition between any shapes.
  function signedDist2D(P, x, y) { const d = boundaryDist(P, x, y); return inside2D(P, x, y) ? -d : d; }
  function loftBody(f, s0, s1) {
    if (!s0 || !s1 || s0.plane !== s1.plane) return null;
    const F = sketchFrame(s0), P0 = compileProfile(s0.entities), P1 = compileProfile(s1.entities);
    if (!P0.loops.length || !P1.loops.length) return null;
    const H = (s1.offset || 0) - (s0.offset || 0);
    if (Math.abs(H) < 1e-9) return null;
    const lo = Math.min(0, H), hi = Math.max(0, H);
    const field = (u, v, w) => { const t = Math.max(0, Math.min(1, w / H)); return (1 - t) * signedDist2D(P0, u, v) + t * signedDist2D(P1, u, v); };
    const bb = [Math.min(P0.bb[0], P1.bb[0]), Math.min(P0.bb[1], P1.bb[1]), Math.max(P0.bb[2], P1.bb[2]), Math.max(P0.bb[3], P1.bb[3])];
    const corners = [];
    for (const u of [bb[0], bb[2]]) for (const v of [bb[1], bb[3]]) for (const w of [lo, hi]) corners.push(F.toWorld(u, v, w));
    const patches = [{ id: f.id + '/side', name: 'lofted surface' },
      { id: f.id + '/cap0', name: 'start face', plane: planeOf(mul(F.P.w, H > 0 ? -1 : 1), [F.toWorld(P0.bb[0], P0.bb[1], 0), F.toWorld(P0.bb[2], P0.bb[1], 0), F.toWorld(P0.bb[2], P0.bb[3], 0), F.toWorld(P0.bb[0], P0.bb[3], 0)]) },
      { id: f.id + '/cap1', name: 'end face', plane: planeOf(mul(F.P.w, H > 0 ? 1 : -1), [F.toWorld(P1.bb[0], P1.bb[1], H), F.toWorld(P1.bb[2], P1.bb[1], H), F.toWorld(P1.bb[2], P1.bb[3], H), F.toWorld(P1.bb[0], P1.bb[3], H)]) }];
    return {
      bbox: bboxOfPoints(corners), patches,
      inside: p => { const l = F.toLocal(p); return l[2] >= lo && l[2] <= hi && field(l[0], l[1], l[2]) <= 0; },
      near: p => {
        const l = F.toLocal(p), w = Math.max(lo, Math.min(hi, l[2])), dw = Math.abs(l[2] - w), fv = field(l[0], l[1], w);
        const e = 1e-3 * Math.max(1, bb[2] - bb[0]);
        const gu = (field(l[0] + e, l[1], w) - field(l[0] - e, l[1], w)) / (2 * e), gv = (field(l[0], l[1] + e, w) - field(l[0], l[1] - e, w)) / (2 * e);
        const side = Math.hypot(fv, dw);
        const ins = fv <= 0;
        const d0 = ins ? Math.abs(l[2]) : Infinity, d1 = ins ? Math.abs(l[2] - H) : Infinity;
        if (side <= d0 && side <= d1) return [side, 0, norm(add(mul(F.P.u, gu), mul(F.P.v, gv)))];
        return d0 < d1 ? [d0, 1, mul(F.P.w, H > 0 ? -1 : 1)] : [d1, 2, mul(F.P.w, H > 0 ? 1 : -1)];
      },
      nearAll: p => {                                     // every surface, for points on the rims
        const l = F.toLocal(p), w = Math.max(lo, Math.min(hi, l[2])), dw = Math.abs(l[2] - w), fv = field(l[0], l[1], w);
        const e = 1e-3 * Math.max(1, bb[2] - bb[0]);
        const gu = (field(l[0] + e, l[1], w) - field(l[0] - e, l[1], w)) / (2 * e), gv = (field(l[0], l[1] + e, w) - field(l[0], l[1] - e, w)) / (2 * e);
        // the blend's slope along the loft tilts the side's normal; dividing by the
        // full gradient turns the blended field into a distance
        const ww = Math.max(lo + e, Math.min(hi - e, w)), gw = (field(l[0], l[1], ww + e) - field(l[0], l[1], ww - e)) / (2 * e);
        const gl = Math.hypot(gu, gv, gw) || 1;
        const s0 = signedDist2D(P0, l[0], l[1]), s1 = signedDist2D(P1, l[0], l[1]);
        return [[Math.hypot(fv / gl, dw), 0, norm(add(add(mul(F.P.u, gu), mul(F.P.v, gv)), mul(F.P.w, gw)))],
          [s0 <= 0 ? Math.abs(l[2]) : Math.hypot(l[2], s0), 1, mul(F.P.w, H > 0 ? -1 : 1)],
          [s1 <= 0 ? Math.abs(l[2] - H) : Math.hypot(l[2] - H, s1), 2, mul(F.P.w, H > 0 ? 1 : -1)]];
      }
    };
  }
  // ---- sweep: a profile carried along a path ------------------------------------
  // The path is the first open polyline of its sketch (corners may be filleted).
  // Frames are parallel-transported along it, so the profile does not twist.
  function pathPoints(sk) {
    const F = sketchFrame(sk), k = (sk.entities || []).findIndex(e => e.type === 'poly' && e.open);
    if (k < 0) return null;
    const out = [];
    for (const sg of openPolySegs(sk.entities[k].pts, sk.entities[k].fillets, k)) {
      const pts = sg.t === 'line' ? [sg.a, sg.b] : Array.from({ length: Math.max(4, Math.ceil(Math.abs(sg.sw) / (Math.PI / 18))) + 1 },
        (z, q, arr) => { const m = Math.max(4, Math.ceil(Math.abs(sg.sw) / (Math.PI / 18))), a = sg.a0 + sg.sw * q / m; return [sg.c[0] + sg.r * Math.cos(a), sg.c[1] + sg.r * Math.sin(a)]; });
      for (const q of pts) { const w = F.toWorld(q[0], q[1], 0); const last = out[out.length - 1]; if (!last || Math.hypot(...sub(w, last)) > 1e-9) out.push(w); }
    }
    return out.length >= 2 ? out : null;
  }
  function sweepBody(f, prof, path) {
    if (!prof || !path) return null;
    const P = compileProfile(prof.entities), pts = pathPoints(path);
    if (!P.loops.length || !pts) return null;
    const PF = sketchFrame(prof);
    // segment tangents and parallel-transported frames (double reflection)
    const n = pts.length - 1, T = [], U = [], V = [], L = [], S0 = [];
    let acc = 0;
    for (let i = 0; i < n; i++) { const d = sub(pts[i + 1], pts[i]), l = Math.hypot(...d); T.push(mul(d, 1 / l)); L.push(l); S0.push(acc); acc += l; }
    // start frame: the profile plane's axes, made perpendicular to the first tangent
    let u0 = sub(PF.P.u, mul(T[0], dot(PF.P.u, T[0])));
    if (Math.hypot(...u0) < 1e-6) u0 = sub(PF.P.v, mul(T[0], dot(PF.P.v, T[0])));
    U.push(norm(u0)); V.push(cross(T[0], U[0]));
    for (let i = 1; i < n; i++) {
      const v1 = sub(pts[i], pts[i - 1]), c1 = dot(v1, v1) || 1;
      const rL = sub(U[i - 1], mul(v1, 2 / c1 * dot(v1, U[i - 1]))), tL = sub(T[i - 1], mul(v1, 2 / c1 * dot(v1, T[i - 1])));
      const v2 = sub(T[i], tL), c2 = dot(v2, v2) || 1;
      const u = c2 < 1e-18 ? rL : sub(rL, mul(v2, 2 / c2 * dot(v2, rL)));
      U.push(norm(u)); V.push(cross(T[i], U[i]));
    }
    // profile coordinates are measured from the path start, in the start frame mapped onto the profile plane
    const P0 = pts[0], base = PF.toLocal(P0);
    const fu = [dot(U[0], PF.P.u), dot(V[0], PF.P.u)], fv = [dot(U[0], PF.P.v), dot(V[0], PF.P.v)];
    const local = p => {                                      // -> [u, v, s, endGap, segment]
      let best = Infinity, bi = 0, bt = 0;
      for (let i = 0; i < n; i++) {
        const q = sub(p, pts[i]), t = Math.max(0, Math.min(L[i], dot(q, T[i]))), r = sub(q, mul(T[i], t)), d = dot(r, r);
        if (d < best) { best = d; bi = i; bt = t; }
      }
      const q = sub(sub(p, pts[bi]), mul(T[bi], bt)), a = dot(q, U[bi]), b = dot(q, V[bi]);
      const along = dot(sub(p, pts[bi]), T[bi]);
      const gap = bi === 0 && along < 0 ? -along : bi === n - 1 && along > L[bi] ? along - L[bi] : 0;
      // (a, b) in the moving frame -> profile-plane coordinates
      return [base[0] + a * fu[0] + b * fu[1], base[1] + a * fv[0] + b * fv[1], S0[bi] + bt, gap, bi, a, b];
    };
    let R = 0;
    for (const lp of P.loops) for (let i = 0; i < lp.X.length; i++) R = Math.max(R, Math.hypot(lp.X[i] - base[0], lp.Y[i] - base[1]));
    const bb = bboxOfPoints(pts);
    const patches = P.segs.map(sg => ({ id: f.id + '/' + sg.id, name: 'swept side (' + segLabel(sg) + ')' }));
    patches.push({ id: f.id + '/cap0', name: 'start face' }, { id: f.id + '/cap1', name: 'end face' });
    const ns = P.segs.length;
    return {
      bbox: [sub(bb[0], [R, R, R]), add(bb[1], [R, R, R])], patches,
      inside: p => { const l = local(p); return l[3] <= 1e-9 && inside2D(P, l[0], l[1]); },
      near: p => {
        const l = local(p);
        let best = Infinity, bi = 0, bn = null, bd2 = Infinity;
        for (let i = 0; i < ns; i++) {
          const r = segDist(P.segs[i], l[0], l[1]);
          if (r[0] < bd2) bd2 = r[0];
          const d = Math.hypot(r[0], l[3]);
          if (d < best) {
            best = d; bi = i;
            // profile-plane normal -> moving frame
            const nu = r[1] * fu[0] + r[2] * fv[0], nv = r[1] * fu[1] + r[2] * fv[1];
            bn = norm(add(mul(U[l[4]], nu), mul(V[l[4]], nv)));
          }
        }
        const s = l[2], inP = inside2D(P, l[0], l[1]);
        const ds = inP ? s : Math.hypot(s, bd2), de = inP ? acc - s : Math.hypot(acc - s, bd2);
        if (l[4] === 0 && ds < best) { best = ds; bi = ns; bn = mul(T[0], -1); }
        if (l[4] === n - 1 && de < best) { best = de; bi = ns + 1; bn = T[n - 1]; }
        return [best, bi, bn];
      },
      nearAll: p => {
        const l = local(p), out = [];
        let bd2 = Infinity;
        for (let i = 0; i < ns; i++) {
          const r = segDist(P.segs[i], l[0], l[1]);
          if (r[0] < bd2) bd2 = r[0];
          const nu = r[1] * fu[0] + r[2] * fv[0], nv = r[1] * fu[1] + r[2] * fv[1];
          out.push([Math.hypot(r[0], l[3]), i, norm(add(mul(U[l[4]], nu), mul(V[l[4]], nv)))]);
        }
        const s = l[2], inP = inside2D(P, l[0], l[1]);
        if (l[4] === 0) out.push([inP ? Math.abs(s) : Math.hypot(s, bd2), ns, mul(T[0], -1)]);
        if (l[4] === n - 1) out.push([inP ? Math.abs(acc - s) : Math.hypot(acc - s, bd2), ns + 1, T[n - 1]]);
        return out;
      }
    };
  }
  // ---- edge blends: fillet / chamfer an edge between two named faces ----------------
  // Plane–plane edges (any dihedral angle) and plane–cylinder rims (cap or hole
  // edges). Works in the cross-section across the edge: the corner wedge between
  // the faces, cut by the chord through the tangent points, minus the fillet
  // circle. A convex edge loses that region, a concave edge gains it.
  function wedge2(u1, u2, r, kind) {
    const c = Math.max(-1, Math.min(1, u1[0] * u2[0] + u1[1] * u2[1])), phi = Math.acos(c);
    if (phi < 1e-3 || phi > Math.PI - 1e-3) return null;
    const bl = Math.hypot(u1[0] + u2[0], u1[1] + u2[1]), bis = [(u1[0] + u2[0]) / bl, (u1[1] + u2[1]) / bl];
    const t = kind === 'chamfer' ? r : r / Math.tan(phi / 2), chord = t * Math.cos(phi / 2), cd = r / Math.sin(phi / 2);
    const ctr = [bis[0] * cd, bis[1] * cd], sgn = u1[0] * u2[1] - u1[1] * u2[0] > 0 ? 1 : -1;
    const inWedge = (x, y) => sgn * (u1[0] * y - u1[1] * x) >= 0 && sgn * (x * u2[1] - y * u2[0]) >= 0;
    return {
      region: (x, y) => {
        if (x * bis[0] + y * bis[1] >= chord || !inWedge(x, y)) return false;
        return kind === 'chamfer' ? true : (x - ctr[0]) ** 2 + (y - ctr[1]) ** 2 > r * r;
      },
      // distance to the blend surface and its normal pointing away from the corner's material (convex case)
      near: (x, y) => {
        if (!inWedge(x, y)) return null;
        if (kind === 'chamfer') return [Math.abs(x * bis[0] + y * bis[1] - chord), -bis[0], -bis[1]];
        const dx = x - ctr[0], dy = y - ctr[1], l = Math.hypot(dx, dy) || 1;
        if (x * bis[0] + y * bis[1] > chord + r) return null;
        return [Math.abs(l - r), dx / l, dy / l];
      },
      probe: [bis[0] * chord * 0.3, bis[1] * chord * 0.3]
    };
  }
  function edgeBlend(A, B, r, kind, ins) {
    if (A && B && A.plane && B.plane) {
      const P1 = A.plane, P2 = B.plane, n1 = P1.normal, n2 = P2.normal;
      let e = cross(n1, n2);
      const le = Math.hypot(...e);
      if (le < 1e-6) return null;
      e = mul(e, 1 / le);
      const m = mul(add(P1.center, P2.center), 0.5), k = dot(n1, n2), r1 = P1.d - dot(n1, m), r2 = P2.d - dot(n2, m), det = 1 - k * k;
      const x0 = add(m, add(mul(n1, (r1 - k * r2) / det), mul(n2, (r2 - k * r1) / det)));
      let u1 = norm(cross(e, n1)); if (dot(u1, sub(P1.center, x0)) < 0) u1 = mul(u1, -1);
      let u2 = norm(cross(e, n2)); if (dot(u2, sub(P2.center, x0)) < 0) u2 = mul(u2, -1);
      const b1 = u1, b2 = norm(cross(e, u1));             // cross-section basis
      const W = wedge2([1, 0], [dot(u2, b1), dot(u2, b2)], r, kind);
      if (!W) return null;
      const ext = P => { let lo = Infinity, hi = -Infinity; for (const q of P.corners || []) { const t = dot(sub(q, x0), e); lo = Math.min(lo, t); hi = Math.max(hi, t); } return P.corners ? [lo, hi] : [-Infinity, Infinity]; };
      const [a0, a1] = ext(P1), [c0, c1] = ext(P2), lo = Math.max(a0, c0), hi = Math.min(a1, c1);
      if (!(hi > lo)) return null;
      const xs = p => { const q = sub(p, x0), s = dot(q, e); return [dot(q, b1), dot(q, b2), s]; };
      const mid = isFinite(lo + hi) ? (lo + hi) / 2 : 0;
      return {
        region: p => { const q = xs(p); return q[2] >= lo - 1e-9 && q[2] <= hi + 1e-9 && W.region(q[0], q[1]); },
        near: p => { const q = xs(p); if (q[2] < lo || q[2] > hi) return null; const r2_ = W.near(q[0], q[1]); return r2_ && [r2_[0], add(mul(b1, r2_[1]), mul(b2, r2_[2]))]; },
        probe: add(x0, add(add(mul(b1, W.probe[0]), mul(b2, W.probe[1])), mul(e, mid)))
      };
    }
    const pl = A && A.plane ? A : B && B.plane ? B : null, cy = A && A.cyl ? A : B && B.cyl ? B : null;
    if (pl && cy) {
      const P = pl.plane, C = cy.cyl, a = norm(C.a);
      if (Math.abs(dot(P.normal, a)) < 0.999) return null;
      const hp = (P.d - dot(P.normal, C.c)) / dot(P.normal, a), R = C.r;
      const rhoOf = q => { const v = sub(q, C.c), t = dot(v, a); return Math.hypot(...sub(v, mul(a, t))); };
      const hmid = (C.h0 + C.h1) / 2, u2 = [0, hmid >= hp ? 1 : -1];
      const ref0 = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0], ea = norm(cross(a, ref0)), dl = 1e-3 * Math.max(1, R);
      // which side of the rim the planar face is on: probe the solid just under it, inside and outside the circle
      let inner = P.rho ? P.rho[0] < R - 1e-6 && P.rho[1] <= R + 1e-6 : rhoOf(P.center) < R;
      if (ins) {
        const at = rho => ins(add(C.c, add(mul(a, hp + u2[1] * dl * 3), mul(ea, rho))));
        const iIn = at(R - dl * 3), iOut = at(R + dl * 3);
        if (iIn !== iOut) inner = iIn; else if (iIn && iOut) inner = false;
      }
      const u1 = [inner ? -1 : 1, 0];
      const W = wedge2(u1, u2, r, kind);
      if (!W) return null;
      const ref = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0], e1 = norm(cross(a, ref));
      const mh = p => { const v = sub(p, C.c), t = dot(v, a), rv = sub(v, mul(a, t)), rho = Math.hypot(...rv); return [rho - R, t - hp, rho > 1e-9 ? mul(rv, 1 / rho) : e1]; };
      return {
        region: p => { const q = mh(p); return W.region(q[0], q[1]); },
        near: p => { const q = mh(p), r2_ = W.near(q[0], q[1]); return r2_ && [r2_[0], add(mul(q[2], r2_[1]), mul(a, r2_[2]))]; },
        probe: add(C.c, add(mul(a, hp + W.probe[1]), mul(e1, R + W.probe[0])))
      };
    }
    return null;
  }
  // exact 1D squared distance transform (Felzenszwalb & Huttenlocher), spacing h
  function edt1(f, n, h) {
    const d = new Float64Array(n), v = new Int32Array(n), z = new Float64Array(n + 1);
    let k = 0; v[0] = 0; z[0] = -Infinity; z[1] = Infinity;
    for (let q = 1; q < n; q++) {
      let sv;
      while (true) {
        const p = v[k];
        sv = ((f[q] + (q * h) ** 2) - (f[p] + (p * h) ** 2)) / (2 * h * (q - p));
        if (sv <= z[k] && k > 0) k--; else break;
      }
      if (sv <= z[k]) { v[0] = q; z[0] = -Infinity; z[1] = Infinity; k = 0; continue; }
      k++; v[k] = q; z[k] = sv; z[k + 1] = Infinity;
    }
    k = 0;
    for (let q = 0; q < n; q++) { while (z[k + 1] < q * h) k++; d[q] = (q - v[k]) ** 2 * h * h + f[v[k]]; }
    return d;
  }
  function edt3(seed, nx, ny, nz, dx, dy, dz) {           // seed: 1 = distance 0; returns squared distances
    const N = nx * ny * nz, D = new Float64Array(N);
    for (let i = 0; i < N; i++) D[i] = seed[i] ? 0 : 1e30;
    const pass = (n, h, idx) => { const f = new Float64Array(n); for (let q = 0; q < n; q++) f[q] = D[idx(q)]; const d = edt1(f, n, h); for (let q = 0; q < n; q++) D[idx(q)] = d[q]; };
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) pass(nx, dx, i => (k * ny + j) * nx + i);
    for (let k = 0; k < nz; k++) for (let i = 0; i < nx; i++) pass(ny, dy, j => (k * ny + j) * nx + i);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) pass(nz, dz, k => (k * ny + j) * nx + i);
    return D;
  }

  // ---- STL: ray-parity inside test, faces = facets grouped by normal --------
  function parseSTL(data) {
    let bytes = null;
    if (typeof data === 'string') {
      if (/^\s*solid[\s\S]*facet/.test(data.slice(0, 2000))) return parseAsciiSTL(data);
      bytes = Uint8Array.from(atob(data), ch => ch.charCodeAt(0));
    } else bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
    const head = String.fromCharCode.apply(null, Array.from(bytes.subarray(0, Math.min(400, bytes.length))));
    if (/^\s*solid/.test(head) && /facet/.test(head)) return parseAsciiSTL(new TextDecoder().decode(bytes));
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const n = dv.getUint32(80, true), out = new Float32Array(n * 9);
    for (let i = 0; i < n; i++) for (let k = 0; k < 9; k++) out[i * 9 + k] = dv.getFloat32(84 + i * 50 + 12 + k * 4, true);
    return out;
  }
  function parseAsciiSTL(text) {
    const v = [], re = /vertex\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)/g;
    let m;
    while ((m = re.exec(text))) v.push(+m[1], +m[2], +m[3]);
    return Float32Array.from(v.slice(0, v.length - v.length % 9));
  }
  function b64(f32) {
    const u8 = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
    let s = '';
    for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
    return (typeof btoa === 'function' ? btoa(s) : Buffer.from(s, 'binary').toString('base64'));
  }
  function unb64(s) {
    const bin = typeof atob === 'function' ? atob(s) : Buffer.from(s, 'base64').toString('binary');
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(u8.buffer);
  }
  const stlCache = new Map();
  function stlBody(f) {
    const key = f.id + ':' + (f.data ? f.data.length : 0) + ':' + f.scale + ':' + f.tx + ':' + f.ty + ':' + f.tz;
    let G = stlCache.get(key);
    if (!G) { G = prepareSTL(f); stlCache.set(key, G); if (stlCache.size > 8) stlCache.delete(stlCache.keys().next().value); }
    const patches = G.groups.map((g, i) => ({ id: f.id + '/f' + i, name: 'face ' + (i + 1) }));
    return { bbox: G.bbox, patches, inside: G.inside, near: G.near,
      warn: G.openEdges ? (f.name || f.id) + ': ' + G.openEdges + ' open edges (not watertight) — solid decided by three-ray voting' : null };
  }
  function prepareSTL(f) {
    const raw = f.data ? unb64(f.data) : new Float32Array(0), s = f.scale || 1, T = [f.tx || 0, f.ty || 0, f.tz || 0];
    const nt = Math.floor(raw.length / 9), V = new Float64Array(nt * 9);
    for (let i = 0; i < nt * 9; i++) V[i] = raw[i] * s + T[i % 3];
    const N = new Float64Array(nt * 3);
    const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (let t = 0; t < nt; t++) {
      const a = [V[t * 9], V[t * 9 + 1], V[t * 9 + 2]], b = [V[t * 9 + 3], V[t * 9 + 4], V[t * 9 + 5]], c = [V[t * 9 + 6], V[t * 9 + 7], V[t * 9 + 8]];
      const n = norm(cross(sub(b, a), sub(c, a)));
      N[t * 3] = n[0]; N[t * 3 + 1] = n[1]; N[t * 3 + 2] = n[2];
      for (const p of [a, b, c]) for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], p[k]); mx[k] = Math.max(mx[k], p[k]); }
    }
    if (!nt) { mn[0] = mn[1] = mn[2] = 0; mx[0] = mx[1] = mx[2] = 0; }
    // faces: flood fill across shared vertices while normals stay within ~20°
    const vkey = (x, y, z) => Math.round(x * 1e4) + ',' + Math.round(y * 1e4) + ',' + Math.round(z * 1e4);
    const byV = new Map();
    for (let t = 0; t < nt; t++) for (let k = 0; k < 3; k++) {
      const kk = vkey(V[t * 9 + k * 3], V[t * 9 + k * 3 + 1], V[t * 9 + k * 3 + 2]);
      let l = byV.get(kk); if (!l) byV.set(kk, l = []); l.push(t);
    }
    const grp = new Int32Array(nt).fill(-1), groups = [];
    for (let t = 0; t < nt; t++) {
      if (grp[t] >= 0) continue;
      const g = groups.length, stack = [t], n0 = [N[t * 3], N[t * 3 + 1], N[t * 3 + 2]];
      grp[t] = g; groups.push({ n: n0 });
      while (stack.length) {
        const u = stack.pop();
        for (let k = 0; k < 3; k++) for (const w of byV.get(vkey(V[u * 9 + k * 3], V[u * 9 + k * 3 + 1], V[u * 9 + k * 3 + 2]))) {
          if (grp[w] >= 0) continue;
          if (N[w * 3] * N[u * 3] + N[w * 3 + 1] * N[u * 3 + 1] + N[w * 3 + 2] * N[u * 3 + 2] < 0.94) continue;
          grp[w] = g; stack.push(w);
        }
      }
    }
    // open edges (used by one facet only): the mesh is not watertight
    const ecount = new Map();
    for (let t = 0; t < nt; t++) for (let k = 0; k < 3; k++) {
      const a = vkey(V[t * 9 + k * 3], V[t * 9 + k * 3 + 1], V[t * 9 + k * 3 + 2]), j = (k + 1) % 3;
      const b = vkey(V[t * 9 + j * 3], V[t * 9 + j * 3 + 1], V[t * 9 + j * 3 + 2]), key = a < b ? a + '|' + b : b + '|' + a;
      ecount.set(key, (ecount.get(key) || 0) + 1);
    }
    let openEdges = 0;
    for (const c of ecount.values()) if (c === 1) openEdges++;
    // per axis: 2D bins over the other two coordinates for a parity ray along it; 3D bins for nearest facet
    const B = Math.max(4, Math.min(64, Math.round(Math.sqrt(nt / 2)))), ext = sub(mx, mn).map(v => v || 1);
    const binsA = [0, 1, 2].map(() => Array.from({ length: B * B }, () => []));
    const bins = binsA[0];
    const bin3 = new Map(), C = Math.max(ext[0], ext[1], ext[2]) / Math.max(4, Math.min(48, Math.round(Math.cbrt(nt)))) || 1;
    for (let t = 0; t < nt; t++) {
      let y0 = Infinity, y1 = -Infinity, z0 = Infinity, z1 = -Infinity;
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let k = 0; k < 3; k++) {
        const y = V[t * 9 + k * 3 + 1], z = V[t * 9 + k * 3 + 2];
        y0 = Math.min(y0, y); y1 = Math.max(y1, y); z0 = Math.min(z0, z); z1 = Math.max(z1, z);
        for (let q = 0; q < 3; q++) { lo[q] = Math.min(lo[q], V[t * 9 + k * 3 + q]); hi[q] = Math.max(hi[q], V[t * 9 + k * 3 + q]); }
      }
      void y0; void y1; void z0; void z1;
      for (let ax = 0; ax < 3; ax++) {
        const o1 = (ax + 1) % 3, o2 = (ax + 2) % 3;
        const a0 = Math.max(0, Math.floor((lo[o1] - mn[o1]) / ext[o1] * B)), a1 = Math.min(B - 1, Math.floor((hi[o1] - mn[o1]) / ext[o1] * B));
        const c0 = Math.max(0, Math.floor((lo[o2] - mn[o2]) / ext[o2] * B)), c1 = Math.min(B - 1, Math.floor((hi[o2] - mn[o2]) / ext[o2] * B));
        for (let a = a0; a <= a1; a++) for (let b = c0; b <= c1; b++) binsA[ax][a * B + b].push(t);
      }
      for (let i = Math.floor(lo[0] / C); i <= Math.floor(hi[0] / C); i++) for (let j = Math.floor(lo[1] / C); j <= Math.floor(hi[1] / C); j++)
        for (let k = Math.floor(lo[2] / C); k <= Math.floor(hi[2] / C); k++) {
          const kk = i + ',' + j + ',' + k; let l = bin3.get(kk); if (!l) bin3.set(kk, l = []); l.push(t);
        }
    }
    // parity of crossings of a ray from p along +axis
    const parity = (p, ax) => {
      const o1 = (ax + 1) % 3, o2 = (ax + 2) % 3;
      const ba = Math.min(B - 1, Math.max(0, Math.floor((p[o1] - mn[o1]) / ext[o1] * B))), bb = Math.min(B - 1, Math.max(0, Math.floor((p[o2] - mn[o2]) / ext[o2] * B)));
      const y = p[o1] + 1.3e-7 * ext[o1], z = p[o2] + 2.1e-7 * ext[o2];   // jitter keeps the ray off shared edges
      let cnt = 0;
      for (const t of binsA[ax][ba * B + bb]) {
        const o = t * 9;
        const ay = V[o + o1], az = V[o + o2], by_ = V[o + 3 + o1], bz_ = V[o + 3 + o2], cy = V[o + 6 + o1], cz = V[o + 6 + o2];
        const d = (by_ - ay) * (cz - az) - (cy - ay) * (bz_ - az);
        if (Math.abs(d) < 1e-18) continue;
        const l1 = ((by_ - y) * (cz - z) - (cy - y) * (bz_ - z)) / d, l2 = ((cy - y) * (az - z) - (ay - y) * (cz - z)) / d, l3 = 1 - l1 - l2;
        if (l1 < 0 || l2 < 0 || l3 < 0) continue;
        if (l1 * V[o + ax] + l2 * V[o + 3 + ax] + l3 * V[o + 6 + ax] > p[ax]) cnt++;
      }
      return (cnt & 1) === 1;
    };
    // watertight: one ray decides; with holes or flipped facets, three axis rays vote
    const inside = p => {
      if (!nt || p[0] < mn[0] || p[0] > mx[0] || p[1] < mn[1] || p[1] > mx[1] || p[2] < mn[2] || p[2] > mx[2]) return false;
      if (!openEdges) return parity(p, 0);
      return (parity(p, 0) ? 1 : 0) + (parity(p, 1) ? 1 : 0) + (parity(p, 2) ? 1 : 0) >= 2;
    };
    const near = p => {
      const ci = Math.floor(p[0] / C), cj = Math.floor(p[1] / C), ck = Math.floor(p[2] / C);
      let best = Infinity, bt = -1;
      for (let rad = 0; rad <= 3 && (bt < 0 || rad <= 1); rad++) {
        for (let i = ci - rad; i <= ci + rad; i++) for (let j = cj - rad; j <= cj + rad; j++) for (let k = ck - rad; k <= ck + rad; k++) {
          if (Math.max(Math.abs(i - ci), Math.abs(j - cj), Math.abs(k - ck)) !== rad) continue;
          const l = bin3.get(i + ',' + j + ',' + k); if (!l) continue;
          for (const t of l) { const d = triDist(p, V, t * 9); if (d < best) { best = d; bt = t; } }
        }
      }
      if (bt < 0) return [Infinity, 0, null];
      return [best, grp[bt], [N[bt * 3], N[bt * 3 + 1], N[bt * 3 + 2]]];
    };
    return { bbox: [mn, mx], groups, inside, near, nt, openEdges };
  }
  function triDist(p, V, o) {                            // point-triangle distance (Ericson)
    const a = [V[o], V[o + 1], V[o + 2]], b = [V[o + 3], V[o + 4], V[o + 5]], c = [V[o + 6], V[o + 7], V[o + 8]];
    const ab = sub(b, a), ac = sub(c, a), ap = sub(p, a);
    const d1 = dot(ab, ap), d2 = dot(ac, ap);
    let q;
    if (d1 <= 0 && d2 <= 0) q = a;
    else {
      const bp = sub(p, b), d3 = dot(ab, bp), d4 = dot(ac, bp);
      if (d3 >= 0 && d4 <= d3) q = b;
      else {
        const vc = d1 * d4 - d3 * d2;
        if (vc <= 0 && d1 >= 0 && d3 <= 0) q = add(a, mul(ab, d1 / (d1 - d3)));
        else {
          const cp = sub(p, c), d5 = dot(ab, cp), d6 = dot(ac, cp);
          if (d6 >= 0 && d5 <= d6) q = c;
          else {
            const vb = d5 * d2 - d1 * d6;
            if (vb <= 0 && d2 >= 0 && d6 <= 0) q = add(a, mul(ac, d2 / (d2 - d6)));
            else {
              const va = d3 * d6 - d5 * d4;
              if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) q = add(b, mul(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
              else { const den = 1 / (va + vb + vc); q = add(a, add(mul(ab, vb * den), mul(ac, vc * den))); }
            }
          }
        }
      }
    }
    return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
  }

  // ---- copies (patterns, mirror) -----------------------------------------
  function transformed(body, id, label, fwd, inv, rotN) {
    // fwd maps source -> copy, inv maps copy -> source; rotN maps a source normal to the copy
    return {
      bbox: bboxOfPoints(cornersOf(body.bbox).map(fwd)),
      patches: body.patches.map(p => ({ id: id + '/' + p.id, name: label + ' · ' + p.name, plane: p.plane ? { normal: rotN(p.plane.normal) } : null })),
      inside: p => body.inside(inv(p)),
      near: p => { const r = body.near(inv(p)); return [r[0], r[1], r[2] ? rotN(r[2]) : null]; },
      nearAll: p => (body.nearAll ? body.nearAll(inv(p)) : [body.near(inv(p))]).map(r => [r[0], r[1], r[2] ? rotN(r[2]) : null])
    };
  }
  function cornersOf(bb) {
    const out = [];
    for (const x of [bb[0][0], bb[1][0]]) for (const y of [bb[0][1], bb[1][1]]) for (const z of [bb[0][2], bb[1][2]]) out.push([x, y, z]);
    return out;
  }
  function rotAbout(axis, c, ang) {
    const k = axis, cs = Math.cos(ang), sn = Math.sin(ang);
    const R = v => add(add(mul(v, cs), mul(cross(k, v), sn)), mul(k, dot(k, v) * (1 - cs)));
    return { R, p: q => add(c, R(sub(q, c))) };
  }

  // ---------------------------------------------------------------- model
  const FEATURE_INFO = {
    sketch: { label: 'Sketch', solid: false },
    extrude: { label: 'Extrude', solid: true },
    revolve: { label: 'Revolve', solid: true },
    box: { label: 'Box', solid: true },
    cylinder: { label: 'Cylinder', solid: true },
    sphere: { label: 'Sphere', solid: true },
    hole: { label: 'Hole', solid: true },
    lpattern: { label: 'LPattern', solid: true },
    cpattern: { label: 'CirPattern', solid: true },
    mirror: { label: 'Mirror', solid: true },
    stl: { label: 'Imported', solid: true },
    loft: { label: 'Loft', solid: true },
    sweep: { label: 'Sweep', solid: true },
    fillet: { label: 'Fillet', solid: true, modifier: true },
    chamfer: { label: 'Chamfer', solid: true, modifier: true },
    shell: { label: 'Shell', solid: true, modifier: true }
  };
  function modelExtent(model) {
    let e = 100;
    for (const f of model.features || []) {
      if (f.suppressed) continue;
      if (f.type === 'box') e = Math.max(e, Math.abs(f.x || 0) + Math.abs(f.sx || 0), Math.abs(f.y || 0) + Math.abs(f.sy || 0), Math.abs(f.z || 0) + Math.abs(f.sz || 0));
      if (f.type === 'extrude' || f.type === 'cylinder') e = Math.max(e, Math.abs(f.depth || f.h || 0));
      if (f.type === 'sketch') for (const en of f.entities || []) {
        if (en.type === 'circle') e = Math.max(e, Math.abs(en.cx) + en.r, Math.abs(en.cy) + en.r);
        if (en.type === 'rect') e = Math.max(e, Math.abs(en.x) + Math.abs(en.w), Math.abs(en.y) + Math.abs(en.h));
        if (en.type === 'poly') for (const p of en.pts) e = Math.max(e, Math.abs(p[0]), Math.abs(p[1]));
      }
    }
    return e;
  }
  // one feature -> { body, op } (null for sketches, missing references, suppressed)
  function bodyOf(f, model, byId, depth) {
    if (!f || f.suppressed || (depth || 0) > 4) return null;
    const sk = f.sketch ? byId[f.sketch] : null;
    let body = null;
    switch (f.type) {
      case 'extrude': body = sk && !sk.suppressed ? extrudeBody(f, sk) : null; break;
      case 'revolve': body = sk && !sk.suppressed ? revolveBody(f, sk) : null; break;
      case 'box': body = boxBody(f); break;
      case 'cylinder': body = cylinderBody(f); break;
      case 'sphere': body = sphereBody(f); break;
      case 'hole': body = holeBody(f, model); break;
      case 'stl': body = stlBody(f); break;
      case 'loft': { const a = byId[f.sketch], b = byId[f.sketch2]; body = a && b && !a.suppressed && !b.suppressed ? loftBody(f, a, b) : null; break; }
      case 'sweep': { const a = byId[f.sketch], b = byId[f.path]; body = a && b && !a.suppressed && !b.suppressed ? sweepBody(f, a, b) : null; break; }
      case 'lpattern': case 'cpattern': case 'mirror': {
        const src = byId[f.src], sb = src && bodyOf(src, model, byId, (depth || 0) + 1);
        if (!sb) return null;
        const copies = [];
        if (f.type === 'lpattern') {
          const d = dirVec(f.dir || '+x'), n = Math.max(2, Math.round(f.count || 2));
          for (let i = 1; i < n; i++) {
            const o = mul(d, (f.spacing || 10) * i);
            for (const [bi, bd] of sb.bodies.entries()) copies.push(transformed(bd, f.id + '#' + i + (bi ? '.' + bi : ''), (f.name || f.id) + ' copy ' + i, q => add(q, o), q => sub(q, o), v => v));
          }
        } else if (f.type === 'cpattern') {
          const a = dirVec(f.axis || '+y'), c = [f.cx || 0, f.cy || 0, f.cz || 0], n = Math.max(2, Math.round(f.count || 4));
          const tot = (f.angle || 360) * Math.PI / 180, step = Math.abs(tot) >= 2 * Math.PI - 1e-6 ? tot / n : tot / (n - 1);
          for (let i = 1; i < n; i++) {
            const F = rotAbout(a, c, step * i), I = rotAbout(a, c, -step * i);
            for (const [bi, bd] of sb.bodies.entries()) copies.push(transformed(bd, f.id + '#' + i + (bi ? '.' + bi : ''), (f.name || f.id) + ' copy ' + i, F.p, I.p, F.R));
          }
        } else {
          const ax = { XY: 2, XZ: 1, YZ: 0 }[f.plane || 'YZ'], o = f.offset || 0;
          const M = q => { const r = q.slice(); r[ax] = 2 * o - r[ax]; return r; };
          const MN = v => { const r = v.slice(); r[ax] = -r[ax]; return r; };
          for (const [bi, bd] of sb.bodies.entries()) copies.push(transformed(bd, f.id + '#1' + (bi ? '.' + bi : ''), (f.name || f.id) + ' mirror', M, M, MN));
        }
        return { op: sb.op, bodies: copies };
      }
      default: return null;
    }
    if (!body) return null;
    const op = f.type === 'hole' ? 'cut' : f.op === 'cut' ? 'cut' : 'boss';
    for (const p of body.patches) p.feat = f.id;
    return { op, bodies: [body] };
  }
  function applyStep(st, p, s) {
    if (st.op === 'boss') return s || st.body.inside(p);
    if (st.op === 'cut') return s && !st.body.inside(p);
    if (st.op === 'blend') return st.mode === 'remove' ? s && !st.region(p) : s || st.region(p);
    return s;                                                // shell: grid only
  }
  // evaluate the whole feature list once; returns the compiled part
  function compile(model) {
    const feats = model.features || [], byId = {};
    for (const f of feats) byId[f.id] = f;
    const steps = [], faces = {}, geom = {}, warnings = [];
    let bb = null;
    const insideUpTo = (p, n) => {
      let s = false;
      for (let i = 0; i < n; i++) s = applyStep(steps[i], p, s);
      return s;
    };
    for (const f of feats) {
      if (!FEATURE_INFO[f.type] || !FEATURE_INFO[f.type].solid || f.suppressed) continue;
      if (f.type === 'fillet' || f.type === 'chamfer') {
        const r = Math.abs(f.r || 1), patches = [];
        (f.edges || []).forEach((pair, i) => {
          const B = edgeBlend(geom[pair[0]], geom[pair[1]], r, f.type, p => insideUpTo(p, steps.length));
          if (!B) { warnings.push((f.name || f.id) + ': edge ' + (i + 1) + ' is not between two planes or a plane and a cylinder rim — skipped'); return; }
          const mode = insideUpTo(B.probe, steps.length) ? 'remove' : 'add', pid = f.id + '/e' + i;
          const body = { patches: [{ id: pid, name: (f.type === 'fillet' ? 'fillet face ' : 'chamfer face ') + (i + 1) }],
            near: p => { const q = B.near(p); return q ? [q[0], 0, mode === 'remove' ? q[1] : mul(q[1], -1)] : [Infinity, 0, null]; } };
          steps.push({ op: 'blend', mode, region: B.region, body, feat: f.id });
          faces[pid] = { id: pid, feat: f.id, op: 'boss', name: (f.name || f.id) + ' · ' + body.patches[0].name, drag: null, plane: null };
          patches.push(pid);
        });
        continue;
      }
      if (f.type === 'shell') {
        steps.push({ op: 'shell', t: Math.abs(f.t || 2), open: new Set(f.open || []), feat: f.id });
        faces[f.id + '/inner'] = { id: f.id + '/inner', feat: f.id, op: 'cut', name: (f.name || f.id) + ' · inner faces', drag: null, plane: null };
        continue;
      }
      const r = bodyOf(f, model, byId, 0);
      if (!r) continue;
      for (const b of r.bodies) {
        steps.push({ op: r.op, body: b, feat: f.id });
        if (b.warn) warnings.push(b.warn);
        for (const p of b.patches) {
          faces[p.id] = { id: p.id, feat: f.id, op: r.op, name: (f.name || f.id) + ' · ' + p.name, drag: p.drag || null, plane: p.plane || null };
          if (p.plane || p.cyl) geom[p.id] = { plane: p.plane && p.plane.corners ? p.plane : null, cyl: p.cyl || null };
        }
        if (r.op === 'boss') bb = bb ? [[Math.min(bb[0][0], b.bbox[0][0]), Math.min(bb[0][1], b.bbox[0][1]), Math.min(bb[0][2], b.bbox[0][2])],
          [Math.max(bb[1][0], b.bbox[1][0]), Math.max(bb[1][1], b.bbox[1][1]), Math.max(bb[1][2], b.bbox[1][2])]] : [b.bbox[0].slice(), b.bbox[1].slice()];
      }
    }
    // point queries treat a shell as a no-op: shells act on the voxel grid (see voxelize)
    const inside = p => insideUpTo(p, steps.length);
    // nearest named face to a point on the voxel skin. n is the skin's outward
    // normal; a patch whose own outward side disagrees is penalised (a cut's
    // surface faces the other way: the part's outside is the cut's inside).
    const faceAt = (p, n, h) => {
      let best = Infinity, id = null;
      for (const st of steps) {
        if (!st.body) continue;
        const r = st.body.near(p);
        if (!(r[0] < Infinity)) continue;
        let d = r[0];
        if (n && r[2]) { const c = dot(n, r[2]) * (st.op === 'cut' ? -1 : 1); if (c < -0.3) d += (h || 1); }
        if (d < best - 1e-9) { best = d; id = st.body.patches[r[1]] ? st.body.patches[r[1]].id : null; }
      }
      // skin a shell opened up, far from every feature surface
      if (shellIds.length && best > Math.max(1.5 * (h || 1), 1e-9)) return shellIds[shellIds.length - 1] + '/inner';
      return id;
    };
    const shellIds = steps.filter(st => st.op === 'shell').map(st => st.feat);
    // Move a point of the voxel skin onto the part's real surface (body-fitted
    // mesh). n is the skin's outward normal there. Projections onto every surface
    // within reach are repeated, which settles points on edges and corners;
    // surfaces facing away (the far side of a thin wall) are ignored. Returns
    // null when the result would not lie on the part's skin.
    const project = (p, n, h, info) => {
      const tol = 0.9 * h, maxMove = 0.85 * h;
      let q = p.slice(), moved = false;
      const near = (st, x) => st.body.nearAll ? st.body.nearAll(x) : [st.body.near(x)];
      for (let it = 0; it < 6; it++) {
        let any = false;
        const cand = [];
        let aligned = false;
        for (const st of steps) {
          if (!st.body) continue;
          for (const r of near(st, q)) {
            if (!(r[0] < tol) || !r[2]) continue;
            const out = st.op === 'cut' ? mul(r[2], -1) : r[2], c = dot(out, n);
            if (c >= 0.3) aligned = true;                   // also when the point already lies on it
            if (c >= -0.2 && r[0] >= 1e-7 * h) cand.push([st, r, out, c]);
          }
        }
        // A surface across the skin normal is either the side wall at a rim (then a
        // surface facing along the skin is in reach too: the cap) or a gently sloping
        // wall seen as a staircase (nothing else in reach). At a rim it only pulls in
        // points that stick out past it, never drags cap points onto it.
        for (const [st, r, out, c] of cand) {
          const d = near(st, q).find(b => b[1] === r[1]);
          if (!d || !(d[0] > 1e-7 * h)) continue;           // already settled on it this pass
          const ins = inside(q);
          if (c < 0.3 && aligned && ins) continue;
          // a step is kept only if it really lands on the surface (the distance to a
          // cap from beyond its edge is not measured along its normal)
          const q2 = add(q, mul(out, ins ? d[0] : -d[0])), back = near(st, q2).find(b => b[1] === r[1]);
          if (!back || !(back[0] < 0.25 * d[0])) continue;
          q = q2; any = moved = true;
        }
        if (!any) break;
      }
      if (!moved) return null;
      if (Math.hypot(...sub(q, p)) > maxMove) return null;
      const e = 0.06 * h;                                 // clear of the polygonised arcs inside() tests against
      if (inside(add(q, mul(n, e))) || !inside(sub(q, mul(n, e)))) return null;
      if (info) {                                         // which surfaces the point settled on
        const ids = [];
        steps.forEach((st, si) => { if (st.body) for (const r of near(st, q)) if (r[0] < 1e-3 * h && r[2]) ids.push(si + ':' + r[1]); });
        info.key = ids.sort().join(',');
      }
      return q;
    };
    return { model, steps, faces, inside, faceAt, insideUpTo, project, bbox: bb || [[0, 0, 0], [100, 100, 100]], empty: !bb, byId, geom, warnings };
  }

  // ------------------------------------------------------------- meshing
  // A structured voxel grid over the part's bounding box. Spacing is fitted to
  // the box so faces at its limits land on cell boundaries exactly; sub > 1
  // samples sub³ points per cell and keeps cells that are at least half solid.
  function voxelize(cm, opt) {
    const o = opt || {}, bb = cm.bbox, ext = sub(bb[1], bb[0]).map(v => Math.max(v, 1e-6));
    let h = Math.max(0.05, o.h || 5), aspect = o.aspect || 1, budget = o.budget || 30000, subS = Math.max(1, Math.min(3, Math.round(o.sub || 2)));
    // solid fraction from a coarse probe, to honour the element budget
    let hits = 0, tot = 0;
    const PR = 14;
    for (let i = 0; i < PR; i++) for (let j = 0; j < PR; j++) for (let k = 0; k < PR; k++) {
      tot++; if (cm.inside([bb[0][0] + ext[0] * (i + 0.5) / PR, bb[0][1] + ext[1] * (j + 0.5) / PR, bb[0][2] + ext[2] * (k + 0.5) / PR])) hits++;
    }
    const frac = Math.max(0.02, hits / tot);
    const dims = hh => [Math.max(1, Math.round(ext[0] / hh)), Math.max(1, Math.round(ext[1] / hh)), Math.max(1, Math.round(ext[2] / (hh * aspect)))];
    let clamped = false, guard = 0, d = dims(h);
    if (!o.noBudget) while (d[0] * d[1] * d[2] * frac > budget && guard++ < 400) { h *= 1.06; d = dims(h); clamped = true; }
    const [nx, ny, nz] = d, dx = ext[0] / nx, dy = ext[1] / ny, dz = ext[2] / nz;
    const solid = new Uint8Array(nx * ny * nz), need = Math.ceil(subS * subS * subS / 2);
    const steps = cm.steps || [], cuts = [];
    steps.forEach((st, i) => { if (st.op === 'shell') cuts.push(i); });
    const ranges = [];
    let from = 0;
    for (const c of cuts) { ranges.push([from, c]); from = c + 1; }
    ranges.push([from, steps.length]);
    // evaluate steps [i0, i1) on every cell, starting from the cell's current state
    const stage = (i0, i1, first) => {
      for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const ci = (k * ny + j) * nx + i, s0 = first ? false : solid[ci] === 1;
        const run = p => { let st_ = s0; for (let q = i0; q < i1; q++) st_ = applyStep(steps[q], p, st_); return st_; };
        let c = 0;
        if (subS === 1) c = run([bb[0][0] + (i + 0.5) * dx, bb[0][1] + (j + 0.5) * dy, bb[0][2] + (k + 0.5) * dz]) ? 1 : 0;
        else for (let a = 0; a < subS && c < need; a++) for (let b = 0; b < subS; b++) for (let e = 0; e < subS; e++)
          if (run([bb[0][0] + (i + (a + 0.5) / subS) * dx, bb[0][1] + (j + (b + 0.5) / subS) * dy, bb[0][2] + (k + (e + 0.5) / subS) * dz])) c++;
        solid[ci] = c >= (subS === 1 ? 1 : need) ? 1 : 0;
      }
    };
    if (!steps.length && cm.inside) { stage(0, 0, true); }
    ranges.forEach((rg, n) => {
      stage(rg[0], rg[1], n === 0);
      if (n < cuts.length) shellGrid(cm, steps[cuts[n]], solid, nx, ny, nz, dx, dy, dz, bb[0]);
    });
    let ne = 0;
    for (let q = 0; q < solid.length; q++) ne += solid[q];
    return { nx, ny, nz, dx, dy, dz, solid, ne, org: bb[0].slice(), h, clamped, frac };
  }

  // Shell on the grid: cells farther than t from the part's skin are removed. Skin
  // on the open faces does not count, so the cavity breaks out through them.
  function shellGrid(cm, st, solid, nx, ny, nz, dx, dy, dz, org) {
    const seed = new Uint8Array(solid.length), h = Math.min(dx, dy, dz);
    const D6 = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const ci = (k * ny + j) * nx + i;
      if (!solid[ci]) continue;
      for (const D of D6) {
        const a = i + D[0], b = j + D[1], c = k + D[2];
        if (a >= 0 && b >= 0 && c >= 0 && a < nx && b < ny && c < nz && solid[(c * ny + b) * nx + a]) continue;
        if (st.open.size) {
          const p = [org[0] + (i + 0.5 + D[0] * 0.5) * dx, org[1] + (j + 0.5 + D[1] * 0.5) * dy, org[2] + (k + 0.5 + D[2] * 0.5) * dz];
          if (st.open.has(cm.faceAt(p, D, h))) continue;   // an opening: not a wall
        }
        seed[ci] = 1; break;
      }
    }
    const D2 = edt3(seed, nx, ny, nz, dx, dy, dz), lim = (st.t - 0.5 * h) ** 2;
    for (let q = 0; q < solid.length; q++) if (solid[q] && D2[q] > lim + 1e-9) solid[q] = 0;
  }

  // ------------------------------------------------- sketch constraints
  // A 2D geometric constraint solver, like a CAD sketcher's. The unknowns are the
  // entities' own numbers (polyline vertices, rectangle x, y, w, h, circle
  // centre and radius); constraints are residuals that must vanish; driving
  // dimensions are constraints with a value. Solved by damped Gauss–Newton
  // (Levenberg–Marquardt) from the current geometry, so unconstrained parts move
  // as little as possible. The rank of the Jacobian gives the remaining degrees
  // of freedom (0 = fully defined).
  //
  // References: point { e, p } (poly vertex p, rect corner 0..3, 'c' circle centre)
  // or { origin: true }; line { e, l } (poly edge l, rect edge 0..3); circle { e }.
  const CONSTRAINTS = {
    fix: { label: 'Fix', refs: ['point'] },
    coincident: { label: 'Coincident', refs: ['point', 'point'] },
    horizontal: { label: 'Horizontal', refs: ['line'] },
    vertical: { label: 'Vertical', refs: ['line'] },
    parallel: { label: 'Parallel', refs: ['line', 'line'] },
    perpendicular: { label: 'Perpendicular', refs: ['line', 'line'] },
    equal: { label: 'Equal', refs: ['line|circle', 'line|circle'] },
    tangent: { label: 'Tangent', refs: ['line', 'circle'] },
    onLine: { label: 'On line', refs: ['point', 'line'] },
    length: { label: 'Length', refs: ['line'], dim: true },
    distance: { label: 'Distance', refs: ['point', 'point'], dim: true },
    hdist: { label: 'Horizontal distance', refs: ['point', 'point'], dim: true },
    vdist: { label: 'Vertical distance', refs: ['point', 'point'], dim: true },
    radius: { label: 'Radius', refs: ['circle'], dim: true },
    diameter: { label: 'Diameter', refs: ['circle'], dim: true },
    angle: { label: 'Angle', refs: ['line', 'line'], dim: true, unit: '°' }
  };
  function sketchModel(ents) {
    const idx = [], x0 = [];
    ents.forEach(e => {
      idx.push(x0.length);
      if (e.type === 'poly') for (const q of e.pts) x0.push(q[0], q[1]);
      else if (e.type === 'rect') x0.push(e.x, e.y, e.w, e.h);
      else if (e.type === 'circle') x0.push(e.cx, e.cy, e.r);
    });
    const ok = r => r && (r.origin || (ents[r.e] && (r.p === undefined || ents[r.e].type !== 'poly' || r.p < ents[r.e].pts.length)));
    const pt = (X, r) => {
      if (r.origin) return [0, 0];
      const e = ents[r.e], o = idx[r.e];
      if (e.type === 'poly') return [X[o + 2 * r.p], X[o + 2 * r.p + 1]];
      if (e.type === 'circle') return [X[o], X[o + 1]];
      const x = X[o], y = X[o + 1], w = X[o + 2], h = X[o + 3];
      return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]][r.p || 0];
    };
    const line = (X, r) => {
      const e = ents[r.e], n = e.type === 'poly' ? e.pts.length : 4;
      return [pt(X, { e: r.e, p: r.l }), pt(X, { e: r.e, p: (r.l + 1) % n })];
    };
    const rad = (X, r) => X[idx[r.e] + 2];
    const write = X0 => { const X = X0.map(v => Math.round(v * 1e9) / 1e9); return ents.map((e, k) => {
      const o = idx[k];
      if (e.type === 'poly') return { ...e, pts: e.pts.map((q, i) => [X[o + 2 * i], X[o + 2 * i + 1]]) };
      if (e.type === 'rect') return { ...e, x: X[o], y: X[o + 1], w: X[o + 2], h: X[o + 3] };
      if (e.type === 'circle') return { ...e, cx: X[o], cy: X[o + 1], r: Math.abs(X[o + 2]) };
      return e;
    }); };
    return { x0, pt, line, rad, write, ok, n: x0.length };
  }
  // the current value of a dimension (what a new dimension would be set to)
  function measure(M, X, c) {
    const P = r => M.pt(X, r), Ln = r => M.line(X, r);
    switch (c.type) {
      case 'length': { const [a, b] = Ln(c.a); return Math.hypot(b[0] - a[0], b[1] - a[1]); }
      case 'distance': { const a = P(c.a), b = P(c.b); return Math.hypot(b[0] - a[0], b[1] - a[1]); }
      case 'hdist': return Math.abs(P(c.b)[0] - P(c.a)[0]);
      case 'vdist': return Math.abs(P(c.b)[1] - P(c.a)[1]);
      case 'radius': return M.rad(X, c.a);
      case 'diameter': return 2 * M.rad(X, c.a);
      case 'angle': { const [a, b] = Ln(c.a), [p, q] = Ln(c.b); const u = [b[0] - a[0], b[1] - a[1]], v = [q[0] - p[0], q[1] - p[1]];
        return Math.abs(Math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])) * 180 / Math.PI; }
    }
    return 0;
  }
  function residuals(M, X, cons, Lc) {
    const out = [], P = r => M.pt(X, r), Ln = r => M.line(X, r);
    const dir = r => { const [a, b] = Ln(r); return [b[0] - a[0], b[1] - a[1]]; };
    const len = r => { const d = dir(r); return Math.hypot(d[0], d[1]) || 1e-12; };
    const size = r => (r.l !== undefined ? len(r) : M.rad(X, r));
    for (const c of cons) {
      if (!M.ok(c.a) || (c.b && !M.ok(c.b))) continue;
      switch (c.type) {
        case 'fix': { const a = P(c.a); out.push(a[0] - c.value[0], a[1] - c.value[1]); break; }
        case 'drag': { const a = P(c.a); out.push((a[0] - c.value[0]) * 0.5, (a[1] - c.value[1]) * 0.5); break; }
        case 'coincident': { const a = P(c.a), b = P(c.b); out.push(a[0] - b[0], a[1] - b[1]); break; }
        case 'horizontal': { const [a, b] = Ln(c.a); out.push(a[1] - b[1]); break; }
        case 'vertical': { const [a, b] = Ln(c.a); out.push(a[0] - b[0]); break; }
        case 'parallel': { const u = dir(c.a), v = dir(c.b); out.push((u[0] * v[1] - u[1] * v[0]) / (len(c.a) * len(c.b)) * Lc); break; }
        case 'perpendicular': { const u = dir(c.a), v = dir(c.b); out.push((u[0] * v[0] + u[1] * v[1]) / (len(c.a) * len(c.b)) * Lc); break; }
        case 'equal': out.push(size(c.a) - size(c.b)); break;
        case 'tangent': {
          const [a, b] = Ln(c.a), C = P({ e: c.b.e, p: 'c' }), d = dir(c.a), l = len(c.a);
          out.push(Math.abs((C[0] - a[0]) * d[1] - (C[1] - a[1]) * d[0]) / l - M.rad(X, c.b)); void b; break;
        }
        case 'onLine': { const q = P(c.a), [a] = Ln(c.b), d = dir(c.b); out.push(((q[0] - a[0]) * d[1] - (q[1] - a[1]) * d[0]) / len(c.b)); break; }
        case 'length': case 'distance': case 'radius': case 'diameter': out.push(measure(M, X, c) - c.value); break;
        case 'hdist': case 'vdist': {
          const a = P(c.a), b = P(c.b), k = c.type === 'hdist' ? 0 : 1, sgn = c.sign || Math.sign(b[k] - a[k]) || 1;
          out.push((b[k] - a[k]) * sgn - c.value); break;
        }
        case 'angle': {
          const u = dir(c.a), v = dir(c.b), ang = Math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1]);
          const want = c.value * Math.PI / 180 * (c.sign || Math.sign(ang) || 1);
          let d = ang - want; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
          out.push(d * Lc); break;
        }
      }
    }
    return out;
  }
  function rankOf(J, m, n, tol) {
    const A = J.map(r => r.slice());
    let rank = 0;
    for (let col = 0; col < n && rank < m; col++) {
      let piv = rank, best = Math.abs(A[rank] ? A[rank][col] : 0);
      for (let r = rank + 1; r < m; r++) if (Math.abs(A[r][col]) > best) { best = Math.abs(A[r][col]); piv = r; }
      if (best < tol) continue;
      [A[rank], A[piv]] = [A[piv], A[rank]];
      for (let r = rank + 1; r < m; r++) { const f = A[r][col] / A[rank][col]; if (f) for (let c = col; c < n; c++) A[r][c] -= f * A[rank][c]; }
      rank++;
    }
    return rank;
  }
  function solveLinear(A, b) {                             // dense Gaussian elimination with partial pivoting
    const n = b.length, M = A.map((r, i) => r.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      [M[c], M[piv]] = [M[piv], M[c]];
      const d = M[c][c] || 1e-30;
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / d; if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) { let v = M[r][n]; for (let k = r + 1; k < n; k++) v -= M[r][k] * x[k]; x[r] = v / (M[r][r] || 1e-30); }
    return x;
  }
  // solve a sketch's constraints; opt.drag = { ref, to: [u, v] } pulls a point while solving
  function solveSketch(sk, opt) {
    const o = opt || {}, ents = sk.entities || [], M = sketchModel(ents);
    const cons = (sk.constraints || []).slice();
    if (o.drag) cons.push({ type: 'drag', a: o.drag.ref, value: o.drag.to });
    let X = M.x0.slice();
    const n = M.n;
    let span = 1;
    for (let i = 0; i < n; i++) span = Math.max(span, Math.abs(X[i]));
    const Lc = Math.max(10, span), F = Y => residuals(M, Y, cons, Lc);
    const jac = (Y, r0) => {
      const J = r0.map(() => new Array(n).fill(0));
      for (let j = 0; j < n; j++) {
        const h = 1e-6 * Math.max(1, Math.abs(Y[j])), Yp = Y.slice(); Yp[j] += h;
        const r = F(Yp);
        for (let i = 0; i < r0.length; i++) J[i][j] = (r[i] - r0[i]) / h;
      }
      return J;
    };
    let r = F(X), nr = Math.hypot(...r, 0), lam = 1e-3, it = 0;
    while (nr > 1e-10 && it++ < 100 && n) {
      const J = jac(X, r), A = [], g = [];
      for (let a = 0; a < n; a++) {
        A.push(new Array(n).fill(0)); let s_ = 0;
        for (let i = 0; i < r.length; i++) s_ += J[i][a] * r[i];
        g.push(-s_);
        for (let b = 0; b <= a; b++) { let t = 0; for (let i = 0; i < r.length; i++) t += J[i][a] * J[i][b]; A[a][b] = A[b][a] = t; }
      }
      let improved = false;
      for (let tries = 0; tries < 8 && !improved; tries++) {
        const Al = A.map((row, i) => row.map((v, j) => (i === j ? v * (1 + lam) + lam * 1e-6 : v)));
        const d = solveLinear(Al, g), Y = X.map((v, i) => v + d[i]), rY = F(Y), nY = Math.hypot(...rY, 0);
        if (nY < nr) { X = Y; r = rY; nr = nY; lam = Math.max(1e-9, lam / 4); improved = true; } else lam *= 6;
      }
      if (!improved) break;
    }
    // degrees of freedom left, from the real constraints only (not the drag)
    const real = (sk.constraints || []);
    const rr = residuals(M, X, real, Lc);
    let dof = n;
    if (rr.length) { const J = (() => { const Jt = rr.map(() => new Array(n).fill(0)); for (let j = 0; j < n; j++) { const h = 1e-6 * Math.max(1, Math.abs(X[j])), Yp = X.slice(); Yp[j] += h; const r2 = residuals(M, Yp, real, Lc); for (let i = 0; i < rr.length; i++) Jt[i][j] = (r2[i] - rr[i]) / h; } return Jt; })();
      dof = n - rankOf(J, rr.length, n, 1e-7); }
    const res = Math.hypot(...rr, 0);
    return { entities: M.write(X), residual: res, ok: res < 1e-6 * Lc, dof, conflict: res >= 1e-6 * Lc };
  }
  // the value a new dimension constraint starts from
  function measureConstraint(sk, c) { const M = sketchModel(sk.entities || []); return measure(M, M.x0, c); }
  // what to call a reference, for lists
  function refLabel(sk, r) {
    if (!r) return '?';
    if (r.origin) return 'origin';
    const e = (sk.entities || [])[r.e], nm = e ? ({ poly: e.open ? 'Path' : 'Polyline', rect: 'Rectangle', circle: 'Circle' }[e.type] || e.type) + ' ' + (r.e + 1) : '?';
    if (r.l !== undefined) return nm + ' edge ' + (r.l + 1);
    if (r.p !== undefined) return nm + (r.p === 'c' ? ' centre' : e && e.type === 'rect' ? ' corner ' + (r.p + 1) : ' point ' + (r.p + 1));
    return nm;
  }
  // constraints left after deleting entity k (references above k shift down)
  function dropEntityConstraints(cons, k) {
    const touch = r => r && !r.origin && r.e === k, shift = r => (r && !r.origin && r.e > k ? { ...r, e: r.e - 1 } : r);
    return (cons || []).filter(c => !touch(c.a) && !touch(c.b)).map(c => ({ ...c, a: shift(c.a), b: shift(c.b) }));
  }
  // SolidWorks-style inference while drawing: nearly horizontal / vertical edges get H / V
  function inferHV(e, k, tolDeg) {
    const out = [], tol = Math.tan((tolDeg || 3) * Math.PI / 180);
    if (e.type !== 'poly') return out;
    const n = e.pts.length, m = e.open ? n - 1 : n;
    for (let i = 0; i < m; i++) {
      const a = e.pts[i], b = e.pts[(i + 1) % n], dx = Math.abs(b[0] - a[0]), dy = Math.abs(b[1] - a[1]);
      if (dx > 1e-9 && dy <= tol * dx) out.push({ type: 'horizontal', a: { e: k, l: i } });
      else if (dy > 1e-9 && dx <= tol * dy) out.push({ type: 'vertical', a: { e: k, l: i } });
    }
    return out;
  }

  // ----------------------------------------------------------- editing
  let seq = 1;
  function nextId(model, type) {
    const used = new Set((model.features || []).map(f => f.id));
    let i = 1, id;
    do { id = type + i++; } while (used.has(id));
    return id;
  }
  function defaultName(model, type) {
    const base = { sketch: 'Sketch', extrude: 'Extrude', revolve: 'Revolve', box: 'Box', cylinder: 'Cylinder', sphere: 'Sphere',
      hole: 'Hole', lpattern: 'LPattern', cpattern: 'CirPattern', mirror: 'Mirror', stl: 'Imported', loft: 'Loft', sweep: 'Sweep',
      fillet: 'Fillet', chamfer: 'Chamfer', shell: 'Shell' }[type] || type;
    const n = (model.features || []).filter(f => f.type === type).length + 1;
    return base + n;
  }
  // move a sketch segment along its outward normal by delta (Instant3D on a side face)
  function moveSegment(sketch, seg, delta) {
    const m = /^e(\d+)\.(l|a|c)(\d*)$/.exec(seg.id);
    if (!m) return sketch;
    const ents = sketch.entities.map(e => JSON.parse(JSON.stringify(e)));
    const e = ents[+m[1]];
    if (!e) return sketch;
    if (m[2] === 'c') e.r = Math.max(0.01, e.r + delta);
    else if (m[2] === 'a') {
      const i = +m[3];
      if (e.type === 'rect') e.r = Math.max(0, (e.r || 0) - delta);
      else { e.fillets = e.fillets || e.pts.map(() => 0); e.fillets[i] = Math.max(0, (e.fillets[i] || 0) - delta); }
    } else {
      const i = +m[3];
      const dx = seg.b[0] - seg.a[0], dy = seg.b[1] - seg.a[1], l = Math.hypot(dx, dy) || 1;
      const off = [dy / l * seg.out * delta, -dx / l * seg.out * delta];
      if (e.type === 'rect') {
        const x0 = Math.min(e.x, e.x + e.w), x1 = Math.max(e.x, e.x + e.w), y0 = Math.min(e.y, e.y + e.h), y1 = Math.max(e.y, e.y + e.h);
        const r = { x0, x1, y0, y1 };
        if (i === 0) r.y0 += off[1]; else if (i === 1) r.x1 += off[0]; else if (i === 2) r.y1 += off[1]; else r.x0 += off[0];
        e.x = Math.min(r.x0, r.x1); e.y = Math.min(r.y0, r.y1); e.w = Math.abs(r.x1 - r.x0); e.h = Math.abs(r.y1 - r.y0);
      } else if (e.type === 'poly') {
        const n = e.pts.length;
        for (const k of [i, (i + 1) % n]) e.pts[k] = [e.pts[k][0] + off[0], e.pts[k][1] + off[1]];
      }
    }
    return Object.assign({}, sketch, { entities: ents });
  }
  // where a segment sits, as one editable number (x = 120 for a vertical edge, R 15 for an arc)
  function segValue(seg) {
    if (seg.t === 'arc') return { label: Math.abs(seg.sw) >= 2 * Math.PI - 1e-9 ? 'Radius' : 'Fillet radius', value: seg.r };
    const dx = seg.b[0] - seg.a[0], dy = seg.b[1] - seg.a[1];
    if (Math.abs(dx) < 1e-9) return { label: 'Position u', value: seg.a[0], axis: 0 };
    if (Math.abs(dy) < 1e-9) return { label: 'Position v', value: seg.a[1], axis: 1 };
    const l = Math.hypot(dx, dy), n = [dy / l * seg.out, -dx / l * seg.out];
    return { label: 'Offset', value: seg.a[0] * n[0] + seg.a[1] * n[1], normal: n };
  }
  function featureDefaults(type, model, ctx) {
    const c = ctx || {}, bb = c.bbox || [[0, 0, 0], [100, 100, 100]], mid = [(bb[0][0] + bb[1][0]) / 2, (bb[0][1] + bb[1][1]) / 2, (bb[0][2] + bb[1][2]) / 2];
    const id = nextId(model, type), name = defaultName(model, type), lastSolid = (model.features || []).filter(f => FEATURE_INFO[f.type] && FEATURE_INFO[f.type].solid).slice(-1)[0];
    switch (type) {
      case 'sketch': return { id, type, name, plane: c.plane || 'XY', offset: c.offset || 0, entities: [] };
      case 'extrude': return { id, type, name, sketch: c.sketch, op: c.op || 'boss', depth: 50, dir: 'normal' };
      case 'revolve': return { id, type, name, sketch: c.sketch, op: c.op || 'boss', axis: 'v', angle: 360 };
      case 'box': return { id, type, name, op: c.op || 'boss', x: 0, y: 0, z: 0, sx: 100, sy: 50, sz: 50 };
      case 'cylinder': return { id, type, name, op: c.op || 'boss', x: 0, y: 0, z: 0, axis: '+y', r: 25, h: 100 };
      case 'sphere': return { id, type, name, op: c.op || 'boss', x: 0, y: 0, z: 0, r: 40 };
      case 'hole': return { id, type, name, x: mid[0], y: bb[1][1], z: mid[2], dir: '-y', d: 10, depth: 20, through: true };
      case 'lpattern': return { id, type, name, src: c.src || (lastSolid && lastSolid.id), dir: '+x', count: 3, spacing: 30 };
      case 'cpattern': return { id, type, name, src: c.src || (lastSolid && lastSolid.id), axis: '+y', cx: mid[0], cy: mid[1], cz: mid[2], count: 4, angle: 360 };
      case 'mirror': return { id, type, name, src: c.src || (lastSolid && lastSolid.id), plane: 'YZ', offset: bb[0][0] };
      case 'fillet': return { id, type, name, r: 5, edges: c.edges || [] };
      case 'chamfer': return { id, type, name, r: 3, edges: c.edges || [] };
      case 'shell': return { id, type, name, t: 3, open: c.open || [] };
      case 'loft': return { id, type, name, op: c.op || 'boss', sketch: c.sketch, sketch2: c.sketch2 };
      case 'sweep': return { id, type, name, op: c.op || 'boss', sketch: c.sketch, path: c.path };
      default: return { id, type, name };
    }
  }
  // the editable fields of a feature, for its PropertyManager
  function featureFields(f) {
    const N = (key, label, step, min, max, unit) => ({ kind: 'num', key, label, step: step || 1, min: min === undefined ? -1e5 : min, max: max === undefined ? 1e5 : max, unit: unit === undefined ? 'mm' : unit });
    const C = (key, label, options) => ({ kind: 'choice', key, label, options });
    const op = C('op', 'Operation', [['boss', 'Boss (add)'], ['cut', 'Cut (remove)']]);
    const dirs = [['+x', '+X'], ['-x', '−X'], ['+y', '+Y'], ['-y', '−Y'], ['+z', '+Z'], ['-z', '−Z']];
    switch (f.type) {
      case 'sketch': return [C('plane', 'Plane', [['XY', 'Front'], ['XZ', 'Top'], ['YZ', 'Right']]), N('offset', 'Offset', 5)];
      case 'extrude': return [op, N('depth', 'Depth', 5, 0.1), C('dir', 'Direction', [['normal', 'Normal'], ['reverse', 'Reverse'], ['mid', 'Mid-plane']])];
      case 'revolve': return [op, C('axis', 'Axis', [['v', 'Sketch vertical'], ['u', 'Sketch horizontal']]), N('angle', 'Angle', 15, 1, 360, '°')];
      case 'box': return [op, N('x', 'Corner x', 5), N('y', 'Corner y', 5), N('z', 'Corner z', 5), N('sx', 'Size x', 5, 0.1), N('sy', 'Size y', 5, 0.1), N('sz', 'Size z', 5, 0.1)];
      case 'cylinder': return [op, N('x', 'Base x', 5), N('y', 'Base y', 5), N('z', 'Base z', 5), C('axis', 'Axis', dirs), N('r', 'Radius', 1, 0.1), N('h', 'Height', 5, 0.1)];
      case 'sphere': return [op, N('x', 'Centre x', 5), N('y', 'Centre y', 5), N('z', 'Centre z', 5), N('r', 'Radius', 1, 0.1)];
      case 'hole': return [N('x', 'Entry x', 5), N('y', 'Entry y', 5), N('z', 'Entry z', 5), C('dir', 'Direction', dirs), N('d', 'Diameter', 1, 0.1),
        C('through', 'End', [[true, 'Through all'], [false, 'Blind']]), N('depth', 'Depth (blind)', 5, 0.1)];
      case 'lpattern': return [{ kind: 'feature', key: 'src', label: 'Feature' }, C('dir', 'Direction', dirs), N('count', 'Instances', 1, 2, 200, ''), N('spacing', 'Spacing', 5)];
      case 'cpattern': return [{ kind: 'feature', key: 'src', label: 'Feature' }, C('axis', 'Axis', dirs.filter((d, i) => i % 2 === 0)), N('cx', 'Axis point x', 5), N('cy', 'Axis point y', 5), N('cz', 'Axis point z', 5),
        N('count', 'Instances', 1, 2, 360, ''), N('angle', 'Total angle', 15, 1, 360, '°')];
      case 'mirror': return [{ kind: 'feature', key: 'src', label: 'Feature' }, C('plane', 'Mirror plane', [['YZ', 'x = offset'], ['XZ', 'y = offset'], ['XY', 'z = offset']]), N('offset', 'Offset', 5)];
      case 'stl': return [op, N('scale', 'Scale', 0.1, 0.0001, 1e4, '×'), N('tx', 'Move x', 5), N('ty', 'Move y', 5), N('tz', 'Move z', 5)];
      case 'fillet': return [N('r', 'Radius', 1, 0.01), { kind: 'edges', key: 'edges', label: 'Edges' }];
      case 'chamfer': return [N('r', 'Distance', 1, 0.01), { kind: 'edges', key: 'edges', label: 'Edges' }];
      case 'shell': return [N('t', 'Wall thickness', 0.5, 0.01), { kind: 'faces', key: 'open', label: 'Faces to remove (openings)' }];
      case 'loft': return [op, { kind: 'sketch', key: 'sketch', label: 'From profile' }, { kind: 'sketch', key: 'sketch2', label: 'To profile' }];
      case 'sweep': return [op, { kind: 'sketch', key: 'sketch', label: 'Profile' }, { kind: 'sketch', key: 'path', label: 'Path (open polyline)' }];
      default: return [];
    }
  }
  // smart dimensions to draw on the model for a feature: { key, label, at: [x, y, z] }
  function featureDims(f, cm) {
    const out = [];
    const st = cm.steps.find(s => s.feat === f.id);
    if (f.type === 'extrude' && st && st.body.frame) {
      const F = st.body.frame, P = st.body.profile, u = (P.bb[0] + P.bb[2]) / 2, v = P.bb[3];
      out.push({ key: 'depth', label: 'D ' + fmt(f.depth), at: F.toWorld(u, v, (st.body.range[0] + st.body.range[1]) / 2) });
    }
    if (f.type === 'box') {
      out.push({ key: 'sx', label: fmt(f.sx), at: [f.x + f.sx / 2, f.y, f.z + f.sz] }, { key: 'sy', label: fmt(f.sy), at: [f.x + f.sx, f.y + f.sy / 2, f.z + f.sz] },
        { key: 'sz', label: fmt(f.sz), at: [f.x + f.sx, f.y, f.z + f.sz / 2] });
    }
    if (f.type === 'cylinder') {
      const a = dirVec(f.axis), top = add([f.x, f.y, f.z], mul(a, f.h));
      out.push({ key: 'r', label: 'R ' + fmt(f.r), at: top }, { key: 'h', label: 'H ' + fmt(f.h), at: add([f.x, f.y, f.z], mul(a, f.h / 2)) });
    }
    if (f.type === 'sphere') out.push({ key: 'r', label: 'R ' + fmt(f.r), at: [f.x, f.y + f.r, f.z] });
    if (f.type === 'hole') out.push({ key: 'd', label: 'Ø' + fmt(f.d), at: [f.x, f.y, f.z] });
    if (f.type === 'lpattern') out.push({ key: 'spacing', label: fmt(f.spacing) + ' × ' + f.count, at: null });

    return out;
  }
  const fmt = v => String(+(+v).toFixed(3));

  // ----------------------------------------------------------- templates
  // Each template is a model plus a default study (supports and loads on named faces).
  function bracket(p) {
    const W = p && p.W || 120, H = p && p.H || 180, D = p && p.D || 60, t = p && p.t || 30, r = p && p.r !== undefined ? p.r : 10;
    return {
      name: 'bracket_L', template: 'bracket', tplParams: { W, H, D, t, r },
      features: [
        { id: 'sketch1', type: 'sketch', name: 'Sketch1', plane: 'XY', offset: 0,
          entities: [{ type: 'poly', pts: [[0, 0], [t, 0], [t, H - t], [W, H - t], [W, H], [W - t, H], [0, H]], fillets: [0, 0, r, 0, 0, 0, 0] }] },
        { id: 'extrude1', type: 'extrude', name: 'Boss-Extrude1', sketch: 'sketch1', op: 'boss', depth: D, dir: 'normal' }
      ],
      studies: [
        { id: 'fix1', type: 'fixed', name: 'Fixed-1 · base', faces: ['extrude1/e0.l0'] },
        { id: 'force1', type: 'force', name: 'Force-1 · arm tip', faces: ['extrude1/e0.l4'], F: [0, -4200, 0] },
        { id: 'press1', type: 'pressure', name: 'Pressure-1 · web', faces: ['extrude1/e0.l6'], P: 0.8 }
      ]
    };
  }
  const TEMPLATES = {
    bracket: { label: 'L-bracket (filleted)', make: () => bracket() },
    cantilever: { label: 'Cantilever beam', make: () => ({
      name: 'cantilever', features: [{ id: 'box1', type: 'box', name: 'Beam', op: 'boss', x: 0, y: 0, z: 0, sx: 600, sy: 60, sz: 40 }],
      studies: [{ id: 'fix1', type: 'fixed', name: 'Fixed-1 · root', faces: ['box1/-x'] },
        { id: 'force1', type: 'force', name: 'Force-1 · tip', faces: ['box1/+x'], F: [0, -2000, 0] }] }) },
    plate: { label: 'Plate with a hole', make: () => ({
      name: 'plate_hole', features: [
        { id: 'sketch1', type: 'sketch', name: 'Sketch1', plane: 'XY', offset: 0,
          entities: [{ type: 'rect', x: 0, y: 0, w: 240, h: 100, r: 0 }, { type: 'circle', cx: 120, cy: 50, r: 20 }] },
        { id: 'extrude1', type: 'extrude', name: 'Boss-Extrude1', sketch: 'sketch1', op: 'boss', depth: 10, dir: 'normal' }],
      studies: [{ id: 'fix1', type: 'fixed', name: 'Fixed-1 · left edge', faces: ['extrude1/e0.l3'] },
        { id: 'force1', type: 'force', name: 'Force-1 · tension', faces: ['extrude1/e0.l1'], F: [20000, 0, 0] }] }) },
    ibeam: { label: 'I-beam', make: () => {
      const b = 100, h = 200, tf = 15, tw = 10;
      return { name: 'i_beam', features: [
        { id: 'sketch1', type: 'sketch', name: 'Sketch1 · I-section', plane: 'YZ', offset: 0, entities: [{ type: 'poly',
          pts: [[-b / 2, 0], [b / 2, 0], [b / 2, tf], [tw / 2, tf], [tw / 2, h - tf], [b / 2, h - tf], [b / 2, h], [-b / 2, h], [-b / 2, h - tf], [-tw / 2, h - tf], [-tw / 2, tf], [-b / 2, tf]],
          fillets: [0, 0, 0, 6, 6, 0, 0, 0, 0, 6, 6, 0] }] },
        { id: 'extrude1', type: 'extrude', name: 'Boss-Extrude1', sketch: 'sketch1', op: 'boss', depth: 1000, dir: 'normal' }],
        studies: [{ id: 'fix1', type: 'fixed', name: 'Fixed-1 · end', faces: ['extrude1/cap0'] },
          { id: 'force1', type: 'force', name: 'Force-1 · top flange', faces: ['extrude1/e0.l6'], F: [0, -20000, 0] }] };
    } },
    shaft: { label: 'Flanged shaft (revolve + hole pattern)', make: () => ({
      name: 'flanged_shaft', features: [
        { id: 'sketch1', type: 'sketch', name: 'Sketch1 · half section', plane: 'XY', offset: 0, entities: [{ type: 'poly',
          pts: [[0, 0], [80, 0], [80, 20], [30, 20], [30, 200], [0, 200]], fillets: [0, 0, 0, 8, 0, 0] }] },
        { id: 'revolve1', type: 'revolve', name: 'Revolve1', sketch: 'sketch1', op: 'boss', axis: 'v', angle: 360 },
        { id: 'hole1', type: 'hole', name: 'Hole1', x: 60, y: 20, z: 0, dir: '-y', d: 12, depth: 20, through: true },
        { id: 'cpattern1', type: 'cpattern', name: 'CirPattern1', src: 'hole1', axis: '+y', cx: 0, cy: 0, cz: 0, count: 6, angle: 360 }],
      studies: [{ id: 'fix1', type: 'fixed', name: 'Fixed-1 · flange base', faces: ['revolve1/e0.l0'] },
        { id: 'force1', type: 'force', name: 'Force-1 · shaft end', faces: ['revolve1/e0.l4'], F: [3000, -10000, 0] }] }) },
    blank: { label: 'Blank part', make: () => ({ name: 'part1', features: [], studies: [] }) }
  };

  const api = { PLANES, AXES, FEATURE_INFO, TEMPLATES, compile, voxelize, compileProfile, inside2D, segDist, entityLoop, pathPoints, edgeBlend, edt3,
    CONSTRAINTS, solveSketch, measureConstraint, refLabel, dropEntityConstraints, inferHV, sketchModel,
    parseSTL, b64, unb64, moveSegment, segValue, featureDefaults, featureFields, featureDims, nextId, defaultName, dirVec,
    sketchFrame, bracket, modelExtent };
  void seq;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.CADK = api;
})(typeof window !== 'undefined' ? window : globalThis);
