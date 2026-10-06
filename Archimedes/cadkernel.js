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
    if (e.type === 'poly') return polyLoop(e.pts, e.fillets, k);
    return [];
  }
  // discretise for inside tests; each edge remembers its source segment
  function compileProfile(entities) {
    const loops = [], segs = [];
    (entities || []).forEach((e, k) => {
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
        pt.plane = { normal: add(mul(F.P.u, n2[0]), mul(F.P.v, n2[1])) };
      }
      return pt;
    });
    const capA = { id: f.id + '/cap0', name: f.dir === 'reverse' ? 'end face' : 'start face', drag: f.dir === 'reverse' ? { kind: 'param', key: 'depth', axis: mul(F.P.w, -1), factor: 1 } : null, plane: { normal: mul(F.P.w, -1) } };
    const capB = { id: f.id + '/cap1', name: f.dir === 'reverse' ? 'start face' : 'end face', drag: f.dir !== 'reverse' ? { kind: 'param', key: 'depth', axis: F.P.w.slice(), factor: f.dir === 'mid' ? 2 : 1 } : null, plane: { normal: F.P.w.slice() } };
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
    const patches = P.segs.map(s => ({ id: f.id + '/' + s.id, name: 'surface (' + segLabel(s) + ')', drag: { kind: 'seg', sketch: sk.id, seg: s } }));
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
      }
    };
  }
  function boxBody(f) {
    const mn = [f.x || 0, f.y || 0, f.z || 0], s = [Math.abs(f.sx || 1), Math.abs(f.sy || 1), Math.abs(f.sz || 1)];
    const mx = add(mn, s);
    const names = ['−X face', '+X face', '−Y face', '+Y face', '−Z face', '+Z face'], keys = ['x', 'y', 'z'], sz = ['sx', 'sy', 'sz'];
    const patches = names.map((n, i) => {
      const ax = Math.floor(i / 2), hi = i % 2 === 1, axis = [0, 0, 0]; axis[ax] = hi ? 1 : -1;
      return { id: f.id + '/' + (hi ? '+' : '-') + 'xyz'[ax], name: n, plane: { normal: axis },
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
      }
    };
  }
  // cylinder from base centre c along unit axis a, radius r, length h
  function cylBody(id, c, a, r, h, labels, drags) {
    const ref = Math.abs(a[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0], e1 = norm(cross(a, ref)), e2 = cross(a, e1);
    const top = add(c, mul(a, h));
    const corners = [];
    for (const q of [c, top]) for (const s1 of [-r, r]) for (const s2 of [-r, r]) corners.push(add(q, add(mul(e1, s1), mul(e2, s2))));
    const patches = [{ id: id + '/wall', name: labels[0], drag: drags[0] }, { id: id + '/cap0', name: labels[1], drag: drags[1], plane: { normal: mul(a, -1) } },
      { id: id + '/cap1', name: labels[2], drag: drags[2], plane: { normal: a.slice() } }];
    return {
      bbox: bboxOfPoints(corners), patches,
      inside: p => { const q = sub(p, c), t = dot(q, a); if (t < 0 || t > h) return false; const rr = sub(q, mul(a, t)); return dot(rr, rr) <= r * r; },
      near: p => {
        const q = sub(p, c), t = dot(q, a), radv = sub(q, mul(a, t)), rho = Math.hypot(radv[0], radv[1], radv[2]);
        const dt = t < 0 ? -t : t > h ? t - h : 0, dr = rho > r ? rho - r : 0;
        const dw = Math.hypot(rho - r, dt), d0 = Math.hypot(t, dr), d1 = Math.hypot(t - h, dr);
        if (dw <= d0 && dw <= d1) return [dw, 0, rho > 1e-9 ? mul(radv, 1 / rho) : e1];
        return d0 < d1 ? [d0, 1, mul(a, -1)] : [d1, 2, a.slice()];
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
    return { bbox: G.bbox, patches, inside: G.inside, near: G.near };
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
    // 2D bins over (y, z) for the +x parity ray, 3D bins for nearest facet
    const B = Math.max(4, Math.min(64, Math.round(Math.sqrt(nt / 2)))), ext = sub(mx, mn).map(v => v || 1);
    const bins = Array.from({ length: B * B }, () => []);
    const bin3 = new Map(), C = Math.max(ext[0], ext[1], ext[2]) / Math.max(4, Math.min(48, Math.round(Math.cbrt(nt)))) || 1;
    for (let t = 0; t < nt; t++) {
      let y0 = Infinity, y1 = -Infinity, z0 = Infinity, z1 = -Infinity;
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let k = 0; k < 3; k++) {
        const y = V[t * 9 + k * 3 + 1], z = V[t * 9 + k * 3 + 2];
        y0 = Math.min(y0, y); y1 = Math.max(y1, y); z0 = Math.min(z0, z); z1 = Math.max(z1, z);
        for (let q = 0; q < 3; q++) { lo[q] = Math.min(lo[q], V[t * 9 + k * 3 + q]); hi[q] = Math.max(hi[q], V[t * 9 + k * 3 + q]); }
      }
      const by0 = Math.max(0, Math.floor((y0 - mn[1]) / ext[1] * B)), by1 = Math.min(B - 1, Math.floor((y1 - mn[1]) / ext[1] * B));
      const bz0 = Math.max(0, Math.floor((z0 - mn[2]) / ext[2] * B)), bz1 = Math.min(B - 1, Math.floor((z1 - mn[2]) / ext[2] * B));
      for (let a = by0; a <= by1; a++) for (let b = bz0; b <= bz1; b++) bins[a * B + b].push(t);
      for (let i = Math.floor(lo[0] / C); i <= Math.floor(hi[0] / C); i++) for (let j = Math.floor(lo[1] / C); j <= Math.floor(hi[1] / C); j++)
        for (let k = Math.floor(lo[2] / C); k <= Math.floor(hi[2] / C); k++) {
          const kk = i + ',' + j + ',' + k; let l = bin3.get(kk); if (!l) bin3.set(kk, l = []); l.push(t);
        }
    }
    const inside = p => {
      if (!nt || p[0] < mn[0] || p[0] > mx[0] || p[1] < mn[1] || p[1] > mx[1] || p[2] < mn[2] || p[2] > mx[2]) return false;
      const by = Math.min(B - 1, Math.max(0, Math.floor((p[1] - mn[1]) / ext[1] * B))), bz = Math.min(B - 1, Math.max(0, Math.floor((p[2] - mn[2]) / ext[2] * B)));
      // jitter keeps the ray off shared edges
      const y = p[1] + 1.3e-7 * ext[1], z = p[2] + 2.1e-7 * ext[2];
      let cnt = 0;
      for (const t of bins[by * B + bz]) {
        const o = t * 9;
        const ay = V[o + 1], az = V[o + 2], by_ = V[o + 4], bz_ = V[o + 5], cy = V[o + 7], cz = V[o + 8];
        const d = (by_ - ay) * (cz - az) - (cy - ay) * (bz_ - az);
        if (Math.abs(d) < 1e-18) continue;
        const l1 = ((by_ - y) * (cz - z) - (cy - y) * (bz_ - z)) / d, l2 = ((cy - y) * (az - z) - (ay - y) * (cz - z)) / d, l3 = 1 - l1 - l2;
        if (l1 < 0 || l2 < 0 || l3 < 0) continue;
        const x = l1 * V[o] + l2 * V[o + 3] + l3 * V[o + 6];
        if (x > p[0]) cnt++;
      }
      return (cnt & 1) === 1;
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
    return { bbox: [mn, mx], groups, inside, near, nt };
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
      near: p => { const r = body.near(inv(p)); return [r[0], r[1], r[2] ? rotN(r[2]) : null]; }
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
    stl: { label: 'Imported', solid: true }
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
  // evaluate the whole feature list once; returns the compiled part
  function compile(model) {
    const feats = model.features || [], byId = {};
    for (const f of feats) byId[f.id] = f;
    const steps = [], faces = {};
    let bb = null;
    for (const f of feats) {
      if (!FEATURE_INFO[f.type] || !FEATURE_INFO[f.type].solid) continue;
      const r = bodyOf(f, model, byId, 0);
      if (!r) continue;
      for (const b of r.bodies) {
        steps.push({ op: r.op, body: b, feat: f.id });
        for (const p of b.patches) faces[p.id] = { id: p.id, feat: f.id, op: r.op, name: (f.name || f.id) + ' · ' + p.name, drag: p.drag || null, plane: p.plane || null };
        if (r.op === 'boss') bb = bb ? [[Math.min(bb[0][0], b.bbox[0][0]), Math.min(bb[0][1], b.bbox[0][1]), Math.min(bb[0][2], b.bbox[0][2])],
          [Math.max(bb[1][0], b.bbox[1][0]), Math.max(bb[1][1], b.bbox[1][1]), Math.max(bb[1][2], b.bbox[1][2])]] : [b.bbox[0].slice(), b.bbox[1].slice()];
      }
    }
    const inside = p => {
      let s = false;
      for (const st of steps) {
        if (st.op === 'boss') { if (!s && st.body.inside(p)) s = true; }
        else if (s && st.body.inside(p)) s = false;
      }
      return s;
    };
    // nearest named face to a point on the voxel skin. n is the skin's outward
    // normal; a patch whose own outward side disagrees is penalised (a cut's
    // surface faces the other way: the part's outside is the cut's inside).
    const faceAt = (p, n, h) => {
      let best = Infinity, id = null;
      for (const st of steps) {
        const r = st.body.near(p);
        if (!(r[0] < Infinity)) continue;
        let d = r[0];
        if (n && r[2]) { const c = dot(n, r[2]) * (st.op === 'cut' ? -1 : 1); if (c < -0.3) d += (h || 1); }
        if (d < best - 1e-9) { best = d; id = st.body.patches[r[1]] ? st.body.patches[r[1]].id : null; }
      }
      return id;
    };
    return { model, steps, faces, inside, faceAt, bbox: bb || [[0, 0, 0], [100, 100, 100]], empty: !bb, byId };
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
    let ne = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      let c = 0;
      if (subS === 1) c = cm.inside([bb[0][0] + (i + 0.5) * dx, bb[0][1] + (j + 0.5) * dy, bb[0][2] + (k + 0.5) * dz]) ? 1 : 0;
      else {
        for (let a = 0; a < subS && c < need; a++) for (let b = 0; b < subS; b++) for (let e = 0; e < subS; e++)
          if (cm.inside([bb[0][0] + (i + (a + 0.5) / subS) * dx, bb[0][1] + (j + (b + 0.5) / subS) * dy, bb[0][2] + (k + (e + 0.5) / subS) * dz])) c++;
      }
      if (c >= (subS === 1 ? 1 : need)) { solid[(k * ny + j) * nx + i] = 1; ne++; }
    }
    return { nx, ny, nz, dx, dy, dz, solid, ne, org: bb[0].slice(), h, clamped, frac };
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
      hole: 'Hole', lpattern: 'LPattern', cpattern: 'CirPattern', mirror: 'Mirror', stl: 'Imported' }[type] || type;
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

  const api = { PLANES, AXES, FEATURE_INFO, TEMPLATES, compile, voxelize, compileProfile, inside2D, segDist, entityLoop,
    parseSTL, b64, unb64, moveSegment, segValue, featureDefaults, featureFields, featureDims, nextId, defaultName, dirVec,
    sketchFrame, bracket, modelExtent };
  void seq;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.CADK = api;
})(typeof window !== 'undefined' ? window : globalThis);
