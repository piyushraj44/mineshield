// Hand-written SVG chart helpers. No libraries. Each chart redraws when its container width changes.

let heatId = 0;
export const PRESSURE_SCALE = { min: 0, max: 420, stops: [[0, '#F4F0D0'], [0.6, '#9FB6CC'], [0.857, '#1F3A5F'], [0.905, '#B83227'], [1, '#7A1A12']] };

export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const p2 = (n) => String(n).padStart(2, '0');
export function fmtT(t) {
  if (t == null || !Number.isFinite(t)) return '—';
  const s = Math.max(0, Math.round(t));
  return `T+${p2(Math.floor(s / 3600))}:${p2(Math.floor((s % 3600) / 60))}:${p2(s % 60)}`;
}
export function fmtNum(v, d = 1) { return v == null || !Number.isFinite(v) ? '—' : v.toFixed(d); }

// Smallest "nice" step (1, 2, 2.5, 5 x 10^n) that keeps the tick count near `count`.
export function niceTicks(lo, hi, count = 4) {
  if (!(hi > lo)) hi = lo + 1;
  const raw = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(raw));
  const m = [1, 2, 2.5, 5, 10].find((k) => k * mag >= raw) ?? 10;
  const step = m * mag;
  const a = Math.floor(lo / step) * step, b = Math.ceil(hi / step) * step, ticks = [];
  for (let v = a; v <= b + step / 2; v += step) ticks.push(+v.toFixed(10));
  return { ticks, lo: ticks[0], hi: ticks[ticks.length - 1] };
}
function timeTicks(t0, t1, count = 5) {
  const want = (t1 - t0) / count;
  const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600].find((s) => s >= want) ?? 3600;
  const out = [];
  for (let v = Math.ceil(t0 / step) * step; v <= t1; v += step) out.push(v);
  return out;
}
const rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
// Colour for value v on a scale {min, max, stops:[[pos, '#hex'], ...]}; linear RGB interpolation.
export function colorFor(sc, v) {
  const f = clamp01((v - sc.min) / (sc.max - sc.min)), st = sc.stops;
  for (let k = 1; k < st.length; k++) {
    if (f <= st[k][0] || k === st.length - 1) {
      const [p0, c0] = st[k - 1], [p1, c1] = st[k];
      const w = p1 === p0 ? 0 : clamp01((f - p0) / (p1 - p0));
      const a = rgb(c0), b = rgb(c1);
      return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * w)).join(',')})`;
    }
  }
  return st[0][1];
}

// Container bookkeeping: one SVG host plus one tooltip per chart element.
function host(el) {
  if (!el._host) {
    el.classList.add('chart');
    el.style.position = 'relative';
    el._host = document.createElement('div');
    el._tip = document.createElement('div');
    el._tip.className = 'chart-tip';
    el._tip.hidden = true;
    el.append(el._host, el._tip);
    el._ro = new ResizeObserver(() => { if (el.clientWidth !== el._w && el._draw) el._draw(); });
    el._ro.observe(el);
  }
  return el._host;
}
function showTip(el, html, ev) {
  const tip = el._tip, r = el.getBoundingClientRect();
  tip.innerHTML = html;
  tip.hidden = false;
  const x = ev.clientX - r.left, y = ev.clientY - r.top, tw = tip.offsetWidth, th = tip.offsetHeight;
  let left = x + 14;
  if (left + tw > el.clientWidth) left = Math.max(0, x - tw - 14);
  let top = y + 14;
  if (top + th > el.clientHeight) top = Math.max(0, y - th - 14);
  tip.style.left = `${left}px`;
  tip.style.top = `${top}px`;
}
const hideTip = (el) => { el._tip.hidden = true; };
const svgPoint = (svg, ev, W, H) => {
  const r = svg.getBoundingClientRect();
  return [(ev.clientX - r.left) * (W / r.width), (ev.clientY - r.top) * (H / r.height)];
};

// ---------- lineChart ----------
// opts: {t, series:[{name, values, color, unit?}], yLabel, xLabel, hLines, bands, stateBands, height, yMin, yMax, unit, states, title}
export function lineChart(el, opts) {
  host(el);
  el._draw = () => drawLine(el, opts);
  el._draw();
}
function drawLine(el, o) {
  const W = el.clientWidth || 600, H = o.height || 240;
  el._w = W;
  const t = o.t || [], series = o.series || [], hl = o.hLines || [];
  const mL = 62, mR = 14, mT = series.length ? 28 : 10, mB = o.xLabel ? 40 : 26;
  const pw = Math.max(60, W - mL - mR), ph = Math.max(40, H - mT - mB);
  const vals = [];
  series.forEach((s) => s.values.forEach((v) => { if (v != null && Number.isFinite(v)) vals.push(v); }));
  hl.forEach((h) => vals.push(h.y));
  let lo = vals.length ? Math.min(...vals) : 0, hi = vals.length ? Math.max(...vals) : 1;
  if (hi <= lo) hi = lo + 1;
  const pad = (hi - lo) * 0.06;
  // Non-negative data keeps a zero floor instead of padding below zero.
  const floor = lo >= 0 && lo - pad < 0 ? 0 : lo - pad;
  const yt = niceTicks(o.yMin ?? floor, o.yMax ?? hi + pad, 4);
  const yMin = yt.lo, yMax = yt.hi;
  const t0 = t.length ? t[0] : 0, t1 = t.length > 1 ? t[t.length - 1] : t0 + 1;
  const X = (x) => mL + ((x - t0) / (t1 - t0)) * pw;
  const Y = (y) => mT + (1 - (y - yMin) / (yMax - yMin)) * ph;
  const p = [];

  const yStep = yt.ticks.length > 1 ? yt.ticks[1] - yt.ticks[0] : 1;
  const yDec = yStep >= 1 ? 0 : yStep >= 0.1 ? 1 : 2; // enough decimals that tick labels stay distinct
  yt.ticks.forEach((v) => {
    p.push(`<line class="c-grid" x1="${mL}" x2="${mL + pw}" y1="${Y(v)}" y2="${Y(v)}"/>`,
      `<text class="c-txt" x="${mL - 8}" y="${Y(v) + 3}" text-anchor="end">${fmtNum(v, yDec)}</text>`);
  });
  const bandX = (a, b) => [Math.max(mL, Math.min(mL + pw, X(a))), Math.max(mL, Math.min(mL + pw, X(b)))];
  (o.stateBands || []).forEach((b) => {
    const [x0, x1] = bandX(b.t0, b.t1);
    if (x1 > x0) p.push(`<rect class="c-state c-state-${b.state}" x="${x0}" y="${mT}" width="${x1 - x0}" height="${ph}"/>`);
  });
  (o.bands || []).forEach((b) => {
    const [x0, x1] = bandX(b.t0, b.t1);
    if (x1 <= x0) return;
    p.push(`<rect class="c-band" x="${x0}" y="${mT}" width="${x1 - x0}" height="${ph}" style="fill:${b.color}"/>`);
    if (b.label && x1 - x0 > 40) p.push(`<text class="c-band-label" x="${x0 + 4}" y="${mT + 12}" style="fill:${b.color}">${esc(b.label)}</text>`);
  });
  p.push(`<line class="c-axis" x1="${mL}" x2="${mL + pw}" y1="${mT + ph}" y2="${mT + ph}"/>`,
    `<line class="c-axis" x1="${mL}" x2="${mL}" y1="${mT}" y2="${mT + ph}"/>`);
  if (t.length > 1) {
    timeTicks(t0, t1).forEach((v) => {
      const x = X(v), a = x < mL + 30 ? 'start' : x > mL + pw - 30 ? 'end' : 'middle';
      p.push(`<line class="c-tick" x1="${x}" x2="${x}" y1="${mT + ph}" y2="${mT + ph + 3}"/>`,
        `<text class="c-txt" x="${x}" y="${mT + ph + 15}" text-anchor="${a}">${fmtT(v)}</text>`);
    });
  }
  if (o.yLabel) p.push(`<text class="c-txt" x="4" y="${mT - 8}">${esc(o.yLabel)}</text>`);
  if (o.xLabel) p.push(`<text class="c-txt" x="${mL + pw}" y="${H - 6}" text-anchor="end">${esc(o.xLabel)}</text>`);
  hl.forEach((h) => {
    if (h.y < yMin || h.y > yMax) return;
    const y = Y(h.y), c = h.color || 'var(--muted)';
    const dash = typeof h.dash === 'string' ? h.dash : h.dash === false ? 'none' : '5 4';
    p.push(`<line class="c-hline" x1="${mL}" x2="${mL + pw}" y1="${y}" y2="${y}" style="stroke:${c}" stroke-dasharray="${dash}"/>`);
    if (h.label) p.push(`<text class="c-hl-label" x="${mL + pw - 4}" y="${y - 4}" text-anchor="end" style="fill:${c}">${esc(h.label)}</text>`);
  });
  series.forEach((s) => {
    const c = s.color || 'var(--accent)';
    let d = '', pen = false;
    s.values.forEach((v, k) => {
      if (v == null || !Number.isFinite(v)) { pen = false; return; }
      const next = s.values[k + 1];
      if (!pen && (next == null)) p.push(`<circle class="c-iso" cx="${X(t[k])}" cy="${Y(v)}" r="1.6" style="fill:${c}"/>`);
      d += `${pen ? 'L' : 'M'}${X(t[k]).toFixed(1)},${Y(v).toFixed(1)}`;
      pen = true;
    });
    if (d) p.push(`<path class="c-line" d="${d}" style="stroke:${c}"/>`);
  });
  let lx = mL;
  for (const s of series) {
    if (lx > W - 80) break; // legend entries that do not fit are dropped; the tooltip still lists all series
    const c = s.color || 'var(--accent)';
    p.push(`<line class="c-line" x1="${lx}" x2="${lx + 14}" y1="12" y2="12" style="stroke:${c}"/>`,
      `<text class="c-txt c-legend" x="${lx + 19}" y="15">${esc(s.name)}</text>`);
    lx += 30 + s.name.length * 6.4;
  }
  p.push(`<rect class="c-hit" x="${mL}" y="${mT}" width="${pw}" height="${ph}"/>`, `<g class="c-hover"></g>`);

  el._host.innerHTML = `<svg class="chart-svg" width="100%" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.title || 'line chart')}">${p.join('')}</svg>`;
  const svg = el._host.querySelector('svg'), hov = svg.querySelector('.c-hover'), hit = svg.querySelector('.c-hit');
  const hide = () => { hov.innerHTML = ''; hideTip(el); };
  hit.addEventListener('mouseleave', hide);
  hit.addEventListener('mousemove', (ev) => {
    if (!t.length) return;
    const [x] = svgPoint(svg, ev, W, H);
    const tv = t0 + ((x - mL) / pw) * (t1 - t0);
    let k = 0, best = Infinity;
    for (let j = 0; j < t.length; j++) { const d = Math.abs(t[j] - tv); if (d < best) { best = d; k = j; } }
    const cx = X(t[k]);
    hov.innerHTML = `<line class="c-cross" x1="${cx}" x2="${cx}" y1="${mT}" y2="${mT + ph}"/>` + series
      .map((s) => (s.values[k] == null ? '' : `<circle class="c-dot" cx="${cx}" cy="${Y(s.values[k])}" r="3" style="fill:${s.color}"/>`)).join('');
    const rows = series.map((s) => {
      const v = s.values[k], unit = s.unit ?? o.unit ?? '';
      const txt = v == null ? '—' : `${fmtNum(v, 1)}${unit ? ' ' + esc(unit) : ''}`;
      return `<div class="tip-row"><span class="sw" style="background:${s.color || 'var(--accent)'}"></span><span>${esc(s.name)}</span><b>${txt}</b></div>`;
    }).join('');
    const st = o.states ? `<div class="tip-muted">state: ${esc(o.states[k])}</div>` : '';
    showTip(el, `<div class="tip-t">${fmtT(t[k])}</div>${rows}${st}`, ev);
  });
}

// ---------- heatmap ----------
// opts: {rows: times (ascending), cols: ids, values: rows x cols (null allowed), colorScale, onColClick, selected, height, legendLabel}
export function heatmap(el, opts) {
  host(el);
  el._draw = () => drawHeat(el, opts);
  el._draw();
}
function drawHeat(el, o) {
  const W = el.clientWidth || 600, H = o.height || 320;
  el._w = W;
  const rows = o.rows || [], cols = o.cols || [], vals = o.values || [], sc = o.colorScale || PRESSURE_SCALE;
  const mL = 72, mR = 8, mT = 6, mB = 60;
  const pw = Math.max(60, W - mL - mR), ph = Math.max(40, H - mT - mB);
  const N = rows.length, C = cols.length, cw = C ? pw / C : pw, rh = N ? ph / N : ph;
  const id = `hz${++heatId}`;
  const p = [`<defs><pattern id="${id}" width="6" height="6" patternUnits="userSpaceOnUse"><rect class="hz-hatch-bg" width="6" height="6"/><path class="hz-hatch" d="M0,6 L6,0 M-3,3 L3,-3 M3,9 L9,3"/></pattern></defs>`];
  const rowLabelStep = Math.max(1, Math.round(N / 6));
  rows.forEach((tt, k) => {
    const y = mT + (N - 1 - k) * rh;
    cols.forEach((c, j) => {
      const v = vals[k] ? vals[k][j] : null, x = mL + j * cw;
      p.push(v == null
        ? `<rect shape-rendering="crispEdges" x="${x}" y="${y}" width="${cw + 0.5}" height="${rh + 0.5}" fill="url(#${id})"/>`
        : `<rect shape-rendering="crispEdges" x="${x}" y="${y}" width="${cw + 0.5}" height="${rh + 0.5}" style="fill:${colorFor(sc, v)}"/>`);
    });
    if ((N - 1 - k) % rowLabelStep === 0) p.push(`<text class="c-txt" x="${mL - 6}" y="${y + rh / 2 + 3}" text-anchor="end">${fmtT(tt)}</text>`);
  });
  const labelStep = Math.max(1, Math.ceil(28 / cw));
  cols.forEach((c, j) => {
    if (j % labelStep === 0) p.push(`<text class="c-txt${o.selected === c ? ' c-sel-txt' : ''}" x="${mL + (j + 0.5) * cw}" y="${mT + ph + 14}" text-anchor="middle">${esc(c)}</text>`);
  });
  if (o.selected && cols.includes(o.selected)) {
    const j = cols.indexOf(o.selected);
    p.push(`<rect class="hz-sel" x="${mL + j * cw + 0.5}" y="${mT + 0.5}" width="${cw - 1}" height="${ph - 1}"/>`);
  }
  // Discrete colour legend (no gradients).
  const ly = mT + ph + 30, steps = 24, lw = Math.min(240, pw - 120);
  for (let s = 0; s < steps; s++) {
    const v = sc.min + ((sc.max - sc.min) * (s + 0.5)) / steps;
    p.push(`<rect x="${mL + (s * lw) / steps}" y="${ly}" width="${lw / steps + 0.5}" height="8" style="fill:${colorFor(sc, v)}"/>`);
  }
  p.push(`<text class="c-txt" x="${mL}" y="${ly + 20}">${sc.min}</text>`,
    `<text class="c-txt" x="${mL + lw}" y="${ly + 20}" text-anchor="end">${sc.max}</text>`,
    `<text class="c-txt" x="${mL + lw + 20}" y="${ly + 8}">${esc(o.legendLabel || 'Shield mean (bar)')}</text>`,
    `<rect x="${mL + lw + 20 + 0}" y="${ly + 14}" width="14" height="8" fill="url(#${id})"/>`,
    `<text class="c-txt" x="${mL + lw + 40}" y="${ly + 21}">no valid data</text>`);
  p.push(`<rect class="c-hit" x="${mL}" y="${mT}" width="${pw}" height="${ph}"/>`);

  el._host.innerHTML = `<svg class="chart-svg" width="100%" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Face pressure heatmap">${p.join('')}</svg>`;
  const svg = el._host.querySelector('svg'), hit = svg.querySelector('.c-hit');
  const cell = (ev) => {
    const [x, y] = svgPoint(svg, ev, W, H);
    const j = Math.floor((x - mL) / cw), row = Math.floor((y - mT) / rh);
    if (j < 0 || j >= C || row < 0 || row >= N) return null;
    const k = N - 1 - row;
    return { j, k, v: vals[k] ? vals[k][j] ?? null : null };
  };
  hit.addEventListener('mouseleave', () => hideTip(el));
  hit.addEventListener('mousemove', (ev) => {
    const c = cell(ev);
    if (!c) return hideTip(el);
    const pct = clamp01((c.v - sc.min) / (sc.max - sc.min)) * 100;
    const body = c.v == null
      ? '<div class="tip-muted">no valid data</div>'
      : `<div class="tip-row"><span>Shield mean</span><b>${fmtNum(c.v, 1)} bar</b></div><div class="tip-bar"><i style="width:${pct.toFixed(1)}%;background:${colorFor(sc, c.v)}"></i></div>`;
    showTip(el, `<div class="tip-t">${esc(cols[c.j])} · ${fmtT(rows[c.k])}</div>${body}`, ev);
  });
  hit.addEventListener('click', (ev) => {
    const c = cell(ev);
    if (c && o.onColClick) o.onColClick(cols[c.j]);
  });
}

// ---------- barChart ----------
// opts: {categories, groups:[{name, values, color}], yMax, yLabel, valueFormat, height}
export function barChart(el, opts) {
  host(el);
  el._draw = () => drawBar(el, opts);
  el._draw();
}
function drawBar(el, o) {
  const W = el.clientWidth || 600, H = o.height || 260;
  el._w = W;
  const mL = 56, mR = 12, mT = 30, mB = 34;
  const pw = Math.max(80, W - mL - mR), ph = Math.max(40, H - mT - mB);
  const cats = o.categories || [], groups = o.groups || [];
  const all = groups.flatMap((g) => g.values).filter((v) => v != null);
  const yt = niceTicks(0, o.yMax ?? Math.max(0.0001, ...all), 4);
  const top = o.yMax ?? yt.hi;
  const Y = (v) => mT + (1 - v / top) * ph;
  const cw = pw / Math.max(1, cats.length), gw = cw * 0.72, bw = Math.min(26, gw / Math.max(1, groups.length));
  const fmt = o.valueFormat || ((v) => fmtNum(v, 1));
  const p = [];
  yt.ticks.filter((v) => v <= top + 1e-9).forEach((v) => {
    p.push(`<line class="c-grid" x1="${mL}" x2="${mL + pw}" y1="${Y(v)}" y2="${Y(v)}"/>`,
      `<text class="c-txt" x="${mL - 8}" y="${Y(v) + 3}" text-anchor="end">${+v.toFixed(2)}</text>`);
  });
  p.push(`<line class="c-axis" x1="${mL}" x2="${mL + pw}" y1="${mT + ph}" y2="${mT + ph}"/>`);
  if (o.yLabel) p.push(`<text class="c-txt" x="4" y="${mT - 10}">${esc(o.yLabel)}</text>`);
  cats.forEach((c, ci) => {
    const x0 = mL + ci * cw + (cw - bw * groups.length) / 2;
    groups.forEach((g, gi) => {
      const v = g.values[ci], x = x0 + gi * bw + 1, c0 = g.color || 'var(--accent)';
      if (v == null) {
        p.push(`<text class="c-txt" x="${x + bw / 2 - 1}" y="${mT + ph - 4}" text-anchor="middle">—</text>`);
        return;
      }
      const y = Y(Math.min(v, top)), h = mT + ph - y;
      p.push(`<rect x="${x}" y="${y}" width="${Math.max(1, bw - 2)}" height="${Math.max(0, h)}" style="fill:${c0}"><title>${esc(g.name)} · ${esc(c)}: ${esc(fmt(v))}</title></rect>`,
        `<text class="c-val" x="${x + bw / 2 - 1}" y="${y - 4}" text-anchor="middle">${esc(fmt(v))}</text>`);
    });
    p.push(`<text class="c-txt" x="${mL + ci * cw + cw / 2}" y="${mT + ph + 16}" text-anchor="middle">${esc(c)}</text>`);
  });
  let lx = mL;
  groups.forEach((g) => {
    p.push(`<rect x="${lx}" y="10" width="10" height="10" style="fill:${g.color || 'var(--accent)'}"/>`,
      `<text class="c-txt" x="${lx + 15}" y="19">${esc(g.name)}</text>`);
    lx += 30 + g.name.length * 6.4;
  });
  el._host.innerHTML = `<svg class="chart-svg" width="100%" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.title || 'bar chart')}">${p.join('')}</svg>`;
}
