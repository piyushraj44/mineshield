// App state, polling loop, rendering and event handlers. All data comes through api.js.
import * as api from './api.js';
import { lineChart, heatmap, barChart, esc, fmtT, fmtNum, colorFor, PRESSURE_SCALE } from './charts.js';

const WINDOW_S = 300;
const LEG_COLORS = ['#1F3A5F', '#C2571A', '#2A8C82', '#7A4FB0'];
const CAT_ORDER = ['process', 'sensor', 'telemetry'];
const CAT_LABEL = { process: 'Process', sensor: 'Sensor', telemetry: 'Telemetry' };
const STATUS_LABEL = { normal: 'Normal', watch: 'Watch', process_alert: 'Process alert', sensor_fault: 'Sensor fault', no_data: 'No data' };
const DEV_SCALE = { min: -40, max: 40, stops: [[0, '#2F6690'], [0.5, '#BDB58C'], [1, '#B83227']] };
const AI_SCALE = { min: 0, max: 2, stops: [[0, '#BDB58C'], [0.5, '#D99A2B'], [1, '#B83227']] };
const EVENT_BAND = { process: 'var(--alert)', sensor: 'var(--sensor)', telemetry: 'var(--muted)' };
const LIMIT_LABEL = {
  sensor_min_bar: 'Sensor range, minimum', sensor_max_bar: 'Sensor range, maximum', setting_bar: 'Setting pressure',
  high_limit_bar: 'High limit (illustrative)', yield_bar: 'Yield', low_limit_bar: 'Low limit (illustrative)',
};
// [feature key, label, unit multiplier]
const FEATURES = [
  ['pressure_bar', 'Pressure (bar)', 1],
  ['dev_own_bar', 'Deviation from own baseline (bar)', 1],
  ['slope_5_bar_s', '5-s gradient (bar/s)', 1],
  ['slope_30_bar_s', '30-s gradient (bar/s)', 1],
  ['imbalance_frac', 'Leg imbalance (%)', 100],
  ['neighbor_dev_bar', 'Deviation from neighbours (bar)', 1],
  ['roll_std_bar', 'Rolling std (bar)', 1],
  ['missing_frac', 'Missing fraction (%)', 100],
];

const $ = (id) => document.getElementById(id);
const idx = (id) => Number(String(id).slice(1)) - 1;
const sidOf = (i) => 'S' + String(i + 1).padStart(2, '0');
const gwOf = (i) => 'GW-' + String(Math.floor(i / 6) + 1).padStart(2, '0');
const numOf = (id) => { const s = $(id).value.trim(); return s === '' ? NaN : Number(s); };
const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);
const kv = (k, v, note = '') => `<div class="kv" title="${esc(note)}"><div class="k">${esc(k)}</div><div class="v num">${esc(v)}</div><div class="q">${esc(note)}</div></div>`;

const S = {
  config: null, state: null, selected: 'S01', running: false, inFlight: false, reqId: 0, timer: null,
  error: null, stale: false, showGT: false, colorBy: 'status', formFilled: false, optN: null, optLegs: null,
  view: 'live', drawerId: null, drawerTimer: null, detail: null, detailJson: '', reveal: false,
  evalData: null, evalLoaded: false, evalBusy: false, evalError: null,
};

// ---------- boot, config, static controls ----------
async function loadConfig() {
  try {
    S.config = await api.getConfig();
  } catch (e) {
    return fail(e);
  }
  buildStatic();
  await fetchState(false);
}

function buildStatic() {
  const c = S.config;
  $('disclaimer').textContent = c.disclaimer;
  $('m-disclaimer').textContent = c.disclaimer;
  $('speed').innerHTML = c.speeds.map((s) => `<option value="${s.steps}">${esc(s.label)} (${s.steps} step${s.steps > 1 ? 's' : ''} per poll)</option>`).join('');
  $('ev-type').innerHTML = CAT_ORDER.map((cat) => `<optgroup label="${CAT_LABEL[cat]}">${c.event_types
    .filter((e) => e.category === cat).map((e) => `<option value="${esc(e.type)}">${esc(e.label)}</option>`).join('')}</optgroup>`).join('');
  const d = c.defaults;
  $('p-seed').value = d.seed;
  $('p-shields').value = d.n_shields;
  $('p-legs').value = d.legs_per_shield;
  $('p-noise').value = d.noise_bar;
  $('p-trend').value = d.load_trend_bar_per_h;
  validateReset();
  buildShieldOptions(d.n_shields);
  buildLegOptions(d.legs_per_shield);
  S.optN = d.n_shields;
  S.optLegs = d.legs_per_shield;
  onTypeChange();
  $('lnk-exp').href = api.exportExperimentUrl;
  $('lnk-tel').href = api.exportTelemetryUrl;
  $('m-area').textContent = c.piston_area_m2;
  $('m-ex').textContent = fmtNum((300 * 1e5 * c.piston_area_m2) / 1e6, 2);
  $('m-limits').innerHTML = `<thead><tr><th>Limit</th><th class="n">Value</th></tr></thead><tbody>${Object.entries(c.limits)
    .map(([k, v]) => `<tr><td>${esc(LIMIT_LABEL[k] || k)}</td><td class="n">${v} bar</td></tr>`).join('')}</tbody>`;
}

function fillReset(sim) {
  $('p-seed').value = sim.seed;
  $('p-shields').value = sim.n_shields;
  $('p-legs').value = sim.legs_per_shield;
  $('p-noise').value = sim.noise_bar;
  $('p-trend').value = sim.load_trend_bar_per_h;
  validateReset();
}

function readReset() {
  const r = S.config.ranges;
  const v = {
    seed: numOf('p-seed'), n_shields: numOf('p-shields'), legs_per_shield: numOf('p-legs'),
    noise_bar: numOf('p-noise'), load_trend_bar_per_h: numOf('p-trend'),
  };
  const errs = [];
  if (!Number.isInteger(v.seed)) errs.push('Seed must be a whole number.');
  if (!Number.isInteger(v.n_shields) || v.n_shields < r.n_shields[0] || v.n_shields > r.n_shields[1]) errs.push(`Shields must be a whole number from ${r.n_shields[0]} to ${r.n_shields[1]}.`);
  if (![2, 4].includes(v.legs_per_shield)) errs.push('Legs per shield must be 2 or 4.');
  if (!(v.noise_bar >= r.noise_bar[0] && v.noise_bar <= r.noise_bar[1])) errs.push(`Sensor noise must be between ${r.noise_bar[0]} and ${r.noise_bar[1]} bar.`);
  if (!(v.load_trend_bar_per_h >= r.load_trend_bar_per_h[0] && v.load_trend_bar_per_h <= r.load_trend_bar_per_h[1])) {
    errs.push(`Roof-load trend must be between ${r.load_trend_bar_per_h[0]} and ${r.load_trend_bar_per_h[1]} bar/h.`);
  }
  return { v, errs };
}

function validateReset() {
  const { errs } = readReset();
  $('p-msg').textContent = errs.join(' ');
  $('btn-reset').disabled = errs.length > 0 || !S.config;
}

async function doReset() {
  const { v, errs } = readReset();
  if (errs.length) return validateReset();
  pause();
  try {
    await api.resetSim(v);
    $('p-msg').textContent = '';
    if (idx(S.selected) >= v.n_shields) S.selected = 'S01';
    await fetchState(false);
  } catch (e) {
    if (e.status === 422) $('p-msg').textContent = e.message;
    else fail(e);
  }
}

// ---------- polling loop and error handling ----------
const speedSteps = () => Number($('speed').value) || 1;

function start() {
  if (!S.config || !S.state) return;
  clearError();
  S.running = true;
  clearInterval(S.timer);
  S.timer = setInterval(tick, S.config.poll_ms);
  renderChips();
}

function pause() {
  S.running = false;
  clearInterval(S.timer);
  S.timer = null;
  renderChips();
}

async function tick() {
  if (!S.running || S.inFlight) return;
  S.inFlight = true;
  try {
    await fetchState(true, speedSteps());
  } finally {
    S.inFlight = false;
  }
}

async function step() {
  if (S.running || S.inFlight || !S.config) return;
  S.inFlight = true;
  try {
    await fetchState(true, 1);
  } finally {
    S.inFlight = false;
  }
}

// reqId makes sure only the newest response is applied (e.g. after a shield change).
async function fetchState(advance, steps = 1) {
  const my = ++S.reqId;
  try {
    const st = advance
      ? await api.stepSim({ steps, shield: S.selected, window_s: WINDOW_S })
      : await api.getState(S.selected, WINDOW_S);
    if (my !== S.reqId) return;
    S.state = st;
    S.stale = false;
    clearError();
    render();
  } catch (e) {
    if (my !== S.reqId) return;
    fail(e);
  }
}

function fail(e) {
  pause();
  S.stale = true;
  S.error = e && e.message ? e.message : String(e);
  $('err-text').textContent = S.state
    ? `Backend error: ${S.error}. Simulation paused; figures below are the last good state.`
    : `Backend error: ${S.error}. No data loaded. Start the backend with: python run.py`;
  $('err-banner').classList.remove('hidden');
  renderChips();
}

function clearError() {
  S.error = null;
  $('err-banner').classList.add('hidden');
}

function retry() {
  clearError();
  if (!S.config) loadConfig();
  else fetchState(false);
}

// ---------- chips and render ----------
function renderChips() {
  const st = S.state, sim = st ? st.sim : null;
  const src = $('chip-source');
  src.textContent = 'Data source: SIMULATED';
  src.title = 'All telemetry comes from the built-in simulator; no physical sensors are connected.';
  $('chip-time').textContent = `Sim time ${sim ? fmtT(sim.t) : '—'}${S.stale && st ? ' (last known)' : ''}`;
  const run = $('chip-run');
  if (S.running) { run.textContent = 'Running'; run.className = 'chip run'; }
  else if (S.error) { run.textContent = 'Paused (error)'; run.className = 'chip err'; }
  else { run.textContent = 'Paused'; run.className = 'chip'; }
  const m = sim ? sim.model : null;
  $('chip-model').textContent = `Model: ${m ? `${m.name} · ${m.status}` : '—'}`;
  const a = st ? st.counts.active_alerts : null;
  $('chip-alerts').textContent = `Active alerts ${a == null ? '—' : a}`;
  $('chip-alerts').className = `chip${a ? ' err' : ''}`;
  const b = $('btn-run');
  b.textContent = S.running ? 'Pause' : 'Start';
  b.setAttribute('aria-pressed', String(S.running));
  b.disabled = !S.state;
  $('btn-step').disabled = S.running || !S.state;
  $('btn-inject').disabled = !S.state;
}

function render() {
  if (!S.state) return;
  if (!S.formFilled) { fillReset(S.state.sim); S.formFilled = true; }
  syncOptions();
  renderChips();
  renderFace();
  renderLegend();
  renderShield();
  renderHeat();
  renderAlerts();
  renderDevices();
  renderGT();
}

function disposeCharts(root) {
  root.querySelectorAll('.chart').forEach((el) => el._ro && el._ro.disconnect());
}

// ---------- face tiles, legend, shield charts, heatmap ----------
function ensureTiles(n) {
  const box = $('face-tiles');
  if (box.childElementCount === n) return;
  box.innerHTML = '';
  for (let i = 0; i < n; i++) {
    const b = document.createElement('button');
    b.className = 'tile';
    b.dataset.id = sidOf(i);
    b.innerHTML = '<div class="id"></div><div class="mean">—</div><div class="st">—</div>';
    b.querySelector('.id').textContent = sidOf(i);
    b.addEventListener('click', () => selectShield(b.dataset.id));
    box.appendChild(b);
  }
}

function renderFace() {
  const st = S.state;
  ensureTiles(st.shields.length);
  const tiles = $('face-tiles').children;
  st.shields.forEach((s, i) => {
    const b = tiles[i];
    let cls = `tile is-${s.status}`;
    if (s.status === 'no_data') cls += ' nodata';
    if (s.id === S.selected) cls += ' sel';
    if (b.className !== cls) b.className = cls;
    let style = '';
    if (S.colorBy === 'dev') style = `--c:${s.deviation_bar == null ? 'var(--nodata)' : colorFor(DEV_SCALE, s.deviation_bar)};--tint:40%`;
    else if (S.colorBy === 'ai') style = `--c:${s.ai_score == null ? 'var(--nodata)' : colorFor(AI_SCALE, s.ai_score)};--tint:40%`;
    if (b.getAttribute('style') !== style) b.setAttribute('style', style);
    b.setAttribute('aria-pressed', String(s.id === S.selected));
    b.title = `${s.id}: ${STATUS_LABEL[s.status]}, ${s.state}. ${s.ai_score == null ? 'No AI score' : `AI score ${s.ai_score} (1.0 = alert threshold)`}`;
    b.querySelector('.mean').textContent = s.mean_bar == null ? '—' : `${fmtNum(s.mean_bar, 0)} bar`;
    b.querySelector('.st').textContent = s.state;
  });
}

function renderLegend() {
  const L = $('face-legend');
  if (L.dataset.key === S.colorBy) return;
  L.dataset.key = S.colorBy;
  if (S.colorBy === 'status') {
    L.innerHTML = Object.entries(STATUS_LABEL).map(([k, v]) => `<span class="is-${k}"><i${k === 'no_data' ? ' class="nodata"' : ''}></i>${esc(v)}</span>`).join('');
  } else if (S.colorBy === 'dev') {
    L.innerHTML = [-40, -20, 0, 20, 40].map((v) => `<span><i style="--c:${colorFor(DEV_SCALE, v)}"></i>${v > 0 ? '+' : ''}${v} bar</span>`).join('')
      + '<span>vs median of loaded shields</span>';
  } else {
    L.innerHTML = [[0, '0'], [0.8, '0.8 watch'], [1, '1.0 alert threshold'], [2, '2.0 or more']]
      .map(([v, t]) => `<span><i style="--c:${colorFor(AI_SCALE, v)}"></i>${esc(t)}</span>`).join('');
  }
}

function stateSegments(ser) {
  const segs = [];
  let cur = null;
  ser.t.forEach((t, k) => {
    const s = ser.state[k];
    if (s === 'loaded') { cur = null; return; }
    if (cur && cur.state === s) { cur.t1 = t + 1; return; }
    cur = { t0: t, t1: t + 1, state: s };
    segs.push(cur);
  });
  return segs;
}

function renderShield() {
  const st = S.state, ser = st.series, L = S.config.limits;
  const sh = st.shields.find((s) => s.id === S.selected) || st.shields[0];
  $('shield-title').textContent = `Shield ${S.selected} — leg pressures`;
  const names = Object.keys(ser.channels);
  const segs = stateSegments(ser);
  const bands = S.showGT ? st.events.filter((e) => e.shields.includes(S.selected)).map((e) => ({
    t0: e.onset_t, t1: e.end_t, color: EVENT_BAND[e.category] || 'var(--muted)', label: e.id,
  })) : [];
  lineChart($('chart-legs'), {
    t: ser.t,
    series: names.map((n, k) => ({ name: n, values: ser.channels[n], color: LEG_COLORS[k % LEG_COLORS.length], unit: 'bar' })),
    yLabel: 'bar', unit: 'bar', height: 250, states: ser.state, stateBands: segs, bands,
    hLines: [
      { y: L.high_limit_bar, label: 'High limit (illustrative)', color: 'var(--alert)', dash: '6 4' },
      { y: L.setting_bar, label: 'Setting pressure', color: 'var(--muted)', dash: '2 3' },
    ],
    title: `Leg pressures of ${S.selected}`,
  });
  lineChart($('chart-ai'), {
    t: ser.t,
    series: [{ name: 'AI score (max over legs)', values: ser.ai_score, color: '#1F3A5F', unit: '' }],
    yLabel: 'score', yMin: 0, height: 120, states: ser.state, stateBands: segs,
    hLines: [{ y: 1.0, label: 'Alert threshold (1.0)', color: 'var(--alert)', dash: '4 3' }],
    title: `AI score of ${S.selected}`,
  });
  const cells = sh.legs.map((x) => kv(x.channel, x.value == null ? '—' : `${fmtNum(x.value, 1)} bar`, x.quality === 'ok' ? 'quality ok' : `quality ${x.quality}`));
  cells.push(kv('Status', STATUS_LABEL[sh.status], 'precedence: no data > sensor fault > process alert > watch > normal'));
  cells.push(kv('Mean of valid legs', sh.mean_bar == null ? '—' : `${fmtNum(sh.mean_bar, 1)} bar`, 'mean of valid legs'));
  cells.push(kv('Force (illustrative)', sh.force_MN == null ? '—' : `${fmtNum(sh.force_MN, 2)} MN`, 'F = P × A, piston area from config'));
  cells.push(kv('Deviation', sh.deviation_bar == null ? '—' : `${sh.deviation_bar > 0 ? '+' : ''}${fmtNum(sh.deviation_bar, 1)} bar`, 'shield mean minus median mean of loaded shields'));
  cells.push(kv('State', sh.state, `${sh.time_in_state_s} s in this state`));
  cells.push(kv('AI score', sh.ai_score == null ? '—' : fmtNum(sh.ai_score, 2), '1.0 = calibrated alert threshold; — when not loaded or no valid data'));
  $('readout').innerHTML = cells.join('');
}

function renderHeat() {
  const hm = S.state.heatmap;
  heatmap($('chart-heat'), {
    rows: hm.t, cols: hm.shields, values: hm.values, colorScale: PRESSURE_SCALE, selected: S.selected,
    onColClick: selectShield, height: 300, legendLabel: 'Shield mean (bar)',
  });
}

function selectShield(id) {
  if (!id || id === S.selected) return;
  S.selected = id;
  $('ev-shield').value = id;
  updateScope();
  if (S.state) renderFace();
  fetchState(false);
}

function syncOptions() {
  const sim = S.state.sim;
  if (S.optN !== sim.n_shields) { S.optN = sim.n_shields; buildShieldOptions(sim.n_shields); }
  if (S.optLegs !== sim.legs_per_shield) { S.optLegs = sim.legs_per_shield; buildLegOptions(sim.legs_per_shield); }
}

function buildShieldOptions(n) {
  const sel = $('ev-shield'), cur = sel.value;
  sel.innerHTML = Array.from({ length: n }, (_, i) => `<option value="${sidOf(i)}">${sidOf(i)}</option>`).join('');
  sel.value = cur && idx(cur) < n ? cur : S.selected;
  updateScope();
}

function buildLegOptions(legs) {
  const sel = $('ev-leg'), cur = sel.value;
  sel.innerHTML = Array.from({ length: legs }, (_, l) => `<option value="${l + 1}">Leg L${l + 1}</option>`).join('');
  sel.value = cur && Number(cur) <= legs ? cur : '1';
}

// ---------- alerts, devices, ground truth ----------
function renderAlerts() {
  const st = S.state, box = $('alerts-list');
  const active = document.activeElement;
  const focusId = active && box.contains(active) ? active.dataset.id : null;
  if (!st.alerts.length) {
    box.innerHTML = `<p class="empty">No alerts. The detector is processing ${st.counts.channels_monitored} channels.</p>`;
    return;
  }
  box.innerHTML = st.alerts.map((a) => `
    <button class="alert-row" data-id="${esc(a.id)}">
      <span class="sev sev-${esc(a.severity)}"></span>
      <span>
        <div class="title">${esc(a.id)} · ${esc(a.title)}</div>
        <div class="meta">
          <span class="pill cat-${esc(a.category)}">${esc(a.category)}</span>
          <span class="num">${esc(a.shields.join(', '))}</span>
          <span title="Prioritisation score 0–100, not a probability">priority <b class="num">${a.priority}</b></span>
          <span class="num">opened ${fmtT(a.opened_t)}</span>
          <span class="status-txt">${a.active ? 'active' : 'closed'}</span>
        </div>
      </span>
    </button>`).join('');
  if (focusId) {
    const again = box.querySelector(`[data-id="${CSS.escape(focusId)}"]`);
    if (again) again.focus();
  }
}

function renderDevices() {
  const rows = S.state.devices.map((d) => `
    <tr>
      <td class="num">${esc(d.id)}</td>
      <td class="num">${esc(d.shields[0])}–${esc(d.shields[d.shields.length - 1])}</td>
      <td><span class="pill is-${esc(d.status)}">${esc(d.status)}</span></td>
      <td class="n">${fmtNum(d.age_s, 0)} s</td>
      <td class="n">${d.missing}</td>
      <td class="n">${d.invalid}</td>
      <td class="n">${fmtNum(d.throughput_rps, 0)}/s</td>
    </tr>`).join('');
  $('devices-tbl').innerHTML = `<thead><tr><th>Device</th><th>Shields</th><th>Status</th><th class="n" title="Data age: time since the last valid reading">Age</th>
    <th class="n" title="Missing readings (cumulative)">Miss.</th><th class="n" title="Invalid readings (cumulative)">Inv.</th>
    <th class="n" title="Valid readings per second over the last 10 s">Rate</th></tr></thead><tbody>${rows}</tbody>`;
}

function renderGT() {
  const ev = S.state.events;
  if (!ev.length) { $('gt-list').innerHTML = '<p class="empty">No injected events.</p>'; return; }
  $('gt-list').innerHTML = `<div class="scroll"><table class="tbl"><thead><tr><th>ID</th><th>Type</th><th>Shields</th>
    <th class="n">Onset</th><th class="n">End</th><th class="n">Magnitude</th><th>State</th></tr></thead><tbody>${ev.map((e) => `
    <tr>
      <td class="num">${esc(e.id)}</td>
      <td>${esc(e.label)}</td>
      <td class="num">${esc(e.shields.join(', '))}</td>
      <td class="n">${fmtT(e.onset_t)}</td>
      <td class="n">${fmtT(e.end_t)}</td>
      <td class="n">${e.magnitude == null ? '—' : `${fmtNum(e.magnitude, 1)} ${esc(e.magnitude_unit || '')}`}</td>
      <td>${e.active ? 'active' : 'ended'}</td>
    </tr>`).join('')}</tbody></table></div>`;
}

// ---------- inject event ----------
function currentType() {
  return S.config.event_types.find((e) => e.type === $('ev-type').value) || null;
}

function onTypeChange() {
  const t = currentType();
  if (!t) return;
  $('ev-mag-wrap').classList.toggle('hidden', t.default_magnitude == null);
  $('ev-mag').value = t.default_magnitude == null ? '' : t.default_magnitude;
  $('ev-mag-unit').textContent = t.magnitude_unit ? `(${t.magnitude_unit})` : '';
  $('ev-dur').value = t.default_duration_s;
  // A sensor spike is a single-sample fault; the simulator fixes its duration at one sample.
  const single = t.type === 'sensor_spike';
  $('ev-dur').disabled = single;
  $('ev-dur').title = single ? 'A sensor spike lasts exactly one sample (1 s)' : '';
  $('ev-leg-wrap').classList.toggle('hidden', t.scope !== 'channel');
  updateScope();
}

function updateScope() {
  const t = currentType();
  if (!t || !S.config) return;
  const i = idx($('ev-shield').value || S.selected);
  const n = S.optN ?? S.config.defaults.n_shields;
  const leg = $('ev-leg').value || '1';
  let text;
  if (t.scope === 'shield') text = `Affects ${sidOf(i)} (all legs)`;
  else if (t.scope === 'channel') text = `Affects ${sidOf(i)}-L${leg} only`;
  else if (t.scope === 'block') text = `Affects ${sidOf(Math.max(0, i - 2))}–${sidOf(Math.min(n - 1, i + 2))} (block ±2)`;
  else {
    const g = Math.floor(i / 6) * 6;
    text = `Affects every shield on ${gwOf(i)} (${sidOf(g)}–${sidOf(Math.min(n - 1, g + 5))})`;
  }
  $('ev-scope').textContent = text;
}

async function doInject() {
  const t = currentType(), msg = $('ev-msg');
  const fail1 = (text) => { msg.className = 'small msg-err'; msg.textContent = text; };
  if (!t) return fail1('Choose an event type.');
  const body = { type: t.type, shield: $('ev-shield').value };
  if (t.scope === 'channel') body.leg = Number($('ev-leg').value);
  if (t.default_magnitude != null) {
    const m = numOf('ev-mag');
    if (!Number.isFinite(m)) return fail1('Enter a magnitude.');
    body.magnitude = m;
  }
  const d = numOf('ev-dur');
  if (!(d >= 1)) return fail1('Duration must be at least 1 s.');
  body.duration_s = d;
  $('btn-inject').disabled = true;
  try {
    const ev = await api.injectEvent(body);
    msg.className = 'small msg-ok';
    msg.textContent = `Injected ${ev.id} (${ev.label}) on ${ev.shields.join(', ')} at ${fmtT(ev.onset_t)}.`;
    await fetchState(false);
  } catch (e) {
    fail1(`Inject failed: ${e.message}`);
  } finally {
    $('btn-inject').disabled = !S.state;
  }
}

// ---------- alert investigation drawer ----------
function openDrawer(id) {
  S.drawerId = id;
  S.detail = null;
  S.detailJson = '';
  S.reveal = false;
  $('drawer').classList.add('open');
  $('drawer-bg').classList.add('open');
  $('drawer').setAttribute('aria-hidden', 'false');
  $('drawer-title').textContent = `Alert ${id}`;
  $('drawer-body').innerHTML = '<p class="muted">Loading alert…</p>';
  clearInterval(S.drawerTimer);
  S.drawerTimer = setInterval(() => { if (S.running && S.drawerId) loadDetail(S.drawerId); }, 2000);
  loadDetail(id);
  $('drawer-close').focus();
}

function closeDrawer() {
  if (!S.drawerId) return;
  disposeCharts($('drawer-body'));
  S.drawerId = null;
  S.detail = null;
  S.detailJson = '';
  clearInterval(S.drawerTimer);
  S.drawerTimer = null;
  $('drawer').classList.remove('open');
  $('drawer-bg').classList.remove('open');
  $('drawer').setAttribute('aria-hidden', 'true');
}

async function loadDetail(id) {
  try {
    const d = await api.getAlert(id);
    if (S.drawerId !== id) return;
    const json = JSON.stringify(d);
    if (json === S.detailJson) return; // unchanged: keep the DOM (and focus) as it is
    S.detail = d;
    S.detailJson = json;
    renderDrawer(d);
  } catch (e) {
    if (S.drawerId !== id) return;
    disposeCharts($('drawer-body'));
    $('drawer-body').innerHTML = `<p class="msg-err">Could not load alert ${esc(id)}: ${esc(e.message)}</p>`;
  }
}

// Rule evidence value: fractions as percentages, scores with 2 decimals, everything else 1 decimal.
const evVal = (v, unit) => (unit === 'fraction' ? pct(v) : fmtNum(v, unit === 'score' ? 2 : 1));

const gtText = (ev) => (ev
  ? `<p>Matched injected event <span class="num">${esc(ev.id)}</span>: ${esc(ev.label)} on ${esc(ev.shields.join(', '))}, onset ${fmtT(ev.onset_t)}, ends ${fmtT(ev.end_t)}.</p>`
  : '<p>No matching injected event (possible false alarm).</p>');

function renderDrawer(d) {
  const body = $('drawer-body');
  disposeCharts(body);
  const chans = Object.keys(d.series.channels);
  const shieldsAi = Object.keys(d.series.ai_score);
  const fch = Object.keys(d.features);
  const table = (head, rows) => `<div class="scroll"><table class="tbl"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table></div>`;
  body.innerHTML = `
    <section>
      <div class="row">
        <span class="pill cat-${esc(d.category)}">${esc(d.category)}</span>
        <span class="pill sev-${esc(d.severity)}">severity ${esc(d.severity)}</span>
        <span class="pill">${d.active ? 'active' : 'closed'}</span>
      </div>
      <h3 style="margin-top:8px; font-size:14px">${esc(d.id)} · ${esc(d.title)}</h3>
      <div class="grid2 small">
        <div>Shields: <span class="num">${esc(d.shields.join(', '))}</span></div>
        <div>Opened: <span class="num">${fmtT(d.opened_t)}</span></div>
        <div>Last update: <span class="num">${fmtT(d.last_t)}</span></div>
        <div title="Prioritisation score 0–100, not a probability">Priority <b class="num">${d.priority}</b> / 100</div>
      </div>
      <p class="hint">Prioritisation score, not a hazard probability.</p>
    </section>
    <section><h3>Explanation</h3><p>${esc(d.explanation)}</p></section>
    <section>
      <h3>Time-series evidence <span class="hint">shaded: alert window ${fmtT(d.opened_t)} to ${fmtT(d.last_t)}</span></h3>
      <div id="d-chart-ch"></div>
      <div id="d-chart-ai" style="margin-top:8px"></div>
    </section>
    <section>
      <h3>Rules that fired</h3>
      ${d.evidence.length ? table('<th>Rule</th><th>Channel</th><th class="n">Time</th><th class="n">Observed vs threshold</th><th>Description</th>',
        d.evidence.map((e) => `<tr><td class="num">${esc(e.rule)}</td><td class="num">${esc(e.channel)}</td><td class="n">${fmtT(e.t)}</td>
          <td class="n">${evVal(e.observed, e.unit)} <span class="muted">vs ${evVal(e.threshold, e.unit)}${e.unit === 'fraction' ? '' : ` ${esc(e.unit)}`}</span></td><td>${esc(e.description)}</td></tr>`).join(''))
        : '<p class="empty">No rules recorded.</p>'}
    </section>
    <section>
      <h3>AI anomaly score <span class="hint">1.0 = calibrated alert threshold</span></h3>
      <div class="grid2">
        ${kv('Score max', fmtNum(d.ai_detail.score_max, 2), 'Highest normalised Isolation Forest score during the alert')}
        ${kv('At time', fmtT(d.ai_detail.t_max), 'time of the highest score (t_max)')}
      </div>
      <p class="hint">Isolation Forest trained on normal-only synthetic data. Threshold ${fmtNum(d.ai_detail.threshold, 1)}.</p>
    </section>
    <section>
      <h3>Feature values</h3>
      ${fch.length ? table(`<th>Feature</th>${fch.map((c) => `<th class="n">${esc(c)}</th>`).join('')}`,
        FEATURES.map(([key, label, mult]) => `<tr><td>${esc(label)}</td>${fch.map((c) => {
          const f = d.features[c], v = f ? f[key] : null;
          return `<td class="n">${v == null ? '—' : fmtNum(v * mult, mult === 100 ? 1 : 2)}</td>`;
        }).join('')}</tr>`).join(''))
        : '<p class="empty">No features.</p>'}
    </section>
    <section>
      <h3>Data quality</h3>
      <p><span class="pill is-${esc(d.data_quality.status)}">${esc(d.data_quality.status)}</span> ${esc(d.data_quality.note)}</p>
    </section>
    <section>
      <h3>Neighbouring shields</h3>
      ${d.neighbors.length ? table('<th>Shield</th><th class="n">Mean</th><th class="n">Deviation</th><th>Status</th>',
        d.neighbors.map((n) => `<tr><td class="num">${esc(n.shield)}</td><td class="n">${n.mean_bar == null ? '—' : `${fmtNum(n.mean_bar, 1)} bar`}</td>
          <td class="n">${n.deviation_bar == null ? '—' : `${n.deviation_bar > 0 ? '+' : ''}${fmtNum(n.deviation_bar, 1)} bar`}</td>
          <td><span class="pill is-${esc(n.status)}">${esc(n.status)}</span></td></tr>`).join(''))
        : '<p class="empty">No neighbouring shields.</p>'}
    </section>
    <section>
      <h3>Recommended verification steps</h3>
      <ol class="steps">${d.verification_steps.map((s) => `<li>${esc(s)}</li>`).join('')}</ol>
    </section>
    <section><h3>Uncertainty</h3><p class="callout">${esc(d.uncertainty)}</p></section>
    <section>
      <h3>Priority formula</h3>
      <div class="formula">${esc(d.priority_formula)}</div>
    </section>
    <section>
      <h3>Synthetic ground truth
        <button id="btn-reveal" class="small">${S.reveal ? 'Hide ground truth' : 'Reveal ground truth'}</button>
      </h3>
      ${S.reveal ? gtText(d.matched_event) : '<p class="hint">Hidden so the alert can be judged blind.</p>'}
    </section>`;

  lineChart($('d-chart-ch'), {
    t: d.series.t,
    series: chans.map((c, k) => ({ name: c, values: d.series.channels[c], color: LEG_COLORS[k % LEG_COLORS.length], unit: 'bar' })),
    yLabel: 'bar', unit: 'bar', height: 260,
    bands: [{ t0: d.opened_t, t1: d.last_t, color: 'var(--alert)', label: 'alert' }],
    title: `${d.id} channels`,
  });
  lineChart($('d-chart-ai'), {
    t: d.series.t,
    series: shieldsAi.map((s, k) => ({ name: s, values: d.series.ai_score[s], color: LEG_COLORS[k % LEG_COLORS.length], unit: '' })),
    yLabel: 'score', yMin: 0, height: 120,
    bands: [{ t0: d.opened_t, t1: d.last_t, color: 'var(--alert)', label: '' }],
    hLines: [{ y: 1.0, label: 'Alert threshold (1.0)', color: 'var(--alert)', dash: '4 3' }],
    title: `${d.id} AI score`,
  });
}

// ---------- evaluation tab ----------
function showView(v) {
  S.view = v;
  document.querySelectorAll('.tab').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === v)));
  document.querySelectorAll('.view').forEach((s) => { s.hidden = s.dataset.view !== v; });
  if (v === 'eval' && !S.evalLoaded) loadEval();
}

async function loadEval() {
  S.evalLoaded = true;
  try {
    S.evalData = await api.getEvaluation();
    S.evalError = null;
  } catch (e) {
    S.evalError = `Could not load evaluation: ${e.message}`;
  }
  renderEval();
}

async function runEval() {
  if (S.evalBusy) return;
  S.evalBusy = true;
  S.evalError = null;
  renderEval();
  try {
    S.evalData = await api.runEvaluation();
  } catch (e) {
    S.evalError = `Evaluation failed: ${e.message}`;
  } finally {
    S.evalBusy = false;
    renderEval();
  }
}

function renderEval() {
  const body = $('eval-body'), btn = $('btn-eval-run'), setup = $('eval-setup');
  disposeCharts(body);
  const d = S.evalData;
  const notes = [];
  if (S.evalBusy) notes.push('<p class="msg-ok">Running… this may take up to 90 s.</p>');
  if (S.evalError) notes.push(`<p class="msg-err">${esc(S.evalError)}</p>`);
  btn.disabled = S.evalBusy;
  if (!d) {
    btn.textContent = 'Run evaluation';
    setup.textContent = '';
    body.innerHTML = notes.join('') || '<p class="empty">Loading evaluation…</p>';
    return;
  }
  btn.textContent = S.evalBusy ? 'Running…' : d.available ? 'Re-run evaluation' : 'Run evaluation';
  if (!d.available) {
    setup.textContent = '';
    body.innerHTML = `${notes.join('')}<section class="panel"><h2>No evaluation results yet</h2>
      <p>${esc(d.message || 'Run the evaluation to produce results.')}</p>
      <p class="hint">Running the suite may take up to 90 s.</p></section>`;
    return;
  }
  const su = d.setup, M = d.methods, cc = d.category_confusion;
  setup.textContent = `${su.episodes} episodes × ${su.episode_duration_s / 60} min · ${su.events_total} injected events · test seeds ${su.test_seeds.join(', ')} · generated ${d.generated_at}`;
  const bestOf = (key, pick) => {
    const vals = M.map((m) => m[key]).filter((v) => v != null);
    return vals.length ? pick(...vals) : null;
  };
  const best = {
    precision: bestOf('precision', Math.max), recall: bestOf('recall', Math.max), f1: bestOf('f1', Math.max),
    fp_per_hour: bestOf('fp_per_hour', Math.min), median_delay_s: bestOf('median_delay_s', Math.min),
    tp_events: bestOf('tp_events', Math.max), fn_events: bestOf('fn_events', Math.min),
    runtime_ms_per_step: bestOf('runtime_ms_per_step', Math.min),
  };
  const cls = (m, key) => (best[key] != null && m[key] === best[key] ? 'n best' : 'n');
  const short = (label) => label.replace(/\s*\(.*\)/, '');
  const methodRows = M.map((m) => `<tr>
    <td title="${esc(m.description)}">${esc(m.label)}</td>
    <td class="${cls(m, 'precision')}">${pct(m.precision)}</td>
    <td class="${cls(m, 'recall')}">${pct(m.recall)}</td>
    <td class="${cls(m, 'f1')}">${pct(m.f1)}</td>
    <td class="${cls(m, 'fp_per_hour')}">${fmtNum(m.fp_per_hour, 2)}</td>
    <td class="${cls(m, 'median_delay_s')}">${fmtNum(m.median_delay_s, 0)}</td>
    <td class="${cls(m, 'tp_events')}">${m.tp_events}</td>
    <td class="${cls(m, 'fn_events')}">${m.fn_events}</td>
    <td class="n">${m.alerts_total}</td>
    <td class="${cls(m, 'runtime_ms_per_step')}">${fmtNum(m.runtime_ms_per_step, 1)}</td>
  </tr>`).join('');
  const types = S.config ? S.config.event_types : [];
  const scenRows = types.map((t) => `<tr><td>${esc(t.label)}</td>${M.map((m) => {
    const c = d.by_scenario && d.by_scenario[m.id] ? d.by_scenario[m.id][t.type] : null;
    if (!c) return '<td class="n">—</td>';
    const bg = `color-mix(in srgb, var(--ok) ${Math.round(c.recall * 40)}%, transparent)`;
    return `<td class="n cell-recall" style="background:${bg}">${c.detected}/${c.events} (${Math.round(c.recall * 100)}%)
      <div class="muted small">median ${c.median_delay_s == null ? '—' : `${fmtNum(c.median_delay_s, 0)} s`}</div></td>`;
  }).join('')}</tr>`).join('');
  const confRows = cc.labels.map((lab, r) => `<tr><th style="text-align:left;font-weight:500">${esc(lab)}</th>${cc.matrix[r]
    .map((v, c) => `<td class="n${r === c ? ' diag' : ''}">${v}</td>`).join('')}</tr>`).join('');
  const diffRows = (d.difficult_cases || []).map((c) => `<tr><td>${esc(c.name)}</td><td>${esc(c.description)}</td>${M
    .map((m) => `<td>${esc((c.results && c.results[m.id]) ?? '—')}</td>`).join('')}</tr>`).join('');
  body.innerHTML = `${notes.join('')}
    <div class="eval-grid">
      <section class="panel wide"><h2>Matching rule</h2>
        <p class="doc-p">${esc(su.matching_rule)}</p>
        <p class="hint">Train seeds ${su.train_seeds.join(', ')} · validation seeds ${su.validation_seeds.join(', ')} · test seeds ${su.test_seeds.join(', ')}.</p>
      </section>
      <section class="panel wide"><h2>Methods</h2>
        <div class="scroll"><table class="tbl"><thead><tr>
          <th>Method</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">F1</th><th class="n">False alerts / h</th>
          <th class="n">Median delay (s)</th><th class="n">TP events</th><th class="n">Missed events</th><th class="n">Alerts</th><th class="n">Runtime (ms/step)</th>
        </tr></thead><tbody>${methodRows}</tbody></table></div>
        <p class="hint">Green marks the best value in each column. Precision, recall and F1 are shown as percentages.</p>
      </section>
      <section class="panel"><h2>Precision, recall and F1</h2><div id="eval-bars"></div></section>
      <section class="panel"><h2>Category confusion (${esc(cc.method)})</h2>
        <div class="scroll"><table class="tbl matrix"><thead><tr><th>Truth ↓ · predicted →</th>${cc.columns.map((c) => `<th class="n">${esc(c)}</th>`).join('')}</tr></thead>
        <tbody>${confRows}</tbody></table></div>
        <p class="hint">Rows are the true category. The last column counts events that were missed.</p>
      </section>
      <section class="panel wide"><h2>Recall by scenario type</h2>
        <div class="scroll"><table class="tbl"><thead><tr><th>Scenario</th>${M.map((m) => `<th class="n">${esc(short(m.label))}</th>`).join('')}</tr></thead>
        <tbody>${scenRows}</tbody></table></div>
        <p class="hint">Cell: detected / injected events (recall), with median delay to the first matching alert. Shading scales with recall.</p>
      </section>
      <section class="panel wide"><h2>Difficult cases</h2>
        <div class="scroll"><table class="tbl"><thead><tr><th>Case</th><th>Description</th>${M.map((m) => `<th>${esc(short(m.label))}</th>`).join('')}</tr></thead>
        <tbody>${diffRows}</tbody></table></div>
      </section>
      <section class="panel wide"><h2>Limitations</h2>
        <ul>${(d.limitations || []).map((x) => `<li>${esc(x)}</li>`).join('')}</ul>
      </section>
    </div>`;
  barChart($('eval-bars'), {
    categories: M.map((m) => short(m.label)),
    groups: [
      { name: 'Precision', values: M.map((m) => m.precision), color: '#1F3A5F' },
      { name: 'Recall', values: M.map((m) => m.recall), color: '#2A8C82' },
      { name: 'F1', values: M.map((m) => m.f1), color: '#C2571A' },
    ],
    yMax: 1, yLabel: 'score', valueFormat: (v) => `${Math.round(v * 100)}%`, height: 280,
  });
}

// ---------- boot ----------
wire();
showView('live');
renderChips();
loadConfig();

function wire() {
  document.querySelectorAll('.tab').forEach((b) => b.addEventListener('click', () => showView(b.dataset.tab)));
  $('btn-run').addEventListener('click', () => (S.running ? pause() : start()));
  $('btn-step').addEventListener('click', step);
  $('btn-reset').addEventListener('click', doReset);
  $('err-retry').addEventListener('click', retry);
  $('face-color').addEventListener('change', (e) => {
    S.colorBy = e.target.value;
    renderLegend();
    if (S.state) renderFace();
  });
  $('show-gt').addEventListener('change', (e) => {
    S.showGT = e.target.checked;
    if (S.state) renderShield();
  });
  ['p-seed', 'p-shields', 'p-legs', 'p-noise', 'p-trend'].forEach((id) => $(id).addEventListener('input', validateReset));
  $('ev-type').addEventListener('change', onTypeChange);
  $('ev-shield').addEventListener('change', updateScope);
  $('ev-leg').addEventListener('change', updateScope);
  $('btn-inject').addEventListener('click', doInject);
  $('alerts-list').addEventListener('click', (e) => {
    const b = e.target.closest('.alert-row');
    if (b) openDrawer(b.dataset.id);
  });
  $('drawer-body').addEventListener('click', (e) => {
    if (e.target.closest('#btn-reveal') && S.detail) {
      S.reveal = !S.reveal;
      renderDrawer(S.detail);
      $('btn-reveal').focus();
    }
  });
  $('drawer-close').addEventListener('click', closeDrawer);
  $('drawer-bg').addEventListener('click', closeDrawer);
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && S.drawerId) closeDrawer(); });
  $('btn-eval-run').addEventListener('click', runEval);
}
