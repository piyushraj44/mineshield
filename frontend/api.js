// The only module that talks to the backend. All calls go to the FastAPI server under /api.
export const exportExperimentUrl = '/api/export/experiment.json';
export const exportTelemetryUrl = '/api/export/telemetry.csv';

async function request(method, path, body) {
  const init = { method, headers: {} };
  if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch (e) {
    throw new Error(`Backend not reachable (${method} ${path})`);
  }
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (e) {
    data = null;
  }
  if (!res.ok) {
    const d = data && data.detail;
    const msg = typeof d === 'string' ? d : d ? JSON.stringify(d) : `HTTP ${res.status}`;
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return data;
}

const qs = (o) => new URLSearchParams(o).toString();

export const getConfig = () => request('GET', '/api/config');
export const resetSim = (body = {}) => request('POST', '/api/sim/reset', body);
export const stepSim = (body) => request('POST', '/api/sim/step', body);
export const getState = (shield, windowS = 300) => request('GET', '/api/state?' + qs({ shield, window_s: windowS }));
export const getAlert = (id) => request('GET', '/api/alerts/' + encodeURIComponent(id));
export const getEvaluation = () => request('GET', '/api/evaluation');
export const runEvaluation = () => request('POST', '/api/evaluation/run');
// Body: {type, shield, leg?, magnitude?, duration_s?} -> EventRecord (docs/API_CONTRACT.md).
export const injectEvent = (body) => request('POST', '/api/sim/inject', body);
