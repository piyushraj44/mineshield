# MineShield AI — API Contract (v0.1)

Single source of truth between `frontend/` and the Python backend (`mineshield/`).
All JSON keys are snake_case. Pressure is always **bar**, force **MN**, time **seconds of simulated time** (`t`).
The backend serves the frontend statically at `/` and the API under `/api`.

## Conventions
- Shield ids: `S01`..`S24` (zero-padded, 1-based). Channel ids: `S05-L1`..`S05-L4` (leg number).
- Gateways: `GW-01` covers S01–S06, `GW-02` S07–S12, `GW-03` S13–S18, `GW-04` S19–S24.
- `t` = simulated seconds since reset (float, 1 sample per second, `dt_s = 1.0`).
- Missing values are JSON `null`.
- Event / alert categories: `"process"` (pressure behaviour), `"sensor"` (measurement fault), `"telemetry"` (communication loss).
- Shield status values: `"normal" | "watch" | "process_alert" | "sensor_fault" | "no_data"`.
- Shield operating states: `"advancing"` (legs lowered, shield moving), `"setting"` (re-set, pressure building), `"loaded"` (carrying roof load).

---

## GET /api/config
Static configuration for building the UI.
```json
{
  "version": "0.1.0",
  "units": {"pressure": "bar", "force": "MN", "time": "s"},
  "limits": {"sensor_min_bar": 0, "sensor_max_bar": 600, "setting_bar": 250,
             "high_limit_bar": 380, "yield_bar": 420, "low_limit_bar": 150},
  "piston_area_m2": 0.0804,
  "defaults": {"seed": 42, "n_shields": 12, "legs_per_shield": 2, "noise_bar": 2.0, "load_trend_bar_per_h": 0},
  "ranges": {"n_shields": [6, 24], "legs_per_shield": [2, 4], "noise_bar": [0.5, 8],
             "load_trend_bar_per_h": [-60, 120]},
  "speeds": [{"label": "2x", "steps": 1}, {"label": "4x", "steps": 2}, {"label": "10x", "steps": 5},
             {"label": "20x", "steps": 10}, {"label": "30x", "steps": 15}],
  "poll_ms": 500,
  "event_types": [
    {"type": "gradual_rise", "label": "Gradual pressure increase", "category": "process", "scope": "shield",
     "default_magnitude": 70, "magnitude_unit": "bar", "default_duration_s": 300},
    {"type": "pressure_spike", "label": "Sudden pressure spike", "category": "process", "scope": "shield",
     "default_magnitude": 90, "magnitude_unit": "bar", "default_duration_s": 40},
    {"type": "leakage", "label": "Leakage-like pressure drop", "category": "process", "scope": "channel",
     "default_magnitude": 0.8, "magnitude_unit": "bar/s", "default_duration_s": 150},
    {"type": "leg_imbalance", "label": "Uneven leg loading", "category": "process", "scope": "shield",
     "default_magnitude": 25, "magnitude_unit": "%", "default_duration_s": 200},
    {"type": "multi_shield_loading", "label": "Localized multi-shield loading", "category": "process", "scope": "block",
     "default_magnitude": 60, "magnitude_unit": "bar", "default_duration_s": 300},
    {"type": "sensor_spike", "label": "Isolated sensor spike", "category": "sensor", "scope": "channel",
     "default_magnitude": 150, "magnitude_unit": "bar", "default_duration_s": 1},
    {"type": "sensor_stuck", "label": "Sensor stuck at constant value", "category": "sensor", "scope": "channel",
     "default_magnitude": null, "magnitude_unit": null, "default_duration_s": 120},
    {"type": "sensor_dropout", "label": "Missing sensor readings", "category": "sensor", "scope": "channel",
     "default_magnitude": null, "magnitude_unit": null, "default_duration_s": 60},
    {"type": "telemetry_interruption", "label": "Gateway telemetry interruption", "category": "telemetry", "scope": "gateway",
     "default_magnitude": null, "magnitude_unit": null, "default_duration_s": 45}
  ],
  "disclaimer": "This prototype uses simulated telemetry and simplified engineering models. It is not a certified mining safety system, does not establish the probability of roof collapse, and must not be used to authorize mining operations or replace qualified personnel and approved safety procedures."
}
```
Scope meaning: `shield` = all legs of one shield; `channel` = one leg channel; `block` = the chosen shield ±2 neighbours; `gateway` = every shield on the chosen shield's gateway.

## POST /api/sim/reset
Body (all optional, defaults from config): `{"seed": 42, "n_shields": 12, "legs_per_shield": 2, "noise_bar": 2.0, "load_trend_bar_per_h": 0}`
Response: `{"ok": true, "sim": SimInfo}`. Resets time to 0, clears alerts/events. Same seed ⇒ identical telemetry.
Invalid input ⇒ HTTP 422 `{"detail": "message"}`.

## POST /api/sim/step
Advances the simulation and returns the full state (saves a round trip). The frontend owns run/pause: it calls this every `poll_ms` while running.
Body: `{"steps": 5, "shield": "S05", "window_s": 300}` (`steps` 1–60, `shield` = currently selected shield, `window_s` 60–900).
Response: **State** (same as GET /api/state).

## POST /api/sim/inject
Injects a synthetic event starting at the next sample.
Body: `{"type": "leakage", "shield": "S05", "leg": 1, "magnitude": 0.8, "duration_s": 150}`
- `leg` (1-based) only for `scope == "channel"` (default 1); ignored otherwise.
- `magnitude` omitted/null ⇒ type default (must be omitted for types whose default is null). `duration_s` omitted ⇒ default (1–3600).
Response: the **EventRecord** itself (not wrapped). Invalid input ⇒ HTTP 422 `{"detail": "message"}`.
`channels` of shield/block/gateway-scoped events list every leg channel of every affected shield.

## GET /api/state?shield=S05&window_s=300
Returns **State** without advancing.

### State
```json
{
  "sim": {
    "t": 1234.0, "samples": 1235, "seed": 42, "n_shields": 12, "legs_per_shield": 2, "dt_s": 1.0,
    "noise_bar": 2.0, "load_trend_bar_per_h": 0,
    "data_source": "simulator",
    "model": {"name": "IsolationForest", "version": "if-0.1.0", "status": "ready",
              "trained_on": "normal-only synthetic seeds 1000-1004", "threshold_note": "score 1.0 = 99.95th percentile of validation normal data"}
  },
  "counts": {"active_alerts": 2, "channels_monitored": 24, "channels_ok": 22, "readings_total": 29640},
  "shields": [
    {"id": "S01", "index": 0, "gateway": "GW-01", "state": "loaded", "time_in_state_s": 123,
     "legs": [{"channel": "S01-L1", "value": 312.4, "quality": "ok"}, {"channel": "S01-L2", "value": 309.9, "quality": "ok"}],
     "mean_bar": 311.2, "force_MN": 5.01, "deviation_bar": -4.3, "ai_score": 0.42, "status": "normal"}
  ],
  "heatmap": {"t": [940.0, 945.0], "shields": ["S01", "S02"], "values": [[311.0, 305.2], [312.1, null]]},
  "series": {"shield": "S05", "t": [935.0, 936.0], "channels": {"S05-L1": [301.2, 302.0], "S05-L2": [299.0, null]},
             "state": ["loaded", "loaded"], "ai_score": [0.31, 0.35]},
  "alerts": [AlertSummary],
  "events": [EventRecord],
  "devices": [{"id": "GW-01", "shields": ["S01", "S02", "S03", "S04", "S05", "S06"], "status": "simulated",
               "last_t": 1234.0, "age_s": 0.0, "missing": 12, "invalid": 0, "throughput_rps": 12.0}]
}
```
Field notes:
- `quality`: `"ok" | "missing" | "invalid"`.
- `mean_bar`: mean of valid legs (null if none). `force_MN` = Σ(valid leg pressure × piston area) — illustrative (F = P × A, 1 bar = 1e5 Pa).
- `deviation_bar`: shield `mean_bar` minus the median `mean_bar` of all **loaded** shields (null if not computable).
- `ai_score`: max normalised Isolation Forest score over the shield's legs at the latest sample; **1.0 = calibrated alert threshold**. null when not scored (shield not loaded or no valid data).
- `status` precedence: `no_data` > `sensor_fault` > `process_alert` > `watch` (ai_score ≥ 0.8) > `normal`.
- `heatmap`: shield mean pressure every 5 s over the window (rows = times in **ascending** order, columns = shields).
- `window_s` is clamped to 60–900; `shield` unknown ⇒ falls back to S01.
- `series`: 1-s resolution for the selected shield over the window.
- `alerts`: newest first, max 50. `events`: newest first (synthetic ground truth, all injected events).
- `devices.status`: `"simulated"` (receiving), `"stale"` (no valid data 3–10 s), `"disconnected"` (>10 s).

### EventRecord (synthetic ground truth)
```json
{"id": "EV-003", "type": "leakage", "label": "Leakage-like pressure drop", "category": "process",
 "onset_t": 1234.0, "end_t": 1384.0, "shields": ["S05"], "channels": ["S05-L1"],
 "magnitude": 0.8, "magnitude_unit": "bar/s", "active": true}
```

### AlertSummary
```json
{"id": "AL-0007", "opened_t": 1240.0, "last_t": 1290.0, "active": true, "category": "process",
 "title": "Rapid pressure rise", "shields": ["S05", "S06"], "channels": ["S05-L1", "S06-L2"],
 "severity": "high", "priority": 72, "rules": ["rapid_rise", "persistent_deviation"], "ai_score_max": 2.3}
```
`severity`: `"high" | "medium" | "low"`. `priority` 0–100 is a **prioritisation score, not a probability**.

## GET /api/alerts/{alert_id}
AlertSummary plus:
```json
{
  "explanation": "Shield S05 shows an unusually rapid pressure increase ...",
  "evidence": [{"rule": "rapid_rise", "channel": "S05-L1", "t": 1241.0, "observed": 5.2, "threshold": 4.0,
                "unit": "bar/s", "description": "5-s pressure gradient"}],
  "features": {"S05-L1": {"pressure_bar": 351.2, "dev_own_bar": 32.1, "slope_5_bar_s": 5.2, "slope_30_bar_s": 1.1,
                          "imbalance_frac": 0.04, "neighbor_dev_bar": 28.0, "roll_std_bar": 6.3, "missing_frac": 0.0}},
  "ai_detail": {"score_max": 2.3, "t_max": 1250.0, "threshold": 1.0},
  "data_quality": {"status": "ok", "note": "All channels of the affected shields reported valid data."},
  "neighbors": [{"shield": "S04", "mean_bar": 318.0, "deviation_bar": 2.1, "status": "normal"}],
  "verification_steps": ["Check ...", "Verify ..."],
  "uncertainty": "This alert is based on simulated readings ...",
  "priority_formula": "priority = 50*rule_weight + 30*min(ai_score_max/2,1) + 20*min((n_shields-1)/2,1)",
  "series": {"t": [1120.0, 1121.0], "channels": {"S05-L1": [300.1, 301.0]}, "ai_score": {"S05": [0.3, 0.4]}},
  "window": [1120.0, 1320.0],
  "matched_event": EventRecord | null
}
```
`series` covers all channels of the alert's shields from `opened_t − 120` to `min(last_t + 30, now)`.
`matched_event` = synthetic ground-truth event overlapping in time and shields (UI hides it behind a "Reveal ground truth" toggle).
Unknown id ⇒ 404.

## GET /api/evaluation
```json
{
  "available": true, "generated_at": "2026-10-13T12:00:00Z",
  "setup": {"test_seeds": [3000, 3001], "episodes": 16, "episode_duration_s": 1800, "events_total": 48,
            "train_seeds": [1000, 1001, 1002, 1003, 1004], "validation_seeds": [2000, 2001],
            "matching_rule": "An alert matches an event if it opens within [onset, end + 30 s] and shares ≥1 shield."},
  "methods": [
    {"id": "fixed_threshold", "label": "Fixed threshold (baseline)", "description": "...",
     "precision": 0.9, "recall": 0.3, "f1": 0.45, "fp_per_hour": 0.2, "median_delay_s": 41.0,
     "tp_events": 14, "fn_events": 34, "alerts_total": 16, "false_alerts": 2, "runtime_ms_per_step": 0.4}
  ],
  "by_scenario": {"fixed_threshold": {"gradual_rise": {"events": 6, "detected": 2, "recall": 0.33, "median_delay_s": 120.0}}},
  "category_confusion": {"method": "mineshield_hybrid", "labels": ["process", "sensor", "telemetry"],
                         "columns": ["process", "sensor", "telemetry", "missed"], "matrix": [[20, 1, 0, 3], [0, 14, 0, 2], [0, 0, 6, 0]]},
  "difficult_cases": [{"name": "Gradual anomaly below fixed threshold", "description": "...",
                       "results": {"fixed_threshold": "missed", "mineshield_hybrid": "detected after 64 s"}}],
  "limitations": ["..."]
}
```
Method ids (in display order): `fixed_threshold`, `rolling_mad`, `isolation_forest`, `mineshield_hybrid`.
If no results file exists: `{"available": false, "message": "Run: python -m mineshield.evaluation"}`.

## POST /api/evaluation/run
Runs the suite (may take ~30–90 s), saves to `data/eval_results.json`, returns the same object as GET.

## GET /api/export/experiment.json
Download: `{"exported_at", "version", "sim": SimInfo, "events": [EventRecord], "alerts": [AlertDetail without series]}`.

## GET /api/export/telemetry.csv
Download. Columns: `t_s,timestamp,seq,device_id,shield_id,channel_id,value,unit,quality,state,gt_label,gt_event_id`.

## POST /api/ingest/validate
Validates readings against the telemetry schema (same validator the simulator path uses).
Body: `{"readings": [TelemetryReading]}` → `{"accepted": 3, "rejected": [{"index": 1, "reason": "unit must be 'bar'"}]}`.

### TelemetryReading
```json
{"timestamp": "2026-01-01T06:20:34Z", "t_s": 1234.0, "seq": 29617, "device_id": "GW-01", "shield_id": "S05",
 "channel_id": "S05-L1", "value": 312.4, "unit": "bar", "quality": "ok"}
```
