# MineShield AI — Backend Model Specification (v0.1)

Implementation spec for the Python package `mineshield/`. Parameters live in `config/default.json` (referred to as `cfg`).
Only dependencies: **numpy, scikit-learn, fastapi, uvicorn** (+ pytest for tests). Standard library otherwise.
All values in this document are **illustrative**, chosen to give plausible signal shapes — not equipment specifications.

---------------------------------------------------------------------------------------------------
## 1. Physics (`mineshield/physics.py`)
- `BAR_TO_PA = 1e5`
- `piston_area_m2(bore_m) = pi * (bore_m/2)**2`  → 0.32 m bore ⇒ 0.080425 m²
- `force_mn(pressure_bar, area_m2) = pressure_bar * 1e5 * area_m2 / 1e6`  (F = P × A; 300 bar ⇒ 2.413 MN per leg)
- Accept numpy arrays; NaN propagates.

## 2. Simulator (`mineshield/simulator.py`)

### 2.1 Indexing
- `n_shields` S, `legs` L. Shield ids `S01..`, channel ids `Sxx-Lj`. Channel order: S01-L1, S01-L2, …, S02-L1, … (channel c ⇒ shield c // L, leg c % L).
- Gateway of shield i (0-based) = `GW-%02d % (i // shields_per_gateway + 1)`.

### 2.2 Random streams (reproducibility independent of injected events)
`rng_setup = np.random.default_rng(seed)`, `rng_noise = np.random.default_rng(seed + 1)`, `rng_trans = np.random.default_rng(seed + 2)`.
Every step draws noise for **all** channels and transient uniforms for **all** shields, whether or not events are active, so injecting an event never shifts the random sequence.

### 2.3 Per-shield setup (drawn once from rng_setup, in this order)
- `phi = rng_setup.uniform(0, 2π)`
- `k_i = 1 + spatial_amplitude * sin(2π (i + 0.5) / S + phi) + rng_setup.uniform(-spatial_jitter, spatial_jitter, S)` — spatial load factor.
- `f_ij = clip(1 + rng_setup.normal(0, leg_factor_sd, (S, L)), 1-leg_factor_clip, 1+leg_factor_clip)` — leg factor.
- Cycle offset `offset_i = i * cycle_s / S`.

### 2.4 Operating cycle (shield advance)
Cycle position `c_i(t) = (t - offset_i) mod cycle_s` (Python `%`, always ≥ 0).
- `c < advance_s` ⇒ state **advancing** (legs lowered).
- `advance_s ≤ c < advance_s + setting_s` ⇒ **setting** (time since set `tau = c - advance_s`).
- otherwise ⇒ **loaded** (`tau = c - advance_s`).
`time_in_state_s` = `c` (advancing), `c - advance_s` (setting), `c - advance_s - setting_s` (loaded).

### 2.5 Roof load (equivalent leg pressure, bar)
`L_i(t) = roof_load_bar * k_i + load_trend_bar_per_h * t / 3600 + T_i(t) + E_i(t)`
- `T_i(t)` normal transients: each step, for each shield, `u = rng_trans.random(S)`, `a = rng_trans.uniform(amp_lo, amp_hi, S)`, `sgn = rng_trans.choice([-1, 1], S)` (always drawn). If `u < transient_rate_per_h/3600`, start a transient `sgn*a` decaying `exp(-(t - t0)/transient_tau_s)`. Implementation: keep a per-shield transient level `T_i`; each step `T_i *= exp(-dt/transient_tau_s)`, then add new kicks. Labelled **normal**.
- `E_i(t)` process-event load contributions (section 2.7).

### 2.6 Leg pressure
Target pressure for leg j of shield i:
- advancing: `P* = lowered_bar`
- setting/loaded: `P* = setting_bar + max(0, L_i(t) * f_ij - setting_bar) * (1 - exp(-tau / tau_load_s))`
  then apply leg-imbalance factor and leakage loss (2.7), then `P* = clip(P*, lowered_bar, yield_bar)` (yield valve caps pressure).
Hydraulic response (first-order lag): `P(t) = P(t-dt) + (P*(t) - P(t-dt)) * (1 - exp(-dt / tau_hyd_s))`. At t = 0, `P = P*`.
Measurement: `m = round((P + rng_noise.normal(0, noise_bar, S*L)) / q) * q` with q = quantization_bar, then clipped to `[sensor_min_bar, sensor_max_bar]`; then sensor events (2.7) applied to `m`.

### 2.7 Events
`Event` dataclass: `id` ("EV-001", sequential per simulator), `type`, `label`, `category`, `onset_t`, `end_t = onset_t + duration_s`, `shields` (ids), `channels` (ids), `magnitude`, `magnitude_unit`, `scope`. `active(t) = onset_t <= t < end_t`. Event catalog constant `EVENT_TYPES` must exactly match `/api/config.event_types` in docs/API_CONTRACT.md.
Scope resolution in `inject(type, shield_id, channel_id=None, magnitude=None, duration_s=None, onset_t=None)`:
- `shield`: all channels of the shield. `channel`: given channel (default leg 1 of the shield). `block`: shields i-2..i+2 clipped to the face. `gateway`: all shields of the shield's gateway.
- magnitude/duration default from catalog; `onset_t` default = time of the next sample produced (`sim.t`). Future onsets allowed (used by evaluation).
- Validation: unknown type / shield / channel ⇒ `ValueError` with a clear message; magnitude must be > 0 when the type has one; duration 1–3600 s.

Process-event effects (`r = t - onset_t`, `d = duration_s`, `m = magnitude`, `rec = recovery_tau_s`):
| type | effect |
|---|---|
| gradual_rise | `E_i += m * min(r/d, 1)` while active; after end `E_i += m * exp(-(t-end)/rec)` |
| pressure_spike | `E_i += m * exp(-r / (d/4))` while active; 0 after end |
| multi_shield_loading | weight w = 1.0 (centre), 0.7 (±1), 0.35 (±2); `E_i += w*m*min(r/60, 1)` while active; after end `w*m*exp(-(t-end)/rec)` |
| leak (leakage) | on its channel, loss `= m * r` while active (subtract from P* before clipping); after end loss `= m*d*exp(-(t-end)/rec)` |
| leg_imbalance | for the shield's legs, multiply P* by `1 - s*(m/200)*min(r/120,1)` (gradual 120 s build-up) with s = +1 for odd legs (L1, L3), −1 for even legs (L2, L4) while active; after end the factor relaxes as `exp(-(t-end)/rec)` back to 1 |
Sensor/telemetry effects on the measurement `m`:
| type | effect |
|---|---|
| sensor_spike | at the single sample `t == onset_t`: `m += magnitude` (duration is 1 s) |
| sensor_stuck | while active: `m =` last reported value of the channel at the sample before onset (store when first applied) |
| sensor_dropout | while active: NaN |
| telemetry_interruption | while active: NaN for every channel of every event shield |
Process effects only apply when the shield is not advancing (advancing always targets lowered_bar). Effects of overlapping events add.
Hidden type `benign_transient` (category "normal", not in EVENT_TYPES, not shown in /api/config): `E_i += m * exp(-r/15)` while active. Used only by evaluation difficult cases; it is never a ground-truth event and labels stay "normal".

### 2.8 Ground-truth labels
Per sample per channel: `gt_label` = type of the most recently injected event active at t that covers the channel (else "normal"), `gt_event_id` likewise (else None). Recovery period after end_t ⇒ "normal". benign_transient ⇒ "normal".

### 2.9 Interface
```python
class FaceSimulator:
    def __init__(self, cfg: dict, seed: int, n_shields: int = 12, legs_per_shield: int = 2,
                 noise_bar: float = 2.0, load_trend_bar_per_h: float = 0.0): ...
    shield_ids: list[str]; channel_ids: list[str]; gateway_ids: list[str]
    channel_shield: np.ndarray  # (C,) int shield index per channel
    t: float                    # time of the NEXT sample (starts 0.0)
    seq: int                    # next sequence number (starts 0); each channel reading consumes one seq
    events: list[Event]         # all injected events (incl. future), excluding benign_transient
    def gateway_of(self, shield_idx: int) -> str
    def inject(...) -> Event
    def step(self) -> Frame
@dataclass
class Frame:
    t: float
    values: np.ndarray        # (C,) measured bar, NaN = missing
    true_values: np.ndarray   # (C,) noise-free hydraulic pressure (debug/tests only, never used by detectors)
    states: list[str]         # (S,) advancing/setting/loaded
    time_in_state: np.ndarray # (S,) seconds
    gt_label: list[str]       # (C,)
    gt_event_id: list         # (C,) str | None
```
`step()` produces the sample at time `self.t`, then advances `self.t += dt_s`.

## 3. Telemetry (`mineshield/telemetry.py`)
- `frame_to_readings(frame, sim) -> list[dict]` — one TelemetryReading per channel (schema in API_CONTRACT.md): `timestamp` = start_timestamp + t (ISO8601 "Z"), `seq` consecutive, `device_id` = gateway, `value` None if NaN, `unit` "bar", `quality` "ok" or "missing".
- `TelemetryValidator(channel_ids, limits)`: `validate(readings) -> (accepted: list[dict], rejected: list[{"index", "reason"}])`. Checks, in order: required fields present (timestamp, t_s, seq, device_id, shield_id, channel_id, value, unit, quality); `unit == "bar"`; channel known; channel prefix matches shield_id; value is number or None (bool rejected); duplicate `seq` already seen (keep last 10 000 seqs) ⇒ reject "duplicate"; `t_s` older than the last accepted t_s for that channel ⇒ reject "stale timestamp". Out-of-range value (outside sensor_min/max) ⇒ **accepted** with `quality = "invalid"` and `value = None` (counted as invalid). Stateful across calls; `reset()`.
- `readings_to_vector(accepted, channel_ids) -> (values (C,), quality list[str])` — NaN for missing/invalid/absent channels; absent channel ⇒ quality "missing".
- `DeviceMonitor(sim)`: `update(t, quality list)`, `status(t) -> list[dict]` per gateway as in the contract: `last_t` = last t with ≥1 valid reading, `age_s = t - last_t`, status simulated (age < 3), stale (3 ≤ age ≤ 10), disconnected (> 10); cumulative `missing` and `invalid` counts; `throughput_rps` = valid readings in the last 10 s / 10.

## 4. Features (`mineshield/features.py`)
`FeatureEngine(n_shields, legs, cfg)`; `update(t, values (C,), states (S,), time_in_state (S,)) -> FeatureFrame`. Keeps a ring buffer of the last `history_s` values per channel (NaN for missing), plus per-channel run counters.
FeatureFrame fields: `t`, `X` (C × 8 float array, columns in FEATURE_NAMES order, NaN = not computable), `values`, `valid` (C bool), `states`, `time_in_state`, `loaded_ok` (C bool: shield loaded and time_in_state ≥ min_loaded_s), `rise_ok` (C bool: loaded (any time in state) OR (setting and time_in_state ≥ min_setting_s) — no blind spot at the setting→loaded transition), `flat_run` (C int), `missing_run` (C int), `spike` (C bool), `scoreable` (C bool), `shield_mean` (S, NaN if no valid leg).
FEATURE_NAMES = `pressure_bar, dev_own_bar, slope_5_bar_s, slope_30_bar_s, imbalance_frac, neighbor_dev_bar, roll_std_bar, missing_frac`.
Definitions (v = current value; h[k] = value k samples ago):
- `pressure_bar = v`
- `dev_own_bar = v − median(valid h[k] for k in [30, 180], restricted to k ≤ time_in_state of the current loaded run)`; NaN unless ≥ baseline_min_samples (20) valid samples and loaded.
- `slope_5_bar_s = (v − h[5]) / 5`.
- `slope_30_bar_s = (median(h[0..4]) − median(h[25..29])) / 25` (robust to single spikes; each median needs ≥ 3 valid).
- `imbalance_frac = (v − mean(other valid legs of the shield)) / mean(all valid legs)`; NaN if no other valid leg.
- `neighbor_dev_bar = shield_mean_i − median(shield_mean_n for neighbours |n−i| ≤ neighbor_radius, n ≠ i, loaded and valid)`; NaN if none. Only computed for loaded shields.
- `roll_std_bar = 1.4826 × MAD(last 10 valid values)` (needs ≥ 5).
- `missing_frac = fraction of NaN in the last 30 samples` (samples before t=0 don't count).
- `flat_run` = number of consecutive samples ending now whose value exactly equals the previous value (0 if v is NaN or differs; 1 means v == h[1]).
- `missing_run` = consecutive NaN samples ending now.
- `spike` (evaluates the previous sample, 1-sample delay): `|h[1] − med| > spike_jump_bar and |v − med| < spike_return_bar` where `med = median(h[2..6])` (needs ≥3 valid and h[1], v valid).
- `scoreable` = valid & loaded_ok & dev_own, slope_5, slope_30 finite. For model input, NaN imbalance/neighbor ⇒ 0.

## 5. Rules (`mineshield/rules.py`)
`RuleEngine(n_channels, legs, cfg)`; `evaluate(ff: FeatureFrame, gateway_of_shield) -> list[Trigger]`. Stateful persistence counters (reset when condition false). `reset()`.
`Trigger` dataclass: `rule, category, shield_idx, channel_idx (int|None), t, observed, threshold, unit, description`.
| rule | category | condition (per channel unless noted) | observed / threshold / unit |
|---|---|---|---|
| high_pressure | process | rise_ok, v > high_pressure_bar for ≥ pressure_persist_s (2) consecutive samples | v / 380 / bar |
| low_pressure | process | loaded_ok, v < low_pressure_bar for ≥ 2 consecutive | v / 150 / bar |
| rapid_rise | process | rise_ok, slope_5 > rapid_rise_bar_s for ≥ 2 consecutive | slope_5 / 4.0 / bar/s |
| rapid_drop | process | loaded_ok, slope_30 < rapid_drop_bar_s (−0.5) for ≥ 2 consecutive | slope_30 / −0.5 / bar/s |
| leg_imbalance | process | loaded_ok, abs(imbalance) > imbalance_frac for ≥ imbalance_persist_s | imbalance / 0.12 / fraction |
| persistent_deviation | process | loaded_ok, abs(dev_own) > deviation_bar for ≥ deviation_persist_s | dev_own / 20 / bar |
| sensor_stuck | sensor | flat_run ≥ stuck_run − 1 (i.e. stuck_run identical values) | flat_run+1 / 6 / samples |
| sensor_spike | sensor | spike | jump / 60 / bar |
| data_missing | sensor | missing_run ≥ missing_persist_s, unless the whole gateway is missing | missing_run / 3 / s |
| telemetry_loss | telemetry | per shield: every channel of its gateway has missing_run ≥ 3; channel_idx None | missing_run / 3 / s |
A trigger is emitted on every sample while its condition (including persistence) holds.
Baseline detectors (also in rules.py, used by evaluation and comparisons):
- `fixed_threshold_triggers(ff)`: only high_pressure and low_pressure (same persistence), category process.
- `RollingMAD`: per channel robust z = (v − median(last 60 valid, excluding v)) / (1.4826·MAD + 0.5); trigger `mad_anomaly` (process) when loaded_ok and |z| > mad_z for ≥ mad_persist_s consecutive samples. Needs ≥ 30 valid samples in the window.

## 6. AI model (`mineshield/model.py`)
`AnomalyModel(cfg)`: StandardScaler + IsolationForest(n_estimators, max_samples, random_state) on the 6 `cfg.model.features` columns.
- `fit(X)`: fit scaler and forest (training rows only).
- `calibrate(X_val)`: `raw = -forest.score_samples(scaled)`; `self.median = median(raw)`, `self.threshold = quantile(raw, calibration_quantile)`.
- `score(X) -> (n,)`: `max(0, (raw − median) / (threshold − median))`. 0 = median of validation normal data, **1.0 = calibrated alert threshold**.
- `save(path)` / `load(path)` via pickle with a `version` string "if-0.1.0".
- `build_training_matrix(cfg, seeds)`: for each seed run a **normal-only** FaceSimulator (default params) for `episode_s`, through FeatureEngine, collect rows where `scoreable`; returns X (NaN imbalance/neighbor ⇒ 0).
- `train_default_model(cfg) -> AnomalyModel`: train seeds ⇒ fit (random subsample to max_train_rows using rng seed 0), validation seeds ⇒ calibrate. Test seeds are never used.
- `AITrigger` (persistence): per channel, `ai_anomaly` (process) when score ≥ 1.0 for ≥ persist_s consecutive scored samples; observed = score, threshold 1.0, unit "score".

## 7. Alerts (`mineshield/alerts.py`)
`AlertCorrelator(sim_meta, cfg)` where sim_meta provides shield_ids, channel_ids, channel_shield, legs, gateway_of. `update(t, triggers, scores (C,), ff) -> list[str]` (ids of newly opened alerts). `reset()`.
1. **Sensor-first suppression** (`use_suppression=True` default): if a channel had a sensor trigger within the last sensor_suppress_s (10 s) drop its process triggers; if a shield has a telemetry trigger, drop all its other triggers.
2. Group surviving triggers by (category, shield). For each group find an **active** alert of the same category that already contains the shield, or (process only) contains a shield within `adjacent_merge` (1) of it ⇒ attach (adds shield, channels, rules). Telemetry: alerts are per gateway (all shields of the gateway). Otherwise open a new alert `AL-0001`, … with `opened_t = t`.
3. On attach/open: `last_t = t`; evidence keeps one entry per (rule, channel) with the most extreme observed value (max |observed|) and its t; track `ai_score_max` over the alert's channels using `scores` (NaN ignored) with `t_max` and a feature snapshot (all FEATURE_NAMES for the alert's channels) at t_max; if no AI score yet, snapshot at the first trigger.
4. An alert becomes inactive when `t − last_t > close_after_s` (checked every update).
5. `rule_weight`: high_pressure 1.0, telemetry_loss 0.7, rapid_rise 0.9, rapid_drop 0.8, low_pressure 0.8, leg_imbalance 0.6, persistent_deviation 0.6, mad_anomaly 0.5, ai_anomaly 0.5, sensor_stuck 0.4, data_missing 0.4, sensor_spike 0.3.
   `priority = round(50*max_rule_weight + 30*min(ai_score_max/2, 1) + 20*min((n_shields − 1)/2, 1))`; severity high ≥ 70, medium ≥ 45, else low. (ai_score_max NaN ⇒ 0.)
6. Title from highest-weight rule: high_pressure "High pressure (above illustrative limit)", low_pressure "Low pressure while loaded", rapid_rise "Rapid pressure rise", rapid_drop "Leak-like pressure decay", leg_imbalance "Leg pressure imbalance", persistent_deviation "Persistent deviation from shield baseline", ai_anomaly "AI-detected abnormal pressure pattern", mad_anomaly "Statistical outlier (rolling MAD)", sensor_stuck "Sensor stuck at constant value", sensor_spike "Isolated sensor spike", data_missing "Missing sensor data", telemetry_loss "Gateway telemetry loss". Process alert with ≥ 2 shields ⇒ prefix "Multi-shield: ".
7. `detail(alert_id, ...)` builds the AlertDetail fields: explanation (template sentences assembled from the actual evidence values, e.g. "Shield S05 (leg S05-L1) rose 5.2 bar/s over 5 s, above the configured 4.0 bar/s rule threshold."; one sentence per rule; AI sentence "The Isolation Forest score peaked at 2.3 (1.0 = calibrated threshold) at T+00:20:50."; sensor categories must say the pattern is more consistent with a measurement/communication fault than a change in roof load; multi-shield must name the adjacent shields), verification_steps (per rule, deduplicated, e.g. stuck ⇒ "Compare with the other leg channel of the same shield", "Inspect transducer cabling/connector and acquisition input", "Perform a field zero/span check"; process ⇒ "Verify shield hydraulic condition (leg seals, valves) and setting pressure", "Check neighbouring shields and face conditions with the deputy/shift engineer", "Confirm readings with a calibrated portable gauge before action"), uncertainty (fixed text: simulated data, illustrative thresholds, not a collapse probability, plus a data-quality clause), data_quality (`ok` if max missing_frac of alert channels at last_t < 0.05, `degraded` < 0.5, else `lost`, with note), priority_formula string as in the contract.

## 8. Engine (`mineshield/engine.py`)
`Pipeline` (shared by live and evaluation): FeatureEngine → RuleEngine → AnomalyModel.score on scoreable rows → AITrigger → AlertCorrelator. `process(values, states, time_in_state, t)`.
`LiveEngine(cfg, model)` wraps FaceSimulator + TelemetryValidator + DeviceMonitor + Pipeline, keeps history arrays for series/heatmap/export (t, values, quality, states, gt_label, gt_event_id, ai scores) for the whole run (cap 6 h). Methods: `reset(**params)`, `step(n)` (n steps: simulate → readings → validate → vector → pipeline), `inject(**kw)`, `state(shield, window_s)`, `alert_detail(id)`, `export_experiment()`, `export_csv()`. Guard all methods with a `threading.Lock`.

## 9. Evaluation (`mineshield/evaluation.py`, run `python -m mineshield.evaluation`)
- Test episodes: seeds `test_seeds_start + k`, k < episodes, each `episode_s` long with default sim params. Events per episode: 3, types assigned round-robin over the 9 EVENT_TYPES across the whole suite (event index e ⇒ type e % 9) so each type appears ≥ 5 times; shield uniform over the face (rng = default_rng(seed + 7)); magnitude = default × uniform(0.7, 1.3) (none for types without magnitude); duration = default; onset for event slot s: uniform in `[200 + s*520, 400 + s*520]`.
- Methods (all fed the same telemetry, all using AlertCorrelator for alert formation, with suppression only for the hybrid):
  1. `fixed_threshold` — fixed_threshold_triggers.
  2. `rolling_mad` — RollingMAD triggers.
  3. `isolation_forest` — AITrigger only.
  4. `mineshield_hybrid` — all rules + AITrigger + sensor-first suppression.
- Event-level matching: an alert matches an event if `onset_t ≤ opened_t ≤ end_t + match_grace_s` and they share ≥ 1 shield. Event detected ⇔ ≥ 1 matching alert; delay = earliest matching opened_t − onset_t. Alert is a false alert if it matches no event. precision = matched alerts / alerts; recall = detected events / events; f1; fp_per_hour = false alerts / (episodes × episode_s / 3600). Ground-truth labels are never inputs to any detector.
- by_scenario per method & type; category_confusion for the hybrid: truth category vs category of the earliest matching alert, or "missed".
- runtime_ms_per_step: wall time of (features + that method's detection + correlation) / total steps.
- Difficult cases (single 1200 s episodes, seed 4000+k, results per method as text "detected after N s" / "missed" / "N false alerts"):
  1. Noise only, no events (noise_bar 5.0) — includes shield advances as abrupt legitimate drops.
  2. Gradual rise that stays below the fixed threshold (gradual_rise, magnitude 30, on the shield with lowest k).
  3. Sensor spike that resembles a pressure jump (sensor_spike 150).
  4. Overlapping events (gradual_rise S05 at 300 s + leakage S06-L1 at 400 s).
  5. Missing data during an event (gradual_rise S05 at 300 s + sensor_dropout S05-L1 at 450 s, 60 s).
  6. Normal transient briefly above the fixed threshold (benign_transient magnitude 70, duration 60, on the shield with highest k).
- Save `data/eval_results.json` in the contract format with `generated_at`, setup, methods, by_scenario, category_confusion, difficult_cases, limitations (honest list).
