# MineShield AI — longwall shield pressure monitoring (research prototype, MVP)

Locally runnable prototype for the project *"Indigenous Development of IoT-Enabled Technology for Monitoring, Analysis, and Interpretation of Longwall Shield Pressures for Improving Safety and Productivity."*

A simulated longwall face (12 shields × 2 hydraulic legs by default) streams leg-pressure telemetry through a validated ingestion path into a layered detector: engineering rules, an Isolation Forest, and alert correlation. Every alert carries an explanation traced to the actual readings, rules and AI score. A reproducible held-out evaluation compares the detector against fixed-threshold and statistical baselines.

> This prototype uses simulated telemetry and simplified engineering models. It is not a certified mining safety system, does not establish the probability of roof collapse, and must not be used to authorize mining operations or replace qualified personnel and approved safety procedures.

## Run

Requires Python 3.11+ (developed on 3.13). Runs fully offline; there are no cloud services or paid APIs.

```bash
pip install -r requirements.txt
```

```bash
python run.py
```

Open http://127.0.0.1:8000. The first start trains the anomaly model on normal-only synthetic data, which takes about 20 s; it is cached in `data/model.pkl` and retrained automatically if the model or simulator config changes.

Other commands:

```bash
python -m pytest tests -q
```

```bash
python -m mineshield.evaluation
```

```bash
python -m mineshield.tuning
```

- The tests are 191 unit and integration tests and take about 35 s.
- `mineshield.evaluation` runs the held-out evaluation (about 60 s) and writes `data/eval_results.json`.
- `mineshield.tuning` repeats the AI-threshold tuning sweep on the tuning seeds (about 5 min). Its last output is saved in `data/tuning_results.txt`.

**Clean restart:** stop the server, then run `python run.py` again. The live simulation always restarts from the default seed. You can also press **Reset** in the UI; the same seed reproduces the same telemetry.

## What the demo shows

1. **Live monitor**
   - Start/pause the simulated face and change speed, seed, shield count (6–24), legs (2/4), noise and roof-load trend.
   - Inject any of 9 events while it runs.
     - Process events: gradual rise, sudden spike, leak-like decay, leg imbalance, multi-shield loading.
     - Sensor events: spike, stuck sensor, dropout.
     - Telemetry event: gateway interruption.
   - The injected event flows through telemetry → face map → heatmap → shield chart → alert list.
2. **Alert investigation:** click an alert to see:
   - the explanation built from actual values;
   - the time-series evidence;
   - the rules that fired, each observed value against its threshold;
   - the AI score and feature values;
   - data quality, neighbouring shields and verification steps;
   - the injected ground truth, hidden behind a reveal button for blind demos.
3. **Evaluation:**
   - Precision, recall and F1 for four methods on held-out seeds.
   - False alerts per hour, detection delay and recall by scenario.
   - Category confusion and the difficult cases.
4. **Method & limits:** the engineering model (F = P × A), illustrative limits, what is simulated, what is measured and what is not validated.

## Architecture

```
FaceSimulator (roof load + setting/advance cycle + hydraulic lag + noise + injected events)
  → telemetry readings (schema: timestamp, seq, device, shield, channel, value, unit, quality)
  → TelemetryValidator (unit, ranges, duplicates, stale timestamps) + DeviceMonitor
  → FeatureEngine (deviation from own baseline, 5 s / robust 30 s gradients, leg imbalance,
                   neighbour deviation, robust spread, missing fraction, stuck/spike detection)
  → RuleEngine (high/low pressure, rapid rise/drop, imbalance, persistent deviation,
                sensor stuck/spike, missing data, gateway loss)
  + AnomalyModel (Isolation Forest, normal-only training, calibrated score: 1.0 = threshold)
  → AlertCorrelator (sensor-first suppression, multi-shield merging, explanation, priority 0–100)
  → FastAPI (/api/...) → vanilla-JS dashboard (frontend/, no build step, no external libraries)
```

| Path |Contents|
|---|---|
| `mineshield/` | Python package: `simulator`, `telemetry`, `features`, `rules`, `model`, `alerts`, `engine`, `api`, `evaluation`, `tuning` |
| `frontend/` | `index.html`, `styles.css`, `app.js`, `charts.js` (hand-written SVG charts), `api.js` |
| `config/default.json` | Every parameter and illustrative threshold |
| `docs/API_CONTRACT.md` | The HTTP/JSON contract |
| `docs/MODEL_SPEC.md` | Equations, features, rules and the evaluation protocol |
| `docs/DEMO_SCRIPT.md` | A 3–5 minute jury demonstration |

Only four runtime dependencies: numpy, scikit-learn, fastapi and uvicorn.

## Engineering model (illustrative, not equipment data)

**Shield cycle.** Each shield cycles through three states:
1. **Advancing:** legs lowered to about 15 bar, for 10 s.
2. **Setting:** pressure builds from the 250 bar setting pressure towards the roof-load level, with a 45 s time constant, for 120 s.
3. **Loaded.**

Shields advance one after another along the face.

**Roof load.**
- It is 320 bar-equivalent, scaled by a spatial factor of about ±6 % per shield and a leg factor of about ±2 %.
- An optional trend and rare normal transients are added.
- The 420 bar yield valve caps it.

**Hydraulic response.** A first-order lag with τ = 2 s.

**Measurement.** Gaussian noise (2 bar by default), 0.1 bar quantisation, and a 0–600 bar sensor range.

**Force.** F = P × A with a 0.32 m bore, so A = 0.0804 m² and 300 bar ≈ 2.41 MN per leg.

**Limits.** All limits are illustrative: high 380 bar, low 150 bar. Real shield forces require the actual leg geometry, the number of legs, the canopy arrangement and manufacturer data.

**Reproducibility.** Each seed uses separate random streams, so injecting an event never changes the noise on unaffected channels.

## Evaluation (synthetic, held-out)

**Protocol.**
- **Model training:** normal-only seeds 1000–1004.
- **Score calibration:** seeds 2000–2001.
- **AI threshold and persistence tuning:** separate episodes with events, seeds 2100–2107.
- **Testing:** seeds 3000–3015, never used for training or tuning. That is 16 episodes of 30 min with 48 events, 5–6 per type.
- **Matching:** an alert matches an event if it opens within [onset, end + 30 s] and shares at least one shield. Metrics are event-level.

**Results** (`data/eval_results.json`, generated by `python -m mineshield.evaluation`):

| Method | Precision | Recall | F1 | False alerts / h | Median delay |
|---|---|---|---|---|---|
| Fixed threshold (380 / 150 bar) | 100 % | 25.0 % | 0.400 | 0.00 | 54 s |
| Rolling MAD (robust z > 6) | 100 % | 27.1 % | 0.426 | 0.00 | 122 s |
| Isolation Forest only | 91.9 % | 50.0 % | 0.648 | 0.38 | 34 s |
| **MineShield hybrid (rules + AI)** | **91.5 %** | **100 %** | **0.956** | **0.62** | **4 s** |

**Category accuracy.** The hybrid assigns every detected event to the right category: 28/28 process, 15/15 sensor, 5/5 telemetry.

**Where it is weaker.**
- Gradual rises and leg imbalances are detected late (median about 100–115 s).
- Some false alerts remain, on the order of one per simulated 1.5 h.
- The Isolation Forest alone misses every sensor and telemetry fault, which is why the rules layer exists.

**Limitations.**
- All numbers come from this project's own simulator, and the detectors were designed knowing it. Real-data performance is unknown.
- The thresholds are illustrative.
- Each event type appears only 5–6 times, so per-type numbers are uncertain.
- The full list is in the Evaluation tab.

## Implemented vs. deferred

**Implemented:** simulator and event injection with ground truth, telemetry schema and validation, device status, rules, Isolation Forest, sensor-vs-process distinction, multi-shield correlation, evidence-linked explanations, the dashboard, exports (experiment JSON, labelled telemetry CSV), and the held-out evaluation.

**Deferred after the MVP:** short-horizon forecasting, MQTT or physical-sensor adapters, productivity and downtime analysis, and a 3-D twin.

**Before any real deployment**, all of the following are needed and none has been done:
- certified (Ex / IECEx / DGMS-approved) hardware;
- calibration against real mine data;
- mine-specific threshold validation;
- a safety assessment.
