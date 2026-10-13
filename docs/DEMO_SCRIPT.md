# MineShield AI — jury demonstration (3–5 min)

Setup: run `python run.py` and open http://127.0.0.1:8000. Press **Reset** with seed 42, then set speed to **20x** and press **Start**. Everything below uses the dashboard only.

Timings come from a verified run at seed 42, quoted in simulated time:
- No alerts during the first 10 simulated minutes.
- A leak injected on S07-L2 alerted after 23 s.
- A stuck sensor on S03-L1 alerted after 4 s.
- A gateway interruption alerted after 2 s.

| Time | What to do | What to say |
|---|---|---|
| 0:00 | Show the face map, the S01 leg chart and the heatmap. | Twelve simulated shields, two leg channels each. Each shield advances, re-sets at 250 bar and builds load. The grey bands are advance and setting periods, where pressure drops are normal and not alarmed. Everything here is simulated; the chip says so. |
| 0:40 | Ask the jury to pick an event type from **Inject event**, e.g. *Leakage-like pressure drop* on S07, leg L2. Keep **Injected events (ground truth)** collapsed. | The detector does not know what was injected; ground truth is hidden. |
| 1:00 | Watch S07's tile turn red and an alert appear. Click the S07 tile to show its legs. | One leg decays while the other holds, which is the leg-imbalance and leak-like gradient pattern. |
| 1:30 | Open the alert. | Walk through the drawer: the explanation is built from the actual numbers; the rules that fired show observed values against thresholds; then the AI score (1.0 = calibrated threshold), feature values, neighbours, verification steps and the uncertainty note. Priority is a ranking score, not a collapse probability. Then press **Reveal ground truth**. |
| 2:30 | Inject *Sensor stuck at constant value* on S03 L1. | It becomes a purple **sensor** alert, not a process alert: identical readings are more consistent with a measurement fault than with roof load. Process rules are suppressed on that channel, so it does not raise a false roof alarm. |
| 3:00 | Optionally inject *Gateway telemetry interruption*. | The gateway shows disconnected and its six shields show no data. The alert says the pressures there are unobserved, not normal. |
| 3:20 | Open the **Evaluation** tab. | 16 held-out episodes and 48 events, never used for training or tuning. The fixed threshold catches 25 %, the Isolation Forest alone 50 %, the hybrid 100 % with 0.62 false alerts per hour and every category correct. Weak spots: gradual rises take about 100 s, and every number is synthetic. |
| 4:20 | Open the **Method & limits** tab. | F = P × A with illustrative limits. What is simulated, what is measured and what is not validated. The next steps are a certified hardware adapter, real mine data for calibration, and mine-specific threshold validation. |

**Contingency.** The evaluation results are saved in `data/eval_results.json` and load without re-running anything. Reset with seed 42 reproduces the same telemetry exactly. No internet is needed.

**Likely questions.**
- **"Is this predicting roof collapse?"**
  - No. It flags abnormal pressure patterns for investigation.
  - The priority is a ranking score, not a probability.
- **"Why not just a threshold?"**
  - On the test suite a 380 bar threshold catches 12 of 48 events.
  - It misses leaks, imbalance and every sensor or telemetry fault.
- **"Why not only AI?"**
  - The Isolation Forest alone misses all sensor and telemetry faults.
  - It barely reacts to an extreme value in a single feature.
  - The hybrid gets its recall from the rules and its context from the AI.
- **"How were thresholds chosen?"**
  - They are illustrative engineering values, set in `config/default.json`.
  - The AI threshold and persistence were chosen on separate tuning seeds (2100–2107), never on the test seeds.
- **"Would it work on a real mine?"**
  - That is unknown until it is calibrated on real shield-leg data from certified sensors.
  - The software already accepts the same telemetry schema from an external source; `/api/ingest/validate` checks it.
