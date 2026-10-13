"""Offline evaluation of the four detection methods on synthetic test episodes (MODEL_SPEC section 9).

Run with ``python -m mineshield.evaluation``: prints a compact results table and writes
``data/eval_results.json`` in the API_CONTRACT ``/api/evaluation`` format.

All four methods see exactly the same telemetry (simulator -> telemetry readings -> validator ->
value vector) and share one FeatureEngine per episode (``Pipeline`` with four heads). Ground-truth
labels and event lists are used only to score the alerts afterwards; they never reach a detector.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .engine import (
    DATA_DIR, HYBRID, METHODS, Pipeline, SimMeta, config_fingerprint, get_model, json_clean, utc_now_iso,
)
from .model import AnomalyModel
from .simulator import EVENT_TYPES, FaceSimulator
from .telemetry import TelemetryValidator, frame_to_readings, readings_to_vector

RESULTS_PATH = DATA_DIR / "eval_results.json"

METHOD_LABELS: dict[str, str] = {
    "fixed_threshold": "Fixed threshold (baseline)",
    "rolling_mad": "Rolling MAD (baseline)",
    "isolation_forest": "Isolation Forest only",
    "mineshield_hybrid": "MineShield hybrid (rules + AI)",
}
CATEGORIES: tuple[str, ...] = ("process", "sensor", "telemetry")

# Event slot geometry (MODEL_SPEC 9): onset of slot s uniform in [200 + 520 s, 400 + 520 s].
SLOT_FIRST_S = 200.0
SLOT_WIDTH_S = 200.0
SLOT_PERIOD_S = 520.0
LONGEST_DEFAULT_DURATION_S = max(float(e["default_duration_s"]) for e in EVENT_TYPES)
MAGNITUDE_JITTER = (0.7, 1.3)

DIFFICULT_SEED_START = 4000
DIFFICULT_EPISODE_S = 1200.0
DIFFICULT_EARLIEST_ONSET_S = 300.0
LOADED_SETTLE_S = 60.0   # difficult-case onsets: >= 60 s into a loaded phase (baselines established)

ProgressFn = Callable[[float, str], None]


# --------------------------------------------------------------------------- matching & metrics
def match_episode(alerts: Sequence[dict], events: Sequence[dict], grace_s: float) -> dict[str, Any]:
    """Event-level matching for one episode.

    An alert matches an event iff ``onset_t <= opened_t <= end_t + grace_s`` and they share >= 1
    shield (alert shields as recorded when the episode ends). An event is detected iff it has >= 1
    matching alert; its delay is the earliest matching ``opened_t - onset_t``. An alert that
    matches no event is a false alert.

    ``alerts``: dicts with ``opened_t``, ``shields`` (and optionally ``id``, ``category``).
    ``events``: dicts with ``onset_t``, ``end_t``, ``shields`` (and optionally ``id``, ``type``,
    ``category``). Returns ``{"events": [...], "alerts": [...]}`` with per-item outcomes.
    """
    a_sh = [set(a.get("shields") or []) for a in alerts]
    matched = [False] * len(alerts)
    ev_out = []
    for ev in events:
        on, end = float(ev["onset_t"]), float(ev["end_t"])
        sh = set(ev.get("shields") or [])
        best = None
        for j, a in enumerate(alerts):
            ot = float(a["opened_t"])
            if on <= ot <= end + grace_s and sh & a_sh[j]:
                matched[j] = True
                if best is None or ot < float(alerts[best]["opened_t"]):
                    best = j
        ev_out.append({
            "id": ev.get("id"), "type": ev.get("type"), "category": ev.get("category"),
            "detected": best is not None,
            "delay_s": None if best is None else float(alerts[best]["opened_t"]) - on,
            "alert_id": None if best is None else alerts[best].get("id"),
            "alert_category": None if best is None else alerts[best].get("category"),
        })
    al_out = [{"id": a.get("id"), "category": a.get("category"), "opened_t": float(a["opened_t"]),
               "matched": matched[j]} for j, a in enumerate(alerts)]
    return {"events": ev_out, "alerts": al_out}


def _r(x: float | None, nd: int = 4) -> float | None:
    return None if x is None or not math.isfinite(float(x)) else round(float(x), nd)


def summarize_matches(matches: Iterable[dict], hours: float) -> dict[str, Any]:
    """Aggregate per-episode ``match_episode`` outputs into the contract metrics.

    precision = matched alerts / alerts (None if there are no alerts), recall = detected events /
    events (None if there are no events), f1 = harmonic mean (0 if both are 0),
    fp_per_hour = false alerts / hours, median_delay_s over detected events (None if none).
    """
    events, alerts = [], []
    for m in matches:
        events.extend(m["events"])
        alerts.extend(m["alerts"])
    n_alerts = len(alerts)
    n_matched = sum(1 for a in alerts if a["matched"])
    n_events = len(events)
    tp = sum(1 for e in events if e["detected"])
    precision = n_matched / n_alerts if n_alerts else None
    recall = tp / n_events if n_events else None
    if precision is None or recall is None:
        f1 = None
    else:
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    delays = [e["delay_s"] for e in events if e["detected"]]
    return {
        "precision": _r(precision),
        "recall": _r(recall),
        "f1": _r(f1),
        "fp_per_hour": _r((n_alerts - n_matched) / hours, 3) if hours > 0 else None,
        "median_delay_s": _r(float(np.median(delays)), 1) if delays else None,
        "tp_events": tp,
        "fn_events": n_events - tp,
        "alerts_total": n_alerts,
        "matched_alerts": n_matched,
        "false_alerts": n_alerts - n_matched,
    }


def by_scenario(matches: Iterable[dict]) -> dict[str, dict[str, Any]]:
    """Per event type: events, detected, recall, median delay."""
    groups: dict[str, list[dict]] = {}
    for m in matches:
        for e in m["events"]:
            groups.setdefault(e["type"], []).append(e)
    out = {}
    for typ in [e["type"] for e in EVENT_TYPES] + sorted(set(groups) - {e["type"] for e in EVENT_TYPES}):
        evs = groups.get(typ)
        if not evs:
            continue
        det = [e for e in evs if e["detected"]]
        out[typ] = {"events": len(evs), "detected": len(det), "recall": _r(len(det) / len(evs)),
                    "median_delay_s": _r(float(np.median([e["delay_s"] for e in det])), 1) if det else None}
    return out


def category_confusion(matches: Iterable[dict], method: str = HYBRID) -> dict[str, Any]:
    """Truth category (rows) vs category of the earliest matching alert, or "missed"."""
    cols = list(CATEGORIES) + ["missed"]
    M = [[0] * len(cols) for _ in CATEGORIES]
    for m in matches:
        for e in m["events"]:
            if e["category"] not in CATEGORIES:
                continue
            r = CATEGORIES.index(e["category"])
            if e["detected"] and e["alert_category"] in CATEGORIES:
                c = cols.index(e["alert_category"])
            else:
                c = len(cols) - 1      # missed
            M[r][c] += 1
    return {"method": method, "labels": list(CATEGORIES), "columns": cols, "matrix": M}


# --------------------------------------------------------------------------- episodes
def _sim_kwargs(cfg: dict, **override: Any) -> dict[str, Any]:
    d = cfg["defaults"]
    kw = {k: d[k] for k in ("n_shields", "legs_per_shield", "noise_bar", "load_trend_bar_per_h")}
    kw.update(override)
    return kw


def n_event_slots(cfg: dict, episode_s: float) -> int:
    """Slots s < events_per_episode whose latest onset plus the longest default duration fits."""
    n = 0
    for s in range(int(cfg["evaluation"]["events_per_episode"])):
        if SLOT_FIRST_S + SLOT_WIDTH_S + s * SLOT_PERIOD_S + LONGEST_DEFAULT_DURATION_S <= episode_s:
            n += 1
    return n


def place_events(sim: FaceSimulator, seed: int, n_slots: int, counter: int) -> int:
    """Inject the test events of one episode; returns the updated global event counter.

    rng = default_rng(seed + 7); per slot, always in this order: shield (uniform over the face),
    leg (uniform, used only by channel-scoped types), onset (uniform in the slot, rounded to the
    1-s sample grid) and a magnitude factor uniform(0.7, 1.3) (applied only when the type has a
    magnitude). Type = EVENT_TYPES[counter % 9]; duration = type default.
    """
    rng = np.random.default_rng(seed + 7)
    S, L = sim.n_shields, sim.legs
    for s in range(n_slots):
        si = int(rng.integers(S))
        leg = int(rng.integers(L))
        lo = SLOT_FIRST_S + s * SLOT_PERIOD_S
        onset = float(np.round(rng.uniform(lo, lo + SLOT_WIDTH_S)))
        factor = float(rng.uniform(*MAGNITUDE_JITTER))
        spec = EVENT_TYPES[counter % len(EVENT_TYPES)]
        counter += 1
        mag = None if spec["default_magnitude"] is None else float(spec["default_magnitude"]) * factor
        sid = sim.shield_ids[si]
        ch = f"{sid}-L{leg + 1}" if spec["scope"] == "channel" else None
        sim.inject(spec["type"], sid, channel_id=ch, magnitude=mag,
                   duration_s=float(spec["default_duration_s"]), onset_t=onset)
    return counter


def run_episode(sim: FaceSimulator, cfg: dict, model: AnomalyModel | None, n_steps: int,
                modes: Sequence[str] = METHODS) -> Pipeline:
    """Feed ``n_steps`` samples through the telemetry path and a multi-head Pipeline.

    The pipeline only receives SimMeta (layout) and per-sample values, operating states and time
    in state; ``frame.gt_label`` / ``frame.gt_event_id`` / ``sim.events`` are never passed on.
    """
    meta = SimMeta.from_sim(sim)
    validator = TelemetryValidator(meta.channel_ids, cfg["limits"])
    pipe = Pipeline(meta, cfg, model, list(modes))
    for _ in range(int(n_steps)):
        fr = sim.step()
        accepted, _rej = validator.validate(frame_to_readings(fr, sim))
        values, _q = readings_to_vector(accepted, meta.channel_ids)
        pipe.process(values, fr.states, fr.time_in_state, fr.t)
    return pipe


def _alerts_of(pipe: Pipeline, mode: str) -> list[dict]:
    return [a.summary() for a in reversed(pipe.heads[mode].correlator.alerts)]


def _events_of(sim: FaceSimulator) -> list[dict]:
    return [e.to_dict(None) for e in sim.events]


# --------------------------------------------------------------------------- difficult cases
def _loaded_onset(sim: FaceSimulator, cfg: dict, si: int, earliest: float, duration: float) -> float:
    """First whole second >= earliest at which shield ``si`` has been loaded for LOADED_SETTLE_S
    and stays loaded for ``duration`` (falls back to ``earliest``)."""
    sc = cfg["simulator"]
    cyc = float(sc["cycle_s"])
    lo = float(sc["advance_s"]) + float(sc["setting_s"]) + LOADED_SETTLE_S
    hi = cyc - duration
    for t in range(int(earliest), int(earliest + cyc)):
        c = (t - float(sim.offset_s[si])) % cyc
        if lo <= c <= hi:
            return float(t)
    return float(earliest)


def _difficult_cases(cfg: dict) -> list[dict[str, Any]]:
    """Case definitions: name, sim overrides and an inject function returning the description."""
    t0 = DIFFICULT_EARLIEST_ONSET_S

    def noise(sim):
        return ("No injected events and sensor noise raised to 5 bar; shield advances (legs lowered to "
                "about 15 bar, then re-set) are legitimate abrupt drops. Every alert is a false alert.")

    def gradual_low(sim):
        si = int(np.argmin(sim.k))
        on = _loaded_onset(sim, cfg, si, t0, 300.0)
        sim.inject("gradual_rise", sim.shield_ids[si], magnitude=30.0, onset_t=on)
        return (f"gradual_rise of 30 bar over 300 s on {sim.shield_ids[si]} (lowest spatial load factor) at "
                f"t={on:.0f} s; the leg pressure stays well below the 380 bar fixed limit.")

    def spike(sim):
        si = 4
        on = _loaded_onset(sim, cfg, si, t0, 1.0)
        sim.inject("sensor_spike", sim.shield_ids[si], channel_id=f"{sim.shield_ids[si]}-L1",
                   magnitude=150.0, onset_t=on)
        return (f"A single-sample +150 bar sensor_spike on {sim.shield_ids[si]}-L1 at t={on:.0f} s while "
                "the shield is loaded; it resembles a sudden pressure jump but is a measurement fault.")

    def overlap(sim):
        sim.inject("gradual_rise", "S05", onset_t=300.0)
        sim.inject("leakage", "S06", channel_id="S06-L1", onset_t=400.0)
        return ("gradual_rise (70 bar, 300 s) on S05 at t=300 s overlapping a leakage (0.8 bar/s, 150 s) "
                "on the adjacent S06-L1 at t=400 s.")

    def missing(sim):
        sim.inject("gradual_rise", "S05", onset_t=300.0)
        sim.inject("sensor_dropout", "S05", channel_id="S05-L1", duration_s=60.0, onset_t=450.0)
        return ("gradual_rise (70 bar, 300 s) on S05 at t=300 s; S05-L1 drops out for 60 s at t=450 s "
                "in the middle of the rise.")

    def benign(sim):
        si = int(np.argmax(sim.k))
        on = _loaded_onset(sim, cfg, si, t0, 60.0)
        sim.inject("benign_transient", sim.shield_ids[si], magnitude=70.0, duration_s=60.0, onset_t=on)
        return (f"A legitimate (normal-labelled) load transient of +70 bar decaying with a 15 s time "
                f"constant on {sim.shield_ids[si]} (highest spatial load factor) at t={on:.0f} s, briefly "
                "above the 380 bar fixed limit. It is not a ground-truth event, so every alert is a false alert.")

    return [
        {"name": "Noise only, no events", "sim": {"noise_bar": 5.0}, "inject": noise},
        {"name": "Gradual rise below the fixed threshold", "sim": {}, "inject": gradual_low},
        {"name": "Sensor spike resembling a pressure jump", "sim": {}, "inject": spike},
        {"name": "Overlapping events", "sim": {}, "inject": overlap},
        {"name": "Missing data during an event", "sim": {}, "inject": missing},
        {"name": "Normal transient briefly above the fixed threshold", "sim": {}, "inject": benign},
    ]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def case_result_text(match: dict[str, Any]) -> str:
    """"detected after N s (category alert)" / "missed" per event (prefixed by the type when there
    are several), plus the number of false alerts ("no false alerts" when an episode has no
    ground-truth events)."""
    evs = match["events"]
    n_false = sum(1 for a in match["alerts"] if not a["matched"])
    fa = "no false alerts" if n_false == 0 else _plural(n_false, "false alert")
    if not evs:
        return fa
    parts = []
    for e in evs:
        s = (f"detected after {e['delay_s']:.0f} s ({e['alert_category']} alert)" if e["detected"]
             else "missed")
        if len(evs) > 1:
            s = f"{str(e['type']).replace('_', ' ')}: {s}"
        parts.append(s)
    if n_false:
        parts.append(fa)
    return "; ".join(parts)


# --------------------------------------------------------------------------- texts
def method_descriptions(cfg: dict) -> dict[str, str]:
    r, m, f = cfg["rules"], cfg["model"], cfg["features"]
    return {
        "fixed_threshold": (
            f"Raises a process alert when a leg stays above the illustrative {r['high_pressure_bar']:g} bar "
            f"limit (or below {r['low_pressure_bar']:g} bar while loaded) for {r['pressure_persist_s']:g} "
            "consecutive samples; it uses no other rule, no AI and no sensor-fault handling."),
        "rolling_mad": (
            f"Raises a process alert when a loaded leg's robust z-score against its rolling "
            f"{r['mad_window_s']:g}-s median (MAD scale) exceeds {r['mad_z']:g} for {r['mad_persist_s']:g} "
            "consecutive samples; it cannot distinguish sensor faults or data loss from pressure changes."),
        "isolation_forest": (
            f"Raises a process alert when the Isolation Forest score (trained on normal-only synthetic data) "
            f"of a loaded leg stays at or above the calibrated threshold 1.0 for {m['persist_s']:g} consecutive "
            "scored samples; samples that cannot be scored (missing data, shields not loaded for at least "
            f"{f['min_loaded_s']:g} s) never alert."),
        "mineshield_hybrid": (
            "Combines the ten engineering rules (process, sensor and telemetry) with the Isolation Forest "
            "trigger, groups triggers into alerts by category and adjacent shields, and drops process "
            "triggers on channels with a recent sensor or telemetry fault (sensor-first suppression)."),
    }


def limitations(cfg: dict, model: AnomalyModel | None) -> list[str]:
    mc, ec = cfg["model"], cfg["evaluation"]
    trained = (model.meta.get("trained_on") if model is not None and model.meta else None) \
        or f"normal-only synthetic seeds {mc['train_seeds']}"
    return [
        "All results come from synthetic telemetry produced by this project's own simulator; nothing has "
        "been validated against real longwall shield-leg pressure data.",
        "The simulator's physics (roof load, setting and yield behaviour, hydraulic lag, event shapes) is "
        "simplified and illustrative, and the detectors were designed with knowledge of the same simulator, "
        "which favours them compared with real data.",
        "All thresholds (380/150 bar limits, gradients, persistence, MAD z, AI threshold) are illustrative "
        "values, not equipment specifications; they were not tuned on the test seeds, but they have not been "
        "independently validated either.",
        f"The Isolation Forest was trained only on {trained} and calibrated so that 1.0 is the "
        f"{100 * float(mc['calibration_quantile']):g}th percentile of validation normal data; its score is a "
        "normalised anomaly score, not a probability.",
        f"Event-level metrics depend on the matching rule (alert opens within [onset, end + "
        f"{float(ec['match_grace_s']):g} s] and shares at least one shield): any overlapping alert counts as a "
        "detection even if it was raised for another reason, and alerts opening later than end + "
        f"{float(ec['match_grace_s']):g} s (e.g. on recovery transients) count as false alerts.",
        "Each event type appears only a handful of times in the suite, so per-type recall and delays have "
        "wide uncertainty; no confidence intervals are reported.",
        "No real sensor or network effects are modelled beyond injected faults: no latency, clock skew, "
        "re-ordering, duplicate delivery, calibration drift or temperature effects; telemetry arrives exactly "
        "once per second.",
        "Test events are placed in non-overlapping slots on one face configuration (default shield count, "
        "legs and noise); other configurations are only covered by the difficult cases.",
        "The baselines can only raise process alerts by design, so the category confusion is reported for the "
        "hybrid only.",
        "Runtime per step is wall time on the machine that ran the evaluation (features, AI scoring where used, "
        "detection and correlation; simulation and telemetry validation excluded) and is indicative only.",
    ]


# --------------------------------------------------------------------------- main entry
def run_evaluation(cfg: dict, model: AnomalyModel | None, episodes: int | None = None,
                   episode_s: float | None = None, include_difficult: bool = True,
                   progress: ProgressFn | None = None, difficult_s: float | None = None) -> dict[str, Any]:
    """Run the test suite (and the difficult cases) for all four methods; contract result dict."""
    t_start = time.perf_counter()
    ec, mc = cfg["evaluation"], cfg["model"]
    dt = float(cfg["simulator"]["dt_s"])
    n_ep = int(ec["episodes"] if episodes is None else episodes)
    ep_s = float(ec["episode_s"] if episode_s is None else episode_s)
    if n_ep < 1:
        raise ValueError("episodes must be >= 1")
    grace = float(ec["match_grace_s"])
    seeds = [int(ec["test_seeds_start"]) + k for k in range(n_ep)]
    n_slots = n_event_slots(cfg, ep_s)
    n_steps = int(round(ep_s / dt))
    cases = _difficult_cases(cfg) if include_difficult else []
    d_s = float(DIFFICULT_EPISODE_S if difficult_s is None else difficult_s)
    total_units = n_ep + len(cases)

    def report(done: int, msg: str) -> None:
        if progress is not None:
            progress(done / total_units if total_units else 1.0, msg)

    matches: dict[str, list[dict]] = {m: [] for m in METHODS}
    time_s = {m: 0.0 for m in METHODS}
    total_steps = 0
    counter = 0
    for k, seed in enumerate(seeds):
        report(k, f"episode {k + 1}/{n_ep} (seed {seed})")
        sim = FaceSimulator(cfg, seed, **_sim_kwargs(cfg))
        counter = place_events(sim, seed, n_slots, counter)
        pipe = run_episode(sim, cfg, model, n_steps)
        events = _events_of(sim)
        for m in METHODS:
            matches[m].append(match_episode(_alerts_of(pipe, m), events, grace))
            time_s[m] += pipe.time_s(m)
        total_steps += pipe.steps

    hours = n_ep * ep_s / 3600.0
    desc = method_descriptions(cfg)
    methods = []
    for m in METHODS:
        row = {"id": m, "label": METHOD_LABELS[m], "description": desc[m]}
        row.update(summarize_matches(matches[m], hours))
        row["runtime_ms_per_step"] = _r(1000.0 * time_s[m] / total_steps, 3) if total_steps else None
        methods.append(row)

    difficult = []
    for i, case in enumerate(cases):
        report(n_ep + i, f"difficult case {i + 1}/{len(cases)}: {case['name']}")
        seed = DIFFICULT_SEED_START + i
        sim = FaceSimulator(cfg, seed, **_sim_kwargs(cfg, **case["sim"]))
        description = case["inject"](sim)
        pipe = run_episode(sim, cfg, model, int(round(d_s / dt)))
        events = _events_of(sim)
        results = {m: case_result_text(match_episode(_alerts_of(pipe, m), events, grace)) for m in METHODS}
        difficult.append({"name": case["name"], "description": description, "seed": seed,
                          "duration_s": d_s, "results": results})
    report(total_units, "done")

    events_total = sum(len(mm["events"]) for mm in matches[HYBRID])
    out = {
        "available": True,
        "generated_at": utc_now_iso(),
        "setup": {
            "test_seeds": seeds,
            "episodes": n_ep,
            "episode_duration_s": ep_s,
            "events_per_episode": n_slots,
            "events_total": events_total,
            "train_seeds": [int(s) for s in mc["train_seeds"]],
            "validation_seeds": [int(s) for s in mc["validation_seeds"]],
            "matching_rule": (f"An alert matches an event if it opens within [onset, end + {grace:g} s] "
                              "and shares ≥1 shield."),
            "sim_params": _sim_kwargs(cfg),
            "difficult_case_seeds": [DIFFICULT_SEED_START + i for i in range(len(cases))],
        },
        "methods": methods,
        "by_scenario": {m: by_scenario(matches[m]) for m in METHODS},
        "category_confusion": category_confusion(matches[HYBRID], HYBRID),
        "difficult_cases": difficult,
        "limitations": limitations(cfg, model),
        "model": dict(model.meta) if model is not None else None,
        "config_fingerprint": config_fingerprint(cfg),
        "runtime_s": round(time.perf_counter() - t_start, 1),
    }
    return json_clean(out)


def save_results(results: dict, path: str | Path | None = None) -> Path:
    """Write the results JSON atomically (tmp file + replace)."""
    p = Path(path) if path is not None else RESULTS_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, allow_nan=False)
    os.replace(tmp, p)
    return p


def load_results(path: str | Path | None = None) -> dict | None:
    p = Path(path) if path is not None else RESULTS_PATH
    if not p.is_file():
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _fmt(x: Any, spec: str) -> str:
    return "-" if x is None else format(x, spec)


def format_table(res: dict) -> str:
    su = res["setup"]
    lines = [
        f"MineShield AI evaluation: {su['episodes']} episodes x {su['episode_duration_s']:g} s, "
        f"{su['events_total']} events, test seeds {su['test_seeds'][0]}-{su['test_seeds'][-1]}, "
        f"runtime {res.get('runtime_s', 0):.1f} s",
        f"{'method':<19}{'prec':>7}{'recall':>8}{'f1':>7}{'FA/h':>7}{'delay_s':>9}{'TP':>5}{'FN':>5}"
        f"{'alerts':>8}{'false':>7}{'ms/step':>9}",
    ]
    for m in res["methods"]:
        lines.append(
            f"{m['id']:<19}{_fmt(m['precision'], '.3f'):>7}{_fmt(m['recall'], '.3f'):>8}{_fmt(m['f1'], '.3f'):>7}"
            f"{_fmt(m['fp_per_hour'], '.2f'):>7}{_fmt(m['median_delay_s'], '.1f'):>9}{m['tp_events']:>5}"
            f"{m['fn_events']:>5}{m['alerts_total']:>8}{m['false_alerts']:>7}"
            f"{_fmt(m['runtime_ms_per_step'], '.3f'):>9}")
    ids = [m["id"] for m in res["methods"]]
    short = {"fixed_threshold": "fixed", "rolling_mad": "mad", "isolation_forest": "iforest",
             "mineshield_hybrid": "hybrid"}
    lines.append("")
    lines.append("Detected/events by scenario (median delay s):")
    lines.append(f"{'type':<24}" + "".join(f"{short.get(i, i):>16}" for i in ids))
    types = [e["type"] for e in EVENT_TYPES]
    for typ in types:
        cells = []
        for i in ids:
            c = res["by_scenario"].get(i, {}).get(typ)
            cells.append("-" if not c else f"{c['detected']}/{c['events']} ({_fmt(c['median_delay_s'], '.0f')})")
        if any(c != "-" for c in cells):
            lines.append(f"{typ:<24}" + "".join(f"{c:>16}" for c in cells))
    cc = res["category_confusion"]
    lines.append("")
    lines.append(f"Category confusion ({cc['method']}), rows = truth: " + " ".join(f"{c:>9}" for c in cc["columns"]))
    for lab, row in zip(cc["labels"], cc["matrix"]):
        lines.append(f"  {lab:<10}" + " ".join(f"{v:>9}" for v in row))
    if res.get("difficult_cases"):
        lines.append("")
        lines.append("Difficult cases:")
        for k, c in enumerate(res["difficult_cases"], 1):
            lines.append(f" {k}. {c['name']}")
            for i in ids:
                lines.append(f"      {i:<19}{c['results'].get(i, '-')}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    from .config import load_config

    ap = argparse.ArgumentParser(prog="python -m mineshield.evaluation",
                                 description="Evaluate the four detection methods on synthetic test episodes.")
    ap.add_argument("--episodes", type=int, default=None, help="number of test episodes (default: config)")
    ap.add_argument("--episode-s", type=float, default=None, help="episode length in s (default: config)")
    ap.add_argument("--no-difficult", action="store_true", help="skip the difficult cases")
    ap.add_argument("--out", default=None, help=f"results file (default: {RESULTS_PATH})")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    model = get_model(cfg)

    def progress(frac: float, msg: str) -> None:
        print(f"[{100 * frac:5.1f}%] {msg}", file=sys.stderr, flush=True)

    res = run_evaluation(cfg, model, episodes=args.episodes, episode_s=args.episode_s,
                         include_difficult=not args.no_difficult, progress=progress)
    path = save_results(res, args.out)
    print(format_table(res))
    print(f"\nResults written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
