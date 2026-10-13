"""Unit tests for mineshield.rules (MODEL_SPEC section 5). Synthetic inputs only, no simulator."""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mineshield.features import FeatureEngine  # noqa: E402
from mineshield.rules import FixedThreshold, RollingMAD, RuleEngine, Trigger  # noqa: E402


def load_cfg():
    with open(ROOT / "config" / "default.json", encoding="utf-8") as fh:
        return json.load(fh)


def gw_of(i):
    return f"GW-{i // 6 + 1:02d}"


def synth_face(S=12, L=2, T=1800, noise=2.0, seed=0):
    """Clean synthetic face: shield cycle (advance 10 s, setting 120 s, loaded), first-order
    hydraulic lag, Gaussian noise, 0.1 bar quantisation. Returns values (T, C), states, tis."""
    rng = np.random.default_rng(seed)
    cycle, adv, setting = 600.0, 10.0, 120.0
    load = 330.0 * (1 + 0.06 * np.sin(2 * np.pi * (np.arange(S) + 0.5) / S))
    fac = 1 + rng.normal(0, 0.02, (S, L))
    offset = np.arange(S) * cycle / S
    vals = np.empty((T, S * L))
    states, tis_all = [], []
    p = None
    for k in range(T):
        c = (k - offset) % cycle
        st = np.where(c < adv, "advancing", np.where(c < adv + setting, "setting", "loaded"))
        tau = c - adv
        tis = np.where(c < adv, c, np.where(c < adv + setting, c - adv, c - adv - setting))
        target = 250.0 + np.maximum(0.0, load[:, None] * fac - 250.0) * (1 - np.exp(-tau[:, None] / 45.0))
        target = np.where((c < adv)[:, None], 15.0, target).reshape(-1)
        p = target if p is None else p + (target - p) * (1 - np.exp(-0.5))
        vals[k] = np.round((p + rng.normal(0, noise, S * L)) / 0.1) * 0.1
        states.append(st.tolist())
        tis_all.append(tis)
    return vals, states, tis_all


def run(seq, S, L, states=None, tis=None, engine_cls=RuleEngine, gw=gw_of, cfg=None):
    """Feed (T, C) values; default: all shields loaded for a long time. Returns triggers per step."""
    cfg = cfg or load_cfg()
    fe = FeatureEngine(S, L, cfg)
    eng = engine_cls(S * L, L, cfg)
    out = []
    for k in range(seq.shape[0]):
        st = states[k] if states is not None else ["loaded"] * S
        ti = tis[k] if tis is not None else np.full(S, 1000.0 + k)
        ff = fe.update(float(k), seq[k], st, ti)
        out.append(eng.evaluate(ff, gw))
    return out


def noisy(T, C, seed=1, sd=2.0, level=300.0):
    rng = np.random.default_rng(seed)
    return np.round((level + rng.normal(0.0, sd, (T, C))) / 0.1) * 0.1


def fired(trigs, rule, channel=None):
    """Sample indices at which ``rule`` fired (optionally on ``channel``)."""
    return [k for k, ts in enumerate(trigs)
            if any(t.rule == rule and (channel is None or t.channel_idx == channel) for t in ts)]


def first(trigs, rule, channel=None):
    ks = fired(trigs, rule, channel)
    return ks[0] if ks else None


# ---------------------------------------------------------------- clean data
def test_no_triggers_on_clean_synthetic_face():
    vals, states, tis = synth_face(T=1800, seed=3)
    trigs = run(vals, 12, 2, states, tis)
    assert sum(len(t) for t in trigs) == 0
    trigs = run(vals, 12, 2, states, tis, engine_cls=RollingMAD)
    assert sum(len(t) for t in trigs) == 0
    trigs = run(vals, 12, 2, states, tis, engine_cls=FixedThreshold)
    assert sum(len(t) for t in trigs) == 0


def test_features_plus_rules_performance():
    """Spec target: 1800 updates on 24 channels in < 1.5 s for features + rules.
    Best of two runs, so a transient load spike on the machine does not fail the test."""
    cfg = load_cfg()
    vals, states, tis = synth_face(T=1800, seed=4)
    times = []
    for _ in range(2):
        fe = FeatureEngine(12, 2, cfg)
        re = RuleEngine(24, 2, cfg)
        t0 = time.perf_counter()
        for k in range(1800):
            ff = fe.update(float(k), vals[k], states[k], tis[k])
            re.evaluate(ff, gw_of)
        times.append(time.perf_counter() - t0)
    assert min(times) < 1.5, f"features + rules took {min(times):.2f} s for 1800 updates"


# ------------------------------------------------------------ process rules
def test_high_pressure_persistence_and_gating():
    seq = noisy(80, 2)
    seq[40:44, 0] = [390.0, 370.0, 390.0, 390.0]
    trigs = run(seq, 1, 2)
    assert fired(trigs, "high_pressure") == [43]          # 2 consecutive samples needed, reset at 41
    tr = next(t for t in trigs[43] if t.rule == "high_pressure")
    assert (tr.category, tr.shield_idx, tr.channel_idx, tr.observed, tr.threshold, tr.unit) == \
        ("process", 0, 0, 390.0, 380.0, "bar")
    # not while advancing (rise_ok False); allowed in setting after min_setting_s
    adv = run(seq, 1, 2, states=[["advancing"]] * 80, tis=[np.array([5.0])] * 80)
    assert fired(adv, "high_pressure") == []
    sett = run(seq, 1, 2, states=[["setting"]] * 80, tis=[np.array([20.0 + k]) for k in range(80)])
    assert fired(sett, "high_pressure") == [43]


def test_low_pressure_only_when_loaded_ok():
    seq = noisy(80, 2)
    seq[50:53, 1] = 140.0
    trigs = run(seq, 1, 2)
    assert fired(trigs, "low_pressure", 1) == [51, 52]
    sett = run(seq, 1, 2, states=[["setting"]] * 80, tis=[np.array([20.0 + k]) for k in range(80)])
    assert fired(sett, "low_pressure") == []


def test_rapid_rise_fires_on_ramp_not_on_single_spike():
    seq = noisy(100, 2, sd=0.5)
    seq[50:70, 0] += 6.0 * np.arange(1, 21)
    seq[70:, 0] += 120.0
    trigs = run(seq, 1, 2)
    ks = fired(trigs, "rapid_rise", 0)
    assert ks and 52 <= ks[0] <= 56
    assert all(t.observed > 4.0 for k in ks for t in trigs[k] if t.rule == "rapid_rise")
    spike = noisy(100, 2, sd=0.5)
    spike[50, 0] += 150.0                                   # slope_5 = +30 bar/s for one sample only
    trigs = run(spike, 1, 2)
    assert fired(trigs, "rapid_rise") == []
    assert not any(t.category == "process" for ts in trigs for t in ts)
    assert fired(trigs, "sensor_spike", 0) == [51]


def test_rapid_drop_on_leak_like_decay():
    seq = noisy(200, 2, sd=1.0)
    seq[60:, 0] -= 1.0 * np.arange(140)                    # -1 bar/s
    trigs = run(seq, 1, 2)
    ks = fired(trigs, "rapid_drop", 0)
    assert ks and ks[0] < 100
    tr = next(t for t in trigs[150] if t.rule == "rapid_drop")
    assert tr.observed == pytest.approx(-1.0, abs=0.15) and tr.threshold == -0.5 and tr.unit == "bar/s"
    assert fired(trigs, "rapid_drop", 1) == []


def test_leg_imbalance_needs_persistence():
    seq = noisy(80, 2, sd=0.5)
    seq[40:, 0] *= 0.85
    seq[40:, 1] *= 1.15
    trigs = run(seq, 1, 2)
    assert first(trigs, "leg_imbalance", 0) == 49           # 10th consecutive sample
    assert first(trigs, "leg_imbalance", 1) == 49
    tr = next(t for t in trigs[60] if t.rule == "leg_imbalance" and t.channel_idx == 0)
    assert tr.observed == pytest.approx(-0.30, abs=0.01) and tr.unit == "fraction"
    # an interruption of the condition resets the counter
    seq2 = seq.copy()
    seq2[45] = [300.0, 300.0]
    trigs2 = run(seq2, 1, 2)
    assert first(trigs2, "leg_imbalance", 0) == 55


def test_persistent_deviation():
    seq = noisy(300, 2, sd=1.0)
    seq[200:, :] += 30.0
    trigs = run(seq, 1, 2, tis=[np.array([float(k)]) for k in range(300)])
    assert first(trigs, "persistent_deviation", 0) == 214     # 15th consecutive sample
    tr = next(t for t in trigs[220] if t.rule == "persistent_deviation" and t.channel_idx == 0)
    assert tr.observed == pytest.approx(30.0, abs=3.0) and tr.threshold == 20.0


# ------------------------------------------------------------- sensor rules
def test_sensor_stuck():
    seq = noisy(60, 4)
    seq[30:40, 2] = 301.3
    trigs = run(seq, 2, 2)
    assert fired(trigs, "sensor_stuck", 2) == [35, 36, 37, 38, 39]   # 6 identical values first at 35
    tr = next(t for t in trigs[35] if t.rule == "sensor_stuck")
    assert (tr.category, tr.observed, tr.threshold, tr.unit) == ("sensor", 6.0, 6.0, "samples")
    assert {t.rule for ts in trigs for t in ts} == {"sensor_stuck"}


def test_sensor_spike_observed_jump():
    seq = noisy(60, 2)
    seq[30, 1] -= 200.0
    trigs = run(seq, 1, 2)
    assert fired(trigs, "sensor_spike") == [31]
    tr = trigs[31][0]
    assert tr.observed == pytest.approx(-200.0, abs=8.0) and tr.threshold == 60.0 and tr.category == "sensor"


def test_data_missing_vs_telemetry_loss():
    S, L = 12, 2
    seq = noisy(40, S * L)
    seq[10:15, 3] = np.nan                       # one channel of GW-01
    seq[20:25, 0:12] = np.nan                    # every channel of GW-01 (shields 0..5)
    seq[30:35, 0:11] = np.nan                    # GW-01 except channel 11
    trigs = run(seq, S, L)
    assert fired(trigs, "data_missing", 3)[:3] == [12, 13, 14]
    assert all(t.category == "sensor" for t in trigs[12])
    # whole gateway: telemetry_loss per shield (channel None), no data_missing on those channels
    assert fired(trigs, "telemetry_loss") == [22, 23, 24]
    tl = [t for t in trigs[22] if t.rule == "telemetry_loss"]
    assert sorted(t.shield_idx for t in tl) == [0, 1, 2, 3, 4, 5]
    assert all(t.channel_idx is None and t.category == "telemetry" and t.observed == 3.0 for t in tl)
    assert not any(t.rule == "data_missing" for k in range(20, 26) for t in trigs[k])
    # one channel of the gateway still reporting: data_missing per channel, no telemetry_loss
    assert not any(t.rule == "telemetry_loss" for k in range(30, 40) for t in trigs[k])
    dm = {t.channel_idx for t in trigs[33] if t.rule == "data_missing"}
    assert dm == set(range(11))
    # other gateways never affected
    assert not any(t.shield_idx >= 6 for ts in trigs for t in ts)


def test_reset_clears_persistence():
    cfg = load_cfg()
    fe = FeatureEngine(1, 2, cfg)
    re = RuleEngine(2, 2, cfg)
    for k in range(40):
        re.evaluate(fe.update(k, [300.0, 300.0 + 0.1 * k], ["loaded"], [1000.0 + k]), gw_of)
    re.evaluate(fe.update(40, [390.0, 300.0], ["loaded"], [1040.0]), gw_of)
    re.reset()
    out = re.evaluate(fe.update(41, [390.0, 300.0], ["loaded"], [1041.0]), gw_of)
    assert not any(t.rule == "high_pressure" for t in out)
    out = re.evaluate(fe.update(42, [390.0, 300.0], ["loaded"], [1042.0]), gw_of)
    assert any(t.rule == "high_pressure" for t in out)


def test_trigger_to_dict():
    tr = Trigger("rapid_rise", "process", 4, 8, 12.0, 5.2, 4.0, "bar/s", "5-s pressure gradient")
    json.dumps(tr.to_dict())
    assert tr.to_dict()["channel_idx"] == 8


# --------------------------------------------------------------- baselines
def test_fixed_threshold_only_pressure_limits():
    seq = noisy(120, 2, sd=0.5)
    seq[40:60, 0] += 6.0 * np.arange(1, 21)      # rapid rise to ~420 bar
    seq[60:, 0] = 300.0
    seq[90:93, 1] = 140.0
    trigs = run(seq, 1, 2, engine_cls=FixedThreshold)
    rules = Counter(t.rule for ts in trigs for t in ts)
    assert set(rules) == {"high_pressure", "low_pressure"}
    assert fired(trigs, "low_pressure", 1) == [91, 92]
    hp = fired(trigs, "high_pressure", 0)
    over = [k for k in range(120) if seq[k, 0] > 380.0]
    assert hp == over[1:]                         # persistence 2
    assert all(t.category == "process" for ts in trigs for t in ts)
    cfg = load_cfg()
    ft = FixedThreshold(2, 2, cfg)
    fe = FeatureEngine(1, 2, cfg)
    ft.evaluate(fe.update(0, [390.0, 300.0], ["loaded"], [100.0]))
    ft.reset()
    assert ft.evaluate(fe.update(1, [390.0, 300.0], ["loaded"], [101.0])) == []


def test_rolling_mad():
    cfg = load_cfg()
    seq = noisy(200, 2, sd=2.0)
    seq[100:106, 0] += 40.0
    seq[10:16, 1] += 40.0                         # too early: < 30 valid samples in the window
    trigs = run(seq, 1, 2, engine_cls=RollingMAD, cfg=cfg)
    assert fired(trigs, "mad_anomaly", 0) == [102, 103, 104, 105]   # persistence 3
    assert fired(trigs, "mad_anomaly", 1) == []
    tr = next(t for t in trigs[102] if t.channel_idx == 0)
    assert tr.category == "process" and tr.threshold == 6.0 and tr.unit == "z" and tr.observed > 6.0
    # not when the shield is not loaded_ok
    sett = run(seq, 1, 2, states=[["setting"]] * 200, tis=[np.array([50.0])] * 200, engine_cls=RollingMAD)
    assert sum(len(t) for t in sett) == 0
    # missing data reduce the window; with < 30 valid samples nothing fires
    seq3 = seq.copy()
    seq3[40:100:2, 0] = np.nan
    seq3[70:100:2, 0] = np.nan
    seq3[71:100:2, 0] = np.nan
    trigs3 = run(seq3, 1, 2, engine_cls=RollingMAD)
    assert fired(trigs3, "mad_anomaly", 0) == []
