"""Unit tests for mineshield.alerts (MODEL_SPEC section 7, API_CONTRACT AlertSummary/AlertDetail)."""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mineshield.alerts import (  # noqa: E402
    FAULT_SENTENCE, PRIORITY_FORMULA, TITLE_WINDOW_S, AlertCorrelator, compute_priority, fmt_t, severity_for,
)
from mineshield.features import FEATURE_INDEX, FEATURE_NAMES, FeatureEngine, FeatureFrame  # noqa: E402
from mineshield.rules import RULE_CATEGORY, RULE_DESCRIPTIONS, RuleEngine, Trigger  # noqa: E402

S, L = 12, 2
C = S * L
SUMMARY_KEYS = {"id", "opened_t", "last_t", "active", "category", "title", "shields", "channels",
                "severity", "priority", "rules", "ai_score_max"}
DETAIL_EXTRA_KEYS = {"explanation", "evidence", "features", "ai_detail", "data_quality", "neighbors",
                     "verification_steps", "uncertainty", "priority_formula", "series", "window",
                     "matched_event"}
EVIDENCE_KEYS = {"rule", "channel", "t", "observed", "threshold", "unit", "description"}


def load_cfg():
    with open(ROOT / "config" / "default.json", encoding="utf-8") as fh:
        return json.load(fh)


def make_meta():
    shield_ids = [f"S{i + 1:02d}" for i in range(S)]
    return SimpleNamespace(
        shield_ids=shield_ids,
        channel_ids=[f"{s}-L{j + 1}" for s in shield_ids for j in range(L)],
        channel_shield=np.arange(C) // L,
        legs=L,
        gateway_of=lambda i: f"GW-{i // 6 + 1:02d}",
    )


def make_ff(t, value=300.0, missing_frac=0.0, overrides=None):
    X = np.zeros((C, 8))
    X[:, FEATURE_INDEX["pressure_bar"]] = value
    X[:, FEATURE_INDEX["roll_std_bar"]] = 1.5
    X[:, FEATURE_INDEX["missing_frac"]] = missing_frac
    for (c, name), v in (overrides or {}).items():
        X[c, FEATURE_INDEX[name]] = v
    return FeatureFrame(
        t=float(t), X=X, values=X[:, 0].copy(), valid=np.isfinite(X[:, 0]), states=["loaded"] * S,
        time_in_state=np.full(S, 100.0), loaded_ok=np.ones(C, bool), rise_ok=np.ones(C, bool),
        flat_run=np.zeros(C, int), missing_run=np.zeros(C, int), spike=np.zeros(C, bool),
        scoreable=np.ones(C, bool), shield_mean=np.full(S, value), spike_jump=np.full(C, np.nan), legs=L)


def trig(rule, channel, t, observed, threshold=None, shield=None, unit="bar"):
    thr = {"rapid_rise": 4.0, "high_pressure": 380.0, "low_pressure": 150.0, "rapid_drop": -0.5,
           "leg_imbalance": 0.12, "persistent_deviation": 20.0, "sensor_stuck": 6.0, "sensor_spike": 60.0,
           "data_missing": 3.0, "telemetry_loss": 3.0, "ai_anomaly": 1.0}.get(rule, 1.0)
    return Trigger(rule=rule, category=RULE_CATEGORY[rule],
                   shield_idx=shield if shield is not None else channel // L,
                   channel_idx=channel, t=float(t), observed=float(observed),
                   threshold=float(threshold if threshold is not None else thr), unit=unit,
                   description=RULE_DESCRIPTIONS[rule])


def corr(**kw):
    return AlertCorrelator(make_meta(), load_cfg(), **kw)


def nan_scores():
    return np.full(C, np.nan)


# --------------------------------------------------------------- lifecycle
def test_open_attach_close_timing():
    ac = corr()
    assert ac.update(100.0, [trig("rapid_rise", 8, 100, 5.2, unit="bar/s")], nan_scores(), make_ff(100)) == ["AL-0001"]
    for t in range(101, 121):
        assert ac.update(float(t), [trig("rapid_rise", 8, t, 5.0, unit="bar/s")], nan_scores(), make_ff(t)) == []
    a = ac.get("AL-0001")
    assert (a.opened_t, a.last_t, a.active) == (100.0, 120.0, True)
    for t in range(121, 181):
        ac.update(float(t), [], nan_scores(), make_ff(t))
    assert a.active                          # 180 - 120 = 60, not > close_after_s
    ac.update(181.0, [], nan_scores(), make_ff(181))
    assert not a.active
    assert ac.update(200.0, [trig("rapid_rise", 8, 200, 4.5, unit="bar/s")], nan_scores(), make_ff(200)) == ["AL-0002"]
    assert [x.id for x in ac.alerts] == ["AL-0002", "AL-0001"]          # newest first
    assert [x.id for x in ac.active_alerts] == ["AL-0002"]
    assert ac.get("AL-9999") is None
    with pytest.raises(KeyError):
        ac.detail("AL-9999", 200.0)
    ac.reset()
    assert ac.alerts == []
    assert ac.update(1.0, [trig("rapid_rise", 8, 1, 5.0)], None, make_ff(1)) == ["AL-0001"]


def test_trigger_at_exactly_close_after_attaches():
    ac = corr()
    ac.update(100.0, [trig("high_pressure", 0, 100, 390)], None, make_ff(100))
    assert ac.update(160.0, [trig("high_pressure", 0, 160, 391)], None, make_ff(160)) == []
    assert ac.update(221.0, [trig("high_pressure", 0, 221, 392)], None, make_ff(221)) == ["AL-0002"]


def test_adjacent_shields_merge_with_multi_shield_title():
    ac = corr()
    ac.update(100.0, [trig("rapid_rise", 8, 100, 5.2, unit="bar/s")], None, make_ff(100))      # S05
    assert ac.update(101.0, [trig("rapid_rise", 10, 101, 4.6, unit="bar/s")], None, make_ff(101)) == []  # S06
    a = ac.get("AL-0001")
    s = a.summary()
    assert s["shields"] == ["S05", "S06"] and s["channels"] == ["S05-L1", "S06-L1"]
    assert s["title"] == "Multi-shield: Rapid pressure rise"
    # S08 is 2 shields away from S06 -> separate alert
    assert ac.update(102.0, [trig("rapid_rise", 14, 102, 4.4, unit="bar/s")], None, make_ff(102)) == ["AL-0002"]
    assert ac.get("AL-0002").summary()["title"] == "Rapid pressure rise"
    # two adjacent shields first triggering in the same sample form one alert
    ac2 = corr()
    opened = ac2.update(5.0, [trig("high_pressure", 22, 5, 390), trig("high_pressure", 20, 5, 385)],
                        None, make_ff(5))
    assert opened == ["AL-0001"] and ac2.get("AL-0001").summary()["shields"] == ["S11", "S12"]
    d = ac2.detail("AL-0001", 6.0)
    assert "S11" in d["explanation"] and "S12" in d["explanation"] and "adjacent" in d["explanation"]


def test_title_uses_rules_first_seen_within_title_window():
    assert TITLE_WINDOW_S == 15.0
    ac = corr()
    for t in range(100, 211):                         # leak-like decay keeps the alert active
        trs = [trig("rapid_drop", 10, t, -0.9, unit="bar/s")]
        if t >= 200:                                  # re-pressurisation 100 s after opening
            trs.append(trig("rapid_rise", 10, t, 5.5, unit="bar/s"))
        ac.update(float(t), trs, None, make_ff(t))
    s = ac.get("AL-0001").summary()
    assert s["title"] == "Leak-like pressure decay"
    assert s["rules"] == ["rapid_rise", "rapid_drop"]
    assert s["priority"] == 45                        # priority still uses the max weight of all rules
    # a higher-weight rule first seen within the window (inclusive) sets the title ...
    ac2 = corr()
    ac2.update(100.0, [trig("rapid_drop", 10, 100, -0.9, unit="bar/s")], None, make_ff(100))
    ac2.update(115.0, [trig("high_pressure", 10, 115, 390)], None, make_ff(115))
    assert ac2.get("AL-0001").summary()["title"] == "High pressure (above illustrative limit)"
    # ... one first seen after the window does not
    ac3 = corr()
    ac3.update(100.0, [trig("rapid_drop", 10, 100, -0.9, unit="bar/s")], None, make_ff(100))
    ac3.update(116.0, [trig("high_pressure", 10, 116, 390), trig("rapid_drop", 10, 116, -0.9, unit="bar/s")],
               None, make_ff(116))
    ac3.update(117.0, [trig("high_pressure", 12, 117, 391)], None, make_ff(117))
    assert ac3.get("AL-0001").summary()["title"] == "Multi-shield: Leak-like pressure decay"


def test_title_window_starts_at_first_engineering_rule():
    # AI opens the alert; high_pressure arrives 40 s later -> engineering title
    ac = corr()
    for t in range(100, 146):
        sc = nan_scores()
        sc[8] = 1.3
        trs = [trig("ai_anomaly", 8, t, 1.3, unit="score")]
        if t >= 140:
            trs.append(trig("high_pressure", 8, t, 392))
        ac.update(float(t), trs, sc, make_ff(t))
        if t == 139:
            assert ac.get("AL-0001").summary()["title"] == "AI-detected abnormal pressure pattern"
    assert ac.get("AL-0001").summary()["title"] == "High pressure (above illustrative limit)"
    # the window is anchored at that first engineering trigger: a later rapid_rise does not relabel
    ac2 = corr()
    for t in range(100, 211):
        trs = [trig("mad_anomaly", 10, t, 7.0, unit="z")]
        if t >= 140:
            trs.append(trig("rapid_drop", 10, t, -0.9, unit="bar/s"))
        if t >= 200:
            trs.append(trig("rapid_rise", 10, t, 5.0, unit="bar/s"))
        ac2.update(float(t), trs, None, make_ff(t))
    s2 = ac2.get("AL-0001").summary()
    assert s2["title"] == "Leak-like pressure decay" and s2["opened_t"] == 100.0
    assert s2["rules"] == ["rapid_rise", "rapid_drop", "mad_anomaly"]


def test_adjacent_alerts_growing_toward_each_other_merge():
    ac = corr()
    sc = nan_scores()
    sc[4] = 1.1
    assert ac.update(100.0, [trig("rapid_drop", 4, 100, -0.8, unit="bar/s")], sc, make_ff(100)) == ["AL-0001"]  # S03
    sc = nan_scores()
    sc[10] = 1.7
    assert ac.update(101.0, [trig("high_pressure", 10, 101, 392)], sc,
                     make_ff(101, overrides={(10, "dev_own_bar"): 44.0})) == ["AL-0002"]                    # S06
    ac.update(102.0, [trig("rapid_drop", 6, 102, -1.2, unit="bar/s")], None, make_ff(102))   # S04 -> AL-0001
    ac.update(103.0, [trig("high_pressure", 10, 103, 395)], None, make_ff(103))              # S06 -> AL-0002
    assert [a.id for a in ac.alerts] == ["AL-0002", "AL-0001"]
    # S05 is adjacent to both alerts: the newer AL-0002 is merged into the older AL-0001
    assert ac.update(104.0, [trig("rapid_drop", 8, 104, -0.9, unit="bar/s")], None, make_ff(104)) == []
    assert [a.id for a in ac.alerts] == ["AL-0001"] and [a.id for a in ac.active_alerts] == ["AL-0001"]
    a = ac.get("AL-0001")
    assert ac.get("AL-0002") is a and ac.aliases == {"AL-0002": "AL-0001"} and a.merged_ids == ["AL-0002"]
    s = a.summary()
    assert s["shields"] == ["S03", "S04", "S05", "S06"]
    assert s["channels"] == ["S03-L1", "S04-L1", "S05-L1", "S06-L1"]
    assert (s["opened_t"], s["last_t"]) == (100.0, 104.0)
    assert s["rules"] == ["high_pressure", "rapid_drop"]
    assert s["ai_score_max"] == 1.7                                    # the higher of the two
    assert s["title"] == "Multi-shield: High pressure (above illustrative limit)"
    assert s["priority"] == compute_priority(1.0, 1.7, 4)
    d = ac.detail("AL-0002", 105.0)                                    # alias resolves
    assert d["id"] == "AL-0001"
    assert d["ai_detail"] == {"score_max": 1.7, "t_max": 101.0, "threshold": 1.0}
    assert d["features"]["S06-L1"]["dev_own_bar"] == 44.0              # snapshot of the higher AI alert
    assert set(d["features"]) == {"S03-L1", "S04-L1", "S05-L1", "S06-L1"}
    ev = {(e["rule"], e["channel"]): e for e in d["evidence"]}
    assert ev[("high_pressure", "S06-L1")]["observed"] == 395.0 and ev[("rapid_drop", "S04-L1")]["observed"] == -1.2
    assert "S03, S04, S05 and S06" in d["explanation"]
    json.dumps(d, allow_nan=False)
    # the merged alert keeps growing
    ac.update(105.0, [trig("rapid_drop", 14, 105, -0.7, unit="bar/s")], None, make_ff(105))  # S08: 2 away
    ac.update(106.0, [trig("rapid_drop", 12, 106, -0.7, unit="bar/s")], None, make_ff(106))  # S07 bridges
    assert [x.id for x in ac.alerts] == ["AL-0001"]
    assert ac.get("AL-0003") is ac.get("AL-0001") and len(ac.get("AL-0001").shields) == 6
    # sensor alerts are never merged
    ac.update(107.0, [trig("sensor_stuck", 4, 107, 6, unit="samples"), trig("sensor_stuck", 6, 107, 6, unit="samples")],
              None, make_ff(107))
    assert sum(x.category == "sensor" for x in ac.alerts) == 2


def test_alert_opened_and_merged_in_same_update_is_not_returned():
    ac = corr()
    assert ac.update(100.0, [trig("rapid_rise", 14, 100, 5.0, unit="bar/s")], None, make_ff(100)) == ["AL-0001"]
    # S06 (2 away from S08) opens AL-0002, then S07 bridges both within the same sample
    opened = ac.update(101.0, [trig("rapid_rise", 10, 101, 4.5, unit="bar/s"),
                               trig("rapid_rise", 12, 101, 4.8, unit="bar/s")], None, make_ff(101))
    assert opened == []
    assert [a.id for a in ac.alerts] == ["AL-0001"]
    assert ac.get("AL-0001").summary()["shields"] == ["S06", "S07", "S08"]
    assert ac.get("AL-0002").id == "AL-0001"


def test_sensor_alerts_are_per_shield_and_not_merged():
    ac = corr()
    ac.update(50.0, [trig("sensor_stuck", 8, 50, 6, unit="samples"),
                     trig("sensor_stuck", 10, 50, 6, unit="samples")], None, make_ff(50))
    assert len(ac.alerts) == 2 and all(a.category == "sensor" for a in ac.alerts)
    # the other leg of the same shield joins the existing sensor alert
    ac.update(51.0, [trig("data_missing", 9, 51, 3, unit="s")], None, make_ff(51))
    a = next(x for x in ac.alerts if 4 in x.shields)
    assert a.summary()["channels"] == ["S05-L1", "S05-L2"]
    assert a.summary()["title"] == "Sensor stuck at constant value"       # tie 0.4: spec order


def test_telemetry_alert_per_gateway():
    ac = corr()
    trs = [trig("telemetry_loss", None, 30, 3, shield=s, unit="s") for s in range(6)]
    assert ac.update(30.0, trs, None, make_ff(30, missing_frac=0.1)) == ["AL-0001"]
    ac.update(31.0, [trig("telemetry_loss", None, 31, 4, shield=s, unit="s") for s in range(6)],
              None, make_ff(31, missing_frac=0.13))
    a = ac.get("AL-0001")
    s = a.summary()
    assert s["category"] == "telemetry" and s["title"] == "Gateway telemetry loss"
    assert s["shields"] == ["S01", "S02", "S03", "S04", "S05", "S06"] and len(s["channels"]) == 12
    assert s["priority"] == 55 and s["severity"] == "medium"      # 50*0.7 + 0 + 20
    d = ac.detail("AL-0001", 31.0)
    assert d["evidence"] == [{"rule": "telemetry_loss", "channel": None, "t": 31.0, "observed": 4.0,
                              "threshold": 3.0, "unit": "s", "description": RULE_DESCRIPTIONS["telemetry_loss"]}]
    assert "GW-01" in d["explanation"] and FAULT_SENTENCE in d["explanation"]
    assert d["data_quality"]["status"] == "degraded"


# ------------------------------------------------------------- suppression
def test_sensor_first_suppression():
    ac = corr()
    ac.update(100.0, [trig("sensor_spike", 8, 100, 150)], None, make_ff(100))
    assert len(ac.alerts) == 1 and ac.alerts[0].category == "sensor"
    ac.update(105.0, [trig("rapid_rise", 8, 105, 30, unit="bar/s")], None, make_ff(105))
    assert len(ac.alerts) == 1 and ac.suppressed == 1                 # dropped (within 10 s)
    ac.update(106.0, [trig("rapid_rise", 9, 106, 5, unit="bar/s")], None, make_ff(106))
    assert ac.alerts[0].category == "process"                         # other channel not suppressed
    assert ac.alerts[0].channels == {9}
    ac.update(110.0, [trig("rapid_rise", 8, 110, 30, unit="bar/s")], None, make_ff(110))
    assert ac.alerts[0].channels == {9} and ac.suppressed == 2         # 110 - 100 = 10 -> still dropped
    ac.update(111.0, [trig("rapid_rise", 8, 111, 30, unit="bar/s")], None, make_ff(111))
    assert ac.alerts[0].channels == {8, 9} and len(ac.alerts) == 2     # 11 s later -> accepted
    # same-sample sensor trigger suppresses the process trigger of that channel
    ac2 = corr()
    ac2.update(5.0, [trig("high_pressure", 3, 5, 400), trig("sensor_stuck", 3, 5, 6, unit="samples")],
               None, make_ff(5))
    assert [a.category for a in ac2.alerts] == ["sensor"]


def test_no_suppression_for_baselines():
    ac = corr(use_suppression=False)
    ac.update(100.0, [trig("sensor_spike", 8, 100, 150)], None, make_ff(100))
    ac.update(105.0, [trig("rapid_rise", 8, 105, 30, unit="bar/s")], None, make_ff(105))
    assert sorted(a.category for a in ac.alerts) == ["process", "sensor"] and ac.suppressed == 0


def test_telemetry_trigger_drops_other_triggers_of_the_shield():
    ac = corr()
    ac.update(40.0, [trig("telemetry_loss", None, 40, 3, shield=s, unit="s") for s in range(6)]
              + [trig("data_missing", 0, 40, 5, unit="s"), trig("rapid_drop", 2, 40, -1.0, unit="bar/s"),
                 trig("rapid_drop", 20, 40, -1.0, unit="bar/s")], None, make_ff(40))
    cats = sorted(a.category for a in ac.alerts)
    assert cats == ["process", "telemetry"]
    proc = next(a for a in ac.alerts if a.category == "process")
    assert proc.shields == {10}


# ----------------------------------------------------------- priority etc.
def test_priority_and_severity_formula():
    # priority = 50*rule_weight + 30*min(ai_score_max/2,1) + 20*min((n_shields-1)/2,1)
    assert PRIORITY_FORMULA == ("priority = 50*rule_weight + 30*min(ai_score_max/2,1) "
                                "+ 20*min((n_shields-1)/2,1)")
    assert compute_priority(1.0, 1.2, 2) == 78           # 50 + 18 + 10
    assert compute_priority(0.8, float("nan"), 1) == 40
    assert compute_priority(0.6, 3.6, 1) == 60           # AI term capped at 30
    assert compute_priority(0.6, 1.8, 1) == 57           # 30 + 27
    assert compute_priority(0.9, 0.8, 4) == 77           # 45 + 12 + 20
    assert compute_priority(1.0, 2.0, 3) == 100
    assert compute_priority(0.5, None, 1) == 25
    assert [severity_for(p) for p in (70, 69, 45, 44)] == ["high", "medium", "medium", "low"]

    ac = corr()
    sc = nan_scores()
    sc[0], sc[2] = 1.2, 0.4
    ac.update(10.0, [trig("high_pressure", 0, 10, 390), trig("high_pressure", 2, 10, 385)], sc, make_ff(10))
    s = ac.get("AL-0001").summary()
    assert s["priority"] == 78 and s["severity"] == "high" and s["ai_score_max"] == 1.2
    ac.update(11.0, [trig("rapid_drop", 20, 11, -0.9, unit="bar/s")], None, make_ff(11))
    s2 = ac.get("AL-0002").summary()
    assert s2["priority"] == 40 and s2["severity"] == "low" and s2["ai_score_max"] is None
    ac.update(12.0, [trig("leg_imbalance", 16, 12, 0.3, unit="fraction"),
                     trig("persistent_deviation", 16, 12, 25.0)], np.full(C, 2.0), make_ff(12))
    s3 = ac.get("AL-0003").summary()
    assert s3["priority"] == 60 and s3["severity"] == "medium"        # 30 + 30*min(2/2,1) + 0
    assert s3["title"] == "Leg pressure imbalance" and s3["rules"] == ["leg_imbalance", "persistent_deviation"]


def test_evidence_keeps_most_extreme_value_and_ai_tracking():
    ac = corr()
    ffs = {t: make_ff(t, overrides={(8, "slope_5_bar_s"): 0.1 * t}) for t in range(100, 106)}
    obs = {100: 4.5, 101: 5.6, 102: 4.9, 103: 4.2, 104: 5.1, 105: 4.4}
    ai = {100: np.nan, 101: 0.8, 102: 1.7, 103: 1.2, 104: np.nan, 105: 1.1}
    for t in range(100, 106):
        sc = nan_scores()
        sc[8] = ai[t]
        ac.update(float(t), [trig("rapid_rise", 8, t, obs[t], unit="bar/s")], sc, ffs[t])
    a = ac.get("AL-0001")
    ev = a.evidence[("rapid_rise", 8)]
    assert (ev["observed"], ev["t"]) == (5.6, 101.0)
    assert a.ai_score_max == 1.7 and a.ai_t_max == 102.0
    d = ac.detail("AL-0001", 106.0)
    assert d["ai_detail"] == {"score_max": 1.7, "t_max": 102.0, "threshold": 1.0}
    assert d["features"]["S05-L1"]["slope_5_bar_s"] == pytest.approx(10.2)   # snapshot at t_max
    assert set(d["features"]["S05-L1"]) == set(FEATURE_NAMES)


def test_feature_snapshot_at_first_trigger_without_ai_score():
    ac = corr()
    ac.update(50.0, [trig("low_pressure", 6, 50, 140)], None, make_ff(50, overrides={(6, "dev_own_bar"): -150.0}))
    ac.update(51.0, [trig("low_pressure", 6, 51, 138), trig("low_pressure", 7, 51, 139)], None,
              make_ff(51, overrides={(6, "dev_own_bar"): -170.0, (7, "dev_own_bar"): -160.0}))
    d = ac.detail("AL-0001", 52.0)
    assert d["features"]["S04-L1"]["dev_own_bar"] == -150.0       # first trigger
    assert d["features"]["S04-L2"]["dev_own_bar"] == -160.0       # channel added later
    assert d["ai_detail"]["score_max"] is None


# ----------------------------------------------------------------- detail
def _process_alert():
    ac = corr()
    sc = nan_scores()
    for t in (1300, 1301, 1302):
        sc[8] = {1300: 0.7, 1301: 2.3, 1302: 1.4}[t]
        ac.update(float(t), [trig("rapid_rise", 8, t, {1300: 4.4, 1301: 5.23, 1302: 4.8}[t], unit="bar/s"),
                             trig("persistent_deviation", 8, t, 32.1)], sc, make_ff(t))
    return ac


def test_summary_and_detail_match_contract():
    ac = _process_alert()
    s = ac.get("AL-0001").summary()
    assert set(s) == SUMMARY_KEYS
    events = [{"id": "EV-001", "type": "gradual_rise", "label": "Gradual pressure increase", "category": "process",
               "onset_t": 1200.0, "end_t": 1500.0, "shields": ["S05"], "channels": ["S05-L1", "S05-L2"],
               "magnitude": 70.0, "magnitude_unit": "bar"},
              {"id": "EV-002", "type": "leakage", "label": "Leakage-like pressure drop", "category": "process",
               "onset_t": 1250.0, "end_t": 1400.0, "shields": ["S09"], "channels": ["S09-L1"],
               "magnitude": 0.8, "magnitude_unit": "bar/s"}]
    calls = []

    def series_provider(alert, t0, t1):
        calls.append((alert.id, t0, t1))
        return {"t": [t0, t1], "channels": {"S05-L1": [300.0, np.nan]}, "ai_score": {"S05": [0.3, np.float64(2.3)]}}

    neighbors = [{"shield": "S04", "mean_bar": 318.0, "deviation_bar": 2.1, "status": "normal"}]
    d = ac.detail("AL-0001", 1310.0, series_provider=series_provider, events=events, neighbors=neighbors)
    assert set(d) == SUMMARY_KEYS | DETAIL_EXTRA_KEYS
    assert calls == [("AL-0001", 1180.0, 1310.0)] and d["window"] == [1180.0, 1310.0]
    assert d["series"]["channels"]["S05-L1"] == [300.0, None]
    assert d["neighbors"] == neighbors
    assert d["matched_event"]["id"] == "EV-001" and d["matched_event"]["active"] is True
    assert d["priority_formula"] == PRIORITY_FORMULA
    assert all(set(e) == EVIDENCE_KEYS for e in d["evidence"])
    assert [e["rule"] for e in d["evidence"]] == ["rapid_rise", "persistent_deviation"]
    assert d["data_quality"]["status"] == "ok"
    json.dumps(d, allow_nan=False)
    # defaults keep the contract shape
    d0 = ac.detail("AL-0001", 1310.0)
    assert d0["series"] == {"t": [], "channels": {}, "ai_score": {}} and d0["neighbors"] == []
    assert d0["matched_event"] is None
    json.dumps(d0, allow_nan=False)
    json.dumps(ac.summaries(), allow_nan=False)


def test_explanation_uses_observed_values():
    ac = _process_alert()
    d = ac.detail("AL-0001", 1310.0)
    text = d["explanation"]
    assert "Shield S05 (leg S05-L1) rose 5.2 bar/s over 5 s at T+00:21:41" in text
    assert "4.0 bar/s rule threshold" in text
    assert "+32.1 bar" in text and "20 bar" in text
    assert "The Isolation Forest score peaked at 2.30 (1.0 = calibrated threshold) at T+00:21:41" in text
    assert "T+00:21:40" in text and "T+00:21:42" in text
    assert "probab" not in text.lower()
    assert FAULT_SENTENCE not in text
    assert "not" in d["uncertainty"] and "probability of roof collapse" in d["uncertainty"]
    steps = d["verification_steps"]
    assert "Confirm readings with a calibrated portable gauge before action" in steps
    assert len(steps) == len(set(steps))


def test_sensor_explanation_and_steps():
    ac = corr()
    ff = make_ff(700, overrides={(8, "pressure_bar"): 312.4})
    ff.values[8] = 312.4
    sc = nan_scores()
    sc[8] = 0.4
    ac.update(700.0, [trig("sensor_stuck", 8, 700, 6, unit="samples")], sc, ff)
    ac.update(701.0, [trig("sensor_stuck", 8, 701, 7, unit="samples")], sc, ff)
    d = ac.detail("AL-0001", 702.0)
    assert FAULT_SENTENCE in d["explanation"]
    assert "Isolation Forest" not in d["explanation"]          # sub-threshold AI peak omitted for sensor alerts
    assert d["ai_score_max"] == 0.4
    assert "312.4 bar for 7 consecutive samples up to T+00:11:41" in d["explanation"]
    assert d["verification_steps"][:3] == ["Compare with the other leg channel of the same shield",
                                           "Inspect transducer cabling/connector and acquisition input",
                                           "Perform a field zero/span check"]
    assert d["severity"] == "low" and d["priority"] == 26          # 50*0.4 + 30*0.4/2


def test_ai_rule_sentence_lists_channels():
    ac = corr()
    for t in (900, 901):
        sc = nan_scores()
        sc[8], sc[9] = 1.25, 1.6 if t == 901 else 1.1
        ac.update(float(t), [trig("ai_anomaly", 8, t, sc[8], unit="score"),
                             trig("ai_anomaly", 9, t, sc[9], unit="score")], sc, make_ff(t))
    d = ac.detail("AL-0001", 902.0)
    assert d["title"] == "AI-detected abnormal pressure pattern" and d["priority"] == 49   # 25 + 30*1.6/2 + 0
    text = d["explanation"]
    n = load_cfg()["model"]["persist_s"]
    assert (f"at or above 1.0 (calibrated threshold) for at least {n} consecutive scored samples "
            "on S05-L1 and S05-L2") in text
    assert "peaked at 1.60 (1.0 = calibrated threshold) at T+00:15:01 on S05-L2." in text


def test_data_quality_levels():
    for mf, status in ((0.0, "ok"), (0.04, "ok"), (0.2, "degraded"), (0.5, "lost"), (0.9, "lost")):
        ac = corr()
        ac.update(10.0, [trig("data_missing", 4, 10, 12, unit="s")], None, make_ff(10, missing_frac=mf))
        dq = ac.detail("AL-0001", 10.0)["data_quality"]
        assert dq["status"] == status
        assert dq["note"]
        if status != "ok":
            assert f"{100 * mf:.0f}%" in dq["note"] and "S03-L1" in dq["note"]


def test_matched_event_with_event_objects():
    class Ev:
        def __init__(self, id_, onset, end, shields, hidden=False):
            self.id, self.type, self.label, self.category = id_, "sensor_spike", "Isolated sensor spike", "sensor"
            self.onset_t, self.end_t, self.shields, self.channels = onset, end, shields, []
            self.magnitude, self.magnitude_unit, self.hidden = 150.0, "bar", hidden

    ac = corr()
    ac.update(101.0, [trig("sensor_spike", 8, 101, 150)], None, make_ff(101))
    evs = [Ev("EV-001", 100.0, 101.0, ["S05"]), Ev("EV-002", 20.0, 40.0, ["S05"]),
           Ev("EV-003", 100.0, 101.0, ["S05"], hidden=True)]
    m = ac.detail("AL-0001", 105.0, events=evs)["matched_event"]
    assert m["id"] == "EV-001" and m["active"] is False
    assert ac.detail("AL-0001", 105.0, events=[evs[1]])["matched_event"] is None


def test_fmt_t():
    assert fmt_t(1250.0) == "T+00:20:50"
    assert fmt_t(3 * 3600 + 61) == "T+03:01:01"
    assert fmt_t(None) == "T+--:--:--"


# ------------------------------------------------- integration (no simulator)
def test_spike_yields_sensor_alert_only():
    """A single-sample sensor spike must not look like a process event end-to-end."""
    cfg = load_cfg()
    rng = np.random.default_rng(11)
    fe, re_ = FeatureEngine(S, L, cfg), RuleEngine(C, L, cfg)
    ac = AlertCorrelator(make_meta(), cfg)
    for k in range(300):
        v = np.round((320.0 + rng.normal(0, 2.0, C)) / 0.1) * 0.1
        if k == 200:
            v[8] += 150.0
        ff = fe.update(float(k), v, ["loaded"] * S, np.full(S, 1000.0 + k))
        ac.update(float(k), re_.evaluate(ff, make_meta().gateway_of), None, ff)
    assert [a.summary()["title"] for a in ac.alerts] == ["Isolated sensor spike"]
    d = ac.detail(ac.alerts[0].id, 300.0)
    assert re.search(r"jumped \+1[45]\d\.\d bar", d["explanation"])
    assert FAULT_SENTENCE in d["explanation"]
