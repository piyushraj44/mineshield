"""Tests for mineshield.evaluation (MODEL_SPEC section 9): matching/metrics, event placement and a
tiny end-to-end run of all four methods."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine_helpers import cfg as load_cfg, small_model, strict_json  # noqa: E402

from mineshield.engine import METHODS  # noqa: E402
from mineshield.evaluation import (  # noqa: E402
    by_scenario, case_result_text, category_confusion, match_episode, n_event_slots, place_events, run_episode,
    save_results, load_results, summarize_matches, _alerts_of,
)
from mineshield.simulator import EVENT_TYPES, FaceSimulator  # noqa: E402

METRIC_KEYS = {"id", "label", "description", "precision", "recall", "f1", "fp_per_hour", "median_delay_s",
               "tp_events", "fn_events", "alerts_total", "false_alerts", "runtime_ms_per_step"}


# --------------------------------------------------------------------------- matching & metrics
def hand_built():
    events = [
        {"id": "EV-001", "type": "gradual_rise", "category": "process", "onset_t": 100.0, "end_t": 200.0,
         "shields": ["S05"]},
        {"id": "EV-002", "type": "sensor_stuck", "category": "sensor", "onset_t": 500.0, "end_t": 560.0,
         "shields": ["S07"]},
        {"id": "EV-003", "type": "telemetry_interruption", "category": "telemetry", "onset_t": 900.0,
         "end_t": 945.0, "shields": ["S01", "S02", "S03", "S04", "S05", "S06"]},
    ]
    alerts = [
        {"id": "A1", "category": "process", "opened_t": 99.0, "shields": ["S05"]},          # before onset
        {"id": "A2", "category": "process", "opened_t": 130.0, "shields": ["S05", "S06"]},  # EV-001, 30 s
        {"id": "A3", "category": "process", "opened_t": 120.0, "shields": ["S04"]},         # no shared shield
        {"id": "A4", "category": "process", "opened_t": 230.0, "shields": ["S05"]},         # end + grace: match
        {"id": "A5", "category": "process", "opened_t": 231.0, "shields": ["S05"]},         # too late
        {"id": "A6", "category": "sensor", "opened_t": 505.0, "shields": ["S07"]},          # EV-002, 5 s
    ]
    return alerts, events


def test_match_episode_hand_built():
    alerts, events = hand_built()
    m = match_episode(alerts, events, grace_s=30.0)
    assert [e["detected"] for e in m["events"]] == [True, True, False]
    assert [e["delay_s"] for e in m["events"]] == [30.0, 5.0, None]
    assert [e["alert_id"] for e in m["events"]] == ["A2", "A6", None]
    assert [a["matched"] for a in m["alerts"]] == [False, True, False, True, False, True]
    # a zero grace turns A4 into a false alert
    m0 = match_episode(alerts, events, grace_s=0.0)
    assert [a["matched"] for a in m0["alerts"]] == [False, True, False, False, False, True]


def test_summarize_matches_metrics():
    alerts, events = hand_built()
    m = match_episode(alerts, events, grace_s=30.0)
    s = summarize_matches([m], hours=0.5)
    assert s["precision"] == 0.5
    assert s["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3), abs=1e-4)
    assert s["fp_per_hour"] == 6.0                      # 3 false alerts in 0.5 h
    assert s["median_delay_s"] == 17.5
    assert (s["tp_events"], s["fn_events"], s["alerts_total"], s["false_alerts"]) == (2, 1, 6, 3)
    assert s["tp_events"] + s["fn_events"] == len(events)
    # two episodes aggregate
    s2 = summarize_matches([m, m], hours=1.0)
    assert (s2["tp_events"], s2["alerts_total"], s2["fp_per_hour"]) == (4, 12, 6.0)
    # undefined ratios are None, not 0
    empty = summarize_matches([match_episode([], events, 30.0)], hours=1.0)
    assert empty["precision"] is None and empty["f1"] is None and empty["recall"] == 0.0
    no_ev = summarize_matches([match_episode(alerts, [], 30.0)], hours=1.0)
    assert no_ev["recall"] is None and no_ev["precision"] == 0.0 and no_ev["false_alerts"] == 6


def test_scenario_and_confusion_tables():
    alerts, events = hand_built()
    m = match_episode(alerts, events, grace_s=30.0)
    bs = by_scenario([m])
    assert bs["gradual_rise"] == {"events": 1, "detected": 1, "recall": 1.0, "median_delay_s": 30.0}
    assert bs["telemetry_interruption"]["detected"] == 0 and bs["telemetry_interruption"]["median_delay_s"] is None
    cc = category_confusion([m])
    assert cc["labels"] == ["process", "sensor", "telemetry"]
    assert cc["columns"] == ["process", "sensor", "telemetry", "missed"]
    assert cc["matrix"] == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]]


def test_case_result_text():
    alerts, events = hand_built()
    one = match_episode(alerts[:2], events[:1], 30.0)
    assert case_result_text(one) == "detected after 30 s (process alert); 1 false alert"
    assert case_result_text(match_episode([], events[:1], 30.0)) == "missed"
    assert case_result_text(match_episode(alerts, [], 30.0)) == "6 false alerts"
    assert case_result_text(match_episode([], [], 30.0)) == "no false alerts"
    two = case_result_text(match_episode(alerts[1:2], events[:2], 30.0))
    assert two == "gradual rise: detected after 30 s (process alert); sensor stuck: missed"


# --------------------------------------------------------------------------- event placement
def test_event_slots_and_placement():
    cfg = load_cfg()
    assert [n_event_slots(cfg, s) for s in (1800, 1220, 900, 600)] == [3, 2, 1, 0]
    sims = [FaceSimulator(cfg, 3000), FaceSimulator(cfg, 3001)]
    counter = 0
    for sim in sims:
        counter = place_events(sim, sim.seed, 3, counter)
    assert counter == 6
    evs = [e for sim in sims for e in sim.events]
    assert [e.type for e in evs] == [t["type"] for t in EVENT_TYPES[:6]]
    defaults = {t["type"]: t for t in EVENT_TYPES}
    for s, e in enumerate(evs):
        slot = s % 3
        assert 200 + slot * 520 <= e.onset_t <= 400 + slot * 520 and e.onset_t == int(e.onset_t)
        d = defaults[e.type]
        assert e.duration_s == d["default_duration_s"]
        if d["default_magnitude"] is not None:
            assert 0.7 * d["default_magnitude"] <= e.magnitude <= 1.3 * d["default_magnitude"]
    # deterministic
    again = FaceSimulator(cfg, 3000)
    place_events(again, 3000, 3, 0)
    assert [(e.type, e.onset_t, e.shields, e.channels, e.magnitude) for e in again.events] == \
        [(e.type, e.onset_t, e.shields, e.channels, e.magnitude) for e in sims[0].events]


class _Poison:
    """Raises on any use: stands in for ground-truth fields that detectors must never read."""

    def _boom(self, *a, **k):
        raise AssertionError("ground truth reached the detection path")

    __iter__ = __len__ = __getitem__ = __eq__ = __bool__ = __array__ = _boom

    def __getattr__(self, name):
        self._boom()


class _GTHidingSim(FaceSimulator):
    def step(self):
        fr = super().step()
        fr.gt_label = _Poison()
        fr.gt_event_id = _Poison()
        fr.true_values = _Poison()
        return fr


def test_ground_truth_never_reaches_detectors():
    cfg = load_cfg()
    plain, hidden = FaceSimulator(cfg, 3005), _GTHidingSim(cfg, 3005)
    for sim in (plain, hidden):
        place_events(sim, 3005, 1, 1)        # pressure_spike
    p1 = run_episode(plain, cfg, small_model(), 450)
    p2 = run_episode(hidden, cfg, small_model(), 450)
    for m in METHODS:
        assert _alerts_of(p1, m) == _alerts_of(p2, m)


# --------------------------------------------------------------------------- end-to-end (tiny suite)
@pytest.fixture(scope="module")
def tiny():
    return __import__("mineshield.evaluation", fromlist=["run_evaluation"]).run_evaluation(
        load_cfg(), small_model(), episodes=2, episode_s=1220, include_difficult=True, difficult_s=520)


def test_tiny_suite_contract_shape(tiny):
    strict_json(tiny)
    assert tiny["available"] is True and tiny["generated_at"].endswith("Z")
    su = tiny["setup"]
    assert su["test_seeds"] == [3000, 3001] and su["episodes"] == 2 and su["episode_duration_s"] == 1220
    assert su["events_total"] == 4 and su["train_seeds"] and su["validation_seeds"]
    assert "shares ≥1 shield" in su["matching_rule"]
    assert [m["id"] for m in tiny["methods"]] == list(METHODS)
    assert tiny["model"]["trained_on"] and len(tiny["config_fingerprint"]) == 16
    assert all(isinstance(x, str) and x for x in tiny["limitations"]) and len(tiny["limitations"]) >= 6


def test_tiny_suite_metrics_consistent(tiny):
    n_ev = tiny["setup"]["events_total"]
    for m in tiny["methods"]:
        assert METRIC_KEYS <= set(m)
        assert isinstance(m["description"], str) and m["description"].endswith(".")
        for k in ("precision", "recall", "f1"):
            assert m[k] is None or 0.0 <= m[k] <= 1.0, (m["id"], k, m[k])
        assert m["recall"] is not None
        assert (m["precision"] is None) == (m["alerts_total"] == 0)
        assert m["tp_events"] + m["fn_events"] == n_ev
        assert 0 <= m["false_alerts"] <= m["alerts_total"]
        assert m["fp_per_hour"] == pytest.approx(m["false_alerts"] / (2 * 1220 / 3600), abs=1e-3)
        assert m["runtime_ms_per_step"] > 0
        sc = tiny["by_scenario"][m["id"]]
        assert sum(v["events"] for v in sc.values()) == n_ev
        assert sum(v["detected"] for v in sc.values()) == m["tp_events"]
        assert list(sc) == [t["type"] for t in EVENT_TYPES[:4]]
    cc = tiny["category_confusion"]
    assert cc["method"] == "mineshield_hybrid" and sum(map(sum, cc["matrix"])) == n_ev
    hyb = next(m for m in tiny["methods"] if m["id"] == "mineshield_hybrid")
    assert sum(cc["matrix"][r][c] for r in range(3) for c in range(3)) == hyb["tp_events"]


def test_tiny_suite_difficult_cases(tiny):
    cases = tiny["difficult_cases"]
    assert len(cases) == 6
    for c in cases:
        assert c["name"] and c["description"] and set(c["results"]) == set(METHODS)
        for txt in c["results"].values():
            assert "detected after" in txt or "missed" in txt or "false alert" in txt
    for m in METHODS:   # no ground-truth events in the noise-only and benign-transient cases
        assert "false alert" in cases[0]["results"][m] and "false alert" in cases[5]["results"][m]
    assert cases[3]["results"]["mineshield_hybrid"].startswith("gradual rise: ")


def test_results_roundtrip(tiny, tmp_path):
    p = save_results(tiny, tmp_path / "eval_results.json")
    assert load_results(p) == tiny
    assert load_results(tmp_path / "missing.json") is None
