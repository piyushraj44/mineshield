"""Tests for mineshield.engine: Pipeline heads, LiveEngine state/alerts/exports, model provider."""

import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine_helpers import (  # noqa: E402
    CSV_HEADER, DETAIL_KEYS, EVENT_KEYS, SHIELD_KEYS, STATE_KEYS, SUMMARY_KEYS, cfg as load_cfg, finite_or_none,
    small_model, strict_json,
)

from mineshield import engine as engine_mod  # noqa: E402
from mineshield.engine import (  # noqa: E402
    METHODS, LiveEngine, Pipeline, SimMeta, config_fingerprint, get_model, load_cached_model, model_is_current,
)
from mineshield.simulator import FaceSimulator  # noqa: E402


def run(eng: LiveEngine, n: int) -> None:
    while n > 0:
        k = min(60, n)
        eng.step(k)
        n -= k


def make_engine(**params) -> LiveEngine:
    eng = LiveEngine(load_cfg(), small_model())
    eng.reset(**params)
    return eng


def loaded_shield(state, min_tis=60, max_tis=330):
    for s in state["shields"]:
        if s["state"] == "loaded" and min_tis <= s["time_in_state_s"] <= max_tis:
            return s["id"]
    raise AssertionError("no suitably loaded shield")


# --------------------------------------------------------------------------- pipeline
def test_pipeline_runs_four_heads_on_one_feature_frame():
    c = load_cfg()
    sim = FaceSimulator(c, 7)
    sim.inject("pressure_spike", "S05", magnitude=150.0, onset_t=400.0)
    pipe = Pipeline(SimMeta.from_sim(sim), c, small_model(), list(METHODS))
    assert list(pipe.heads) == list(METHODS)
    assert pipe.correlator is pipe.heads["fixed_threshold"].correlator
    assert pipe.heads["mineshield_hybrid"].correlator.use_suppression is True
    assert all(not pipe.heads[m].correlator.use_suppression for m in METHODS[:3])
    assert len({id(h.correlator) for h in pipe.heads.values()}) == 4
    for _ in range(460):
        fr = sim.step()
        res = pipe.process(fr.values, fr.states, fr.time_in_state, fr.t)
        assert res.scores.shape == (sim.n_channels,)
        assert set(res.opened) == set(METHODS)
    assert pipe.steps == 460
    for m in METHODS:
        assert pipe.runtime_ms_per_step(m) > 0
    # the AI heads include the scoring time, the others do not
    assert pipe.time_s("isolation_forest") >= pipe.feature_time_s + pipe.score_time_s
    assert pipe.time_s("fixed_threshold") < pipe.feature_time_s + pipe.score_time_s + pipe.heads["fixed_threshold"].time_s + 1e-12
    # the hybrid saw the spike as a process alert on S05
    hyb = pipe.heads["mineshield_hybrid"].correlator.alerts
    assert any(a.category == "process" and 4 in a.shields and a.opened_t >= 400 for a in hyb)
    # non-AI heads carry no AI information
    assert all(not np.isfinite(a.ai_score_max) for a in pipe.heads["fixed_threshold"].correlator.alerts)


def test_pipeline_rejects_unknown_mode():
    c = load_cfg()
    sim = FaceSimulator(c, 1)
    with pytest.raises(ValueError):
        Pipeline(SimMeta.from_sim(sim), c, small_model(), "magic")


# --------------------------------------------------------------------------- reset / step
def test_reset_produces_first_sample_and_sim_info():
    eng = make_engine()
    st = eng.state("S05", 300)
    assert set(st) == STATE_KEYS
    sim = st["sim"]
    assert sim["t"] == 0.0 and sim["samples"] == 1
    assert sim["seed"] == 42 and sim["n_shields"] == 12 and sim["legs_per_shield"] == 2
    assert sim["data_source"] == "simulator" and sim["model"]["status"] == "ready"
    assert len(st["shields"]) == 12 and set(st["shields"][0]) == SHIELD_KEYS
    assert st["series"]["t"] == [0.0] and st["heatmap"]["t"] == [0.0]
    assert st["counts"]["channels_monitored"] == 24 and st["counts"]["readings_total"] == 24
    strict_json(st)


def test_reset_validation_keeps_previous_run():
    eng = make_engine(seed=3, n_shields=8)
    run(eng, 10)
    for bad in ({"n_shields": 30}, {"legs_per_shield": 3}, {"noise_bar": 100.0}, {"seed": -1},
                {"load_trend_bar_per_h": 1000}, {"bogus": 1}):
        with pytest.raises(ValueError):
            eng.reset(**bad)
    st = eng.state()
    assert st["sim"]["seed"] == 3 and st["sim"]["n_shields"] == 8 and st["sim"]["t"] == 10.0


def test_step_advances_time_and_rejects_bad_counts():
    eng = make_engine()
    out = eng.step(5)
    assert out["t"] == 5.0 and out["samples"] == 6
    st = eng.step_state(15, "S02", 300)
    assert st["sim"]["t"] == 20.0 and st["series"]["shield"] == "S02"
    for bad in (0, 61, 2.5, True, "3"):
        with pytest.raises(ValueError):
            eng.step(bad)


def test_same_seed_gives_identical_telemetry():
    a, b = make_engine(seed=11), make_engine(seed=11)
    run(a, 120)
    run(b, 120)
    sa, sb = a.state("S03", 120), b.state("S03", 120)
    assert sa["series"] == sb["series"] and sa["heatmap"] == sb["heatmap"]
    c = make_engine(seed=12)
    run(c, 120)
    assert c.state("S03", 120)["series"]["channels"] != sa["series"]["channels"]


def test_window_clamp_heatmap_and_fallback_shield():
    eng = make_engine()
    run(eng, 400)
    st = eng.state("S99", 10)                  # unknown shield, window below the minimum
    assert st["series"]["shield"] == "S01"
    assert len(st["series"]["t"]) == 60 and st["series"]["t"][-1] == 400.0
    assert len(st["series"]["channels"]["S01-L1"]) == 60 and len(st["series"]["ai_score"]) == 60
    st = eng.state("S04", 5000)                # above the maximum -> 900 (only 401 samples exist)
    assert len(st["series"]["t"]) == 401
    hm = eng.state("S04", 300)["heatmap"]
    assert hm["t"] == sorted(hm["t"]) and all(t % 5 == 0 for t in hm["t"])
    assert len(hm["t"]) == 60 and hm["t"][-1] == 400.0
    assert len(hm["values"][0]) == 12 and hm["shields"][0] == "S01"


def test_shield_rows_are_consistent():
    eng = make_engine()
    run(eng, 300)
    st = eng.state()
    loaded_means = [s["mean_bar"] for s in st["shields"] if s["state"] == "loaded" and s["mean_bar"] is not None]
    med = float(np.median(loaded_means))
    area = np.pi * 0.16 ** 2
    for s in st["shields"]:
        vals = [leg["value"] for leg in s["legs"] if leg["quality"] == "ok"]
        assert s["mean_bar"] == pytest.approx(np.mean(vals), abs=0.01)
        assert s["force_MN"] == pytest.approx(sum(vals) * 1e5 * area / 1e6, abs=1e-3)
        assert s["deviation_bar"] == pytest.approx(s["mean_bar"] - med, abs=0.02)
        assert finite_or_none(s["ai_score"])
        if s["state"] != "loaded":
            assert s["ai_score"] is None
        assert s["status"] in ("normal", "watch")


# --------------------------------------------------------------------------- events & alerts
def test_inject_validation():
    eng = make_engine()
    with pytest.raises(ValueError, match="unknown event type"):
        eng.inject("earthquake", "S01")
    with pytest.raises(ValueError, match="unknown event type"):
        eng.inject("benign_transient", "S01")            # hidden evaluation-only type
    with pytest.raises(ValueError, match="unknown shield"):
        eng.inject("gradual_rise", "S13")
    with pytest.raises(ValueError, match="magnitude must be omitted"):
        eng.inject("sensor_stuck", "S01", magnitude=5.0)
    with pytest.raises(ValueError, match="leg"):
        eng.inject("leakage", "S01", leg=3)
    with pytest.raises(ValueError, match="magnitude"):
        eng.inject("gradual_rise", "S01", magnitude=-5.0)
    with pytest.raises(ValueError, match="duration_s"):
        eng.inject("gradual_rise", "S01", duration_s=0)
    assert eng.state()["events"] == []


def test_inject_returns_event_record_starting_at_next_sample():
    eng = make_engine()
    run(eng, 20)
    ev = eng.inject("leakage", "S05", leg=2, magnitude=0.8, duration_s=150)
    assert set(ev) == EVENT_KEYS
    assert ev["id"] == "EV-001" and ev["onset_t"] == 21.0 and ev["end_t"] == 171.0
    assert ev["channels"] == ["S05-L2"] and ev["shields"] == ["S05"] and ev["active"] is True
    ev2 = eng.inject("multi_shield_loading", "S01", leg=7)  # leg ignored for block scope
    assert ev2["shields"] == ["S01", "S02", "S03"] and len(ev2["channels"]) == 6
    st = eng.state()
    assert [e["id"] for e in st["events"]] == ["EV-002", "EV-001"]    # newest first


def test_pressure_spike_opens_process_alert_with_full_detail():
    eng = make_engine()
    run(eng, 300)
    sid = loaded_shield(eng.state())
    ev = eng.inject("pressure_spike", sid, magnitude=150)
    run(eng, 25)
    st = eng.state(sid, 300)
    alerts = [a for a in st["alerts"] if sid in a["shields"] and a["category"] == "process"]
    assert alerts, st["alerts"]
    a = alerts[-1]                                       # earliest
    assert set(a) == SUMMARY_KEYS
    assert ev["onset_t"] <= a["opened_t"] <= ev["onset_t"] + 10
    assert next(s for s in st["shields"] if s["id"] == sid)["status"] == "process_alert"
    d = eng.alert_detail(a["id"])
    assert set(d) == DETAIL_KEYS
    strict_json(d)
    assert d["matched_event"]["id"] == ev["id"]
    t0, t1 = d["window"]
    assert t0 == max(0.0, a["opened_t"] - 120) and t1 <= st["sim"]["t"]
    assert d["series"]["t"][0] == t0 and d["series"]["t"][-1] == t1
    legs = eng.state()["sim"]["legs_per_shield"]
    assert set(d["series"]["channels"]) == {f"{s}-L{j}" for s in d["shields"] for j in range(1, legs + 1)}
    assert set(d["series"]["ai_score"]) == set(d["shields"])
    nb = {n["shield"] for n in d["neighbors"]}
    idx = [int(s[1:]) for s in d["shields"]]
    expected = {f"S{j:02d}" for i in idx for j in range(i - 2, i + 3) if 1 <= j <= 12} - set(d["shields"])
    assert nb == expected
    with pytest.raises(KeyError):
        eng.alert_detail("AL-9999")


def test_sensor_dropout_opens_sensor_alert():
    eng = make_engine()
    run(eng, 200)
    ev = eng.inject("sensor_dropout", "S03", leg=2)
    run(eng, 10)
    st = eng.state("S03", 300)
    sens = [a for a in st["alerts"] if a["category"] == "sensor" and "S03" in a["shields"]]
    assert sens and ev["onset_t"] <= sens[-1]["opened_t"] <= ev["onset_t"] + 5
    s3 = next(s for s in st["shields"] if s["id"] == "S03")
    assert s3["status"] == "sensor_fault"
    assert [leg["quality"] for leg in s3["legs"]] == ["ok", "missing"] and s3["legs"][1]["value"] is None
    assert st["series"]["channels"]["S03-L2"][-1] is None
    d = eng.alert_detail(sens[-1]["id"])
    assert d["matched_event"]["id"] == ev["id"]
    strict_json(d)


def test_telemetry_interruption_marks_gateway_and_shields():
    eng = make_engine()
    run(eng, 100)
    ev = eng.inject("telemetry_interruption", "S02")
    assert ev["shields"] == ["S01", "S02", "S03", "S04", "S05", "S06"]
    run(eng, 6)
    st = eng.state()
    gw = next(d for d in st["devices"] if d["id"] == "GW-01")
    assert gw["status"] == "stale"
    run(eng, 9)
    st = eng.state()
    gw = next(d for d in st["devices"] if d["id"] == "GW-01")
    assert gw["status"] == "disconnected" and gw["age_s"] > 10
    assert next(d for d in st["devices"] if d["id"] == "GW-02")["status"] == "simulated"
    for s in st["shields"][:6]:
        assert s["status"] == "no_data" and s["mean_bar"] is None and s["force_MN"] is None
    tel = [a for a in st["alerts"] if a["category"] == "telemetry"]
    assert tel and set(tel[0]["shields"]) == set(ev["shields"])
    assert st["counts"]["channels_ok"] == 12
    strict_json(st)


# --------------------------------------------------------------------------- exports
def test_csv_export_rows_and_ground_truth():
    eng = make_engine(n_shields=6)
    run(eng, 30)
    ev = eng.inject("sensor_stuck", "S02", leg=1, duration_s=5)
    run(eng, 10)
    lines = list(eng.iter_csv())
    assert lines[0].rstrip("\n") == CSV_HEADER
    rows = [ln.rstrip("\n").split(",") for ln in lines[1:]]
    assert len(rows) == 41 * 12 and all(len(r) == 12 for r in rows)
    assert rows[0][:4] == ["0.0", "2026-01-01T06:00:00Z", "0", "GW-01"]
    assert [r[2] for r in rows[:13]] == [str(i) for i in range(13)]
    gt = [r for r in rows if r[11] == ev["id"]]
    assert len(gt) == 5 and all(r[5] == "S02-L1" and r[10] == "sensor_stuck" for r in gt)
    assert {float(r[0]) for r in gt} == {31.0, 32.0, 33.0, 34.0, 35.0}
    assert all(r[7] == "bar" and r[8] in ("ok", "missing", "invalid") for r in rows)
    assert all(r[9] in ("advancing", "setting", "loaded") for r in rows)


def test_history_is_capped():
    eng = LiveEngine(load_cfg(), small_model(), history_s=100)
    eng.reset(n_shields=6)
    run(eng, 149)
    lines = list(eng.iter_csv())
    assert len(lines) == 1 + 100 * 12
    assert lines[1].startswith("50.0,")
    st = eng.state("S01", 300)
    assert st["series"]["t"][0] == 50.0 and len(st["series"]["t"]) == 100


def test_export_experiment():
    eng = make_engine()
    run(eng, 300)
    sid = loaded_shield(eng.state())
    eng.inject("pressure_spike", sid, magnitude=150)
    run(eng, 20)
    ex = eng.export_experiment()
    strict_json(ex)
    assert set(ex) == {"exported_at", "version", "sim", "events", "alerts"}
    assert ex["exported_at"].endswith("Z") and ex["sim"]["t"] == 320.0
    assert len(ex["events"]) == 1 and ex["alerts"]
    for a in ex["alerts"]:
        assert "series" not in a and set(a) == DETAIL_KEYS - {"series"}
    opened = [a["opened_t"] for a in ex["alerts"]]
    assert opened == sorted(opened)


# --------------------------------------------------------------------------- model provider
def test_get_model_trains_once_then_loads(tmp_path, monkeypatch, caplog):
    calls = []

    def fake_train(cfg):
        calls.append(1)
        return copy.deepcopy(small_model())

    monkeypatch.setattr(engine_mod, "train_default_model", fake_train)
    c = load_cfg()
    path = tmp_path / "sub" / "model.pkl"
    assert not model_is_current(c, path)
    with caplog.at_level(logging.INFO, logger="mineshield"):
        m1 = get_model(c, path)
    assert calls == [1] and path.is_file()
    assert any("Training" in r.getMessage() for r in caplog.records)
    assert m1.meta["config_fingerprint"] == config_fingerprint(c)
    assert model_is_current(c, path)
    m2 = get_model(c, path)
    assert calls == [1] and m2.ready and m2.threshold == pytest.approx(m1.threshold)
    # changing a model-relevant config section invalidates the cache
    c2 = load_cfg()
    c2["model"]["n_estimators"] = 151
    assert config_fingerprint(c2) != config_fingerprint(c)
    assert load_cached_model(c2, path) is None
    get_model(c2, path)
    assert calls == [1, 1]
    # a corrupt file is ignored and retrained
    path.write_bytes(b"not a pickle")
    get_model(c, path)
    assert calls == [1, 1, 1]
