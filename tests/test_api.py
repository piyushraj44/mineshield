"""Tests for mineshield.api (docs/API_CONTRACT.md) using fastapi.testclient.TestClient."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine_helpers import (  # noqa: E402
    CSV_HEADER, DETAIL_KEYS, EVENT_KEYS, STATE_KEYS, cfg as load_cfg, small_model, strict_loads,
)

from mineshield.api import create_app  # noqa: E402
from mineshield.simulator import EVENT_TYPES  # noqa: E402


@pytest.fixture(scope="module")
def results_path(tmp_path_factory):
    return tmp_path_factory.mktemp("eval") / "eval_results.json"


@pytest.fixture(scope="module")
def client(results_path):
    app = create_app(load_cfg(), model=small_model(), results_path=results_path,
                     evaluation_options={"episodes": 1, "episode_s": 720, "include_difficult": False})
    with TestClient(app) as c:
        yield c


def jget(client, url, **kw):
    r = client.get(url, **kw)
    assert r.status_code == 200, r.text
    return strict_loads(r.text)


def jpost(client, url, body=None, status=200):
    r = client.post(url, json=body)
    assert r.status_code == status, r.text
    return strict_loads(r.text)


def reset(client, **body):
    return jpost(client, "/api/sim/reset", body)


def step(client, steps, shield="S01", window_s=300):
    return jpost(client, "/api/sim/step", {"steps": steps, "shield": shield, "window_s": window_s})


def advance(client, n, shield="S01", window_s=300):
    st = None
    while n > 0:
        k = min(60, n)
        st = step(client, k, shield, window_s)
        n -= k
    return st


def assert_422(r):
    assert r.status_code == 422, r.text
    d = r.json()["detail"]
    assert isinstance(d, str) and d
    return d


# --------------------------------------------------------------------------- config / reset / step
def test_config_matches_contract(client):
    c = jget(client, "/api/config")
    assert set(c) == {"version", "units", "limits", "piston_area_m2", "defaults", "ranges", "speeds", "poll_ms",
                      "event_types", "disclaimer"}
    assert c["piston_area_m2"] == 0.0804
    assert c["units"] == {"pressure": "bar", "force": "MN", "time": "s"}
    assert c["defaults"] == {"seed": 42, "n_shields": 12, "legs_per_shield": 2, "noise_bar": 2.0,
                             "load_trend_bar_per_h": 0}
    assert c["event_types"] == EVENT_TYPES and len(c["event_types"]) == 9
    assert [s["steps"] for s in c["speeds"]] == [1, 2, 5, 10, 15] and c["poll_ms"] == 500


def test_startup_state_has_first_sample(client):
    st = jget(client, "/api/state")
    assert set(st) == STATE_KEYS
    assert st["sim"]["samples"] >= 1 and st["sim"]["data_source"] == "simulator"


def test_reset_reproducibility_and_step(client):
    r = reset(client, seed=7, n_shields=10, legs_per_shield=4, noise_bar=3.0, load_trend_bar_per_h=30)
    assert r["ok"] is True
    sim = r["sim"]
    assert (sim["t"], sim["samples"], sim["seed"], sim["n_shields"], sim["legs_per_shield"]) == (0.0, 1, 7, 10, 4)
    st1 = step(client, 30, "S04")
    assert st1["sim"]["t"] == 30.0 and st1["sim"]["samples"] == 31
    assert len(st1["shields"]) == 10 and len(st1["shields"][0]["legs"]) == 4
    assert set(st1["series"]["channels"]) == {f"S04-L{j}" for j in range(1, 5)}
    reset(client, seed=7, n_shields=10, legs_per_shield=4, noise_bar=3.0, load_trend_bar_per_h=30)
    st2 = step(client, 30, "S04")
    assert st2["series"] == st1["series"] and st2["heatmap"] == st1["heatmap"]
    assert st2["shields"] == st1["shields"]
    reset(client, seed=8, n_shields=10, legs_per_shield=4, noise_bar=3.0, load_trend_bar_per_h=30)
    st3 = step(client, 30, "S04")
    assert st3["series"]["channels"] != st1["series"]["channels"]
    # GET /api/state does not advance
    assert jget(client, "/api/state", params={"shield": "S04", "window_s": 300})["sim"]["t"] == 30.0
    # defaults when the body is empty
    assert reset(client)["sim"]["seed"] == 42


def test_reset_rejects_bad_input_with_string_detail(client):
    for body in ({"n_shields": 99}, {"legs_per_shield": 3}, {"noise_bar": 0.1}, {"seed": -4},
                 {"load_trend_bar_per_h": 500}, {"seed": "abc"}, {"n_shields": 12.5}, {"color": "red"}):
        d = assert_422(client.post("/api/sim/reset", json=body))
        assert d
    d = assert_422(client.post("/api/sim/reset", json={"n_shields": 99}))
    assert "n_shields" in d
    r = client.post("/api/sim/reset", content=b"{not json", headers={"Content-Type": "application/json"})
    assert_422(r)


def test_step_rejects_bad_steps(client):
    reset(client)
    for steps in (0, 61, "many"):
        assert_422(client.post("/api/sim/step", json={"steps": steps, "shield": "S01", "window_s": 300}))
    st = step(client, 1, "S77", 10)               # unknown shield and small window are tolerated
    assert st["series"]["shield"] == "S01" and st["sim"]["t"] == 1.0


# --------------------------------------------------------------------------- inject / alerts
def test_inject_rejects_bad_input_with_string_detail(client):
    reset(client)
    bad = [
        {"type": "volcano", "shield": "S01"},
        {"type": "gradual_rise", "shield": "S40"},
        {"type": "sensor_stuck", "shield": "S01", "magnitude": 3},
        {"type": "leakage", "shield": "S01", "leg": 5},
        {"type": "gradual_rise", "shield": "S01", "duration_s": 0},
        {"type": "gradual_rise", "shield": "S01", "duration_s": 4000},
        {"type": "gradual_rise", "shield": "S01", "magnitude": 0},
        {"type": "gradual_rise"},
        {"type": "benign_transient", "shield": "S01"},
    ]
    for body in bad:
        assert_422(client.post("/api/sim/inject", json=body))
    assert jget(client, "/api/state")["events"] == []


def test_inject_spike_produces_process_alert_and_detail(client):
    reset(client)
    st = advance(client, 300)
    sid = next(s["id"] for s in st["shields"] if s["state"] == "loaded" and 60 <= s["time_in_state_s"] <= 330)
    ev = jpost(client, "/api/sim/inject", {"type": "pressure_spike", "shield": sid, "magnitude": 150, "duration_s": 40})
    assert set(ev) == EVENT_KEYS and ev["onset_t"] == 301.0 and ev["shields"] == [sid]
    st = jget(client, "/api/state", params={"shield": sid})
    assert st["events"][0]["id"] == ev["id"]
    st = advance(client, 20, sid)
    alerts = [a for a in st["alerts"] if a["category"] == "process" and sid in a["shields"]]
    assert alerts, st["alerts"]
    a = alerts[-1]
    assert ev["onset_t"] <= a["opened_t"] <= ev["onset_t"] + 10
    d = jget(client, f"/api/alerts/{a['id']}")
    assert set(d) == DETAIL_KEYS
    assert d["matched_event"]["id"] == ev["id"]
    assert isinstance(d["explanation"], str) and d["evidence"] and d["verification_steps"]
    assert d["series"]["t"] and set(d["series"]["ai_score"]) == set(d["shields"])


def test_sensor_dropout_alert_via_api(client):
    reset(client)
    advance(client, 120)
    ev = jpost(client, "/api/sim/inject", {"type": "sensor_dropout", "shield": "S09", "leg": 1})
    assert ev["channels"] == ["S09-L1"] and ev["end_t"] - ev["onset_t"] == 60
    st = advance(client, 8, "S09")
    sens = [a for a in st["alerts"] if a["category"] == "sensor" and "S09" in a["shields"]]
    assert sens
    d = jget(client, f"/api/alerts/{sens[-1]['id']}")
    assert d["category"] == "sensor" and d["matched_event"]["id"] == ev["id"]


def test_telemetry_interruption_via_api(client):
    reset(client)
    advance(client, 60)
    ev = jpost(client, "/api/sim/inject", {"type": "telemetry_interruption", "shield": "S08", "duration_s": 45})
    assert ev["shields"] == [f"S{i:02d}" for i in range(7, 13)]
    st = advance(client, 6)
    assert next(d for d in st["devices"] if d["id"] == "GW-02")["status"] in ("stale", "disconnected")
    st = advance(client, 10)
    gw = next(d for d in st["devices"] if d["id"] == "GW-02")
    assert gw["status"] == "disconnected"
    assert all(s["status"] == "no_data" for s in st["shields"][6:12])
    assert any(a["category"] == "telemetry" for a in st["alerts"])


def test_unknown_alert_is_404(client):
    r = client.get("/api/alerts/AL-4242")
    assert r.status_code == 404 and isinstance(r.json()["detail"], str)


# --------------------------------------------------------------------------- exports / ingest
def test_csv_export(client):
    reset(client, n_shields=6)
    advance(client, 9)
    r = client.get("/api/export/telemetry.csv")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    assert "mineshield_telemetry.csv" in r.headers["content-disposition"]
    lines = r.text.splitlines()
    assert lines[0] == CSV_HEADER
    assert len(lines) == 1 + 10 * 12


def test_experiment_export(client):
    reset(client)
    advance(client, 30)
    r = client.get("/api/export/experiment.json")
    assert r.status_code == 200
    assert "mineshield_experiment.json" in r.headers["content-disposition"]
    assert r.headers["content-disposition"].startswith("attachment")
    ex = strict_loads(r.text)
    assert set(ex) == {"exported_at", "version", "sim", "events", "alerts"} and ex["sim"]["t"] == 30.0


def test_ingest_validate(client):
    good = {"timestamp": "2026-01-01T06:20:34Z", "t_s": 1234.0, "seq": 29617, "device_id": "GW-01",
            "shield_id": "S05", "channel_id": "S05-L1", "value": 312.4, "unit": "bar", "quality": "ok"}
    r = jpost(client, "/api/ingest/validate", {"readings": [good]})
    assert r == {"accepted": 1, "rejected": []}
    bad_unit = dict(good, seq=29618, unit="psi")
    unknown = dict(good, seq=29619, channel_id="S99-L1", shield_id="S99")
    r = jpost(client, "/api/ingest/validate", {"readings": [good, bad_unit, unknown, "junk"]})
    assert r["accepted"] == 1
    assert r["rejected"][0] == {"index": 1, "reason": "unit must be 'bar'"}
    assert [x["index"] for x in r["rejected"]] == [1, 2, 3]
    # a fresh validator per request: the same seq is accepted again
    assert jpost(client, "/api/ingest/validate", {"readings": [good]})["accepted"] == 1
    assert_422(client.post("/api/ingest/validate", json={"readings": "nope"}))


# --------------------------------------------------------------------------- evaluation / frontend
def test_evaluation_endpoints(client, results_path):
    assert not results_path.exists()
    assert jget(client, "/api/evaluation") == {"available": False, "message": "Run: python -m mineshield.evaluation"}
    lock = client.app.state.eval_lock
    assert lock.acquire(blocking=False)
    try:
        r = client.post("/api/evaluation/run")
        assert r.status_code == 409 and isinstance(r.json()["detail"], str)
    finally:
        lock.release()
    res = jpost(client, "/api/evaluation/run")
    assert res["available"] is True and [m["id"] for m in res["methods"]] == [
        "fixed_threshold", "rolling_mad", "isolation_forest", "mineshield_hybrid"]
    assert results_path.exists()
    assert jget(client, "/api/evaluation") == res


def test_frontend_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "<html" in r.text.lower()
    assert client.get("/api.js").status_code == 200
    assert client.get("/api/does-not-exist").status_code == 404
