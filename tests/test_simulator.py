import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mineshield.config import PROJECT_ROOT, load_config, validate_sim_params
from mineshield.simulator import EVENT_TYPES, Event, FaceSimulator, Frame

EVENT_RECORD_KEYS = {"id", "type", "label", "category", "onset_t", "end_t", "shields", "channels",
                     "magnitude", "magnitude_unit", "active"}


def run(sim: FaceSimulator, n: int) -> SimpleNamespace:
    """Step ``n`` times; row k of every array is the sample at t = k (dt = 1 s)."""
    frames = [sim.step() for _ in range(n)]
    return SimpleNamespace(
        frames=frames,
        t=np.array([f.t for f in frames]),
        values=np.array([f.values for f in frames]),
        true=np.array([f.true_values for f in frames]),
        states=np.array([f.states for f in frames]),
        tis=np.array([f.time_in_state for f in frames]),
        labels=np.array([f.gt_label for f in frames]),
        ids=np.array([f.gt_event_id for f in frames], dtype=object),
    )


def pair(cfg, injections, n=900, seed=42, **sim_kw):
    """Run a baseline simulator and an identical one with ``injections`` (list of inject kwargs)."""
    base = FaceSimulator(cfg, seed, **sim_kw)
    sim = FaceSimulator(cfg, seed, **sim_kw)
    events = [sim.inject(**kw) for kw in injections]
    return run(base, n), run(sim, n), sim, events


def ch(sim, cid):
    return sim.channel_ids.index(cid)


def shield_channels(sim, sid):
    return [c for c, cid in enumerate(sim.channel_ids) if cid.startswith(sid + "-")]


# ---------------------------------------------------------------- config

def test_load_config_default_and_relative_path():
    a = load_config()
    b = load_config("config/default.json")
    assert a == b
    assert a["simulator"]["dt_s"] == 1.0
    a["simulator"]["dt_s"] = 99
    assert load_config()["simulator"]["dt_s"] == 1.0  # fresh dict each call
    assert (PROJECT_ROOT / "mineshield").is_dir()


def test_validate_sim_params_accepts_defaults(cfg):
    d = cfg["defaults"]
    out = validate_sim_params(cfg, d["seed"], d["n_shields"], d["legs_per_shield"], d["noise_bar"],
                              d["load_trend_bar_per_h"])
    assert out == {"seed": 42, "n_shields": 12, "legs_per_shield": 2, "noise_bar": 2.0, "load_trend_bar_per_h": 0.0}
    validate_sim_params(cfg, 0, 24, 4, 8, -60)
    validate_sim_params(cfg, 7, 6, 2, 0.5, 120.0)


@pytest.mark.parametrize("args, match", [
    ((-1, 12, 2, 2.0, 0), "seed"),
    ((1.5, 12, 2, 2.0, 0), "seed"),
    ((True, 12, 2, 2.0, 0), "seed"),
    ((42, 5, 2, 2.0, 0), "n_shields"),
    ((42, 25, 2, 2.0, 0), "n_shields"),
    ((42, 12.0, 2, 2.0, 0), "n_shields"),
    ((42, 12, 3, 2.0, 0), "legs_per_shield must be 2 or 4"),
    ((42, 12, 1, 2.0, 0), "legs_per_shield must be 2 or 4"),
    ((42, 12, 2, 0.1, 0), "noise_bar"),
    ((42, 12, 2, 9, 0), "noise_bar"),
    ((42, 12, 2, float("nan"), 0), "noise_bar"),
    ((42, 12, 2, 2.0, 121), "load_trend_bar_per_h"),
    ((42, 12, 2, 2.0, -61), "load_trend_bar_per_h"),
    ((42, 12, 2, 2.0, "0"), "load_trend_bar_per_h"),
])
def test_validate_sim_params_rejects(cfg, args, match):
    with pytest.raises(ValueError, match=match):
        validate_sim_params(cfg, *args)


# ---------------------------------------------------------------- catalogue / indexing

def test_event_types_match_api_contract():
    text = (PROJECT_ROOT / "docs" / "API_CONTRACT.md").read_text(encoding="utf-8")
    block = re.search(r"## GET /api/config.*?```json\n(.*?)```", text, re.S).group(1)
    assert EVENT_TYPES == json.loads(block)["event_types"]
    assert "benign_transient" not in [e["type"] for e in EVENT_TYPES]


def test_indexing_and_gateways(cfg):
    sim = FaceSimulator(cfg, 1, n_shields=24, legs_per_shield=2)
    assert sim.shield_ids[0] == "S01" and sim.shield_ids[-1] == "S24"
    assert sim.channel_ids[:4] == ["S01-L1", "S01-L2", "S02-L1", "S02-L2"]
    assert sim.channel_shield.tolist() == [c // 2 for c in range(48)]
    assert sim.gateway_ids == ["GW-01", "GW-02", "GW-03", "GW-04"]
    assert sim.gateway_of(5) == "GW-01" and sim.gateway_of(6) == "GW-02" and sim.gateway_of(23) == "GW-04"
    assert FaceSimulator(cfg, 1).gateway_ids == ["GW-01", "GW-02"]


def test_frame_shapes_time_and_seq(cfg):
    sim = FaceSimulator(cfg, 42)
    f0, f1 = sim.step(), sim.step()
    assert isinstance(f0, Frame)
    assert (f0.t, f1.t, sim.t) == (0.0, 1.0, 2.0)
    assert (f0.seq_start, f1.seq_start, sim.seq) == (0, 24, 48)
    assert f0.values.shape == f0.true_values.shape == (24,)
    assert len(f0.states) == 12 and f0.time_in_state.shape == (12,)
    assert f0.gt_label == ["normal"] * 24 and f0.gt_event_id == [None] * 24
    # At t = 0 the hydraulic pressure equals its target (no lag yet); quantised to 0.1 bar.
    assert np.allclose(f0.values * 10, np.round(f0.values * 10))


# ---------------------------------------------------------------- reproducibility

def test_same_seed_bit_identical_and_different_seed_differs(cfg):
    inj = [dict(type="gradual_rise", shield_id="S05", onset_t=100),
           dict(type="sensor_dropout", shield_id="S02", channel_id="S02-L2", onset_t=50),
           dict(type="sensor_stuck", shield_id="S09", onset_t=200)]
    a = FaceSimulator(cfg, 42)
    b = FaceSimulator(cfg, 42)
    for kw in inj:
        a.inject(**kw)
        b.inject(**kw)
    ra, rb = run(a, 600), run(b, 600)
    np.testing.assert_array_equal(ra.values, rb.values)  # NaN-aware, exact
    np.testing.assert_array_equal(ra.true, rb.true)
    assert (ra.labels == rb.labels).all()
    rc = run(FaceSimulator(cfg, 43), 600)
    assert not np.array_equal(run(FaceSimulator(cfg, 42), 600).values, rc.values)


def test_injected_event_leaves_other_channels_bit_identical(cfg):
    base, ev, sim, _ = pair(cfg, [dict(type="gradual_rise", shield_id="S05", onset_t=300),
                                  dict(type="leakage", shield_id="S10", channel_id="S10-L2", onset_t=400)])
    affected = shield_channels(sim, "S05") + [ch(sim, "S10-L2")]
    others = [c for c in range(sim.n_channels) if c not in affected]
    np.testing.assert_array_equal(base.values[:, others], ev.values[:, others])
    np.testing.assert_array_equal(base.true[:, others], ev.true[:, others])
    assert not np.array_equal(base.values[:, affected], ev.values[:, affected])
    # Nothing changes before the (future) onset.
    np.testing.assert_array_equal(base.values[:300], ev.values[:300])


# ---------------------------------------------------------------- cycle and plausibility

def test_cycle_states_and_lowered_pressure_when_advancing(cfg):
    sim = FaceSimulator(cfg, 42)
    sc = cfg["simulator"]
    r = run(sim, 700)
    i = 3  # S04, offset = 3 * 600 / 12 = 150 s
    off = int(sim.offset_s[i])
    assert off == 150
    adv, setting = sc["advance_s"], sc["setting_s"]
    assert r.states[off - 1, i] == "loaded"
    assert r.states[off, i] == "advancing" and r.tis[off, i] == 0
    assert r.states[off + adv - 1, i] == "advancing" and r.tis[off + adv - 1, i] == adv - 1
    assert r.states[off + adv, i] == "setting" and r.tis[off + adv, i] == 0
    assert r.states[off + adv + setting - 1, i] == "setting"
    assert r.states[off + adv + setting, i] == "loaded" and r.tis[off + adv + setting, i] == 0
    legs = shield_channels(sim, "S04")
    # Legs are lowered by the end of the advance (first-order lag from ~330 bar towards lowered_bar).
    assert np.all(r.true[off + adv - 1, legs] < sc["lowered_bar"] + 5)
    assert np.all(r.true[off + adv - 1, legs] >= sc["lowered_bar"])
    assert np.all(np.abs(r.values[off + adv - 1, legs] - sc["lowered_bar"]) < 15)
    assert np.all(r.true[off - 1, legs] > cfg["limits"]["setting_bar"])
    # Shield S01 (offset 0) is advancing at t = 0 and starts exactly at the lowered pressure.
    assert r.states[0, 0] == "advancing"
    assert np.all(r.true[0, :2] == sc["lowered_bar"])


def test_loaded_pressures_plausible_and_ranges_hold(cfg):
    sim = FaceSimulator(cfg, 7)
    r = run(sim, 1800)
    lim, sc = cfg["limits"], cfg["simulator"]
    loaded = np.repeat(r.states == "loaded", sim.legs, axis=1)
    assert loaded.mean() > 0.5
    assert r.true[loaded].min() > lim["setting_bar"] and r.true[loaded].max() < lim["yield_bar"]
    assert r.values[loaded].min() > lim["setting_bar"] and r.values[loaded].max() < lim["yield_bar"]
    assert not np.isnan(r.values).any()
    assert r.values.min() >= lim["sensor_min_bar"] and r.values.max() <= lim["sensor_max_bar"]
    assert r.true.min() >= sc["lowered_bar"] and r.true.max() <= lim["yield_bar"]


def test_ranges_hold_under_extreme_events(cfg):
    sim = FaceSimulator(cfg, 3, noise_bar=8.0, load_trend_bar_per_h=120)
    sim.inject("gradual_rise", "S05", magnitude=900, onset_t=100)
    sim.inject("leakage", "S08", channel_id="S08-L2", magnitude=50, onset_t=100)
    sim.inject("leg_imbalance", "S10", magnitude=400, onset_t=100)
    sim.inject("sensor_spike", "S02", magnitude=5000, onset_t=200)
    r = run(sim, 900)
    lim = cfg["limits"]
    assert np.nanmin(r.values) >= lim["sensor_min_bar"] and np.nanmax(r.values) <= lim["sensor_max_bar"]
    assert r.true.min() >= cfg["simulator"]["lowered_bar"] and r.true.max() <= lim["yield_bar"]
    assert r.true.max() == pytest.approx(lim["yield_bar"], abs=1e-6)  # yield valve caps the gradual rise
    assert r.values[200, ch(sim, "S02-L1")] == lim["sensor_max_bar"]  # spike saturates the transducer


# ---------------------------------------------------------------- process events

ONSET = 400  # S03..S07 are loaded (not advancing) for t in [430, 700) with the default 12-shield face


def test_gradual_rise_raises_shield(cfg):
    base, ev, sim, (e,) = pair(cfg, [dict(type="gradual_rise", shield_id="S05", onset_t=ONSET)])
    legs = shield_channels(sim, "S05")
    diff = ev.true[:, legs] - base.true[:, legs]
    assert np.all(diff[:ONSET] == 0)
    assert np.all(diff[ONSET + 150] > 20) and np.all(diff[ONSET + 299] > 40)
    assert np.all(diff[ONSET + 299] > diff[ONSET + 100])  # ramps up over the duration
    assert np.all(diff[int(e.end_t) + 99] < 2)  # recovers after the end (recovery_tau_s = 20 s)


def test_pressure_spike_raises_then_decays(cfg):
    base, ev, sim, _ = pair(cfg, [dict(type="pressure_spike", shield_id="S05", onset_t=ONSET)])
    legs = shield_channels(sim, "S05")
    diff = ev.true[:, legs] - base.true[:, legs]
    assert np.all(diff[ONSET + 3] > 30)
    assert np.all(diff[ONSET + 3] > diff[ONSET + 30])
    assert np.all(np.abs(diff[ONSET + 60]) < 1)


def test_leakage_lowers_only_its_channel(cfg):
    base, ev, sim, (e,) = pair(cfg, [dict(type="leakage", shield_id="S05", channel_id="S05-L1", onset_t=ONSET)])
    c1, c2 = ch(sim, "S05-L1"), ch(sim, "S05-L2")
    diff = ev.true[:, c1] - base.true[:, c1]
    assert diff[ONSET + 149] < -80  # 0.8 bar/s over ~150 s
    assert np.all(np.diff(diff[ONSET + 3:ONSET + 149]) < 0)
    others = [c for c in range(sim.n_channels) if c != c1]
    np.testing.assert_array_equal(ev.values[:, others], base.values[:, others])
    assert e.channels == ["S05-L1"] and e.shields == ["S05"]


def test_leg_imbalance_separates_legs(cfg):
    base, ev, sim, _ = pair(cfg, [dict(type="leg_imbalance", shield_id="S05", onset_t=ONSET)])
    c1, c2 = ch(sim, "S05-L1"), ch(sim, "S05-L2")
    t = ONSET + 100
    assert ev.true[t, c1] < base.true[t, c1] - 20
    assert ev.true[t, c2] > base.true[t, c2] + 10
    gap_ev = ev.true[t, c2] - ev.true[t, c1]
    gap_base = base.true[t, c2] - base.true[t, c1]
    assert gap_ev - gap_base > 40


def test_multi_shield_loading_centre_more_than_pm2(cfg):
    base, ev, sim, (e,) = pair(cfg, [dict(type="multi_shield_loading", shield_id="S05", magnitude=30,
                                          onset_t=ONSET)])
    assert e.shields == ["S03", "S04", "S05", "S06", "S07"]
    t = 600
    d = {sid: (ev.true[t, shield_channels(sim, sid)] - base.true[t, shield_channels(sim, sid)]).mean()
         for sid in ["S02", "S03", "S04", "S05", "S06", "S07", "S08"]}
    assert d["S05"] > d["S04"] > d["S03"] > 5
    assert d["S05"] > d["S06"] > d["S07"] > 5
    assert d["S05"] > 2 * d["S03"] and d["S05"] > 2 * d["S07"]
    assert d["S02"] == 0 and d["S08"] == 0


def test_block_scope_clipped_at_face_edge(cfg):
    sim = FaceSimulator(cfg, 1)
    assert sim.inject("multi_shield_loading", "S01").shields == ["S01", "S02", "S03"]
    assert sim.inject("multi_shield_loading", "S12").shields == ["S10", "S11", "S12"]


# ---------------------------------------------------------------- sensor / telemetry events

def test_sensor_spike_single_sample(cfg):
    base, ev, sim, (e,) = pair(cfg, [dict(type="sensor_spike", shield_id="S05", channel_id="S05-L2",
                                          onset_t=ONSET, duration_s=30)])
    c = ch(sim, "S05-L2")
    assert e.end_t - e.onset_t == 1.0  # fixed 1 s duration
    diff = ev.values - base.values
    changed = np.argwhere(diff != 0)
    assert changed.tolist() == [[ONSET, c]]
    assert diff[ONSET, c] == pytest.approx(150.0)
    np.testing.assert_array_equal(ev.true, base.true)
    assert ev.labels[ONSET, c] == "sensor_spike" and ev.labels[ONSET + 1, c] == "normal"


def test_sensor_stuck_constant(cfg):
    base, ev, sim, (e,) = pair(cfg, [dict(type="sensor_stuck", shield_id="S05", onset_t=ONSET)])
    c = ch(sim, "S05-L1")  # default leg 1
    assert e.channels == ["S05-L1"]
    stuck = ev.values[ONSET:ONSET + 120, c]
    assert np.all(stuck == base.values[ONSET - 1, c])
    assert ev.values[ONSET + 120, c] == base.values[ONSET + 120, c]
    assert np.unique(base.values[ONSET:ONSET + 120, c]).size > 10
    np.testing.assert_array_equal(ev.true, base.true)


def test_sensor_dropout_nan(cfg):
    base, ev, sim, _ = pair(cfg, [dict(type="sensor_dropout", shield_id="S05", channel_id="S05-L2",
                                       onset_t=ONSET)])
    c = ch(sim, "S05-L2")
    nan_rows, nan_cols = np.nonzero(np.isnan(ev.values))
    assert set(nan_cols.tolist()) == {c}
    assert nan_rows.tolist() == list(range(ONSET, ONSET + 60))


def test_telemetry_interruption_whole_gateway_only(cfg):
    base, ev, sim, (e,) = pair(cfg, [dict(type="telemetry_interruption", shield_id="S03", onset_t=ONSET)])
    assert e.shields == ["S01", "S02", "S03", "S04", "S05", "S06"]
    gw1 = list(range(0, 12))
    gw2 = list(range(12, 24))
    assert np.all(np.isnan(ev.values[ONSET:ONSET + 45, gw1]))
    assert not np.isnan(ev.values[ONSET - 1, gw1]).any() and not np.isnan(ev.values[ONSET + 45, gw1]).any()
    assert not np.isnan(ev.values[:, gw2]).any()
    np.testing.assert_array_equal(ev.values[:, gw2], base.values[:, gw2])
    assert set(ev.labels[ONSET + 10, gw1]) == {"telemetry_interruption"}
    assert set(ev.labels[ONSET + 10, gw2]) == {"normal"}


# ---------------------------------------------------------------- labels

def test_labels_only_during_onset_to_end(cfg):
    _, ev, sim, (e,) = pair(cfg, [dict(type="gradual_rise", shield_id="S05", onset_t=ONSET)])
    legs = shield_channels(sim, "S05")
    in_window = (ev.t >= e.onset_t) & (ev.t < e.end_t)
    for c in range(sim.n_channels):
        expect = np.where(in_window & (c in legs), "gradual_rise", "normal")
        assert (ev.labels[:, c] == expect).all()
        ids = ev.ids[:, c]
        assert all((i == e.id) if w and c in legs else i is None for i, w in zip(ids, in_window))
    assert e.id == "EV-001"


def test_most_recent_event_wins_label(cfg):
    _, ev, sim, (a, b) = pair(cfg, [dict(type="gradual_rise", shield_id="S05", onset_t=ONSET),
                                    dict(type="sensor_dropout", shield_id="S05", channel_id="S05-L1",
                                         onset_t=ONSET + 50)])
    c1, c2 = ch(sim, "S05-L1"), ch(sim, "S05-L2")
    t = ONSET + 60
    assert ev.labels[t, c1] == "sensor_dropout" and ev.ids[t, c1] == b.id == "EV-002"
    assert ev.labels[t, c2] == "gradual_rise" and ev.ids[t, c2] == a.id
    assert ev.labels[ONSET + 120, c1] == "gradual_rise"


def test_benign_transient_hidden_and_labelled_normal(cfg):
    base, ev, sim, (e,) = pair(cfg, [dict(type="benign_transient", shield_id="S05", onset_t=ONSET)])
    assert sim.events == [] and e.hidden and e.category == "normal"
    assert (ev.labels == "normal").all() and all(i is None for i in ev.ids.ravel())
    legs = shield_channels(sim, "S05")
    assert np.all(ev.true[ONSET + 5, legs] - base.true[ONSET + 5, legs] > 20)
    # Visible ids are not consumed by hidden events.
    assert sim.inject("gradual_rise", "S01").id == "EV-001"


# ---------------------------------------------------------------- parameters

def test_load_trend_raises_mean_pressure(cfg):
    a = run(FaceSimulator(cfg, 42, load_trend_bar_per_h=0), 1800)
    b = run(FaceSimulator(cfg, 42, load_trend_bar_per_h=120), 1800)
    loaded = np.repeat(a.states[1500:] == "loaded", 2, axis=1)
    assert (b.states == a.states).all()
    assert b.true[1500:][loaded].mean() - a.true[1500:][loaded].mean() > 25
    assert np.all(b.true[1:] >= a.true[1:] - 1e-9)


def test_four_leg_mode(cfg):
    base, ev, sim, (e,) = pair(cfg, [dict(type="leg_imbalance", shield_id="S05", onset_t=ONSET)],
                               n=600, legs_per_shield=4)
    assert sim.n_channels == 48 and ev.values.shape == (600, 48)
    assert sim.channel_ids[:5] == ["S01-L1", "S01-L2", "S01-L3", "S01-L4", "S02-L1"]
    assert e.channels == ["S05-L1", "S05-L2", "S05-L3", "S05-L4"]
    t = ONSET + 100
    d = {j: ev.true[t, ch(sim, f"S05-L{j}")] - base.true[t, ch(sim, f"S05-L{j}")] for j in (1, 2, 3, 4)}
    assert d[1] < -20 and d[3] < -20 and d[2] > 10 and d[4] > 10
    leak = FaceSimulator(cfg, 42, legs_per_shield=4).inject("leakage", "S05", channel_id="S05-L4")
    assert leak.channels == ["S05-L4"]


# ---------------------------------------------------------------- inject API

def test_inject_defaults_and_event_record(cfg):
    sim = FaceSimulator(cfg, 42)
    for _ in range(10):
        sim.step()
    e = sim.inject("leakage", "S05")
    assert isinstance(e, Event)
    assert (e.onset_t, e.end_t, e.magnitude, e.channels) == (10.0, 160.0, 0.8, ["S05-L1"])
    d = e.to_dict(50.0)
    assert set(d) == EVENT_RECORD_KEYS
    assert d == {"id": "EV-001", "type": "leakage", "label": "Leakage-like pressure drop", "category": "process",
                 "onset_t": 10.0, "end_t": 160.0, "shields": ["S05"], "channels": ["S05-L1"],
                 "magnitude": 0.8, "magnitude_unit": "bar/s", "active": True}
    assert e.to_dict(160.0)["active"] is False and e.to_dict(9.0)["active"] is False
    assert e.to_dict()["active"] is False
    json.dumps(d)
    s = sim.inject("sensor_stuck", "S02", magnitude=5)  # magnitude ignored for types without one
    assert s.magnitude is None and s.to_dict(10)["magnitude_unit"] is None
    assert [ev.id for ev in sim.events] == ["EV-001", "EV-002"]


@pytest.mark.parametrize("kwargs, match", [
    (dict(type="earthquake", shield_id="S05"), "unknown event type"),
    (dict(type="gradual_rise", shield_id="S13"), "unknown shield"),
    (dict(type="gradual_rise", shield_id=5), "unknown shield"),
    (dict(type="leakage", shield_id="S05", channel_id="S05-L3"), "unknown channel"),
    (dict(type="leakage", shield_id="S05", channel_id="S06-L1"), "does not belong to shield"),
    (dict(type="gradual_rise", shield_id="S05", magnitude=0), "magnitude"),
    (dict(type="gradual_rise", shield_id="S05", magnitude=-5), "magnitude"),
    (dict(type="gradual_rise", shield_id="S05", magnitude=float("nan")), "magnitude"),
    (dict(type="gradual_rise", shield_id="S05", magnitude="70"), "magnitude"),
    (dict(type="gradual_rise", shield_id="S05", duration_s=0.5), "duration_s"),
    (dict(type="gradual_rise", shield_id="S05", duration_s=3601), "duration_s"),
    (dict(type="gradual_rise", shield_id="S05", onset_t=-1), "in the past"),
    (dict(type="gradual_rise", shield_id="S05", onset_t=float("inf")), "onset_t"),
])
def test_inject_invalid_arguments(cfg, kwargs, match):
    sim = FaceSimulator(cfg, 42)
    with pytest.raises(ValueError, match=match):
        sim.inject(**kwargs)
    assert sim.events == []


@pytest.mark.parametrize("kwargs", [
    dict(seed=-1), dict(n_shields=0), dict(legs_per_shield=3), dict(noise_bar=-1.0),
    dict(load_trend_bar_per_h=float("nan")),
])
def test_simulator_rejects_invalid_construction(cfg, kwargs):
    params = dict(seed=1) | kwargs
    with pytest.raises(ValueError):
        FaceSimulator(cfg, **params)
