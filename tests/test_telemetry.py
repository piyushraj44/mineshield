import json

import numpy as np
import pytest

from mineshield.simulator import FaceSimulator
from mineshield.telemetry import (
    REQUIRED_FIELDS,
    DeviceMonitor,
    TelemetryValidator,
    format_timestamp,
    frame_to_readings,
    readings_to_vector,
)

DEVICE_KEYS = {"id", "shields", "status", "last_t", "age_s", "missing", "invalid", "throughput_rps"}


def first_readings(cfg):
    sim = FaceSimulator(cfg, 42)
    return sim, frame_to_readings(sim.step(), sim)


def make_validator(sim, cfg):
    return TelemetryValidator(sim.channel_ids, cfg["limits"])


def reading(channel="S01-L1", t=10.0, seq=1000, value=300.0, quality="ok", **kw):
    shield = channel.split("-")[0]
    r = {"timestamp": format_timestamp("2026-01-01T06:00:00Z", t), "t_s": t, "seq": seq, "device_id": "GW-01",
         "shield_id": shield, "channel_id": channel, "value": value, "unit": "bar", "quality": quality}
    r.update(kw)
    return r


# ---------------------------------------------------------------- frame_to_readings

def test_frame_to_readings_schema(cfg):
    sim = FaceSimulator(cfg, 42)
    sim.inject("sensor_dropout", "S05", channel_id="S05-L2", onset_t=1)
    f0, f1 = sim.step(), sim.step()
    r0, r1 = frame_to_readings(f0, sim), frame_to_readings(f1, sim)
    assert len(r0) == len(r1) == sim.n_channels == 24
    assert all(list(r) == list(REQUIRED_FIELDS) for r in r0 + r1)
    assert [r["seq"] for r in r0 + r1] == list(range(48))
    assert r0[0] == {"timestamp": "2026-01-01T06:00:00Z", "t_s": 0.0, "seq": 0, "device_id": "GW-01",
                     "shield_id": "S01", "channel_id": "S01-L1", "value": float(f0.values[0]), "unit": "bar",
                     "quality": "ok"}
    assert type(r0[0]["value"]) is float
    assert r1[0]["timestamp"] == "2026-01-01T06:00:01Z" and r1[0]["t_s"] == 1.0
    assert (r0[12]["device_id"], r0[12]["shield_id"], r0[12]["channel_id"]) == ("GW-02", "S07", "S07-L1")
    c = sim.channel_ids.index("S05-L2")
    assert r1[c]["value"] is None and r1[c]["quality"] == "missing"
    assert r0[c]["quality"] == "ok"
    json.dumps(r0 + r1)  # JSON-serialisable (no numpy scalars, no NaN)


def test_timestamp_format():
    assert format_timestamp("2026-01-01T06:00:00Z", 1234.0) == "2026-01-01T06:20:34Z"
    assert format_timestamp("2026-01-01T06:00:00Z", 0.5) == "2026-01-01T06:00:00.500Z"


# ---------------------------------------------------------------- validator

def test_validator_accepts_simulator_readings(cfg):
    sim, readings = first_readings(cfg)
    acc, rej = make_validator(sim, cfg).validate(readings)
    assert rej == []
    assert acc == readings


@pytest.mark.parametrize("mutate, reason", [
    (lambda r: r.pop("unit"), "missing field(s): unit"),
    (lambda r: (r.pop("value"), r.pop("seq")), "missing field(s): seq, value"),
    (lambda r: r.update(unit="psi"), "unit must be 'bar'"),
    (lambda r: r.update(unit="psi", channel_id="S99-L1"), "unit must be 'bar'"),  # checks run in order
    (lambda r: r.update(channel_id="S99-L1", shield_id="S99"), "unknown channel_id 'S99-L1'"),
    (lambda r: r.update(channel_id="S01-L3"), "unknown channel_id 'S01-L3'"),
    (lambda r: r.update(shield_id="S02"), "channel_id 'S01-L1' does not belong to shield_id 'S02'"),
    (lambda r: r.update(value="312.4"), "value must be a number or null"),
    (lambda r: r.update(value=True), "value must be a number or null"),
    (lambda r: r.update(seq="7"), "seq must be an integer"),
    (lambda r: r.update(t_s=None), "t_s must be a finite number"),
])
def test_validator_rejects_bad_readings(cfg, mutate, reason):
    sim, readings = first_readings(cfg)
    bad = dict(readings[0])
    mutate(bad)
    acc, rej = make_validator(sim, cfg).validate([bad, readings[1]])
    assert rej == [{"index": 0, "reason": reason}]
    assert [r["channel_id"] for r in acc] == ["S01-L2"]


def test_validator_rejects_non_object(cfg):
    sim, _ = first_readings(cfg)
    acc, rej = make_validator(sim, cfg).validate(["oops"])
    assert acc == [] and rej == [{"index": 0, "reason": "reading must be an object"}]


def test_validator_duplicate_seq_across_and_within_calls(cfg):
    sim, readings = first_readings(cfg)
    v = make_validator(sim, cfg)
    v.validate(readings)
    acc, rej = v.validate([readings[3]])
    assert acc == [] and rej == [{"index": 0, "reason": "duplicate"}]
    acc, rej = v.validate([reading(seq=5000, t=2.0), reading(channel="S02-L1", seq=5000, t=2.0)])
    assert len(acc) == 1 and rej == [{"index": 1, "reason": "duplicate"}]
    v.reset()
    acc, rej = v.validate([readings[3]])
    assert len(acc) == 1 and rej == []


def test_validator_seq_memory_is_bounded(cfg):
    sim, _ = first_readings(cfg)
    v = make_validator(sim, cfg)
    v.validate([reading(seq=0, t=0.0)])
    v.validate([reading(seq=s, t=float(s)) for s in range(1, 10_001)])
    acc, rej = v.validate([reading(seq=0, t=20_000.0)])  # seq 0 has been forgotten
    assert rej == [] and len(acc) == 1
    acc, rej = v.validate([reading(seq=10_000, t=20_001.0)])
    assert rej == [{"index": 0, "reason": "duplicate"}]


def test_validator_stale_timestamp_per_channel(cfg):
    sim, _ = first_readings(cfg)
    v = make_validator(sim, cfg)
    acc, rej = v.validate([reading(t=10.0, seq=1), reading(t=9.0, seq=2), reading(channel="S02-L1", t=9.0, seq=3),
                           reading(t=10.0, seq=4)])
    assert rej == [{"index": 1, "reason": "stale timestamp"}]
    assert [r["seq"] for r in acc] == [1, 3, 4]
    _, rej = v.validate([reading(t=9.5, seq=5)])
    assert rej == [{"index": 0, "reason": "stale timestamp"}]


def test_validator_out_of_range_accepted_as_invalid(cfg):
    sim, _ = first_readings(cfg)
    v = make_validator(sim, cfg)
    acc, rej = v.validate([reading(value=700.0, seq=1), reading(channel="S01-L2", value=-5, seq=2),
                           reading(channel="S02-L1", value=None, quality="missing", seq=3),
                           reading(channel="S02-L2", value=600, seq=4)])
    assert rej == []
    assert [(r["value"], r["quality"]) for r in acc] == [(None, "invalid"), (None, "invalid"), (None, "missing"),
                                                         (600.0, "ok")]
    assert v.counts == {"accepted": 4, "rejected": 0, "invalid": 2}


# ---------------------------------------------------------------- readings_to_vector

def test_readings_to_vector(cfg):
    sim, _ = first_readings(cfg)
    v = make_validator(sim, cfg)
    acc, _ = v.validate([reading("S01-L1", value=312.5, seq=1), reading("S01-L2", value=900.0, seq=2),
                         reading("S02-L1", value=None, quality="missing", seq=3)])
    values, quality = readings_to_vector(acc, sim.channel_ids)
    assert values.shape == (24,) and len(quality) == 24
    assert values[0] == 312.5
    assert np.isnan(values[1:]).all()
    assert quality[:4] == ["ok", "invalid", "missing", "missing"]  # S02-L2 absent => missing
    assert set(quality[4:]) == {"missing"}


def test_round_trip_simulator_to_vector(cfg):
    sim = FaceSimulator(cfg, 5)
    sim.inject("sensor_dropout", "S04", channel_id="S04-L1")
    f = sim.step()
    acc, rej = make_validator(sim, cfg).validate(frame_to_readings(f, sim))
    values, quality = readings_to_vector(acc, sim.channel_ids)
    assert rej == []
    np.testing.assert_array_equal(values, f.values)
    assert quality == ["missing" if np.isnan(x) else "ok" for x in f.values]


# ---------------------------------------------------------------- DeviceMonitor

def test_device_monitor_status_through_interruption(cfg):
    sim = FaceSimulator(cfg, 42)
    onset, dur = 100, 45
    sim.inject("telemetry_interruption", "S03", onset_t=onset)  # gateway GW-01 = S01..S06
    validator = make_validator(sim, cfg)
    mon = DeviceMonitor(sim)
    hist = {}
    for _ in range(200):
        f = sim.step()
        acc, rej = validator.validate(frame_to_readings(f, sim))
        assert rej == []
        _, quality = readings_to_vector(acc, sim.channel_ids)
        mon.update(f.t, quality)
        hist[int(f.t)] = {d["id"]: d for d in mon.status(f.t)}

    def gw1(t):
        return hist[t]["GW-01"]

    assert set(gw1(0)) == DEVICE_KEYS
    assert gw1(0)["shields"] == ["S01", "S02", "S03", "S04", "S05", "S06"]
    assert gw1(onset - 1)["status"] == "simulated" and gw1(onset - 1)["age_s"] == 0.0
    assert gw1(onset + 1)["status"] == "simulated"            # age 2 s
    assert gw1(onset + 2)["status"] == "stale" and gw1(onset + 2)["age_s"] == 3.0
    assert gw1(onset + 9)["status"] == "stale"                # age 10 s
    assert gw1(onset + 10)["status"] == "disconnected" and gw1(onset + 10)["age_s"] == 11.0
    assert gw1(onset + dur - 1)["status"] == "disconnected"
    assert gw1(onset + dur - 1)["last_t"] == onset - 1
    assert gw1(onset + dur)["status"] == "simulated" and gw1(onset + dur)["last_t"] == onset + dur
    assert all(h["GW-02"]["status"] == "simulated" for h in hist.values())
    statuses = [gw1(t)["status"] for t in range(200)]
    assert statuses == (["simulated"] * (onset + 2) + ["stale"] * 8 + ["disconnected"] * (dur - 10)
                        + ["simulated"] * (200 - onset - dur))
    assert gw1(199)["missing"] == dur * 12 and hist[199]["GW-02"]["missing"] == 0
    assert gw1(199)["invalid"] == 0
    assert gw1(onset - 1)["throughput_rps"] == 12.0           # 12 channels x 10 samples / 10 s
    assert gw1(5)["throughput_rps"] == pytest.approx(7.2)     # only t = 0..5 so far
    assert gw1(onset + 20)["throughput_rps"] == 0.0
    assert gw1(onset + dur + 5)["throughput_rps"] == pytest.approx(7.2)
    json.dumps(list(hist[150].values()))


def test_device_monitor_initial_and_invalid_counts(cfg):
    sim = FaceSimulator(cfg, 1)
    mon = DeviceMonitor(sim)
    st = mon.status(0.0)
    assert [d["id"] for d in st] == ["GW-01", "GW-02"]
    assert all(d["status"] == "disconnected" and d["last_t"] is None and d["age_s"] is None for d in st)
    q = ["ok"] * 24
    q[0], q[13] = "invalid", "missing"
    mon.update(0.0, q)
    st = {d["id"]: d for d in mon.status(0.0)}
    assert (st["GW-01"]["invalid"], st["GW-01"]["missing"]) == (1, 0)
    assert (st["GW-02"]["invalid"], st["GW-02"]["missing"]) == (0, 1)
    assert st["GW-01"]["throughput_rps"] == pytest.approx(1.1)
    with pytest.raises(ValueError):
        mon.update(1.0, ["ok"] * 5)
