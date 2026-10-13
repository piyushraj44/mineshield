"""Telemetry readings: conversion from simulator frames, schema validation and gateway monitoring (spec section 3)."""

from __future__ import annotations

import math
from collections import deque
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from .simulator import FaceSimulator, Frame

REQUIRED_FIELDS = ("timestamp", "t_s", "seq", "device_id", "shield_id", "channel_id", "value", "unit", "quality")
UNIT = "bar"
QUALITY_OK = "ok"
QUALITY_MISSING = "missing"
QUALITY_INVALID = "invalid"
SEQ_MEMORY = 10_000

STALE_AFTER_S = 3.0          # device age at which a gateway becomes "stale"
DISCONNECTED_AFTER_S = 10.0  # device age above which a gateway is "disconnected"
THROUGHPUT_WINDOW_S = 10.0


@lru_cache(maxsize=8)
def _parse_start(start_timestamp: str) -> datetime:
    dt = datetime.fromisoformat(start_timestamp.replace("Z", "+00:00"))
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def format_timestamp(start_timestamp: str, t: float) -> str:
    """ISO 8601 UTC timestamp ("...Z") of simulated time ``t`` seconds after ``start_timestamp``."""
    dt = _parse_start(start_timestamp) + timedelta(seconds=float(t))
    if dt.microsecond == 0:
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return dt.isoformat(timespec="milliseconds") + "Z"


def frame_to_readings(frame: "Frame", sim: "FaceSimulator") -> list[dict[str, Any]]:
    """One contract TelemetryReading per channel of ``frame`` (seq consecutive from ``frame.seq_start``)."""
    ts = format_timestamp(sim.start_timestamp, frame.t)
    t_s = float(frame.t)
    seq0 = int(frame.seq_start)
    shield_ids = sim.shield_ids
    gateways = sim.shield_gateway
    readings = []
    for c, (cid, s, v) in enumerate(zip(sim.channel_ids, sim.channel_shield.tolist(), frame.values.tolist())):
        missing = v != v  # NaN
        readings.append({
            "timestamp": ts,
            "t_s": t_s,
            "seq": seq0 + c,
            "device_id": gateways[s],
            "shield_id": shield_ids[s],
            "channel_id": cid,
            "value": None if missing else v,
            "unit": UNIT,
            "quality": QUALITY_MISSING if missing else QUALITY_OK,
        })
    return readings


def _is_number(x: Any) -> bool:
    tx = type(x)
    if tx is float or tx is int:  # fast path; ABC checks below are comparatively slow
        return True
    return isinstance(x, Real) and not isinstance(x, bool)


def _is_integer(x: Any) -> bool:
    return type(x) is int or (isinstance(x, Integral) and not isinstance(x, bool))


class TelemetryValidator:
    """Stateful schema / sequence validator for TelemetryReading dicts."""

    def __init__(self, channel_ids: Sequence[str], limits: dict[str, Any]) -> None:
        self.channel_ids = list(channel_ids)
        self._channel_shield = {cid: cid.rsplit("-", 1)[0] for cid in self.channel_ids}
        self.sensor_min = float(limits["sensor_min_bar"])
        self.sensor_max = float(limits["sensor_max_bar"])
        self.reset()

    def reset(self) -> None:
        """Forget seen sequence numbers, per-channel timestamps and counters."""
        self._seen_order: deque[int] = deque()
        self._seen: set[int] = set()
        self._last_t: dict[str, float] = {}
        self.counts = {"accepted": 0, "rejected": 0, "invalid": 0}

    def _remember(self, seq: int) -> None:
        self._seen.add(seq)
        self._seen_order.append(seq)
        if len(self._seen_order) > SEQ_MEMORY:
            self._seen.discard(self._seen_order.popleft())

    def _check(self, r: Any) -> str | None:
        """Return a rejection reason, or None if the reading passes the schema checks."""
        if not isinstance(r, dict):
            return "reading must be an object"
        missing = [f for f in REQUIRED_FIELDS if f not in r]
        if missing:
            return "missing field(s): " + ", ".join(missing)
        if r["unit"] != UNIT:
            return "unit must be 'bar'"
        ch = r["channel_id"]
        if not isinstance(ch, str) or ch not in self._channel_shield:
            return f"unknown channel_id {ch!r}"
        if r["shield_id"] != self._channel_shield[ch]:
            return f"channel_id {ch!r} does not belong to shield_id {r['shield_id']!r}"
        if r["value"] is not None and not _is_number(r["value"]):
            return "value must be a number or null"
        if not _is_integer(r["seq"]):
            return "seq must be an integer"
        if not _is_number(r["t_s"]) or not math.isfinite(float(r["t_s"])):
            return "t_s must be a finite number"
        if int(r["seq"]) in self._seen:
            return "duplicate"
        last = self._last_t.get(ch)
        if last is not None and float(r["t_s"]) < last:
            return "stale timestamp"
        return None

    def _normalise(self, r: dict[str, Any]) -> dict[str, Any]:
        out = dict(r)
        out["seq"] = int(r["seq"])
        out["t_s"] = float(r["t_s"])
        v = r["value"]
        if v is not None:
            try:
                v = float(v)
            except OverflowError:  # integer too large for a float: out of range
                v = math.inf
        flagged = r["quality"] in (QUALITY_MISSING, QUALITY_INVALID)
        if v is None or math.isnan(v):
            out["value"] = None
            out["quality"] = QUALITY_INVALID if r["quality"] == QUALITY_INVALID else QUALITY_MISSING
        elif not self.sensor_min <= v <= self.sensor_max:
            out["value"] = None
            out["quality"] = QUALITY_INVALID
        elif flagged:
            # Device-flagged reading: keep its flag, drop the number.
            out["value"] = None
            out["quality"] = r["quality"]
        else:
            out["value"] = v
            out["quality"] = QUALITY_OK
        return out

    def validate(self, readings: Iterable[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return ``(accepted, rejected)``; rejected items are ``{"index", "reason"}``."""
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for i, r in enumerate(readings):
            reason = self._check(r)
            if reason is not None:
                rejected.append({"index": i, "reason": reason})
                continue
            out = self._normalise(r)
            self._remember(out["seq"])
            self._last_t[out["channel_id"]] = out["t_s"]
            if out["quality"] == QUALITY_INVALID:
                self.counts["invalid"] += 1
            accepted.append(out)
        self.counts["accepted"] += len(accepted)
        self.counts["rejected"] += len(rejected)
        return accepted, rejected


def readings_to_vector(accepted: Iterable[dict[str, Any]], channel_ids: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """Channel-ordered values (NaN unless ok) and qualities; channels without a reading are "missing"."""
    index = {cid: c for c, cid in enumerate(channel_ids)}
    values = np.full(len(index), np.nan)
    quality = [QUALITY_MISSING] * len(index)
    for r in accepted:
        c = index.get(r.get("channel_id"))
        if c is None:
            continue
        v = r.get("value")
        q = r.get("quality", QUALITY_OK)
        if v is None or q != QUALITY_OK:
            values[c] = np.nan
            quality[c] = q if q in (QUALITY_MISSING, QUALITY_INVALID) else QUALITY_MISSING
        else:
            values[c] = float(v)
            quality[c] = QUALITY_OK
    return values, quality


class DeviceMonitor:
    """Per-gateway receive status, missing/invalid counters and valid-reading throughput."""

    def __init__(self, sim: "FaceSimulator") -> None:
        self.gateway_ids: list[str] = list(sim.gateway_ids)
        gw_index = {gw: g for g, gw in enumerate(self.gateway_ids)}
        self._shields = {gw: [sid for sid, sgw in zip(sim.shield_ids, sim.shield_gateway) if sgw == gw]
                         for gw in self.gateway_ids}
        self._channel_gw = np.array([gw_index[sim.shield_gateway[s]] for s in sim.channel_shield.tolist()],
                                    dtype=np.intp)
        self.reset()

    def reset(self) -> None:
        G = len(self.gateway_ids)
        self._last_t: list[float | None] = [None] * G
        self._missing = np.zeros(G, dtype=np.int64)
        self._invalid = np.zeros(G, dtype=np.int64)
        self._recent: deque[tuple[float, np.ndarray]] = deque()

    def update(self, t: float, quality: Sequence[str]) -> None:
        """Record the channel qualities (channel order of the simulator) of the sample at ``t``."""
        q = np.asarray(quality, dtype=object)
        if q.shape != self._channel_gw.shape:
            raise ValueError(f"quality must have {len(self._channel_gw)} entries, got {q.size}")
        G = len(self.gateway_ids)
        n_ok = np.bincount(self._channel_gw, weights=(q == QUALITY_OK), minlength=G).astype(np.int64)
        self._missing += np.bincount(self._channel_gw, weights=(q == QUALITY_MISSING), minlength=G).astype(np.int64)
        self._invalid += np.bincount(self._channel_gw, weights=(q == QUALITY_INVALID), minlength=G).astype(np.int64)
        for g in np.flatnonzero(n_ok).tolist():
            self._last_t[g] = float(t)
        self._recent.append((float(t), n_ok))
        while self._recent and self._recent[0][0] <= t - THROUGHPUT_WINDOW_S:
            self._recent.popleft()

    def status(self, t: float) -> list[dict[str, Any]]:
        """Contract ``devices`` list evaluated at time ``t``."""
        valid = np.zeros(len(self.gateway_ids), dtype=np.int64)
        for ts, n_ok in self._recent:
            if t - THROUGHPUT_WINDOW_S < ts <= t:
                valid += n_ok
        out = []
        for g, gw in enumerate(self.gateway_ids):
            last = self._last_t[g]
            if last is None:
                age, status = None, "disconnected"
            else:
                age = float(t) - last
                if age < STALE_AFTER_S:
                    status = "simulated"
                elif age <= DISCONNECTED_AFTER_S:
                    status = "stale"
                else:
                    status = "disconnected"
            out.append({
                "id": gw,
                "shields": list(self._shields[gw]),
                "status": status,
                "last_t": last,
                "age_s": age,
                "missing": int(self._missing[g]),
                "invalid": int(self._invalid[g]),
                "throughput_rps": float(valid[g]) / THROUGHPUT_WINDOW_S,
            })
        return out
