"""Engineering rules and baseline detectors (MODEL_SPEC section 5).

* ``RuleEngine`` evaluates the ten hybrid rules on a ``FeatureFrame`` with stateful persistence
  counters (a counter resets as soon as its condition is false). A trigger is emitted on every
  sample while its condition, including persistence, holds.
* ``FixedThreshold`` is the fixed-threshold baseline (high_pressure + low_pressure only, same
  persistence). It is the stateful form of the spec's ``fixed_threshold_triggers(ff)``.
* ``RollingMAD`` is the rolling robust-z baseline (``mad_anomaly``).

Persistence durations given in seconds are converted to samples with ``cfg["simulator"]["dt_s"]``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .features import FEATURE_INDEX, MAD_TO_SD, FeatureFrame, nanmedian_cols

RULE_CATEGORY: dict[str, str] = {
    "high_pressure": "process",
    "low_pressure": "process",
    "rapid_rise": "process",
    "rapid_drop": "process",
    "leg_imbalance": "process",
    "persistent_deviation": "process",
    "sensor_stuck": "sensor",
    "sensor_spike": "sensor",
    "data_missing": "sensor",
    "telemetry_loss": "telemetry",
    "mad_anomaly": "process",
    "ai_anomaly": "process",
}

RULE_DESCRIPTIONS: dict[str, str] = {
    "high_pressure": "Leg pressure above the illustrative high-pressure limit",
    "low_pressure": "Leg pressure below the low-pressure limit while loaded",
    "rapid_rise": "5-s pressure gradient",
    "rapid_drop": "30-s robust pressure gradient (median-based)",
    "leg_imbalance": "Leg pressure minus mean of the other legs, as a fraction of the shield mean",
    "persistent_deviation": "Deviation from the leg's own baseline in the current loaded run",
    "sensor_stuck": "Consecutive identical readings",
    "sensor_spike": "Single-sample jump from the recent median, returning on the next sample",
    "data_missing": "Consecutive missing samples",
    "telemetry_loss": "Every channel of the gateway missing",
    "mad_anomaly": "Robust z-score against the rolling median (MAD scale)",
    "ai_anomaly": "Normalised Isolation Forest score (1.0 = calibrated threshold)",
}

HYBRID_RULES: tuple[str, ...] = (
    "high_pressure", "low_pressure", "rapid_rise", "rapid_drop", "leg_imbalance",
    "persistent_deviation", "sensor_stuck", "sensor_spike", "data_missing", "telemetry_loss",
)

PERSISTENT_RULES: tuple[str, ...] = (
    "high_pressure", "low_pressure", "rapid_rise", "rapid_drop", "leg_imbalance", "persistent_deviation",
)
# Feature column observed by each persistent rule (same order as PERSISTENT_RULES).
_P_COLS = [FEATURE_INDEX[n] for n in ("pressure_bar", "pressure_bar", "slope_5_bar_s", "slope_30_bar_s",
                                      "imbalance_frac", "dev_own_bar")]


@dataclass
class Trigger:
    """One rule firing on one sample (``channel_idx`` is None for shield-level rules)."""

    rule: str
    category: str
    shield_idx: int
    channel_idx: int | None
    t: float
    observed: float
    threshold: float
    unit: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _samples(cfg: dict, seconds: float) -> int:
    dt = float(cfg.get("simulator", {}).get("dt_s", 1.0))
    return max(1, int(round(float(seconds) / dt)))


def _make(rule: str, shield: int, channel: int | None, t: float, observed: float,
          threshold: float, unit: str) -> Trigger:
    return Trigger(rule=rule, category=RULE_CATEGORY[rule], shield_idx=int(shield),
                   channel_idx=None if channel is None else int(channel), t=float(t),
                   observed=float(observed), threshold=float(threshold), unit=unit,
                   description=RULE_DESCRIPTIONS[rule])


class RuleEngine:
    """Hybrid engineering rules (table in MODEL_SPEC section 5).

    ``rules`` optionally restricts evaluation to a subset of rule names (used by baselines).
    ``evaluate(ff, gateway_of_shield)`` accepts a callable ``shield_idx -> gateway id``, a
    sequence of gateway ids indexed by shield, or None (gateways of
    ``cfg["simulator"]["shields_per_gateway"]`` consecutive shields).
    """

    def __init__(self, n_channels: int, legs: int, cfg: dict, rules: Sequence[str] | None = None):
        rc = cfg["rules"]
        self.C = int(n_channels)
        self.L = int(legs)
        if self.C % self.L:
            raise ValueError("n_channels must be a multiple of legs")
        self.S = self.C // self.L
        self.dt = float(cfg.get("simulator", {}).get("dt_s", 1.0))
        names = tuple(rules) if rules is not None else HYBRID_RULES
        unknown = [r for r in names if r not in HYBRID_RULES]
        if unknown:
            raise ValueError(f"unknown rule(s): {unknown}")
        self.enabled = frozenset(names)

        self.high_bar = float(rc["high_pressure_bar"])
        self.low_bar = float(rc["low_pressure_bar"])
        self.rise_bar_s = float(rc["rapid_rise_bar_s"])
        self.drop_bar_s = float(rc["rapid_drop_bar_s"])
        self.imb_frac = float(rc["imbalance_frac"])
        self.dev_bar = float(rc["deviation_bar"])
        self.n_pressure = _samples(cfg, rc["pressure_persist_s"])
        self.n_imb = _samples(cfg, rc["imbalance_persist_s"])
        self.n_dev = _samples(cfg, rc["deviation_persist_s"])
        self.stuck_run = int(rc["stuck_run"])
        self.spike_jump_bar = float(rc["spike_jump_bar"])
        self.n_missing = _samples(cfg, rc["missing_persist_s"])
        self.shields_per_gateway = int(cfg.get("simulator", {}).get("shields_per_gateway", self.S))
        self._gw_labels: list | None = None
        self._gw_channels: list[np.ndarray] = []
        self._gw_shields: list[np.ndarray] = []
        self._gw_of_channel = np.zeros(self.C, dtype=np.intp)

        # Persistence-gated process rules: one counter row each, evaluated as a (6, C) matrix.
        self._p_rules = PERSISTENT_RULES
        self._p_need = np.array([self.n_pressure, self.n_pressure, self.n_pressure, self.n_pressure,
                                 self.n_imb, self.n_dev], dtype=np.int64)[:, None]
        self._p_enabled = np.array([r in self.enabled for r in PERSISTENT_RULES])[:, None]
        self._p_thr = (self.high_bar, self.low_bar, self.rise_bar_s, self.drop_bar_s, self.imb_frac, self.dev_bar)
        self._p_unit = ("bar", "bar", "bar/s", "bar/s", "fraction", "bar")
        self.reset()

    def reset(self) -> None:
        self._cnt = np.zeros((len(PERSISTENT_RULES), self.C), dtype=np.int64)

    def _gateways(self, gateway_of_shield: Callable[[int], Any] | Sequence | None) -> None:
        if gateway_of_shield is None:
            labels = [i // self.shields_per_gateway for i in range(self.S)]
        elif callable(gateway_of_shield):
            labels = [gateway_of_shield(i) for i in range(self.S)]
        else:
            labels = list(gateway_of_shield)
            if len(labels) != self.S:
                raise ValueError("gateway_of_shield sequence must have one entry per shield")
        if labels == self._gw_labels:
            return
        self._gw_labels = labels
        order = list(dict.fromkeys(labels))
        shields = np.arange(self.S)
        self._gw_shields = [shields[[lab == g for lab in labels]] for g in order]
        self._gw_channels = [(s[:, None] * self.L + np.arange(self.L)[None, :]).reshape(-1)
                             for s in self._gw_shields]
        for gi, ch in enumerate(self._gw_channels):
            self._gw_of_channel[ch] = gi

    # --------------------------------------------------------------- evaluate
    def evaluate(self, ff: FeatureFrame, gateway_of_shield=None) -> list[Trigger]:
        t = ff.t
        X = ff.X
        L = self.L
        en = self.enabled
        out: list[Trigger] = []

        # Process rules with persistence (counters reset as soon as the condition is false).
        obs = X[:, _P_COLS].T                       # (6, C): v, v, slope_5, slope_30, imbalance, dev_own
        rise_ok, loaded_ok = ff.rise_ok, ff.loaded_ok
        with np.errstate(invalid="ignore"):
            cond = np.empty((len(PERSISTENT_RULES), self.C), dtype=bool)
            cond[0] = rise_ok & (obs[0] > self.high_bar)
            cond[1] = loaded_ok & (obs[1] < self.low_bar)
            cond[2] = rise_ok & (obs[2] > self.rise_bar_s)
            cond[3] = loaded_ok & (obs[3] < self.drop_bar_s)
            cond[4] = loaded_ok & (np.abs(obs[4]) > self.imb_frac)
            cond[5] = loaded_ok & (np.abs(obs[5]) > self.dev_bar)
        cond &= self._p_enabled
        self._cnt = (self._cnt + 1) * cond
        fire = self._cnt >= self._p_need
        if fire.any():
            for r, c in zip(*np.nonzero(fire)):
                out.append(_make(self._p_rules[r], c // L, c, t, obs[r, c], self._p_thr[r], self._p_unit[r]))

        if "sensor_stuck" in en:
            for c in np.flatnonzero(ff.flat_run >= self.stuck_run - 1):
                out.append(_make("sensor_stuck", c // L, c, t, ff.flat_run[c] + 1, self.stuck_run, "samples"))
        if "sensor_spike" in en:
            for c in np.flatnonzero(ff.spike):
                out.append(_make("sensor_spike", c // L, c, t, ff.spike_jump[c], self.spike_jump_bar, "bar"))

        if "data_missing" in en or "telemetry_loss" in en:
            mr = ff.missing_run
            if not mr.any():
                return out
            self._gateways(gateway_of_shield)
            gw_missing = np.zeros(self.C, dtype=bool)
            for g, ch in enumerate(self._gw_channels):
                mrg = mr[ch]
                if (mrg >= 1).all():          # whole gateway currently missing
                    gw_missing[ch] = True
                    if "telemetry_loss" in en and (mrg >= self.n_missing).all():
                        obs = float(mrg.min()) * self.dt
                        for s in self._gw_shields[g]:
                            out.append(_make("telemetry_loss", s, None, t, obs, self.n_missing * self.dt, "s"))
            if "data_missing" in en:
                for c in np.flatnonzero((mr >= self.n_missing) & ~gw_missing):
                    out.append(_make("data_missing", c // L, c, t, mr[c] * self.dt,
                                     self.n_missing * self.dt, "s"))
        return out


class FixedThreshold:
    """Fixed-threshold baseline: only high_pressure and low_pressure (same persistence/gating).

    Stateful because of the persistence counters; ``evaluate(ff)`` is the spec's
    ``fixed_threshold_triggers(ff)``.
    """

    RULES = ("high_pressure", "low_pressure")

    def __init__(self, n_channels: int, legs: int, cfg: dict):
        self._engine = RuleEngine(n_channels, legs, cfg, rules=self.RULES)

    def evaluate(self, ff: FeatureFrame, gateway_of_shield=None) -> list[Trigger]:
        return self._engine.evaluate(ff, gateway_of_shield)

    def reset(self) -> None:
        self._engine.reset()


class RollingMAD:
    """Rolling robust-z baseline (``mad_anomaly``, category process).

    z = (v - median(W)) / (1.4826 * MAD(W) + 0.5) where W = valid values of the previous
    ``mad_window_s`` samples (h[1..60], excluding v). Needs >= half the window (30 of 60) valid.
    Triggers when loaded_ok and |z| > mad_z for >= mad_persist_s consecutive samples.
    """

    Z_FLOOR_BAR = 0.5  # additive scale floor from the spec formula

    def __init__(self, n_channels: int, legs: int, cfg: dict):
        rc = cfg["rules"]
        self.C = int(n_channels)
        self.L = int(legs)
        self.window = _samples(cfg, rc["mad_window_s"])
        self.min_valid = max(1, self.window // 2)
        self.mad_z = float(rc["mad_z"])
        self.n_persist = _samples(cfg, rc["mad_persist_s"])
        self.H = self.window + 1
        self.reset()

    def reset(self) -> None:
        self._buf = np.full((2 * self.H, self.C), np.nan)
        self._pos = -1
        self._cnt = np.zeros(self.C, dtype=np.int64)

    def evaluate(self, ff: FeatureFrame, gateway_of_shield=None) -> list[Trigger]:
        v = ff.values
        self._pos = (self._pos + 1) % self.H
        self._buf[self._pos] = v
        self._buf[self._pos + self.H] = v
        p = self._pos + self.H
        W = self._buf[p - self.window: p]  # h[window..1]
        med, n = nanmedian_cols(W)
        mad, _ = nanmedian_cols(np.abs(W - med))
        with np.errstate(invalid="ignore"):
            z = (v - med) / (MAD_TO_SD * mad + self.Z_FLOOR_BAR)
            cond = ff.loaded_ok & (n >= self.min_valid) & (np.abs(z) > self.mad_z)
        self._cnt = np.where(cond, self._cnt + 1, 0)
        out = []
        for c in np.flatnonzero(self._cnt >= self.n_persist):
            out.append(_make("mad_anomaly", c // self.L, c, ff.t, z[c], self.mad_z, "z"))
        return out
