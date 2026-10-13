"""Longwall face simulator: shield cycles, roof load, leg pressures and injected events (spec section 2)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import numpy as np

# Event catalogue, identical (keys, values, order) to /api/config.event_types in docs/API_CONTRACT.md.
EVENT_TYPES: list[dict[str, Any]] = [
    {"type": "gradual_rise", "label": "Gradual pressure increase", "category": "process", "scope": "shield",
     "default_magnitude": 70, "magnitude_unit": "bar", "default_duration_s": 300},
    {"type": "pressure_spike", "label": "Sudden pressure spike", "category": "process", "scope": "shield",
     "default_magnitude": 90, "magnitude_unit": "bar", "default_duration_s": 40},
    {"type": "leakage", "label": "Leakage-like pressure drop", "category": "process", "scope": "channel",
     "default_magnitude": 0.8, "magnitude_unit": "bar/s", "default_duration_s": 150},
    {"type": "leg_imbalance", "label": "Uneven leg loading", "category": "process", "scope": "shield",
     "default_magnitude": 25, "magnitude_unit": "%", "default_duration_s": 200},
    {"type": "multi_shield_loading", "label": "Localized multi-shield loading", "category": "process", "scope": "block",
     "default_magnitude": 60, "magnitude_unit": "bar", "default_duration_s": 300},
    {"type": "sensor_spike", "label": "Isolated sensor spike", "category": "sensor", "scope": "channel",
     "default_magnitude": 150, "magnitude_unit": "bar", "default_duration_s": 1},
    {"type": "sensor_stuck", "label": "Sensor stuck at constant value", "category": "sensor", "scope": "channel",
     "default_magnitude": None, "magnitude_unit": None, "default_duration_s": 120},
    {"type": "sensor_dropout", "label": "Missing sensor readings", "category": "sensor", "scope": "channel",
     "default_magnitude": None, "magnitude_unit": None, "default_duration_s": 60},
    {"type": "telemetry_interruption", "label": "Gateway telemetry interruption", "category": "telemetry",
     "scope": "gateway", "default_magnitude": None, "magnitude_unit": None, "default_duration_s": 45},
]

# Hidden evaluation-only type: a legitimate (normal) load transient, never a ground-truth event.
BENIGN_TRANSIENT: dict[str, Any] = {
    "type": "benign_transient", "label": "Benign load transient", "category": "normal", "scope": "shield",
    "default_magnitude": 70, "magnitude_unit": "bar", "default_duration_s": 60,
}

_CATALOG: dict[str, dict[str, Any]] = {e["type"]: e for e in EVENT_TYPES}
_CATALOG[BENIGN_TRANSIENT["type"]] = BENIGN_TRANSIENT

STATE_ADVANCING = "advancing"
STATE_SETTING = "setting"
STATE_LOADED = "loaded"
_STATE_NAMES = np.array([STATE_ADVANCING, STATE_SETTING, STATE_LOADED], dtype=object)

# Fixed shape constants of the event-effect model (spec 2.7; not part of the config).
PRESSURE_SPIKE_TAU_DIVISOR = 4.0        # pressure_spike decays with tau = d / 4
MULTI_SHIELD_RAMP_S = 60.0              # multi_shield_loading ramp time
MULTI_SHIELD_WEIGHTS = (1.0, 0.7, 0.35)  # weight by |shield - centre| = 0, 1, 2
IMBALANCE_RAMP_S = 120.0                # leg_imbalance ramp time (develops gradually)
IMBALANCE_PERCENT_DIVISOR = 200.0       # m % total split -> +/- m/2 % per leg group
BENIGN_TAU_S = 15.0                     # benign_transient decay
MIN_DURATION_S = 1.0
MAX_DURATION_S = 3600.0
NORMAL_LABEL = "normal"


@dataclass
class Event:
    """An injected event (synthetic ground truth). ``active(t)`` is ``onset_t <= t < end_t``."""

    id: str
    type: str
    label: str
    category: str
    onset_t: float
    end_t: float
    shields: list[str]
    channels: list[str]
    magnitude: float | None
    magnitude_unit: str | None
    scope: str
    hidden: bool = False
    # Internal indices used by the simulator.
    shield_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.intp), repr=False, compare=False)
    channel_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.intp), repr=False, compare=False)
    shield_weights: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False, compare=False)
    stuck_value: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def duration_s(self) -> float:
        return self.end_t - self.onset_t

    def active(self, t: float) -> bool:
        return self.onset_t <= t < self.end_t

    def to_dict(self, t: float | None = None) -> dict[str, Any]:
        """Contract EventRecord; ``active`` is evaluated at ``t`` (False when ``t`` is None)."""
        return {
            "id": self.id,
            "type": self.type,
            "label": self.label,
            "category": self.category,
            "onset_t": float(self.onset_t),
            "end_t": float(self.end_t),
            "shields": list(self.shields),
            "channels": list(self.channels),
            "magnitude": None if self.magnitude is None else float(self.magnitude),
            "magnitude_unit": self.magnitude_unit,
            "active": False if t is None else self.active(t),
        }


@dataclass
class Frame:
    """One simulated sample of the whole face at time ``t``."""

    t: float
    values: np.ndarray        # (C,) measured bar, NaN = missing
    true_values: np.ndarray   # (C,) noise-free hydraulic pressure (debug/tests only)
    states: list[str]         # (S,) advancing / setting / loaded
    time_in_state: np.ndarray  # (S,) seconds
    gt_label: list[str]       # (C,)
    gt_event_id: list         # (C,) str | None
    seq_start: int = 0        # seq of the first channel reading of this frame


def _is_int(x: Any) -> bool:
    return isinstance(x, Integral) and not isinstance(x, bool)


def _is_finite_real(x: Any) -> bool:
    return isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(float(x))


class FaceSimulator:
    """Simulates leg pressures of ``n_shields`` powered supports with ``legs_per_shield`` legs each."""

    def __init__(self, cfg: dict, seed: int, n_shields: int = 12, legs_per_shield: int = 2,
                 noise_bar: float = 2.0, load_trend_bar_per_h: float = 0.0) -> None:
        if not _is_int(seed) or seed < 0:
            raise ValueError(f"seed must be an integer >= 0, got {seed!r}")
        if not _is_int(n_shields) or n_shields < 1:
            raise ValueError(f"n_shields must be an integer >= 1, got {n_shields!r}")
        if not _is_int(legs_per_shield) or legs_per_shield not in (2, 4):
            raise ValueError(f"legs_per_shield must be 2 or 4, got {legs_per_shield!r}")
        if not _is_finite_real(noise_bar) or noise_bar < 0:
            raise ValueError(f"noise_bar must be a finite number >= 0, got {noise_bar!r}")
        if not _is_finite_real(load_trend_bar_per_h):
            raise ValueError(f"load_trend_bar_per_h must be a finite number, got {load_trend_bar_per_h!r}")

        self.cfg = cfg
        sc = cfg["simulator"]
        lim = cfg["limits"]

        self.seed = int(seed)
        self.n_shields = S = int(n_shields)
        self.legs = self.legs_per_shield = L = int(legs_per_shield)
        self.n_channels = C = S * L
        self.noise_bar = float(noise_bar)
        self.load_trend_bar_per_h = float(load_trend_bar_per_h)
        self.dt_s = float(sc["dt_s"])
        self.start_timestamp: str = sc["start_timestamp"]

        # Parameters.
        self._lowered = float(sc["lowered_bar"])
        self._roof = float(sc["roof_load_bar"])
        self._cycle_s = float(sc["cycle_s"])
        self._advance_s = float(sc["advance_s"])
        self._setting_s = float(sc["setting_s"])
        self._tau_load = float(sc["tau_load_s"])
        self._rec = float(sc["recovery_tau_s"])
        self._q = float(sc["quantization_bar"])
        inv_q = 1.0 / self._q
        # Divide by an exact integer 1/q where possible so values are the nearest doubles to the decimals.
        self._q_inv = float(round(inv_q)) if abs(inv_q - round(inv_q)) < 1e-9 else None
        self._trans_prob = float(sc["transient_rate_per_h"]) * self.dt_s / 3600.0
        self._trans_lo, self._trans_hi = (float(v) for v in sc["transient_amp_bar"])
        self._trans_decay = math.exp(-self.dt_s / float(sc["transient_tau_s"]))
        self._hyd_alpha = 1.0 - math.exp(-self.dt_s / float(sc["tau_hyd_s"]))
        self._spg = int(sc["shields_per_gateway"])
        self._setting_bar = float(lim["setting_bar"])
        self._yield = float(lim["yield_bar"])
        self._sensor_min = float(lim["sensor_min_bar"])
        self._sensor_max = float(lim["sensor_max_bar"])

        # Indexing (2.1).
        self.shield_ids: list[str] = [f"S{i + 1:02d}" for i in range(S)]
        self.channel_ids: list[str] = [f"{sid}-L{j + 1}" for sid in self.shield_ids for j in range(L)]
        self.channel_shield: np.ndarray = np.arange(C, dtype=np.intp) // L
        self.shield_gateway: list[str] = [self.gateway_of(i) for i in range(S)]
        self.gateway_ids: list[str] = list(dict.fromkeys(self.shield_gateway))
        self._shield_index = {sid: i for i, sid in enumerate(self.shield_ids)}
        self._channel_index = {cid: c for c, cid in enumerate(self.channel_ids)}
        self._leg_sign = np.where(np.arange(L) % 2 == 0, 1.0, -1.0)  # +1 for L1/L3, -1 for L2/L4

        # Random streams (2.2) and per-shield setup (2.3), drawn in spec order.
        rng_setup = np.random.default_rng(self.seed)
        self._rng_noise = np.random.default_rng(self.seed + 1)
        self._rng_trans = np.random.default_rng(self.seed + 2)
        i = np.arange(S)
        phi = rng_setup.uniform(0.0, 2.0 * np.pi)
        amp = float(sc["spatial_amplitude"])
        jit = float(sc["spatial_jitter"])
        self.k: np.ndarray = 1.0 + amp * np.sin(2.0 * np.pi * (i + 0.5) / S + phi) + rng_setup.uniform(-jit, jit, S)
        clip = float(sc["leg_factor_clip"])
        self.leg_factor: np.ndarray = np.clip(1.0 + rng_setup.normal(0.0, float(sc["leg_factor_sd"]), (S, L)),
                                              1.0 - clip, 1.0 + clip)
        self.offset_s: np.ndarray = i * self._cycle_s / S

        # Dynamic state.
        self.t: float = 0.0
        self.seq: int = 0
        self.samples: int = 0
        self.events: list[Event] = []
        self._all_events: list[Event] = []
        self._n_visible = 0
        self._n_hidden = 0
        self._T = np.zeros(S)
        self._p_true: np.ndarray | None = None
        self._last_values: np.ndarray | None = None

    # ------------------------------------------------------------------ indexing
    def gateway_of(self, shield_idx: int) -> str:
        """Gateway id of a 0-based shield index."""
        return "GW-%02d" % (shield_idx // self._spg + 1)

    # ------------------------------------------------------------------ events
    def inject(self, type: str, shield_id: str, channel_id: str | None = None, magnitude: float | None = None,
               duration_s: float | None = None, onset_t: float | None = None) -> Event:
        """Schedule an event (spec 2.7). Raises ``ValueError`` on invalid arguments."""
        event_type = type
        if not isinstance(event_type, str) or event_type not in _CATALOG:
            valid = ", ".join(e["type"] for e in EVENT_TYPES)
            raise ValueError(f"unknown event type {event_type!r}; valid types: {valid}")
        spec = _CATALOG[event_type]
        hidden = spec is BENIGN_TRANSIENT

        if not isinstance(shield_id, str) or shield_id not in self._shield_index:
            raise ValueError(f"unknown shield {shield_id!r}; valid shields: "
                             f"{self.shield_ids[0]}..{self.shield_ids[-1]}")
        si = self._shield_index[shield_id]

        # Magnitude: required > 0 for types that have one, ignored otherwise.
        if spec["default_magnitude"] is None:
            mag = None
        else:
            mag = spec["default_magnitude"] if magnitude is None else magnitude
            if not _is_finite_real(mag) or mag <= 0:
                raise ValueError(f"magnitude for {event_type} must be a finite number > 0 "
                                 f"({spec['magnitude_unit']}), got {magnitude!r}")
            mag = float(mag)

        # Duration: 1..3600 s; sensor_spike is a single-sample event with a fixed 1 s duration.
        dur = spec["default_duration_s"] if duration_s is None else duration_s
        if not _is_finite_real(dur) or not MIN_DURATION_S <= dur <= MAX_DURATION_S:
            raise ValueError(f"duration_s must be a number between {MIN_DURATION_S:g} and {MAX_DURATION_S:g} s, "
                             f"got {duration_s!r}")
        dur = float(spec["default_duration_s"]) if event_type == "sensor_spike" else float(dur)

        if onset_t is None:
            onset = self.t
        else:
            if not _is_finite_real(onset_t):
                raise ValueError(f"onset_t must be a finite number, got {onset_t!r}")
            onset = float(onset_t)
            if onset < self.t:
                raise ValueError(f"onset_t {onset:g} is in the past (next sample is t={self.t:g})")

        # Scope resolution.
        scope = spec["scope"]
        weights = None
        if scope == "channel":
            if channel_id is None:
                channel_id = f"{shield_id}-L1"
            if not isinstance(channel_id, str) or channel_id not in self._channel_index:
                legs = ", ".join(f"{shield_id}-L{j + 1}" for j in range(self.legs))
                raise ValueError(f"unknown channel {channel_id!r}; valid channels for {shield_id}: {legs}")
            ci = self._channel_index[channel_id]
            if self.channel_shield[ci] != si:
                raise ValueError(f"channel {channel_id!r} does not belong to shield {shield_id!r}")
            shield_idx = np.array([si], dtype=np.intp)
            channel_idx = np.array([ci], dtype=np.intp)
        else:
            if scope == "shield":
                shield_idx = np.array([si], dtype=np.intp)
            elif scope == "block":
                shield_idx = np.arange(max(0, si - 2), min(self.n_shields - 1, si + 2) + 1, dtype=np.intp)
                weights = np.array([MULTI_SHIELD_WEIGHTS[abs(int(n) - si)] for n in shield_idx])
            elif scope == "gateway":
                gw = self.shield_gateway[si]
                shield_idx = np.array([n for n in range(self.n_shields) if self.shield_gateway[n] == gw],
                                      dtype=np.intp)
            else:  # pragma: no cover - catalogue is static
                raise ValueError(f"unsupported scope {scope!r}")
            channel_idx = (shield_idx[:, None] * self.legs + np.arange(self.legs)).ravel()
        if weights is None:
            weights = np.ones(len(shield_idx))

        if hidden:
            self._n_hidden += 1
            ev_id = "BT-%03d" % self._n_hidden
        else:
            self._n_visible += 1
            ev_id = "EV-%03d" % self._n_visible
        ev = Event(
            id=ev_id, type=event_type, label=spec["label"], category=spec["category"],
            onset_t=onset, end_t=onset + dur,
            shields=[self.shield_ids[n] for n in shield_idx],
            channels=[self.channel_ids[c] for c in channel_idx],
            magnitude=mag, magnitude_unit=spec["magnitude_unit"], scope=scope, hidden=hidden,
            shield_idx=shield_idx, channel_idx=channel_idx, shield_weights=weights,
        )
        self._all_events.append(ev)
        if not hidden:
            self.events.append(ev)
        return ev

    def _process_effects(self, t: float) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, list[Event]]:
        """Event load E (S,), leg factor (S,L) or None, leak loss (S,L) or None, and active sensor events."""
        S, L = self.n_shields, self.legs
        E = np.zeros(S)
        imb_dev: np.ndarray | None = None
        loss: np.ndarray | None = None
        sensor: list[Event] = []
        rec = self._rec
        for ev in self._all_events:
            if t < ev.onset_t:
                continue
            active = t < ev.end_t
            r = t - ev.onset_t
            d = ev.duration_s
            m = ev.magnitude
            typ = ev.type
            if ev.category in ("sensor", "telemetry"):
                if active:
                    sensor.append(ev)
            elif typ == "gradual_rise":
                E[ev.shield_idx] += m * min(r / d, 1.0) if active else m * math.exp(-(t - ev.end_t) / rec)
            elif typ == "pressure_spike":
                if active:
                    E[ev.shield_idx] += m * math.exp(-r / (d / PRESSURE_SPIKE_TAU_DIVISOR))
            elif typ == "multi_shield_loading":
                amount = m * min(r / MULTI_SHIELD_RAMP_S, 1.0) if active else m * math.exp(-(t - ev.end_t) / rec)
                E[ev.shield_idx] += ev.shield_weights * amount
            elif typ == "benign_transient":
                if active:
                    E[ev.shield_idx] += m * math.exp(-r / BENIGN_TAU_S)
            elif typ == "leakage":
                if loss is None:
                    loss = np.zeros(S * L)
                loss[ev.channel_idx] += m * r if active else m * d * math.exp(-(t - ev.end_t) / rec)
            elif typ == "leg_imbalance":
                if imb_dev is None:
                    imb_dev = np.zeros((S, L))
                frac = (m / IMBALANCE_PERCENT_DIVISOR) * min((r if active else d) / IMBALANCE_RAMP_S, 1.0)
                if not active:
                    frac *= math.exp(-(t - ev.end_t) / rec)
                imb_dev[ev.shield_idx, :] -= self._leg_sign * frac
        factor = None if imb_dev is None else 1.0 + imb_dev
        return E, factor, (None if loss is None else loss.reshape(S, L)), sensor

    # ------------------------------------------------------------------ stepping
    def step(self) -> Frame:
        """Produce the sample at ``self.t`` and advance ``self.t`` by ``dt_s``."""
        t = self.t
        S, L, C = self.n_shields, self.legs, self.n_channels

        # Operating cycle (2.4).
        c = np.mod(t - self.offset_s, self._cycle_s)
        advancing = c < self._advance_s
        setting = ~advancing & (c < self._advance_s + self._setting_s)
        tau = c - self._advance_s
        time_in_state = np.where(advancing, c, np.where(setting, tau, tau - self._setting_s))
        state_idx = np.where(advancing, 0, np.where(setting, 1, 2))

        # Normal transients (2.5): all draws happen every step regardless of events.
        self._T *= self._trans_decay
        u = self._rng_trans.random(S)
        a = self._rng_trans.uniform(self._trans_lo, self._trans_hi, S)
        sgn = self._rng_trans.choice([-1, 1], S)
        self._T += np.where(u < self._trans_prob, sgn * a, 0.0)

        # Roof load and target pressure (2.5, 2.6).
        E, imb_factor, loss, sensor_events = self._process_effects(t)
        load = self._roof * self.k + self.load_trend_bar_per_h * t / 3600.0 + self._T + E
        growth = 1.0 - np.exp(-np.maximum(tau, 0.0) / self._tau_load)
        target = self._setting_bar + np.maximum(0.0, load[:, None] * self.leg_factor - self._setting_bar) \
            * growth[:, None]
        if imb_factor is not None:
            target = target * imb_factor
        if loss is not None:
            target = target - loss
        target[advancing, :] = self._lowered
        target = np.clip(target, self._lowered, self._yield).ravel()

        # First-order hydraulic response.
        if self._p_true is None:
            p = target
        else:
            p = self._p_true + (target - self._p_true) * self._hyd_alpha
            # Mathematically a no-op (convex combination); guards the range invariant against rounding.
            p = np.clip(p, self._lowered, self._yield)
        self._p_true = p

        # Measurement: noise, quantisation, sensor range.
        noisy = (p + self._rng_noise.normal(0.0, self.noise_bar, C)) / self._q
        m = np.round(noisy) / self._q_inv if self._q_inv is not None else np.round(noisy) * self._q
        np.clip(m, self._sensor_min, self._sensor_max, out=m)

        # Sensor / telemetry effects: additive spikes, then stuck values, then missing data.
        if sensor_events:
            for ev in sensor_events:
                if ev.type == "sensor_spike" and t < ev.onset_t + self.dt_s:
                    m[ev.channel_idx] += ev.magnitude
            np.clip(m, self._sensor_min, self._sensor_max, out=m)
            for ev in sensor_events:
                if ev.type == "sensor_stuck":
                    if ev.stuck_value is None:
                        prev = self._last_values[ev.channel_idx] if self._last_values is not None \
                            else np.full(len(ev.channel_idx), np.nan)
                        ev.stuck_value = np.where(np.isnan(prev), m[ev.channel_idx], prev)
                    m[ev.channel_idx] = ev.stuck_value
            for ev in sensor_events:
                if ev.type in ("sensor_dropout", "telemetry_interruption"):
                    m[ev.channel_idx] = np.nan

        # Ground-truth labels (2.8): the most recently injected active event covering the channel wins.
        gt_label = [NORMAL_LABEL] * C
        gt_event_id: list[str | None] = [None] * C
        for ev in self.events:
            if ev.onset_t <= t < ev.end_t:
                for ci in ev.channel_idx.tolist():
                    gt_label[ci] = ev.type
                    gt_event_id[ci] = ev.id

        frame = Frame(
            t=t, values=m, true_values=p.copy(), states=_STATE_NAMES[state_idx].tolist(),
            time_in_state=time_in_state, gt_label=gt_label, gt_event_id=gt_event_id, seq_start=self.seq,
        )
        self._last_values = m.copy()
        self.seq += C
        self.samples += 1
        self.t = self.samples * self.dt_s
        return frame
