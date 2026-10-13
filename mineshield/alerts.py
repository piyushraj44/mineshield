"""Alert correlation, prioritisation and explanation (MODEL_SPEC section 7).

``AlertCorrelator.update`` turns per-sample triggers into alerts:

1. sensor-first suppression (optional): process triggers on a channel with a sensor trigger in the
   last ``sensor_suppress_s`` are dropped; a shield with a telemetry trigger keeps only that one;
2. triggers are grouped by (category, shield) and attached to an active alert of the same category
   containing the shield (process: or a shield within ``adjacent_merge``; telemetry: same
   gateway), otherwise a new alert ``AL-0001``, ... is opened;
3. evidence, AI-score maximum and feature snapshots are tracked on every attach;
4. a process group that attaches to one active process alert while also lying within
   ``adjacent_merge`` of another active process alert merges the newer alert into the older one
   (the newer id stays resolvable through an alias map);
5. alerts close when ``t - last_t > close_after_s``.

The title comes from the highest-weight engineering rule (any rule other than ai_anomaly /
mad_anomaly) first seen within ``TITLE_WINDOW_S`` of the alert's FIRST engineering-rule trigger, so
later consequences (e.g. re-pressurisation after a leak) do not relabel the alert and an alert opened
by a statistical trigger takes its title from the first engineering evidence. Only an alert without
any engineering rule is titled from ai_anomaly / mad_anomaly.

``priority`` is a prioritisation score (0-100), never a probability; the AI score is a
normalised anomaly score where 1.0 is the calibrated threshold. All outputs are JSON-safe.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .features import FEATURE_INDEX, FEATURE_NAMES, FeatureFrame
from .rules import Trigger

RULE_WEIGHTS: dict[str, float] = {
    "high_pressure": 1.0,
    "telemetry_loss": 0.7,
    "rapid_rise": 0.9,
    "rapid_drop": 0.8,
    "low_pressure": 0.8,
    "leg_imbalance": 0.6,
    "persistent_deviation": 0.6,
    "mad_anomaly": 0.5,
    "ai_anomaly": 0.5,
    "sensor_stuck": 0.4,
    "data_missing": 0.4,
    "sensor_spike": 0.3,
}
_RULE_ORDER = {r: i for i, r in enumerate(RULE_WEIGHTS)}  # tie-break: spec listing order

RULE_TITLES: dict[str, str] = {
    "high_pressure": "High pressure (above illustrative limit)",
    "low_pressure": "Low pressure while loaded",
    "rapid_rise": "Rapid pressure rise",
    "rapid_drop": "Leak-like pressure decay",
    "leg_imbalance": "Leg pressure imbalance",
    "persistent_deviation": "Persistent deviation from shield baseline",
    "ai_anomaly": "AI-detected abnormal pressure pattern",
    "mad_anomaly": "Statistical outlier (rolling MAD)",
    "sensor_stuck": "Sensor stuck at constant value",
    "sensor_spike": "Isolated sensor spike",
    "data_missing": "Missing sensor data",
    "telemetry_loss": "Gateway telemetry loss",
}
MULTI_SHIELD_PREFIX = "Multi-shield: "
PRIORITY_FORMULA = "priority = 50*rule_weight + 30*min(ai_score_max/2,1) + 20*min((n_shields-1)/2,1)"
# The normalised Isolation Forest score saturates near ~1.9 (bounded path lengths), hence /2.
PRIORITY_AI_FULL_SCORE = 2.0
TITLE_WINDOW_S = 15.0   # title window, measured from the first engineering-rule trigger
STATISTICAL_RULES = frozenset({"ai_anomaly", "mad_anomaly"})   # not "engineering" rules for titles
FAULT_SENTENCE = ("The pattern is more consistent with a measurement/communication fault "
                  "than a change in roof load.")
AI_THRESHOLD = 1.0
SERIES_PRE_S = 120.0    # contract: series from opened_t - 120
SERIES_POST_S = 30.0    # ... to min(last_t + 30, now)
DQ_OK_BELOW = 0.05
DQ_DEGRADED_BELOW = 0.5
_CATEGORY_ORDER = {"telemetry": 0, "sensor": 1, "process": 2}
_MF = FEATURE_INDEX["missing_frac"]

_PROCESS_COMMON_STEPS = [
    "Verify shield hydraulic condition (leg seals, valves) and setting pressure",
    "Check neighbouring shields and face conditions with the deputy/shift engineer",
    "Confirm readings with a calibrated portable gauge before action",
]
_RULE_STEPS: dict[str, list[str]] = {
    "high_pressure": ["Check the yield-valve setting and the recent pressure history of the affected legs"],
    "low_pressure": ["Check that the shield is set firmly against the roof and reached setting pressure"],
    "rapid_rise": ["Review the pressure trend of the affected and adjacent shields over the last minutes"],
    "rapid_drop": ["Inspect the leg circuit for leakage (hoses, fittings, leg seals, check valve)"],
    "leg_imbalance": ["Compare the individual leg pressures and inspect canopy/base contact for uneven loading"],
    "persistent_deviation": ["Compare the current pressure with the shield's earlier loaded cycles"],
    "mad_anomaly": ["Review the pressure trend of the flagged channels"],
    "ai_anomaly": ["Review the pressure trend and feature values of the flagged channels"],
    "sensor_stuck": [
        "Compare with the other leg channel of the same shield",
        "Inspect transducer cabling/connector and acquisition input",
        "Perform a field zero/span check",
    ],
    "sensor_spike": [
        "Compare with the other leg channel of the same shield",
        "Inspect transducer cabling/connector and acquisition input",
        "Check whether the spike recurs before acting on this channel",
    ],
    "data_missing": [
        "Check transducer power and the channel's acquisition input",
        "Inspect transducer cabling/connector and acquisition input",
        "Compare with the other leg channel of the same shield",
    ],
    "telemetry_loss": [
        "Check gateway power supply and network link",
        "Confirm whether the other gateways are still reporting",
        "Obtain local pressure readings at the affected shields until telemetry is restored",
    ],
}


# ------------------------------------------------------------------ helpers
def fmt_t(t: float | None) -> str:
    """Simulated time as ``T+hh:mm:ss``."""
    if t is None or not math.isfinite(float(t)):
        return "T+--:--:--"
    s = int(round(float(t)))
    sign = "-" if s < 0 else "+"
    s = abs(s)
    return f"T{sign}{s // 3600:02d}:{(s // 60) % 60:02d}:{s % 60:02d}"


def _finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _num(x: Any, nd: int = 4) -> float | None:
    return round(float(x), nd) if _finite(x) else None


def _clean(obj: Any) -> Any:
    """Recursively convert numpy types to Python and NaN/inf to None (JSON-safe)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def compute_priority(max_rule_weight: float, ai_score_max: float | None, n_shields: int) -> int:
    """``round(50*w + 30*min(ai/2, 1) + 20*min((n_shields-1)/2, 1))``; NaN/None AI score -> 0."""
    ai = max(float(ai_score_max), 0.0) if _finite(ai_score_max) else 0.0
    n = max(int(n_shields) - 1, 0)
    return int(round(50.0 * float(max_rule_weight) + 30.0 * min(ai / PRIORITY_AI_FULL_SCORE, 1.0)
                     + 20.0 * min(n / 2.0, 1.0)))


def severity_for(priority: int) -> str:
    return "high" if priority >= 70 else "medium" if priority >= 45 else "low"


def _join(names: Sequence[str]) -> str:
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _join_more(names: Sequence[str], k: int) -> str:
    """Join at most ``k`` names, then "and N more"."""
    names = list(names)
    if len(names) <= k:
        return _join(names)
    return ", ".join(names[:k]) + f" and {len(names) - k} more"


class _Meta:
    """Normalised view of ``sim_meta`` (FaceSimulator or any object/dict with the same fields)."""

    def __init__(self, m: Any, shields_per_gateway: int):
        def get(name: str, default: Any = None) -> Any:
            if isinstance(m, dict):
                return m.get(name, default)
            return getattr(m, name, default)

        self.shield_ids: list[str] = list(get("shield_ids"))
        self.channel_ids: list[str] = list(get("channel_ids"))
        self.S, self.C = len(self.shield_ids), len(self.channel_ids)
        legs = get("legs", None) or get("legs_per_shield", None) or self.C // max(self.S, 1)
        self.legs = int(legs)
        cs = get("channel_shield", None)
        self.channel_shield = (np.asarray(cs, dtype=int) if cs is not None
                               else np.arange(self.C) // self.legs)
        gw = get("gateway_of", None)
        if gw is None:
            self.gateway = [f"GW-{i // shields_per_gateway + 1:02d}" for i in range(self.S)]
        elif callable(gw):
            self.gateway = [gw(i) for i in range(self.S)]
        else:
            self.gateway = list(gw)
        self.shield_channels: list[list[int]] = [[] for _ in range(self.S)]
        for c, s in enumerate(self.channel_shield):
            self.shield_channels[int(s)].append(c)


# -------------------------------------------------------------------- alert
class Alert:
    """One correlated alert. Sets hold shield / channel indices; see ``summary()`` for JSON."""

    def __init__(self, alert_id: str, category: str, t: float, meta: _Meta, gateway: str | None = None):
        self.id = alert_id
        self.category = category
        self.opened_t = float(t)
        self.last_t = float(t)
        self.active = True
        self.gateway = gateway
        self.shields: set[int] = set()
        self.channels: set[int] = set()
        self.rules: set[str] = set()
        self.rule_first_t: dict[str, float] = {}   # time each rule was first attached
        self.merged_ids: list[str] = []            # ids of alerts merged into this one
        self.evidence: dict[tuple[str, int | None], dict[str, Any]] = {}
        self.ai_score_max: float = math.nan
        self.ai_t_max: float | None = None
        self.ai_channel_max: int | None = None
        self.features: dict[int, np.ndarray] = {}
        self.features_t: float | None = None
        self.missing_frac: dict[int, float] = {}
        self.n_triggers = 0
        self._meta = meta

    @staticmethod
    def _best_rule(rules: Iterable[str]) -> str | None:
        rules = list(rules)
        if not rules:
            return None
        return max(rules, key=lambda r: (RULE_WEIGHTS.get(r, 0.0), -_RULE_ORDER.get(r, 99)))

    @property
    def top_rule(self) -> str | None:
        """Highest-weight rule over the whole alert (drives the priority's rule weight)."""
        return self._best_rule(self.rules)

    @property
    def title_rule(self) -> str | None:
        """Highest-weight engineering rule first seen within TITLE_WINDOW_S of the first
        engineering-rule trigger; ai_anomaly / mad_anomaly only when no engineering rule exists."""
        eng = {r: self.rule_first_t.get(r, self.opened_t) for r in self.rules if r not in STATISTICAL_RULES}
        if not eng:
            return self._best_rule(self.rules)
        t0 = min(eng.values())
        return self._best_rule(r for r, t in eng.items() if t <= t0 + TITLE_WINDOW_S)

    @property
    def max_rule_weight(self) -> float:
        return max((RULE_WEIGHTS.get(r, 0.0) for r in self.rules), default=0.0)

    @property
    def multi_shield(self) -> bool:
        return self.category == "process" and len(self.shields) >= 2

    @property
    def title(self) -> str:
        top = self.title_rule
        base = RULE_TITLES.get(top, top or "Alert")
        return MULTI_SHIELD_PREFIX + base if self.multi_shield else base

    @property
    def priority(self) -> int:
        return compute_priority(self.max_rule_weight, self.ai_score_max, len(self.shields))

    @property
    def severity(self) -> str:
        return severity_for(self.priority)

    def sorted_rules(self) -> list[str]:
        return sorted(self.rules, key=lambda r: (-RULE_WEIGHTS.get(r, 0.0), _RULE_ORDER.get(r, 99)))

    def summary(self) -> dict[str, Any]:
        """Contract AlertSummary."""
        m = self._meta
        return {
            "id": self.id,
            "opened_t": float(self.opened_t),
            "last_t": float(self.last_t),
            "active": bool(self.active),
            "category": self.category,
            "title": self.title,
            "shields": [m.shield_ids[s] for s in sorted(self.shields)],
            "channels": [m.channel_ids[c] for c in sorted(self.channels)],
            "severity": self.severity,
            "priority": self.priority,
            "rules": self.sorted_rules(),
            "ai_score_max": _num(self.ai_score_max, 3),
        }


# --------------------------------------------------------------- correlator
class AlertCorrelator:
    """Correlates triggers into alerts (MODEL_SPEC section 7).

    ``sim_meta`` provides ``shield_ids``, ``channel_ids``, ``channel_shield``, ``legs`` and
    ``gateway_of`` (callable shield_idx -> gateway id, or a per-shield sequence); a FaceSimulator
    works directly. ``use_suppression=False`` disables sensor-first suppression (baselines).
    """

    def __init__(self, sim_meta: Any, cfg: dict, use_suppression: bool = True):
        ac = cfg["alerts"]
        rc = cfg["rules"]
        self.close_after_s = float(ac["close_after_s"])
        self.sensor_suppress_s = float(ac["sensor_suppress_s"])
        self.adjacent_merge = int(ac["adjacent_merge"])
        self.use_suppression = bool(use_suppression)
        self.match_grace_s = float(cfg.get("evaluation", {}).get("match_grace_s", 0.0))
        spg = int(cfg.get("simulator", {}).get("shields_per_gateway", 6))
        self.meta = _Meta(sim_meta, spg)
        fc = cfg.get("features", {})
        self._txt = {
            "pressure_persist": rc["pressure_persist_s"],
            "imbalance_persist": rc["imbalance_persist_s"],
            "deviation_persist": rc["deviation_persist_s"],
            "slope_short": fc.get("slope_short_s", 5),
            "slope_long": fc.get("slope_long_s", 30),
            "missing_window": fc.get("missing_window_s", 30),
            "mad_window": rc.get("mad_window_s", 60),
            "ai_persist": cfg.get("model", {}).get("persist_s", 3),
        }
        self.reset()

    def reset(self) -> None:
        self._alerts: list[Alert] = []           # in opening order
        self._active: list[Alert] = []           # active subset, in opening order
        self._by_id: dict[str, Alert] = {}
        self._alias: dict[str, str] = {}         # merged-away id -> surviving id
        self._next_id = 1
        self._last_sensor_t = np.full(self.meta.C, -np.inf)
        self.suppressed = 0                       # number of triggers dropped by suppression

    # ------------------------------------------------------------- access
    @property
    def alerts(self) -> list[Alert]:
        """All alerts, newest first."""
        return self._alerts[::-1]

    @property
    def active_alerts(self) -> list[Alert]:
        return self._active[::-1]

    def get(self, alert_id: str) -> Alert | None:
        """Alert by id; the id of an alert merged into another resolves to the surviving alert."""
        return self._by_id.get(self._alias.get(alert_id, alert_id))

    @property
    def aliases(self) -> dict[str, str]:
        """Merged-away alert id -> id of the alert it was merged into."""
        return dict(self._alias)

    def summaries(self, limit: int | None = None) -> list[dict[str, Any]]:
        out = self.alerts
        if limit is not None:
            out = out[: int(limit)]
        return [a.summary() for a in out]

    # ------------------------------------------------------------- update
    def update(self, t: float, triggers: Iterable[Trigger], scores=None, ff: FeatureFrame | None = None) -> list[str]:
        """Process one sample's triggers; returns the ids of newly opened alerts (excluding any
        opened and merged away within this same call)."""
        t = float(t)
        if self._active:
            still = []
            for a in self._active:
                if t - a.last_t > self.close_after_s:
                    a.active = False
                else:
                    still.append(a)
            self._active = still

        trigs = list(triggers or [])
        if self.use_suppression and trigs:
            trigs = self._suppress(t, trigs)

        groups: dict[tuple[str, int], list[Trigger]] = {}
        for tr in trigs:
            groups.setdefault((tr.category, int(tr.shield_idx)), []).append(tr)

        opened: list[str] = []
        touched: dict[str, Alert] = {}
        for cat, s in sorted(groups, key=lambda k: (_CATEGORY_ORDER.get(k[0], 9), k[1])):
            alert = self._find(cat, s)
            if alert is None:
                alert = self._open(cat, s, t)
                opened.append(alert.id)
            elif cat == "process":
                alert = self._merge_adjacent(alert, s)
            self._attach(alert, s, groups[(cat, s)], t, ff)
            touched[alert.id] = alert
        sc = None if scores is None else np.asarray(scores, dtype=float)
        for aid, alert in touched.items():
            if aid not in self._alias:
                self._track(alert, t, sc, ff)
        return [aid for aid in opened if aid not in self._alias]

    def _suppress(self, t: float, trigs: list[Trigger]) -> list[Trigger]:
        tel_shields = {int(tr.shield_idx) for tr in trigs if tr.category == "telemetry"}
        for tr in trigs:
            if tr.category == "sensor" and tr.channel_idx is not None:
                self._last_sensor_t[tr.channel_idx] = t
        # Interpretation: a telemetry loss is also a data-quality event for every channel of the
        # shield, so process triggers right after the outage are held back for the same window.
        for s in tel_shields:
            self._last_sensor_t[self.meta.shield_channels[s]] = t
        kept = []
        for tr in trigs:
            if tr.category != "telemetry" and int(tr.shield_idx) in tel_shields:
                self.suppressed += 1
                continue
            if (tr.category == "process" and tr.channel_idx is not None
                    and t - self._last_sensor_t[tr.channel_idx] <= self.sensor_suppress_s):
                self.suppressed += 1
                continue
            kept.append(tr)
        return kept

    def _find(self, cat: str, s: int) -> Alert | None:
        cands = [a for a in self._active if a.category == cat]
        if not cands:
            return None
        if cat == "telemetry":
            g = self.meta.gateway[s]
            return next((a for a in cands if a.gateway == g), None)
        for a in cands:
            if s in a.shields:
                return a
        if cat == "process":
            best, best_key = None, None
            for a in cands:
                d = min(abs(s - x) for x in a.shields)
                if d <= self.adjacent_merge and (best_key is None or (d, a.opened_t) < best_key):
                    best, best_key = a, (d, a.opened_t)
            return best
        return None

    def _merge_adjacent(self, alert: Alert, s: int) -> Alert:
        """If shield ``s`` is also within ``adjacent_merge`` of other active process alerts, merge
        the newer alerts into the oldest one and return the survivor."""
        others = [a for a in self._active
                  if a is not alert and a.category == "process" and a.shields
                  and min(abs(s - x) for x in a.shields) <= self.adjacent_merge]
        if not others:
            return alert
        group = sorted([alert] + others, key=lambda a: (a.opened_t, self._alerts.index(a)))
        keep = group[0]
        for old in group[1:]:
            self._merge_into(keep, old)
        return keep

    def _merge_into(self, keep: Alert, old: Alert) -> None:
        keep.shields |= old.shields
        keep.channels |= old.channels
        keep.rules |= old.rules
        for r, t0 in old.rule_first_t.items():
            keep.rule_first_t[r] = min(keep.rule_first_t.get(r, t0), t0)
        for key, ev in old.evidence.items():
            prev = keep.evidence.get(key)
            if prev is None or ev["_mag"] > prev["_mag"]:
                keep.evidence[key] = ev
        keep.opened_t = min(keep.opened_t, old.opened_t)
        # AI maximum and its feature snapshot from whichever alert scored higher; channels missing
        # from that snapshot are filled from the other alert's snapshot.
        old_higher = _finite(old.ai_score_max) and (not _finite(keep.ai_score_max)
                                                    or old.ai_score_max > keep.ai_score_max)
        hi, lo = (old, keep) if old_higher else (keep, old)
        feats = dict(hi.features)
        for c, f in lo.features.items():
            feats.setdefault(c, f)
        keep.ai_score_max, keep.ai_t_max, keep.ai_channel_max = hi.ai_score_max, hi.ai_t_max, hi.ai_channel_max
        keep.features, keep.features_t = feats, hi.features_t if hi.features_t is not None else lo.features_t
        # data quality refers to the latest attach
        late, early = (old, keep) if old.last_t > keep.last_t else (keep, old)
        mf = dict(early.missing_frac)
        mf.update(late.missing_frac)
        keep.missing_frac = mf
        keep.last_t = max(keep.last_t, old.last_t)
        keep.n_triggers += old.n_triggers
        keep.merged_ids.extend([old.id] + old.merged_ids)
        old.active = False
        self._alerts.remove(old)
        self._active.remove(old)
        del self._by_id[old.id]
        self._alias[old.id] = keep.id
        for k, v in self._alias.items():
            if v == old.id:
                self._alias[k] = keep.id

    def _open(self, cat: str, s: int, t: float) -> Alert:
        aid = f"AL-{self._next_id:04d}"
        self._next_id += 1
        if cat == "telemetry":
            g = self.meta.gateway[s]
            alert = Alert(aid, cat, t, self.meta, gateway=g)
            for i, gi in enumerate(self.meta.gateway):
                if gi == g:
                    alert.shields.add(i)
                    alert.channels.update(self.meta.shield_channels[i])
        else:
            alert = Alert(aid, cat, t, self.meta, gateway=self.meta.gateway[s])
        self._alerts.append(alert)
        self._active.append(alert)
        self._by_id[aid] = alert
        return alert

    def _attach(self, a: Alert, s: int, trs: list[Trigger], t: float, ff: FeatureFrame | None) -> None:
        a.last_t = t
        a.shields.add(s)
        for tr in trs:
            a.n_triggers += 1
            a.rules.add(tr.rule)
            a.rule_first_t.setdefault(tr.rule, t)
            c = None if tr.channel_idx is None else int(tr.channel_idx)
            if c is not None:
                a.channels.add(c)
            key = (tr.rule, c)
            obs = float(tr.observed)
            prev = a.evidence.get(key)
            mag = abs(obs) if math.isfinite(obs) else -1.0
            if prev is None or mag > prev["_mag"]:
                value = float(ff.values[c]) if (ff is not None and c is not None) else math.nan
                a.evidence[key] = {
                    "rule": tr.rule, "channel_idx": c, "shield_idx": int(tr.shield_idx),
                    "t": float(tr.t), "observed": obs, "threshold": float(tr.threshold),
                    "unit": tr.unit, "description": tr.description, "_mag": mag, "_value": value,
                }

    def _track(self, a: Alert, t: float, scores: np.ndarray | None, ff: FeatureFrame | None) -> None:
        chans = sorted(a.channels)
        if scores is not None and chans:
            sc = scores[chans]
            fin = np.isfinite(sc)
            if fin.any():
                j = int(np.argmax(np.where(fin, sc, -np.inf)))
                if not math.isfinite(a.ai_score_max) or sc[j] > a.ai_score_max:
                    a.ai_score_max = float(sc[j])
                    a.ai_t_max = t
                    a.ai_channel_max = chans[j]
                    if ff is not None:
                        a.features = {c: ff.X[c].copy() for c in chans}
                        a.features_t = t
        if ff is not None:
            for c in chans:
                if c not in a.features:      # first trigger (no AI score yet) or newly added channel
                    a.features[c] = ff.X[c].copy()
                    if a.features_t is None:
                        a.features_t = t
            a.missing_frac = {c: float(ff.X[c, _MF]) for c in chans}

    # ------------------------------------------------------------- detail
    def detail(self, alert_id: str, now_t: float,
               series_provider: Callable[[Alert, float, float], dict] | None = None,
               events: Iterable[Any] | None = None,
               neighbors: Sequence[dict] | None = None) -> dict[str, Any]:
        """Contract AlertDetail. Raises KeyError for an unknown id; a merged-away id returns the
        detail of the alert it was merged into.

        ``series_provider(alert, t0, t1) -> {"t", "channels", "ai_score"}`` fills ``series`` for
        the window ``[max(0, opened_t - 120), min(last_t + 30, now_t)]``; ``events`` (Event objects
        or EventRecord dicts) are matched to fill ``matched_event``; ``neighbors`` is passed
        through. Without them, empty defaults keep the contract shape.
        """
        a = self.get(alert_id)
        if a is None:
            raise KeyError(alert_id)
        m = self.meta
        dq = self._data_quality(a)
        d = a.summary()
        d["explanation"] = self._explain(a)
        d["evidence"] = [
            {"rule": e["rule"],
             "channel": None if e["channel_idx"] is None else m.channel_ids[e["channel_idx"]],
             "t": e["t"], "observed": _num(e["observed"]), "threshold": _num(e["threshold"]),
             "unit": e["unit"], "description": e["description"]}
            for e in self._sorted_evidence(a)
        ]
        d["features"] = {m.channel_ids[c]: {n: _num(a.features[c][j]) for j, n in enumerate(FEATURE_NAMES)}
                         for c in sorted(a.features)}
        d["ai_detail"] = {"score_max": _num(a.ai_score_max, 3), "t_max": a.ai_t_max, "threshold": AI_THRESHOLD}
        d["data_quality"] = dq
        d["neighbors"] = list(neighbors) if neighbors is not None else []
        d["verification_steps"] = self._verification_steps(a)
        d["uncertainty"] = self._uncertainty(a, dq)
        d["priority_formula"] = PRIORITY_FORMULA
        t0 = max(0.0, a.opened_t - SERIES_PRE_S)
        t1 = max(t0, min(a.last_t + SERIES_POST_S, float(now_t)))
        d["window"] = [t0, t1]
        d["series"] = (series_provider(a, t0, t1) if series_provider is not None
                       else {"t": [], "channels": {}, "ai_score": {}})
        d["matched_event"] = self.match_event(a, events, now_t) if events is not None else None
        return _clean(d)

    def match_event(self, a: Alert, events: Iterable[Any], now_t: float | None = None) -> dict | None:
        """Ground-truth event sharing >= 1 shield: preferably one with
        ``onset <= opened_t <= end + match_grace_s`` (latest onset), else any time overlap."""
        shields = {self.meta.shield_ids[s] for s in a.shields}
        best, best_key = None, None
        for ev in events:
            rec = _event_record(ev, now_t)
            if rec is None or not shields.intersection(rec.get("shields") or []):
                continue
            on, end = float(rec["onset_t"]), float(rec["end_t"])
            if on <= a.opened_t <= end + self.match_grace_s:
                key = (0, -on)
            elif on <= a.last_t and end + self.match_grace_s >= a.opened_t:
                key = (1, on)
            else:
                continue
            if best_key is None or key < best_key:
                best, best_key = rec, key
        return best

    # ------------------------------------------------------------ texts
    def _sorted_evidence(self, a: Alert) -> list[dict[str, Any]]:
        """Evidence in the order the rules first fired (the story of the alert), then by weight."""
        return sorted(a.evidence.values(),
                      key=lambda e: (a.rule_first_t.get(e["rule"], a.opened_t),
                                     -RULE_WEIGHTS.get(e["rule"], 0.0), _RULE_ORDER.get(e["rule"], 99),
                                     -1 if e["channel_idx"] is None else e["channel_idx"]))

    def _rule_sentence(self, a: Alert, rule: str, ev: dict[str, Any]) -> str:
        m, x = self.meta, self._txt
        sh = m.shield_ids[ev["shield_idx"]]
        ch = None if ev["channel_idx"] is None else m.channel_ids[ev["channel_idx"]]
        o, th, when = ev["observed"], ev["threshold"], fmt_t(ev["t"])
        if rule == "high_pressure":
            return (f"Shield {sh} (leg {ch}) reached {o:.1f} bar at {when}, above the illustrative "
                    f"{th:.0f} bar high-pressure limit for at least {x['pressure_persist']:g} consecutive samples.")
        if rule == "low_pressure":
            return (f"Shield {sh} (leg {ch}) fell to {o:.1f} bar at {when} while loaded, below the configured "
                    f"{th:.0f} bar low-pressure limit for at least {x['pressure_persist']:g} consecutive samples.")
        if rule == "rapid_rise":
            return (f"Shield {sh} (leg {ch}) rose {o:.1f} bar/s over {x['slope_short']:g} s at {when}, "
                    f"above the configured {th:.1f} bar/s rule threshold.")
        if rule == "rapid_drop":
            return (f"Shield {sh} (leg {ch}) decayed at {o:.2f} bar/s (robust {x['slope_long']:g}-s slope) at {when}, "
                    f"steeper than the configured {th:.2f} bar/s rule threshold.")
        if rule == "leg_imbalance":
            return (f"Leg {ch} differed from the mean of the other leg(s) of shield {sh} by {100 * o:+.1f}% "
                    f"of the shield mean at {when}, beyond the configured {100 * th:.0f}% imbalance limit "
                    f"for at least {x['imbalance_persist']:g} s.")
        if rule == "persistent_deviation":
            return (f"Leg {ch} of shield {sh} was {o:+.1f} bar away from its own baseline in the current loaded "
                    f"run at {when}, beyond the configured {th:.0f} bar for at least {x['deviation_persist']:g} s.")
        if rule == "mad_anomaly":
            return (f"Leg {ch} of shield {sh} reached a robust z-score of {o:+.1f} against its "
                    f"{x['mad_window']:g}-s rolling median at {when}, beyond the configured {th:.1f}.")
        if rule == "ai_anomaly":
            chans = [m.channel_ids[e["channel_idx"]] for e in self._sorted_evidence(a)
                     if e["rule"] == "ai_anomaly" and e["channel_idx"] is not None]
            return (f"The Isolation Forest score was at or above 1.0 (calibrated threshold) for at least "
                    f"{x['ai_persist']:g} consecutive scored samples on {_join_more(chans, 5)}.")
        if rule == "sensor_stuck":
            val = ev.get("_value", math.nan)
            what = f"the identical value {val:.1f} bar" if _finite(val) else "identical values"
            return (f"Channel {ch} (shield {sh}) reported {what} for {o:.0f} consecutive samples up to {when}, "
                    f"at or above the configured {th:.0f}-sample stuck-sensor limit.")
        if rule == "sensor_spike":
            return (f"Channel {ch} (shield {sh}) jumped {o:+.1f} bar from its recent median for a single sample "
                    f"just before {when} and returned on the next sample (spike threshold {th:.0f} bar).")
        if rule == "data_missing":
            return (f"Channel {ch} (shield {sh}) returned no valid reading for {o:.0f} s in a row up to "
                    f"{when}, at or above the configured {th:.0f} s limit.")
        if rule == "telemetry_loss":
            gw_shields = [m.shield_ids[i] for i, g in enumerate(m.gateway) if g == a.gateway]
            span = f"{gw_shields[0]}-{gw_shields[-1]}" if len(gw_shields) > 1 else _join(gw_shields)
            return (f"Every channel on gateway {a.gateway} (shields {span}) stopped reporting; the outage had "
                    f"lasted {o:.0f} s by {when} (limit {th:.0f} s).")
        return f"Rule {rule} fired on {ch or sh} at {when} (observed {o:.3g} {ev['unit']}, threshold {th:.3g})."

    def _explain(self, a: Alert) -> str:
        m = self.meta
        parts: list[str] = []
        if a.multi_shield:
            names = [m.shield_ids[s] for s in sorted(a.shields)]
            parts.append(f"Multi-shield pattern: triggers on adjacent shields {_join(names)} were merged into one "
                         f"alert (shields within {self.adjacent_merge} of each other).")
        by_rule: dict[str, list[dict[str, Any]]] = {}
        for e in self._sorted_evidence(a):
            by_rule.setdefault(e["rule"], []).append(e)
        for rule in by_rule:  # chronological: by_rule preserves _sorted_evidence order
            evs = by_rule[rule]
            if not evs:
                continue
            best = max(evs, key=lambda e: e["_mag"])
            sentence = self._rule_sentence(a, rule, best)
            others = [m.channel_ids[e["channel_idx"]] for e in evs
                      if e is not best and e["channel_idx"] is not None]
            if others and rule != "ai_anomaly":    # the AI sentence already lists its channels
                sentence += f" The same rule also fired on {_join_more(others, 4)}."
            parts.append(sentence)
        # AI peak: always for process alerts (a sub-threshold peak is informative there); for
        # sensor/telemetry alerts only when it reached the threshold.
        if _finite(a.ai_score_max) and (a.category == "process" or a.ai_score_max >= AI_THRESHOLD):
            where = f" on {m.channel_ids[a.ai_channel_max]}" if a.ai_channel_max is not None else ""
            s = (f"The Isolation Forest score peaked at {a.ai_score_max:.2f} (1.0 = calibrated threshold) "
                 f"at {fmt_t(a.ai_t_max)}{where}")
            parts.append(s + ("." if a.ai_score_max >= AI_THRESHOLD else ", below the AI alert threshold."))
        if a.category in ("sensor", "telemetry"):
            parts.append(FAULT_SENTENCE)
        if a.category == "telemetry":
            parts.append("Pressures on these shields are unobserved during the outage, so the absence of "
                         "process alerts there does not show that conditions were normal.")
        status = "still active" if a.active else f"closed after {self.close_after_s:g} s without new triggers"
        parts.append(f"Opened at {fmt_t(a.opened_t)}, last trigger at {fmt_t(a.last_t)} ({status}).")
        return " ".join(parts)

    def _data_quality(self, a: Alert) -> dict[str, str]:
        m = self.meta
        mf = {c: v for c, v in a.missing_frac.items() if c in a.channels and _finite(v)}
        when = fmt_t(a.last_t)
        win = self._txt["missing_window"]
        if not mf:
            return {"status": "ok", "note": "No missing-data information was recorded for this alert."}
        worst = max(mf, key=lambda c: mf[c])
        mx = mf[worst]
        if mx < DQ_OK_BELOW:
            status = "ok"
            note = (f"All channels of the alert reported valid data in the {win:g} s before the last trigger "
                    f"({when}); the worst channel missed {100 * mx:.0f}% of samples.")
        elif mx < DQ_DEGRADED_BELOW:
            status = "degraded"
            note = (f"{m.channel_ids[worst]} missed {100 * mx:.0f}% of samples in the {win:g} s before the last "
                    f"trigger ({when}); features on that channel are less reliable.")
        else:
            status = "lost"
            note = (f"{m.channel_ids[worst]} missed {100 * mx:.0f}% of samples in the {win:g} s before the last "
                    f"trigger ({when}); data for the affected channels are largely unavailable.")
        if a.category == "sensor":
            note += " This measures completeness only; the flagged sensor fault may still make reported values unreliable."
        return {"status": status, "note": note}

    def _verification_steps(self, a: Alert) -> list[str]:
        steps: list[str] = []
        for rule in a.sorted_rules():
            steps.extend(_RULE_STEPS.get(rule, []))
        if a.category == "process":
            steps.extend(_PROCESS_COMMON_STEPS)
        return list(dict.fromkeys(steps))

    def _uncertainty(self, a: Alert, dq: dict[str, str]) -> str:
        return ("This alert is based on simulated readings and illustrative thresholds, not equipment "
                "specifications or a certified safety system. The priority is a prioritisation score and the "
                "AI score is a normalised anomaly score (1.0 = calibrated threshold); neither is a probability "
                "of roof collapse or of any failure. "
                f"Data quality for the alert's channels is {dq['status']}: {dq['note']}")


def _event_record(ev: Any, now_t: float | None) -> dict[str, Any] | None:
    """EventRecord dict from an Event object or dict (hidden events are ignored)."""
    if isinstance(ev, dict):
        if ev.get("hidden"):
            return None
        rec = dict(ev)
    else:
        if getattr(ev, "hidden", False):
            return None
        if hasattr(ev, "to_dict"):
            rec = ev.to_dict(now_t)
        else:
            keys = ("id", "type", "label", "category", "onset_t", "end_t", "shields", "channels",
                    "magnitude", "magnitude_unit")
            rec = {k: getattr(ev, k, None) for k in keys}
    if "active" not in rec:
        rec["active"] = bool(now_t is not None and float(rec["onset_t"]) <= float(now_t) < float(rec["end_t"]))
    rec.pop("hidden", None)
    rec.pop("scope", None)
    return _clean(rec)
