"""Detection pipeline and live simulation engine (MODEL_SPEC section 8).

* ``Pipeline`` is the detection path shared by the live service and the evaluation:
  FeatureEngine -> (AnomalyModel scores) -> one or more ``DetectorHead`` objects. A head is one
  detection method (its trigger sources plus its own ``AlertCorrelator``), so the evaluation can
  compute features and AI scores once per sample and feed the same ``FeatureFrame`` to all four
  methods. Per-stage wall time is recorded for ``runtime_ms_per_step``.
* ``LiveEngine`` wraps FaceSimulator -> frame_to_readings -> TelemetryValidator ->
  readings_to_vector -> DeviceMonitor + hybrid Pipeline, keeps a capped history (6 h of simulated
  time) and builds the API_CONTRACT payloads (State, AlertDetail, exports). Every public method
  holds one ``threading.Lock``.
* ``get_model(cfg)`` loads ``data/model.pkl`` when its config fingerprint matches, otherwise trains
  the default model and saves it.

Ground-truth labels from the simulator are stored for the CSV export and the "Reveal ground truth"
view only; detectors only ever see measured values, operating states and time in state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from . import __version__
from .alerts import AlertCorrelator
from .config import PROJECT_ROOT, validate_sim_params
from .features import FeatureEngine, FeatureFrame
from .model import MODEL_VERSION, AITrigger, AnomalyModel, train_default_model
from .physics import force_mn, piston_area_m2
from .rules import FixedThreshold, RollingMAD, RuleEngine, Trigger
from .simulator import EVENT_TYPES, FaceSimulator
from .telemetry import (
    QUALITY_INVALID, QUALITY_MISSING, QUALITY_OK, DeviceMonitor, TelemetryValidator, format_timestamp,
    frame_to_readings, readings_to_vector,
)

log = logging.getLogger("mineshield")

METHODS: tuple[str, ...] = ("fixed_threshold", "rolling_mad", "isolation_forest", "mineshield_hybrid")
HYBRID = "mineshield_hybrid"
AI_METHODS = frozenset({"isolation_forest", HYBRID})

DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = DATA_DIR / "model.pkl"

SIM_PARAM_KEYS: tuple[str, ...] = ("seed", "n_shields", "legs_per_shield", "noise_bar", "load_trend_bar_per_h")
HISTORY_S = 6 * 3600.0          # history kept for series / heatmap / export
MAX_STEPS = 60                  # per step() call (contract: steps 1-60)
WINDOW_MIN_S, WINDOW_MAX_S, WINDOW_DEFAULT_S = 60.0, 900.0, 300.0
HEATMAP_STEP_S = 5.0
NEIGHBOR_SPAN = 2               # neighbours listed in AlertDetail: shields +/-2 of the alert shields
MAX_ALERTS_IN_STATE = 50
CSV_COLUMNS = ("t_s", "timestamp", "seq", "device_id", "shield_id", "channel_id", "value", "unit",
               "quality", "state", "gt_label", "gt_event_id")

_PUBLIC_TYPES: dict[str, dict[str, Any]] = {e["type"]: e for e in EVENT_TYPES}
_STATE_NAMES = ("advancing", "setting", "loaded")
_STATE_CODE = {s: i for i, s in enumerate(_STATE_NAMES)}
_LOADED = _STATE_CODE["loaded"]
_QUALITY_NAMES = (QUALITY_OK, QUALITY_MISSING, QUALITY_INVALID)
_QUALITY_CODE = {q: i for i, q in enumerate(_QUALITY_NAMES)}


# --------------------------------------------------------------------------- helpers
def _is_int(x: Any) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _opt(x: Any, nd: int) -> float | None:
    """Rounded float, or None for None / NaN / inf."""
    if x is None:
        return None
    f = float(x)
    return round(f, nd) if math.isfinite(f) else None


def _jlist(a: Any, nd: int | None = None) -> list:
    """1-D or 2-D numeric array -> nested list with NaN/inf -> None (JSON-safe)."""
    arr = np.asarray(a, dtype=float)
    if nd is not None:
        arr = np.round(arr, nd)
    fin = np.isfinite(arr)
    out = arr.tolist()
    if fin.all():
        return out
    if arr.ndim == 1:
        return [v if ok else None for v, ok in zip(out, fin.tolist())]
    return [[v if ok else None for v, ok in zip(row, frow)] for row, frow in zip(out, fin.tolist())]


def json_clean(obj: Any) -> Any:
    """Recursively convert numpy types to Python and NaN/inf to None."""
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_clean(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rowmax(a: np.ndarray) -> np.ndarray:
    """NaN-ignoring max over the last axis (NaN only where all are NaN), without warnings."""
    return np.fmax.reduce(a, axis=-1)


def _rowmean(a: np.ndarray) -> np.ndarray:
    """NaN-ignoring mean over the last axis (NaN where no finite value), without warnings."""
    fin = np.isfinite(a)
    cnt = fin.sum(axis=-1)
    s = np.where(fin, a, 0.0).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


# --------------------------------------------------------------------------- metadata
@dataclass(eq=False)
class SimMeta:
    """Face layout handed to detectors (no access to the simulator or its ground truth)."""

    shield_ids: list[str]
    channel_ids: list[str]
    channel_shield: np.ndarray
    legs: int
    gateway_of: list[str]           # gateway id per shield index

    @classmethod
    def from_sim(cls, sim: FaceSimulator) -> "SimMeta":
        return cls(list(sim.shield_ids), list(sim.channel_ids), np.array(sim.channel_shield, copy=True),
                   int(sim.legs), list(sim.shield_gateway))

    @property
    def n_shields(self) -> int:
        return len(self.shield_ids)

    @property
    def n_channels(self) -> int:
        return len(self.channel_ids)


# --------------------------------------------------------------------------- pipeline
class DetectorHead:
    """One detection method: its trigger sources and its own AlertCorrelator.

    fixed_threshold   -> FixedThreshold (high/low pressure)
    rolling_mad       -> RollingMAD
    isolation_forest  -> AITrigger only
    mineshield_hybrid -> RuleEngine (all rules) + AITrigger, sensor-first suppression ON
    Suppression is used only by the hybrid (MODEL_SPEC 9). Heads that do not use the AI pass no
    scores to their correlator, so their alerts carry no AI information.
    """

    def __init__(self, mode: str, meta: SimMeta, cfg: dict):
        if mode not in METHODS:
            raise ValueError(f"unknown detection mode {mode!r}; valid modes: {', '.join(METHODS)}")
        C, L = meta.n_channels, meta.legs
        self.mode = mode
        self.uses_ai = mode in AI_METHODS
        self.rules: RuleEngine | FixedThreshold | None = None
        self.mad: RollingMAD | None = None
        self.ai: AITrigger | None = None
        if mode == "fixed_threshold":
            self.rules = FixedThreshold(C, L, cfg)
        elif mode == "rolling_mad":
            self.mad = RollingMAD(C, L, cfg)
        elif mode == HYBRID:
            self.rules = RuleEngine(C, L, cfg)
        if self.uses_ai:
            self.ai = AITrigger(C, L, cfg)
        self.correlator = AlertCorrelator(meta, cfg, use_suppression=(mode == HYBRID))
        self._gateways = list(meta.gateway_of)
        self.last_triggers: list[Trigger] = []
        self.time_s = 0.0           # wall time of this head's detection + correlation

    def process(self, ff: FeatureFrame, scores: np.ndarray) -> list[str]:
        t0 = time.perf_counter()
        trig: list[Trigger] = []
        if self.rules is not None:
            trig.extend(self.rules.evaluate(ff, self._gateways))
        if self.mad is not None:
            trig.extend(self.mad.evaluate(ff))
        if self.ai is not None:
            trig.extend(self.ai.evaluate(ff.t, scores))
        self.last_triggers = trig
        opened = self.correlator.update(ff.t, trig, scores if self.uses_ai else None, ff)
        self.time_s += time.perf_counter() - t0
        return opened


@dataclass
class StepResult:
    ff: FeatureFrame
    scores: np.ndarray                                   # (C,) normalised AI score, NaN = not scored
    opened: dict[str, list[str]] = field(default_factory=dict)   # mode -> newly opened alert ids


class Pipeline:
    """FeatureEngine -> AnomalyModel.score -> detector head(s).

    ``mode`` is one method id or a sequence of ids (one head each, sharing the FeatureFrame and AI
    scores). The first mode is the primary head (``pipeline.correlator``).
    """

    def __init__(self, meta: SimMeta | Any, cfg: dict, model: AnomalyModel | None,
                 mode: str | Sequence[str] = HYBRID):
        if isinstance(meta, FaceSimulator):
            meta = SimMeta.from_sim(meta)
        elif not isinstance(meta, SimMeta):
            gw = getattr(meta, "gateway_of")
            gws = [gw(i) for i in range(len(meta.shield_ids))] if callable(gw) else list(gw)
            meta = SimMeta(list(meta.shield_ids), list(meta.channel_ids), np.asarray(meta.channel_shield),
                           int(getattr(meta, "legs")), gws)
        modes = [mode] if isinstance(mode, str) else list(mode)
        if not modes:
            raise ValueError("at least one detection mode is required")
        if len(set(modes)) != len(modes):
            raise ValueError("detection modes must be unique")
        self.meta = meta
        self.cfg = cfg
        self.model = model
        self.features = FeatureEngine(meta.n_shields, meta.legs, cfg)
        self.heads: dict[str, DetectorHead] = {m: DetectorHead(m, meta, cfg) for m in modes}
        self.modes = modes
        self._score = model is not None and model.ready
        self.steps = 0
        self.feature_time_s = 0.0
        self.score_time_s = 0.0

    @property
    def head(self) -> DetectorHead:
        return self.heads[self.modes[0]]

    @property
    def correlator(self) -> AlertCorrelator:
        return self.head.correlator

    def process(self, values, states, time_in_state, t: float) -> StepResult:
        t0 = time.perf_counter()
        ff = self.features.update(t, values, states, time_in_state)
        t1 = time.perf_counter()
        if self._score:
            scores = self.model.score_frame(ff)
        else:
            scores = np.full(ff.n_channels, np.nan)
        self.score_time_s += time.perf_counter() - t1
        self.feature_time_s += t1 - t0
        opened = {m: h.process(ff, scores) for m, h in self.heads.items()}
        self.steps += 1
        return StepResult(ff=ff, scores=scores, opened=opened)

    def time_s(self, mode: str) -> float:
        """Total wall time attributable to ``mode``: features (+ AI scoring) + detection + correlation."""
        h = self.heads[mode]
        return self.feature_time_s + (self.score_time_s if h.uses_ai else 0.0) + h.time_s

    def runtime_ms_per_step(self, mode: str) -> float | None:
        return 1000.0 * self.time_s(mode) / self.steps if self.steps else None


# --------------------------------------------------------------------------- model provider
def config_fingerprint(cfg: dict) -> str:
    """Hash of the config sections that determine the trained model, plus the model version."""
    payload = {"simulator": cfg["simulator"], "features": cfg["features"], "model": cfg["model"],
               "model_version": MODEL_VERSION}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_cached_model(cfg: dict, path: str | Path | None = None) -> AnomalyModel | None:
    """The saved model if it exists, loads and matches the config fingerprint; else None."""
    p = Path(path) if path is not None else MODEL_PATH
    if not p.is_file():
        return None
    try:
        m = AnomalyModel.load(p)
    except Exception as exc:  # corrupt / incompatible file: retrain
        log.warning("Ignoring model file %s (%s)", p, exc)
        return None
    if m.meta.get("config_fingerprint") != config_fingerprint(cfg) or not m.ready:
        return None
    return m


def model_is_current(cfg: dict, path: str | Path | None = None) -> bool:
    return load_cached_model(cfg, path) is not None


def get_model(cfg: dict, path: str | Path | None = None) -> AnomalyModel:
    """Load ``data/model.pkl`` if current, otherwise train the default model and save it."""
    p = Path(path) if path is not None else MODEL_PATH
    m = load_cached_model(cfg, p)
    if m is not None:
        return m
    log.info("Training the Isolation Forest on normal-only synthetic data (seeds %s) -> %s",
             cfg["model"]["train_seeds"], p)
    m = train_default_model(cfg)
    m.meta["config_fingerprint"] = config_fingerprint(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    m.save(p)
    return m


def model_info(model: AnomalyModel | None) -> dict[str, Any]:
    """Contract ``sim.model`` block."""
    if model is None:
        return {"name": "IsolationForest", "version": MODEL_VERSION, "status": "unavailable",
                "trained_on": None, "threshold_note": None}
    meta = model.meta or {}
    q = 100 * float(getattr(model, "calibration_quantile", 0.995))
    return {
        "name": "IsolationForest",
        "version": model.version,
        "status": "ready" if model.ready else "not_calibrated",
        "trained_on": meta.get("trained_on", "normal-only synthetic data"),
        "threshold_note": meta.get("threshold_note", f"score 1.0 = {q:g}th percentile of validation normal data"),
    }


# --------------------------------------------------------------------------- history
class _History:
    """Per-sample ring buffer indexed by sample number k (t = k * dt)."""

    def __init__(self, capacity: int, C: int, S: int):
        self.cap = int(capacity)
        self.t = np.full(self.cap, np.nan)
        self.seq = np.zeros(self.cap, dtype=np.int64)
        self.values = np.full((self.cap, C), np.nan)
        self.quality = np.zeros((self.cap, C), dtype=np.int8)
        self.state = np.zeros((self.cap, S), dtype=np.int8)
        self.score = np.full((self.cap, C), np.nan, dtype=np.float32)
        self.gt_label = np.zeros((self.cap, C), dtype=np.int16)
        self.gt_event = np.full((self.cap, C), -1, dtype=np.int32)
        self.n = 0

    @property
    def first_k(self) -> int:
        return max(0, self.n - self.cap)

    @property
    def last_k(self) -> int:
        return self.n - 1

    def append(self, t, seq, values, quality, state, score, gt_label, gt_event) -> None:
        i = self.n % self.cap
        self.t[i] = t
        self.seq[i] = seq
        self.values[i] = values
        self.quality[i] = quality
        self.state[i] = state
        self.score[i] = score
        self.gt_label[i] = gt_label
        self.gt_event[i] = gt_event
        self.n += 1

    def krange(self, k0: int, k1: int) -> np.ndarray:
        """Available sample numbers in [k0, k1] (ascending)."""
        a, b = max(self.first_k, int(k0)), min(self.last_k, int(k1))
        return np.arange(a, b + 1) if b >= a else np.zeros(0, dtype=np.int64)

    def slots(self, ks: np.ndarray) -> np.ndarray:
        return ks % self.cap


# --------------------------------------------------------------------------- live engine
class LiveEngine:
    """Simulator-backed live service (one face, one hybrid pipeline). Thread-safe."""

    def __init__(self, cfg: dict, model: AnomalyModel | None, history_s: float = HISTORY_S):
        self.cfg = cfg
        self.model = model
        self._lock = threading.Lock()
        self._history_s = float(history_s)
        self._area = float(piston_area_m2(float(cfg["piston_bore_m"])))
        self._watch = float(cfg["model"]["watch_score"])
        self._dt = float(cfg["simulator"]["dt_s"])
        self.sim: FaceSimulator | None = None

    # ------------------------------------------------------------------ public API
    def reset(self, **params: Any) -> dict[str, Any]:
        """Validate params (ValueError on bad input), rebuild everything, produce the t = 0 sample.
        Returns SimInfo."""
        with self._lock:
            self._reset(params)
            return self._sim_info()

    def step(self, n: int = 1) -> dict[str, Any]:
        """Advance ``n`` samples (1-60). Returns {"t", "samples", "opened"}."""
        n = self._check_steps(n)
        with self._lock:
            self._ensure()
            opened = self._advance(n)
            return {"t": self._t_last, "samples": self.sim.samples, "opened": opened}

    def step_state(self, n: int, shield: str | None = None, window_s: Any = WINDOW_DEFAULT_S) -> dict[str, Any]:
        """Advance ``n`` samples and return the State, atomically."""
        n = self._check_steps(n)
        with self._lock:
            self._ensure()
            self._advance(n)
            return self._state(shield, window_s)

    def inject(self, type: str, shield: str, leg: int | None = None, magnitude: float | None = None,
               duration_s: float | None = None) -> dict[str, Any]:
        """Inject a synthetic event starting at the next sample. Returns the EventRecord."""
        with self._lock:
            self._ensure()
            spec = _PUBLIC_TYPES.get(type) if isinstance(type, str) else None
            if spec is None:
                raise ValueError(f"unknown event type {type!r}; valid types: {', '.join(_PUBLIC_TYPES)}")
            if spec["default_magnitude"] is None and magnitude is not None:
                raise ValueError(f"magnitude must be omitted for {type} (this event type has no magnitude)")
            if not isinstance(shield, str) or shield not in self._shield_index:
                raise ValueError(f"unknown shield {shield!r}; valid shields: "
                                 f"{self.meta.shield_ids[0]}..{self.meta.shield_ids[-1]}")
            channel_id = None
            if spec["scope"] == "channel":
                leg = 1 if leg is None else leg
                L = self.meta.legs
                if not _is_int(leg) or not 1 <= int(leg) <= L:
                    raise ValueError(f"leg must be an integer between 1 and {L}, got {leg!r}")
                channel_id = f"{shield}-L{int(leg)}"
            ev = self.sim.inject(type, shield, channel_id=channel_id, magnitude=magnitude, duration_s=duration_s)
            return self._event_dict(ev)

    def state(self, shield: str | None = None, window_s: Any = WINDOW_DEFAULT_S) -> dict[str, Any]:
        with self._lock:
            self._ensure()
            return self._state(shield, window_s)

    def alert_detail(self, alert_id: str) -> dict[str, Any]:
        """Contract AlertDetail. Raises KeyError for an unknown alert id."""
        with self._lock:
            self._ensure()
            return self._alert_detail(alert_id, with_series=True)

    def export_experiment(self) -> dict[str, Any]:
        """{"exported_at", "version", "sim", "events", "alerts"}; events and alerts in chronological
        order, alerts as AlertDetail without ``series``."""
        with self._lock:
            self._ensure()
            corr = self.pipeline.correlator
            alerts = []
            for a in reversed(corr.alerts):           # corr.alerts is newest first
                try:
                    alerts.append(self._alert_detail(a.id, with_series=False))
                except KeyError:
                    continue
            return json_clean({
                "exported_at": utc_now_iso(),
                "version": __version__,
                "sim": self._sim_info(),
                "events": [self._event_dict(e) for e in self.sim.events],
                "alerts": alerts,
            })

    def iter_csv(self) -> Iterator[str]:
        """CSV lines (header first, each ending in newline) for the kept history. The history is
        snapshotted under the lock; lines are generated afterwards without holding it."""
        with self._lock:
            self._ensure()
            h = self.history
            ks = h.krange(h.first_k, h.last_k)
            sl = h.slots(ks)
            snap = {
                "t": h.t[sl].copy(), "seq": h.seq[sl].copy(), "values": h.values[sl].copy(),
                "quality": h.quality[sl].copy(), "state": h.state[sl].copy(),
                "gt_label": h.gt_label[sl].copy(), "gt_event": h.gt_event[sl].copy(),
                "labels": list(self._labels), "event_ids": list(self._event_ids),
                "channel_ids": list(self.meta.channel_ids), "shield_ids": list(self.meta.shield_ids),
                "channel_shield": self.meta.channel_shield.tolist(), "gateways": list(self.meta.gateway_of),
                "start": self.sim.start_timestamp,
            }
        return _csv_lines(snap)

    @property
    def channel_ids(self) -> list[str]:
        with self._lock:
            self._ensure()
            return list(self.meta.channel_ids)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _check_steps(n: Any) -> int:
        if not _is_int(n) or not 1 <= int(n) <= MAX_STEPS:
            raise ValueError(f"steps must be an integer between 1 and {MAX_STEPS}, got {n!r}")
        return int(n)

    def _ensure(self) -> None:
        if self.sim is None:
            self._reset({})

    def _reset(self, params: dict[str, Any]) -> None:
        unknown = sorted(set(params) - set(SIM_PARAM_KEYS))
        if unknown:
            raise ValueError(f"unknown parameter(s): {', '.join(unknown)}; "
                             f"allowed: {', '.join(SIM_PARAM_KEYS)}")
        d = self.cfg["defaults"]
        merged = {k: (params[k] if params.get(k) is not None else d[k]) for k in SIM_PARAM_KEYS}
        p = validate_sim_params(self.cfg, **merged)
        sim = FaceSimulator(self.cfg, p["seed"], n_shields=p["n_shields"], legs_per_shield=p["legs_per_shield"],
                            noise_bar=p["noise_bar"], load_trend_bar_per_h=p["load_trend_bar_per_h"])
        meta = SimMeta.from_sim(sim)
        S, C = meta.n_shields, meta.n_channels
        # Build everything first so a failure leaves the previous run intact.
        validator = TelemetryValidator(meta.channel_ids, self.cfg["limits"])
        monitor = DeviceMonitor(sim)
        pipeline = Pipeline(meta, self.cfg, self.model, HYBRID)
        history = _History(max(1, int(round(self._history_s / self._dt))), C, S)

        self.params = p
        self.sim = sim
        self.meta = meta
        self.validator = validator
        self.monitor = monitor
        self.pipeline = pipeline
        self.history = history
        self._shield_index = {sid: i for i, sid in enumerate(meta.shield_ids)}
        self._labels: list[str] = ["normal"] + [e["type"] for e in EVENT_TYPES]
        self._label_code = {lab: i for i, lab in enumerate(self._labels)}
        self._event_ids: list[str] = []
        self._event_code: dict[str, int] = {}
        self._readings_total = 0
        self._t_last = 0.0
        self._last_values = np.full(C, np.nan)
        self._last_quality: list[str] = [QUALITY_MISSING] * C
        self._last_states: list[str] = ["advancing"] * S
        self._last_tis = np.zeros(S)
        self._last_scores = np.full(C, np.nan)
        self._advance(1)

    def _advance(self, n: int) -> list[str]:
        sim, meta = self.sim, self.meta
        opened: list[str] = []
        for _ in range(n):
            fr = sim.step()
            readings = frame_to_readings(fr, sim)
            accepted, rejected = self.validator.validate(readings)
            values, quality = readings_to_vector(accepted, meta.channel_ids)
            self.monitor.update(fr.t, quality)
            res = self.pipeline.process(values, fr.states, fr.time_in_state, fr.t)
            opened.extend(res.opened.get(HYBRID, []))
            self._readings_total += len(readings)
            self.history.append(
                fr.t, fr.seq_start, values,
                [_QUALITY_CODE.get(q, 1) for q in quality],
                [_STATE_CODE[s] for s in fr.states],
                res.scores,
                [self._label_of(lab) for lab in fr.gt_label],
                [-1 if e is None else self._event_of(e) for e in fr.gt_event_id],
            )
            self._t_last = float(fr.t)
            self._last_values = values
            self._last_quality = quality
            self._last_states = list(fr.states)
            self._last_tis = np.asarray(fr.time_in_state, dtype=float)
            self._last_scores = res.scores
        return opened

    def _label_of(self, label: str) -> int:
        code = self._label_code.get(label)
        if code is None:
            code = self._label_code[label] = len(self._labels)
            self._labels.append(label)
        return code

    def _event_of(self, ev_id: str) -> int:
        code = self._event_code.get(ev_id)
        if code is None:
            code = self._event_code[ev_id] = len(self._event_ids)
            self._event_ids.append(ev_id)
        return code

    # ------------------------------------------------------------------ payload builders
    def _sim_info(self) -> dict[str, Any]:
        p = self.params
        return {
            "t": self._t_last,
            "samples": int(self.sim.samples),
            "seed": p["seed"],
            "n_shields": p["n_shields"],
            "legs_per_shield": p["legs_per_shield"],
            "dt_s": self._dt,
            "noise_bar": p["noise_bar"],
            "load_trend_bar_per_h": p["load_trend_bar_per_h"],
            "data_source": "simulator",
            "model": model_info(self.model),
        }

    def _event_dict(self, ev) -> dict[str, Any]:
        d = ev.to_dict(self._t_last)
        # "active" = not yet ended at the latest sample; an event injected for the next sample
        # (onset = t_last + dt) is therefore reported active immediately.
        d["active"] = bool(self._t_last < ev.end_t)
        return d

    def _shield_rows(self) -> list[dict[str, Any]]:
        meta = self.meta
        S, L = meta.n_shields, meta.legs
        V = self._last_values.reshape(S, L)
        valid = np.isfinite(V)
        cnt = valid.sum(axis=1)
        sums = np.where(valid, V, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.nan)
        force = np.where(cnt > 0, force_mn(sums, self._area), np.nan)
        loaded = np.array([s == "loaded" for s in self._last_states])
        ref = loaded & np.isfinite(mean)
        med = float(np.median(mean[ref])) if ref.any() else math.nan
        dev = mean - med
        ai = _rowmax(self._last_scores.reshape(S, L))

        cat: dict[str, set[int]] = {"process": set(), "sensor": set(), "telemetry": set()}
        for a in self.pipeline.correlator.active_alerts:
            cat.setdefault(a.category, set()).update(int(s) for s in a.shields)

        rows = []
        q = self._last_quality
        vals = _jlist(self._last_values, 2)
        for i in range(S):
            if cnt[i] == 0:
                status = "no_data"
            elif i in cat["sensor"] or i in cat["telemetry"]:
                status = "sensor_fault"
            elif i in cat["process"]:
                status = "process_alert"
            elif math.isfinite(ai[i]) and ai[i] >= self._watch:
                status = "watch"
            else:
                status = "normal"
            chans = range(i * L, (i + 1) * L)
            rows.append({
                "id": meta.shield_ids[i],
                "index": i,
                "gateway": meta.gateway_of[i],
                "state": self._last_states[i],
                "time_in_state_s": int(round(float(self._last_tis[i]))),
                "legs": [{"channel": meta.channel_ids[c], "value": vals[c], "quality": q[c]} for c in chans],
                "mean_bar": _opt(mean[i], 2),
                "force_MN": _opt(force[i], 3),
                "deviation_bar": _opt(dev[i], 2),
                "ai_score": _opt(ai[i], 3),
                "status": status,
            })
        return rows

    def _window(self, window_s: Any) -> float:
        try:
            w = float(window_s)
        except (TypeError, ValueError):
            w = WINDOW_DEFAULT_S
        if not math.isfinite(w):
            w = WINDOW_DEFAULT_S
        return min(max(w, WINDOW_MIN_S), WINDOW_MAX_S)

    def _state(self, shield: str | None, window_s: Any) -> dict[str, Any]:
        meta, h = self.meta, self.history
        S, L = meta.n_shields, meta.legs
        w = self._window(window_s)
        si = self._shield_index.get(shield, 0) if isinstance(shield, str) else 0
        sid = meta.shield_ids[si]
        n_win = int(round(w / self._dt))
        ks = h.krange(h.last_k - n_win + 1, h.last_k)
        sl = h.slots(ks)
        ts = (ks * self._dt).astype(float)

        # series (1 s) of the selected shield
        cols = np.arange(si * L, (si + 1) * L)
        vals = h.values[sl][:, cols]
        series = {
            "shield": sid,
            "t": ts.tolist(),
            "channels": {meta.channel_ids[c]: _jlist(vals[:, j], 2) for j, c in enumerate(cols)},
            "state": [_STATE_NAMES[c] for c in h.state[sl, si].tolist()],
            "ai_score": _jlist(_rowmax(h.score[sl][:, cols].astype(float)), 3),
        }

        # heatmap: shield mean every 5 s (aligned to multiples of 5 s), rows ascending in t
        hstep = max(1, int(round(HEATMAP_STEP_S / self._dt)))
        kh = ks[ks % hstep == 0]
        hv = h.values[h.slots(kh)].reshape(len(kh), S, L)
        heatmap = {"t": (kh * self._dt).astype(float).tolist(), "shields": list(meta.shield_ids),
                   "values": _jlist(_rowmean(hv), 2)}

        corr = self.pipeline.correlator
        q = self._last_quality
        state = {
            "sim": self._sim_info(),
            "counts": {
                "active_alerts": len(corr.active_alerts),
                "channels_monitored": meta.n_channels,
                "channels_ok": int(sum(1 for x in q if x == QUALITY_OK)),
                "readings_total": int(self._readings_total),
            },
            "shields": self._shield_rows(),
            "heatmap": heatmap,
            "series": series,
            "alerts": corr.summaries(limit=MAX_ALERTS_IN_STATE),
            "events": [self._event_dict(e) for e in reversed(self.sim.events)],
            "devices": self.monitor.status(self._t_last),
        }
        return json_clean(state)

    def _alert_series(self, alert, t0: float, t1: float) -> dict[str, Any]:
        meta, h = self.meta, self.history
        L = meta.legs
        shields = sorted(int(s) for s in alert.shields)
        ks = h.krange(math.ceil(t0 / self._dt - 1e-9), math.floor(t1 / self._dt + 1e-9))
        sl = h.slots(ks)
        V = h.values[sl]
        SC = h.score[sl].astype(float)
        channels: dict[str, list] = {}
        ai: dict[str, list] = {}
        for s in shields:
            cols = np.arange(s * L, (s + 1) * L)
            for c in cols:
                channels[meta.channel_ids[c]] = _jlist(V[:, c], 2)
            ai[meta.shield_ids[s]] = _jlist(_rowmax(SC[:, cols]), 3)
        return {"t": (ks * self._dt).astype(float).tolist(), "channels": channels, "ai_score": ai}

    def _neighbors(self, shield_ids: Sequence[str], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        idx = {self._shield_index[s] for s in shield_ids if s in self._shield_index}
        S = self.meta.n_shields
        nb = sorted({j for i in idx for j in range(i - NEIGHBOR_SPAN, i + NEIGHBOR_SPAN + 1)
                     if 0 <= j < S and j not in idx})
        return [{"shield": rows[j]["id"], "mean_bar": rows[j]["mean_bar"],
                 "deviation_bar": rows[j]["deviation_bar"], "status": rows[j]["status"]} for j in nb]

    def _alert_detail(self, alert_id: str, with_series: bool) -> dict[str, Any]:
        corr = self.pipeline.correlator
        if not isinstance(alert_id, str):
            raise KeyError(alert_id)
        try:
            d = corr.detail(alert_id, self._t_last,
                            series_provider=self._alert_series if with_series else None,
                            events=self.sim.events)
        except KeyError:
            if corr.get(alert_id) is None:
                raise KeyError(alert_id) from None
            raise
        d["neighbors"] = self._neighbors(d.get("shields", []), self._shield_rows())
        if not with_series:
            d.pop("series", None)
        return json_clean(d)


def _csv_lines(snap: dict[str, Any]) -> Iterator[str]:
    yield ",".join(CSV_COLUMNS) + "\n"
    chans = snap["channel_ids"]
    cs = snap["channel_shield"]
    shield_of = [snap["shield_ids"][s] for s in cs]
    gw_of = [snap["gateways"][s] for s in cs]
    labels, ev_ids = snap["labels"], snap["event_ids"]
    start = snap["start"]
    C = len(chans)
    for r in range(len(snap["t"])):
        t = float(snap["t"][r])
        ts = format_timestamp(start, t)
        seq0 = int(snap["seq"][r])
        vals = snap["values"][r].tolist()
        qual = snap["quality"][r].tolist()
        st = snap["state"][r].tolist()
        gl = snap["gt_label"][r].tolist()
        ge = snap["gt_event"][r].tolist()
        t_s = repr(t)
        lines = []
        for c in range(C):
            v = vals[c]
            vs = repr(round(v, 4)) if v == v else ""
            e = ge[c]
            lines.append(f"{t_s},{ts},{seq0 + c},{gw_of[c]},{shield_of[c]},{chans[c]},{vs},bar,"
                         f"{_QUALITY_NAMES[qual[c]]},{_STATE_NAMES[st[cs[c]]]},{labels[gl[c]]},"
                         f"{'' if e < 0 else ev_ids[e]}\n")
        yield from lines
