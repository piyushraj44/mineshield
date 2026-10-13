"""Isolation Forest anomaly model and AI trigger (MODEL_SPEC section 6).

The normalised score is NOT a probability: 0 = median raw score of the validation normal data,
1.0 = calibrated alert threshold (``calibration_quantile`` of validation normal data; 99.95th pct by default, chosen with ``python -m mineshield.tuning``).

Per-step scoring of ~24 rows through scikit-learn's ``IsolationForest.score_samples`` costs
~13 ms (joblib dispatch over 150 trees). ``AnomalyModel`` therefore packs the fitted trees into
numpy arrays and scores small batches with a vectorised traversal (~0.2 ms). The packed scorer is
verified against scikit-learn on probe points after every fit/load and disabled if it ever
disagrees; large batches (calibration) always use scikit-learn directly.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_INDEX, FeatureEngine, FeatureFrame
from .rules import Trigger, _make, _samples

MODEL_VERSION = "if-0.1.0"
DEFAULT_MODEL_FEATURES: list[str] = [
    "pressure_bar", "dev_own_bar", "slope_5_bar_s", "slope_30_bar_s", "imbalance_frac", "neighbor_dev_bar",
]
ZERO_FILL_FEATURES = frozenset({"imbalance_frac", "neighbor_dev_bar"})
FAST_MAX_ROWS = 1024       # batches up to this size use the packed numpy scorer
_FAST_CHUNK = 512
_VERIFY_TOL = 1e-9


def model_input(ff: FeatureFrame, features: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Model matrix for the scoreable rows of ``ff``.

    Returns ``(Xm (n, k), idx (n,))`` where ``idx[i]`` is the channel of row ``i``. NaN
    imbalance / neighbour deviation are replaced by 0 (spec); scoreable rows always have finite
    pressure, dev_own and slopes, and any other non-finite value (custom feature lists only) is 0.
    """
    names = list(features) if features is not None else DEFAULT_MODEL_FEATURES
    cols = [FEATURE_INDEX[n] for n in names]
    idx = np.flatnonzero(ff.scoreable)
    Xm = ff.X[np.ix_(idx, cols)]
    if Xm.size:
        Xm[~np.isfinite(Xm)] = 0.0
    return Xm, idx


def _avg_path_length(n: np.ndarray) -> np.ndarray:
    """Average path length of an unsuccessful BST search, c(n) (Liu et al.)."""
    n = np.asarray(n, dtype=float)
    out = np.zeros_like(n)
    big = n > 2
    out[n == 2] = 1.0
    out[big] = 2.0 * (np.log(n[big] - 1.0) + np.euler_gamma) - 2.0 * (n[big] - 1.0) / n[big]
    return out


class _PackedForest:
    """Fitted isolation trees as padded numpy arrays for fast small-batch scoring."""

    def __init__(self, forest: IsolationForest, n_features: int):
        trees = [est.tree_ for est in forest.estimators_]
        feats = getattr(forest, "estimators_features_", None)
        if feats is not None and any(not np.array_equal(np.sort(f), np.arange(n_features)) for f in feats):
            raise ValueError("feature subsampling is not supported by the packed scorer")
        T = len(trees)
        M = max(tr.node_count for tr in trees)
        self.T, self.M = T, M
        left = np.zeros((T, M), dtype=np.intp)
        right = np.zeros((T, M), dtype=np.intp)
        feat = np.zeros((T, M), dtype=np.intp)
        thr = np.zeros((T, M))
        leafval = np.zeros((T, M))
        max_depth = 0
        for i, tr in enumerate(trees):
            n = tr.node_count
            cl, cr = tr.children_left, tr.children_right
            depth = np.zeros(n, dtype=np.intp)
            for node in range(n):  # parents precede children in sklearn's depth-first layout
                if cl[node] != -1:
                    depth[cl[node]] = depth[node] + 1
                    depth[cr[node]] = depth[node] + 1
            is_leaf = cl == -1
            idx = np.arange(n)
            # Leaves point to themselves so extra traversal iterations are no-ops.
            left[i, :n] = np.where(is_leaf, idx, cl)
            right[i, :n] = np.where(is_leaf, idx, cr)
            feat[i, :n] = np.where(is_leaf, 0, tr.feature)
            thr[i, :n] = tr.threshold
            leafval[i, :n] = depth + _avg_path_length(tr.n_node_samples)
            max_depth = max(max_depth, int(depth.max()))
        base = (np.arange(T) * M)[None, :]
        self.base = base
        self.left = left.reshape(-1)
        self.right = right.reshape(-1)
        self.feat = feat.reshape(-1)
        self.thr = thr.reshape(-1)
        self.leafval = leafval.reshape(-1)
        self.max_depth = max_depth
        self.denominator = T * float(_avg_path_length(np.array([forest.max_samples_]))[0])

    def raw(self, Xs: np.ndarray) -> np.ndarray:
        """Anomaly score 2**(-E[h]/c) (== -forest.score_samples) for scaled rows ``Xs``."""
        Xf = np.asarray(Xs, dtype=np.float32).astype(np.float64)  # sklearn compares float32 inputs
        out = np.empty(Xf.shape[0])
        for s in range(0, Xf.shape[0], _FAST_CHUNK):
            Xc = Xf[s: s + _FAST_CHUNK]
            m = Xc.shape[0]
            flat_x = Xc.reshape(-1)
            row_off = (np.arange(m) * Xc.shape[1])[:, None]
            node = np.broadcast_to(self.base, (m, self.T)).copy()   # flat node index
            for _ in range(self.max_depth):
                x = flat_x[row_off + self.feat[node]]
                node = np.where(x <= self.thr[node], self.left[node], self.right[node]) + self.base
            depths = self.leafval[node].sum(axis=1)
            out[s: s + m] = 2.0 ** (-depths / self.denominator)
        return out


class AnomalyModel:
    """StandardScaler + IsolationForest on the ``cfg["model"]["features"]`` columns."""

    def __init__(self, cfg: dict | None = None):
        mc = (cfg or {}).get("model", {})
        self.features: list[str] = list(mc.get("features", DEFAULT_MODEL_FEATURES))
        self.n_estimators = int(mc.get("n_estimators", 150))
        self.max_samples = mc.get("max_samples", 256)
        self.random_state = mc.get("random_state", 0)
        self.calibration_quantile = float(mc.get("calibration_quantile", 0.995))
        self.version = MODEL_VERSION
        self.scaler: StandardScaler | None = None
        self.forest: IsolationForest | None = None
        self.median: float | None = None
        self.threshold: float | None = None
        self.meta: dict[str, Any] = {}
        self._packed: _PackedForest | None = None

    # --------------------------------------------------------------- fitting
    @property
    def ready(self) -> bool:
        return self.forest is not None and self.threshold is not None

    def _check_X(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.features):
            raise ValueError(f"X must have shape (n, {len(self.features)})")
        return X

    def fit(self, X) -> "AnomalyModel":
        X = self._check_X(X)
        if X.shape[0] < 2:
            raise ValueError("need at least 2 training rows")
        self.scaler = StandardScaler().fit(X)
        self.forest = IsolationForest(n_estimators=self.n_estimators, max_samples=self.max_samples,
                                      random_state=self.random_state)
        self.forest.fit((X - self.scaler.mean_) / self.scaler.scale_)
        self.median = self.threshold = None
        self.meta["n_train_rows"] = int(X.shape[0])
        self._pack()
        return self

    def _pack(self) -> None:
        """Build and verify the packed scorer; fall back to scikit-learn if it disagrees."""
        self._packed = None
        try:
            packed = _PackedForest(self.forest, len(self.features))
        except ValueError:
            return
        probe = np.random.default_rng(12345).normal(0.0, 2.0, (256, len(self.features)))
        probe[:8] *= 25.0  # include far outliers
        ref = -self.forest.score_samples(probe)
        if np.max(np.abs(packed.raw(probe) - ref)) <= _VERIFY_TOL:
            self._packed = packed

    def raw_scores(self, X) -> np.ndarray:
        """Raw Isolation Forest anomaly score (= -score_samples; higher = more abnormal)."""
        if self.forest is None:
            raise RuntimeError("model is not fitted")
        X = self._check_X(X)
        if X.shape[0] == 0:
            return np.zeros(0)
        # Same arithmetic as StandardScaler.transform (X -= mean_; X /= scale_) without the
        # per-call validation overhead.
        Xs = (X - self.scaler.mean_) / self.scaler.scale_
        if self._packed is not None and X.shape[0] <= FAST_MAX_ROWS:
            return self._packed.raw(Xs)
        return -self.forest.score_samples(Xs)

    def calibrate(self, X_val) -> "AnomalyModel":
        raw = self.raw_scores(X_val)
        if raw.size == 0:
            raise ValueError("calibration needs at least one row")
        self.median = float(np.median(raw))
        self.threshold = float(np.quantile(raw, self.calibration_quantile))
        self.meta["n_validation_rows"] = int(raw.size)
        return self

    # --------------------------------------------------------------- scoring
    def score(self, X) -> np.ndarray:
        """Normalised score max(0, (raw - median) / (threshold - median)); 1.0 = threshold."""
        if self.threshold is None or self.median is None:
            raise RuntimeError("model is not calibrated")
        raw = self.raw_scores(X)
        span = self.threshold - self.median
        if not span > 0:
            span = 1e-12
        return np.maximum(0.0, (raw - self.median) / span)

    def score_frame(self, ff: FeatureFrame) -> np.ndarray:
        """Per-channel scores ``(C,)`` for a FeatureFrame; NaN where the row is not scoreable."""
        out = np.full(ff.n_channels, np.nan)
        Xm, idx = model_input(ff, self.features)
        if idx.size:
            out[idx] = self.score(Xm)
        return out

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        payload = {
            "version": self.version,
            "features": self.features,
            "params": {"n_estimators": self.n_estimators, "max_samples": self.max_samples,
                       "random_state": self.random_state, "calibration_quantile": self.calibration_quantile},
            "scaler": self.scaler,
            "forest": self.forest,
            "median": self.median,
            "threshold": self.threshold,
            "meta": self.meta,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyModel":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict) or payload.get("version") != MODEL_VERSION:
            got = payload.get("version") if isinstance(payload, dict) else type(payload).__name__
            raise ValueError(f"incompatible model file {path}: version {got!r}, expected {MODEL_VERSION!r}")
        m = cls(None)
        m.features = list(payload["features"])
        p = payload["params"]
        m.n_estimators = p["n_estimators"]
        m.max_samples = p["max_samples"]
        m.random_state = p["random_state"]
        m.calibration_quantile = p["calibration_quantile"]
        m.scaler = payload["scaler"]
        m.forest = payload["forest"]
        m.median = payload["median"]
        m.threshold = payload["threshold"]
        m.meta = dict(payload.get("meta", {}))
        if m.forest is not None:
            m._pack()
        return m


class AITrigger:
    """``ai_anomaly`` (process) when a channel's score >= 1.0 for >= persist_s consecutive
    scored samples. A sample without a score (NaN: not scoreable) resets the counter."""

    THRESHOLD = 1.0

    def __init__(self, n_channels: int, legs: int, cfg: dict):
        self.C = int(n_channels)
        self.L = int(legs)
        self.n_persist = _samples(cfg, cfg["model"]["persist_s"])
        self.reset()

    def reset(self) -> None:
        self._cnt = np.zeros(self.C, dtype=np.int64)

    def evaluate(self, t: float, scores) -> list[Trigger]:
        sc = np.asarray(scores, dtype=float)
        with np.errstate(invalid="ignore"):
            cond = sc >= self.THRESHOLD
        self._cnt = np.where(cond, self._cnt + 1, 0)
        return [_make("ai_anomaly", c // self.L, c, t, sc[c], self.THRESHOLD, "score")
                for c in np.flatnonzero(self._cnt >= self.n_persist)]


# ------------------------------------------------------------------ training
def build_training_matrix(cfg: dict, seeds: Sequence[int], episode_s: float | None = None) -> np.ndarray:
    """Run a normal-only FaceSimulator (default params) per seed through the FeatureEngine and
    stack the scoreable rows (model feature columns, NaN imbalance/neighbour -> 0)."""
    from .simulator import FaceSimulator  # imported lazily: only needed for training

    mc = cfg["model"]
    feats = list(mc["features"])
    dt = float(cfg.get("simulator", {}).get("dt_s", 1.0))
    n_steps = int(round(float(episode_s if episode_s is not None else mc["episode_s"]) / dt))
    d = cfg.get("defaults", {})
    kwargs = {k: d[k] for k in ("n_shields", "legs_per_shield", "noise_bar", "load_trend_bar_per_h") if k in d}
    blocks: list[np.ndarray] = []
    for seed in seeds:
        sim = FaceSimulator(cfg, int(seed), **kwargs)
        S = len(sim.shield_ids)
        fe = FeatureEngine(S, len(sim.channel_ids) // S, cfg)
        for _ in range(n_steps):
            fr = sim.step()
            ff = fe.update(fr.t, fr.values, fr.states, fr.time_in_state)
            Xm, _ = model_input(ff, feats)
            if Xm.shape[0]:
                blocks.append(Xm)
    return np.vstack(blocks) if blocks else np.empty((0, len(feats)))


def _seed_text(seeds: Sequence[int]) -> str:
    s = [int(x) for x in seeds]
    if len(s) > 1 and s == list(range(s[0], s[0] + len(s))):
        return f"{s[0]}-{s[-1]}"
    return ", ".join(str(x) for x in s)


def train_default_model(cfg: dict) -> AnomalyModel:
    """Train on ``train_seeds`` (random subsample to ``max_train_rows`` with rng seed 0) and
    calibrate on ``validation_seeds``. Test seeds are never used."""
    mc = cfg["model"]
    t0 = time.perf_counter()
    X_train = build_training_matrix(cfg, mc["train_seeds"])
    n_all = int(X_train.shape[0])
    max_rows = int(mc["max_train_rows"])
    if n_all > max_rows:
        rng = np.random.default_rng(0)
        X_train = X_train[np.sort(rng.choice(n_all, size=max_rows, replace=False))]
    model = AnomalyModel(cfg).fit(X_train)
    X_val = build_training_matrix(cfg, mc["validation_seeds"])
    model.calibrate(X_val)
    model.meta.update({
        "trained_on": f"normal-only synthetic seeds {_seed_text(mc['train_seeds'])}",
        "validation_seeds": [int(s) for s in mc["validation_seeds"]],
        "n_train_rows_available": n_all,
        "threshold_note": f"score 1.0 = {100 * model.calibration_quantile:g}th percentile of validation normal data",
        "train_time_s": round(time.perf_counter() - t0, 2),
    })
    return model
