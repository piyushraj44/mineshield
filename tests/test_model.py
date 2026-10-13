"""Unit tests for mineshield.model (MODEL_SPEC section 6). Synthetic Gaussian data, no simulator."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mineshield.features import FEATURE_INDEX, FeatureFrame  # noqa: E402
from mineshield.model import (  # noqa: E402
    DEFAULT_MODEL_FEATURES, MODEL_VERSION, AITrigger, AnomalyModel, model_input,
)

MEAN = np.array([320.0, 0.0, 0.0, 0.0, 0.0, 0.0])
SD = np.array([15.0, 4.0, 0.6, 0.15, 0.02, 6.0])


def load_cfg():
    with open(ROOT / "config" / "default.json", encoding="utf-8") as fh:
        return json.load(fh)


def gaussian(n, seed):
    return MEAN + SD * np.random.default_rng(seed).normal(size=(n, 6))


@pytest.fixture(scope="module")
def model():
    cfg = load_cfg()
    cfg["model"]["calibration_quantile"] = 0.995  # pin: these tests check the score arithmetic, not the tuned value
    m = AnomalyModel(cfg).fit(gaussian(20000, 1))
    m.calibrate(gaussian(30000, 2))
    return m


def test_config_columns():
    cfg = load_cfg()
    assert AnomalyModel(cfg).features == cfg["model"]["features"] == DEFAULT_MODEL_FEATURES


def test_calibrated_score_semantics(model):
    X_val = gaussian(30000, 2)
    s_val = model.score(X_val)
    # 0 = validation median, 1.0 = 99.5th percentile of the validation data
    assert np.mean(s_val >= 1.0) == pytest.approx(0.005, abs=0.0005)
    assert np.mean(s_val == 0.0) == pytest.approx(0.5, abs=0.01)
    assert model.threshold > model.median
    # held-out normal rows from the same distribution: ~0.5 % at or above 1.0
    s_test = model.score(gaussian(40000, 3))
    assert 0.002 <= np.mean(s_test >= 1.0) <= 0.009
    assert np.all(s_test >= 0.0)
    # an obvious outlier scores far above the threshold and above every held-out normal row.
    # (Isolation Forest raw scores are bounded by the shortest possible path, so the normalised
    # score saturates, here at ~2.2.)
    outlier = (MEAN + 8.0 * SD)[None, :]
    assert model.score(outlier)[0] > 2.0
    assert model.score(outlier)[0] > s_test.max()


def test_score_is_monotone_in_raw(model):
    X = gaussian(500, 9)
    raw = model.raw_scores(X)
    s = model.score(X)
    order = np.argsort(raw)
    assert np.all(np.diff(s[order]) >= -1e-12)
    assert np.allclose(s, np.maximum(0, (raw - model.median) / (model.threshold - model.median)))


def test_fast_scorer_matches_sklearn(model):
    X = np.vstack([gaussian(700, 4), MEAN + 12 * SD * np.random.default_rng(5).normal(size=(50, 6))])
    Xs = model.scaler.transform(X)
    ref = -model.forest.score_samples(Xs)
    assert model._packed is not None
    assert np.allclose(model.raw_scores(X), ref, rtol=0, atol=1e-12)   # packed path (<= 1024 rows)
    assert np.allclose(model._packed.raw(Xs), ref, rtol=0, atol=1e-12)


def test_save_load_round_trip(model, tmp_path):
    X = np.vstack([gaussian(2000, 6), (MEAN + 8 * SD)[None, :]])
    path = tmp_path / "model.pkl"
    model.save(path)
    loaded = AnomalyModel.load(path)
    assert loaded.version == MODEL_VERSION == "if-0.1.0"
    assert loaded.median == model.median and loaded.threshold == model.threshold
    assert np.array_equal(loaded.score(X), model.score(X))
    assert np.array_equal(loaded.score(X[:10]), model.score(X[:10]))
    with open(path, "rb") as fh:
        payload = pickle.load(fh)
    payload["version"] = "if-0.0.1"
    bad = tmp_path / "bad.pkl"
    with open(bad, "wb") as fh:
        pickle.dump(payload, fh)
    with pytest.raises(ValueError):
        AnomalyModel.load(bad)


def test_unfitted_and_uncalibrated_errors():
    cfg = load_cfg()
    m = AnomalyModel(cfg)
    with pytest.raises(RuntimeError):
        m.raw_scores(gaussian(3, 0))
    m.fit(gaussian(500, 0))
    with pytest.raises(RuntimeError):
        m.score(gaussian(3, 0))
    with pytest.raises(ValueError):
        m.fit(np.zeros((10, 5)))


def _frame(X, scoreable):
    C = X.shape[0]
    return FeatureFrame(
        t=10.0, X=X, values=X[:, 0].copy(), valid=np.isfinite(X[:, 0]), states=["loaded"] * (C // 2),
        time_in_state=np.full(C // 2, 100.0), loaded_ok=np.ones(C, bool), rise_ok=np.ones(C, bool),
        flat_run=np.zeros(C, int), missing_run=np.zeros(C, int), spike=np.zeros(C, bool),
        scoreable=np.asarray(scoreable, bool), shield_mean=np.full(C // 2, 300.0),
        spike_jump=np.full(C, np.nan), legs=2)


def test_model_input_and_score_frame(model):
    X = np.full((4, 8), np.nan)
    X[:, FEATURE_INDEX["pressure_bar"]] = [320.0, 321.0, 322.0, np.nan]
    X[:, FEATURE_INDEX["dev_own_bar"]] = [1.0, 2.0, 3.0, np.nan]
    X[:, FEATURE_INDEX["slope_5_bar_s"]] = 0.1
    X[:, FEATURE_INDEX["slope_30_bar_s"]] = 0.01
    X[:, FEATURE_INDEX["imbalance_frac"]] = [0.01, np.nan, 0.02, np.nan]
    X[:, FEATURE_INDEX["neighbor_dev_bar"]] = [np.nan, 3.0, 4.0, np.nan]
    ff = _frame(X, [True, True, False, False])
    Xm, idx = model_input(ff)
    assert idx.tolist() == [0, 1]
    assert Xm.shape == (2, 6)
    assert Xm[0].tolist() == [320.0, 1.0, 0.1, 0.01, 0.01, 0.0]    # NaN neighbour -> 0
    assert Xm[1].tolist() == [321.0, 2.0, 0.1, 0.01, 0.0, 3.0]     # NaN imbalance -> 0
    assert np.isnan(ff.X[0, FEATURE_INDEX["neighbor_dev_bar"]])     # frame not modified
    s = model.score_frame(ff)
    assert s.shape == (4,) and np.isfinite(s[:2]).all() and np.isnan(s[2:]).all()
    assert np.allclose(s[:2], model.score(Xm))


def test_ai_trigger_persistence():
    cfg = load_cfg()
    cfg["model"]["persist_s"] = 3  # pin the persistence this hand-built sequence was designed for
    ai = AITrigger(4, 2, cfg)
    seq = [1.2, 1.3, np.nan, 1.1, 1.2, 1.5, 0.9, 1.0, 1.0, 1.0]
    fired = []
    for k, x in enumerate(seq):
        out = ai.evaluate(float(k), [x, 0.2, np.nan, 5.0 if k == 9 else 0.0])
        fired.append([(t.channel_idx, t.shield_idx) for t in out])
        for t in out:
            assert (t.rule, t.category, t.threshold, t.unit) == ("ai_anomaly", "process", 1.0, "score")
    # NaN (not scored) and < 1.0 reset; >= 3 consecutive scores >= 1.0 needed
    assert [k for k, f in enumerate(fired) if (0, 0) in f] == [5, 9]
    assert all((3, 1) not in f for f in fired)            # single high score never fires
    ai.reset()
    assert ai.evaluate(0.0, [2.0, 2.0, 2.0, 2.0]) == []
