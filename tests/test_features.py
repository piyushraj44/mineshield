"""Unit tests for mineshield.features (MODEL_SPEC section 4). Synthetic inputs only, no simulator."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mineshield.features import (  # noqa: E402
    FEATURE_INDEX, FEATURE_NAMES, FeatureEngine, FeatureFrame, nanmedian_cols,
)


def load_cfg():
    with open(ROOT / "config" / "default.json", encoding="utf-8") as fh:
        return json.load(fh)


def feed(fe, seq, states=None, tis=None, t0=0.0):
    """Feed rows of ``seq`` (T, C). All shields 'loaded' with time_in_state 1000 + k unless given.

    ``states`` / ``tis`` may be callables k -> list / array. Returns the FeatureFrames."""
    out = []
    for k in range(seq.shape[0]):
        st = states(k) if callable(states) else (states or ["loaded"] * fe.S)
        ti = tis(k) if callable(tis) else (np.full(fe.S, 1000.0 + k) if tis is None else tis)
        out.append(fe.update(t0 + k, seq[k], st, ti))
    return out


def col(ff, name):
    return ff.X[:, FEATURE_INDEX[name]]


def test_feature_names_match_spec():
    assert FEATURE_NAMES == ["pressure_bar", "dev_own_bar", "slope_5_bar_s", "slope_30_bar_s",
                             "imbalance_frac", "neighbor_dev_bar", "roll_std_bar", "missing_frac"]


def test_nanmedian_cols_matches_numpy():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(17, 9))
    a[rng.random(a.shape) < 0.3] = np.nan
    a[:, 4] = np.nan  # all-NaN column
    med, n = nanmedian_cols(a)
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning):
        ref = np.nanmedian(a, axis=0)
    assert np.allclose(med, ref, equal_nan=True)
    assert np.isnan(med[4]) and n[4] == 0
    assert np.array_equal(n, np.sum(~np.isnan(a), axis=0))


def test_frame_shapes_and_pressure():
    cfg = load_cfg()
    fe = FeatureEngine(3, 2, cfg)
    vals = np.array([300.0, 301.0, np.nan, 299.0, 310.0, 305.0])
    ff = fe.update(0.0, vals, ["loaded"] * 3, np.full(3, 100.0))
    assert isinstance(ff, FeatureFrame)
    assert ff.X.shape == (6, 8)
    assert np.allclose(col(ff, "pressure_bar"), vals, equal_nan=True)
    assert ff.valid.tolist() == [True, True, False, True, True, True]
    assert np.allclose(ff.shield_mean, [300.5, 299.0, 307.5])
    cf = ff.channel_features(2)
    assert cf["pressure_bar"] is None and set(cf) == set(FEATURE_NAMES)


def test_constant_and_linear_ramp_slopes():
    cfg = load_cfg()
    fe = FeatureEngine(1, 2, cfg)
    T = 60
    k = np.arange(T, dtype=float)
    seq = np.column_stack([300.0 + 0.5 * k, np.full(T, 300.0)])
    frames = feed(fe, seq)
    # slope_5 needs h[5]; slope_30 needs >= 3 valid in h[25..29]
    assert np.isnan(col(frames[4], "slope_5_bar_s")[0])
    assert col(frames[5], "slope_5_bar_s")[0] == pytest.approx(0.5)
    assert np.isnan(col(frames[26], "slope_30_bar_s")[0])
    assert np.isfinite(col(frames[27], "slope_30_bar_s")[0])
    for ff in frames[30:]:
        assert col(ff, "slope_5_bar_s")[0] == pytest.approx(0.5)
        assert col(ff, "slope_30_bar_s")[0] == pytest.approx(0.5)
        assert col(ff, "slope_5_bar_s")[1] == 0.0
        assert col(ff, "slope_30_bar_s")[1] == 0.0


def test_dev_own_needs_enough_samples_of_current_loaded_run():
    cfg = load_cfg()
    fe = FeatureEngine(1, 2, cfg)
    # run 1 loaded at 400 bar (100 s), advancing 10 s, setting 30 s, then run 2 loaded at 300 bar
    phases = [("loaded", 100, 400.0), ("advancing", 10, 15.0), ("setting", 30, 300.0), ("loaded", 120, 300.0)]
    states, tis, vals = [], [], []
    for st, n, v in phases:
        for j in range(n):
            states.append(st)
            tis.append(float(j))
            vals.append(v)
    vals = np.array(vals)
    vals[140 + 60] = 312.5  # run-2 loaded sample with time_in_state 60
    seq = np.column_stack([vals, vals])
    frames = feed(fe, seq, states=lambda k: [states[k]], tis=lambda k: np.array([tis[k]]))
    run2 = 140
    for ff in frames[100:run2]:                   # not loaded -> NaN
        assert np.isnan(col(ff, "dev_own_bar")).all()
    # baseline k in [30, 180] restricted to k <= time_in_state: 20 samples first at tis = 49
    assert np.isnan(col(frames[run2 + 48], "dev_own_bar")).all()
    assert col(frames[run2 + 49], "dev_own_bar")[0] == 0.0   # run-1 values (400 bar) are excluded
    assert col(frames[run2 + 60], "dev_own_bar")[0] == pytest.approx(12.5)
    # run 1: also NaN before 20 baseline samples, then 0
    assert np.isnan(col(frames[48], "dev_own_bar")[0])
    assert col(frames[49], "dev_own_bar")[0] == 0.0


def test_dev_own_ignores_last_30_samples():
    cfg = load_cfg()
    fe = FeatureEngine(1, 2, cfg)
    vals = np.full(260, 300.0)
    vals[200:] = 330.0
    frames = feed(fe, np.column_stack([vals, vals]), tis=lambda k: np.array([float(k)]))
    # 29 samples after the step the baseline window h[30..180] is still entirely pre-step
    assert col(frames[229], "dev_own_bar")[0] == pytest.approx(30.0)
    # dev_own ignores missing samples in the baseline but needs v valid
    fe2 = FeatureEngine(1, 2, cfg)
    vals2 = np.full(100, 300.0)
    vals2[99] = np.nan
    frames2 = feed(fe2, np.column_stack([vals2, np.full(100, 300.0)]), tis=lambda k: np.array([float(k)]))
    assert np.isnan(col(frames2[99], "dev_own_bar")[0]) and col(frames2[99], "dev_own_bar")[1] == 0.0


def test_imbalance_two_legs():
    cfg = load_cfg()
    fe = FeatureEngine(1, 2, cfg)
    ff = fe.update(0.0, [330.0, 300.0], ["loaded"], [100.0])
    imb = col(ff, "imbalance_frac")
    assert imb[0] == pytest.approx(30.0 / 315.0)
    assert imb[1] == pytest.approx(-30.0 / 315.0)
    ff = fe.update(1.0, [330.0, np.nan], ["loaded"], [101.0])
    assert np.isnan(col(ff, "imbalance_frac")).all()   # no other valid leg


def test_imbalance_four_legs():
    cfg = load_cfg()
    fe = FeatureEngine(1, 4, cfg)
    ff = fe.update(0.0, [400.0, 300.0, 300.0, 300.0], ["loaded"], [100.0])
    imb = col(ff, "imbalance_frac")
    assert imb[0] == pytest.approx((400.0 - 300.0) / 325.0)
    assert imb[1] == pytest.approx((300.0 - 1000.0 / 3.0) / 325.0)
    ff = fe.update(1.0, [400.0, 300.0, 300.0, np.nan], ["loaded"], [101.0])
    imb = col(ff, "imbalance_frac")
    assert imb[0] == pytest.approx(100.0 / (1000.0 / 3.0))
    assert np.isnan(imb[3])


def test_neighbor_deviation():
    cfg = load_cfg()
    assert cfg["features"]["neighbor_radius"] == 2
    fe = FeatureEngine(5, 2, cfg)
    means = [300.0, 310.0, 340.0, 300.0, 305.0]
    vals = np.repeat(means, 2)
    ff = fe.update(0.0, vals, ["loaded"] * 5, np.full(5, 100.0))
    nd = col(ff, "neighbor_dev_bar")
    assert nd[4] == pytest.approx(340.0 - np.median([300, 310, 300, 305]))  # shield 2, leg 1
    assert nd[5] == nd[4]                                                    # same for every leg
    assert nd[0] == pytest.approx(300.0 - np.median([310, 340]))             # edge shield
    # a non-loaded neighbour is excluded and the non-loaded shield itself gets NaN
    ff = fe.update(1.0, vals, ["loaded", "loaded", "loaded", "setting", "loaded"], np.full(5, 101.0))
    nd = col(ff, "neighbor_dev_bar")
    assert nd[4] == pytest.approx(340.0 - np.median([300, 310, 305]))
    assert np.isnan(nd[6]) and np.isnan(nd[7])
    # a neighbour without valid legs is excluded; a shield without loaded/valid neighbours -> NaN
    v2 = vals.copy()
    v2[2:4] = np.nan
    ff = fe.update(2.0, v2, ["loaded", "loaded", "loaded", "advancing", "advancing"], np.full(5, 102.0))
    nd = col(ff, "neighbor_dev_bar")
    assert nd[4] == pytest.approx(340.0 - 300.0)
    assert np.isnan(nd[2])


def test_missing_frac_and_missing_run():
    cfg = load_cfg()
    fe = FeatureEngine(1, 2, cfg)
    v = np.full(40, 300.0)
    v[[2, 5, 7, 35]] = np.nan
    v[36:39] = np.nan
    frames = feed(fe, np.column_stack([v, np.full(40, 300.0)]))
    # samples before t = 0 do not count
    assert col(frames[0], "missing_frac")[0] == 0.0
    assert col(frames[2], "missing_frac")[0] == pytest.approx(1 / 3)
    assert col(frames[9], "missing_frac")[0] == pytest.approx(3 / 10)
    assert col(frames[29], "missing_frac")[0] == pytest.approx(3 / 30)
    assert col(frames[34], "missing_frac")[0] == pytest.approx(2 / 30)     # window k = 5..34: 5, 7
    assert col(frames[35], "missing_frac")[0] == pytest.approx(2 / 30)     # window 6..35: 7, 35
    assert col(frames[39], "missing_frac")[0] == pytest.approx(4 / 30)     # 35, 36, 37, 38
    assert col(frames[39], "missing_frac")[1] == 0.0
    assert [int(f.missing_run[0]) for f in frames[33:40]] == [0, 0, 1, 2, 3, 4, 0]


def test_flat_run():
    cfg = load_cfg()
    fe = FeatureEngine(1, 2, cfg)
    v = np.array([300.0, 301.0, 301.0, 301.0, 302.0, np.nan, 302.0, 302.0, 302.0])
    frames = feed(fe, np.column_stack([v, v + 0.1 * np.arange(len(v))]))
    assert [int(f.flat_run[0]) for f in frames] == [0, 0, 1, 2, 0, 0, 0, 1, 2]
    assert all(int(f.flat_run[1]) == 0 for f in frames[:5])


def test_roll_std_uses_last_valid_values():
    cfg = load_cfg()
    fe = FeatureEngine(1, 2, cfg)
    rng = np.random.default_rng(5)
    v = 300.0 + rng.normal(0.0, 3.0, 40)
    v[[1, 3, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]] = np.nan
    frames = feed(fe, np.column_stack([v, np.full(40, 300.0)]))
    valid_idx = [k for k in range(40) if np.isfinite(v[k])]
    # needs >= 5 valid values: the 5th valid value is at sample valid_idx[4]
    assert np.isnan(col(frames[valid_idx[4] - 1], "roll_std_bar")[0])
    assert np.isfinite(col(frames[valid_idx[4]], "roll_std_bar")[0])
    for k in (19, 30, 39):
        last10 = v[[i for i in valid_idx if i <= k][-10:]]
        med = np.median(last10)
        expect = 1.4826 * np.median(np.abs(last10 - med))
        assert col(frames[k], "roll_std_bar")[0] == pytest.approx(expect)
    assert col(frames[39], "roll_std_bar")[1] == 0.0


def _noisy(T, C, seed=1, sd=1.0, level=300.0):
    rng = np.random.default_rng(seed)
    return np.round((level + rng.normal(0.0, sd, (T, C))) / 0.1) * 0.1


def test_spike_detected_on_single_sample_spike():
    cfg = load_cfg()
    fe = FeatureEngine(2, 2, cfg)
    seq = _noisy(80, 4)
    seq[50, 1] += 150.0
    seq[60, 2] -= 120.0
    frames = feed(fe, seq)
    spikes = [(k, c) for k, f in enumerate(frames) for c in np.flatnonzero(f.spike)]
    assert spikes == [(51, 1), (61, 2)]          # 1-sample delay
    assert frames[51].spike_jump[1] == pytest.approx(150.0, abs=6.0)
    assert frames[61].spike_jump[2] == pytest.approx(-120.0, abs=6.0)


def test_spike_not_detected_on_sustained_step():
    cfg = load_cfg()
    fe = FeatureEngine(2, 2, cfg)
    seq = _noisy(120, 4)
    seq[50:, 1] += 150.0
    frames = feed(fe, seq)
    assert not any(f.spike.any() for f in frames)


def test_slope_30_robust_to_single_spike():
    cfg = load_cfg()
    fe = FeatureEngine(1, 2, cfg)
    seq = np.full((90, 2), 300.0)
    seq[50, 0] = 450.0
    frames = feed(fe, seq)
    for ff in frames[30:]:
        assert col(ff, "slope_30_bar_s")[0] == 0.0
    assert col(frames[50], "slope_5_bar_s")[0] == pytest.approx(30.0)
    assert col(frames[55], "slope_5_bar_s")[0] == pytest.approx(-30.0)


def test_loaded_ok_rise_ok_and_scoreable():
    cfg = load_cfg()
    fe = FeatureEngine(5, 2, cfg)
    states = ["loaded", "loaded", "setting", "setting", "advancing"]
    tis = np.array([29.0, 30.0, 14.0, 15.0, 5.0])
    ff = fe.update(0.0, np.full(10, 300.0), states, tis)
    assert ff.loaded_ok.tolist() == [False, False, True, True, False, False, False, False, False, False]
    # rise rules have no blind spot at the setting -> loaded transition
    assert ff.rise_ok.tolist() == [True, True, True, True, False, False, True, True, False, False]
    assert not ff.scoreable.any()     # dev_own and slopes not yet computable
    # after a long loaded run with valid data every channel becomes scoreable
    fe = FeatureEngine(2, 2, cfg)
    frames = feed(fe, _noisy(60, 4), tis=lambda k: np.full(2, float(k)))
    assert not frames[48].scoreable.any()
    assert frames[49].scoreable.all()
    v = _noisy(1, 4)[0]
    v[3] = np.nan
    ff = fe.update(60.0, v, ["loaded", "loaded"], np.full(2, 60.0))
    assert ff.scoreable.tolist() == [True, True, True, False]


def test_reset_and_validation():
    cfg = load_cfg()
    fe = FeatureEngine(1, 2, cfg)
    feed(fe, np.full((40, 2), 300.0))
    fe.reset()
    ff = fe.update(0.0, [300.0, 300.0], ["loaded"], [100.0])
    assert np.isnan(col(ff, "slope_5_bar_s")).all() and int(ff.flat_run[0]) == 0
    assert fe.recent(3).shape == (3, 2)
    with pytest.raises(ValueError):
        fe.update(1.0, [300.0], ["loaded"], [100.0])
    with pytest.raises(ValueError):
        fe.update(1.0, [300.0, 300.0], ["loaded", "loaded"], [100.0])
