"""Streaming per-channel feature extraction (MODEL_SPEC section 4).

``FeatureEngine.update`` consumes one sample of the whole face (all channels) and returns a
``FeatureFrame``. All computations are vectorised over channels with numpy; the engine keeps a
ring buffer of the last ``history_s`` samples per channel (NaN = missing) plus small run counters.

Notation (as in the spec): ``v`` is the current value and ``h[k]`` the value ``k`` samples ago
(``h[0] == v``). Samples before t = 0 are treated as missing and are never counted in fractions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

FEATURE_NAMES: list[str] = [
    "pressure_bar",
    "dev_own_bar",
    "slope_5_bar_s",
    "slope_30_bar_s",
    "imbalance_frac",
    "neighbor_dev_bar",
    "roll_std_bar",
    "missing_frac",
]
FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}

# Fixed by the spec text (not configurable): 1.4826 * MAD ~ standard deviation for normal data,
# robust medians need >= 3 valid samples, roll_std needs >= 5 valid values, and the spike test
# compares h[1] / v against median(h[2..6]).
MAD_TO_SD = 1.4826
MIN_MEDIAN_VALID = 3
MIN_ROLL_VALID = 5
SPIKE_MED_LAGS = (2, 6)


_COLS: dict[int, np.ndarray] = {}


def nanmedian_cols(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise median ignoring NaN, for a 2-D array ``(K, C)``.

    Returns ``(median (C,), n_valid (C,))``; the median is NaN where a column has no valid value.
    Much faster than ``np.nanmedian(axis=0)`` for the small windows used here.
    """
    K, C = a.shape
    cols = _COLS.get(C)
    if cols is None:
        cols = _COLS[C] = np.arange(C)
    s = np.sort(a, axis=0)  # NaN sorts last
    n = K - np.isnan(s).sum(axis=0)
    lo = (n - 1) // 2
    np.maximum(lo, 0, out=lo)  # n == 0 -> row 0, which is NaN -> median NaN
    med = (s[lo, cols] + s[n // 2, cols]) * 0.5
    return med, n


def _json_num(x: Any, nd: int = 4) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


@dataclass
class FeatureFrame:
    """Features of every channel at one sample.

    ``X`` is ``(C, 8)`` in ``FEATURE_NAMES`` order, NaN = not computable. Per-channel boolean / int
    arrays have shape ``(C,)``; ``states``, ``time_in_state`` and ``shield_mean`` are per shield.
    ``spike_jump`` (extension) holds ``h[1] - median(h[2..6])`` used by the spike test (NaN if not
    computable); the sensor_spike rule reports it as the observed value.
    """

    t: float
    X: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    states: list
    time_in_state: np.ndarray
    loaded_ok: np.ndarray
    rise_ok: np.ndarray
    flat_run: np.ndarray
    missing_run: np.ndarray
    spike: np.ndarray
    scoreable: np.ndarray
    shield_mean: np.ndarray
    spike_jump: np.ndarray
    legs: int = 2

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_shields(self) -> int:
        return int(self.shield_mean.shape[0])

    def feature(self, name: str) -> np.ndarray:
        """Column of feature ``name`` (view, shape (C,))."""
        return self.X[:, FEATURE_INDEX[name]]

    def channel_features(self, c: int) -> dict[str, float | None]:
        """JSON-safe dict of all features of channel ``c`` (NaN -> None)."""
        return {name: _json_num(self.X[c, j]) for j, name in enumerate(FEATURE_NAMES)}


class FeatureEngine:
    """Streaming feature extractor: one ``update`` per sample (MODEL_SPEC section 4).

    ``cfg`` is the full configuration dict; ``cfg["features"]`` provides the windows and
    ``cfg["rules"]`` the spike thresholds. Windows given in seconds are converted to samples with
    ``cfg["simulator"]["dt_s"]`` (1 s by default, so seconds == samples).
    """

    def __init__(self, n_shields: int, legs: int, cfg: dict):
        fc = cfg["features"]
        rc = cfg["rules"]
        self.S = int(n_shields)
        self.L = int(legs)
        if self.S < 1 or self.L < 1:
            raise ValueError("n_shields and legs must be >= 1")
        self.C = self.S * self.L
        self.dt = float(cfg.get("simulator", {}).get("dt_s", 1.0))

        def ns(seconds: float) -> int:
            return max(1, int(round(float(seconds) / self.dt)))

        bw = [ns(x) for x in fc["baseline_window_s"]]
        self.k_base_lo, self.k_base_hi = min(bw), max(bw)            # k in [30, 180]
        self.base_min = int(fc["baseline_min_samples"])               # 20
        self.k_short = ns(fc["slope_short_s"])                        # 5
        self.k_long = ns(fc["slope_long_s"])                          # 30
        if self.k_long <= self.k_short:
            raise ValueError("slope_long_s must exceed slope_short_s")
        self.roll_n = ns(fc["roll_window_s"])                         # last 10 valid values
        self.miss_n = ns(fc["missing_window_s"])                      # 30 samples
        self.radius = int(fc["neighbor_radius"])
        self.min_loaded_s = float(fc["min_loaded_s"])
        self.min_setting_s = float(fc["min_setting_s"])
        self.spike_jump_bar = float(rc["spike_jump_bar"])
        self.spike_return_bar = float(rc["spike_return_bar"])

        # Rows needed per update (oldest lag + 1) and ring length (>= history_s).
        self.span = max(self.k_base_hi, self.k_long, self.miss_n, SPIKE_MED_LAGS[1], self.k_short) + 1
        self.H = max(ns(fc["history_s"]), self.span)

        # Precomputed index helpers.
        self._shield_of = np.arange(self.C) // self.L
        self._kvec = np.arange(self.k_base_hi, self.k_base_lo - 1, -1, dtype=float)  # oldest first
        offs = np.array([o for o in range(-self.radius, self.radius + 1) if o != 0], dtype=np.intp)
        nb = np.arange(self.S)[:, None] + offs[None, :]
        self._nb_inrange = (nb >= 0) & (nb < self.S)
        self._nb_idx = np.clip(nb, 0, self.S - 1)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Forget all history (as at t = 0)."""
        self._buf = np.full((2 * self.H, self.C), np.nan)  # double-written ring -> contiguous slices
        self._pos = -1
        self.n = 0                                          # samples seen
        self._vbuf = np.full((self.roll_n, self.C), np.nan)  # last valid values per channel
        self._vcount = np.zeros(self.C, dtype=np.int64)
        self.flat_run = np.zeros(self.C, dtype=np.int64)
        self.missing_run = np.zeros(self.C, dtype=np.int64)

    def recent(self, n: int) -> np.ndarray:
        """Last ``n`` samples (``n <= history``), oldest first, shape ``(n, C)``; NaN = missing."""
        n = int(n)
        if not 1 <= n <= self.H:
            raise ValueError(f"n must be in [1, {self.H}]")
        if self._pos < 0:
            return np.full((n, self.C), np.nan)
        p = self._pos + self.H
        return self._buf[p - n + 1: p + 1].copy()

    # ----------------------------------------------------------------- update
    def update(self, t: float, values, states, time_in_state) -> FeatureFrame:
        S, L, C = self.S, self.L, self.C
        v = np.array(values, dtype=float).reshape(-1)
        if v.shape[0] != C:
            raise ValueError(f"expected {C} values, got {v.shape[0]}")
        valid = np.isfinite(v)
        v[~valid] = np.nan
        st = [str(s) for s in states]
        if len(st) != S:
            raise ValueError(f"expected {S} states, got {len(st)}")
        loaded_s = np.array([s == "loaded" for s in st])
        setting_s = np.array([s == "setting" for s in st])
        tis = np.asarray(time_in_state, dtype=float).reshape(-1)
        if tis.shape[0] != S:
            raise ValueError(f"expected {S} time_in_state values, got {tis.shape[0]}")

        # ring buffer write (double-written so the last `span` rows are one contiguous slice)
        self._pos = (self._pos + 1) % self.H
        self._buf[self._pos] = v
        self._buf[self._pos + self.H] = v
        self.n += 1
        p = self._pos + self.H
        R = self._buf[p - self.span + 1: p + 1]  # R[-1 - k] == h[k]
        h1 = R[-2]

        # run counters
        self.missing_run = np.where(valid, 0, self.missing_run + 1)
        self.flat_run = np.where(valid & (v == h1), self.flat_run + 1, 0)

        so = self._shield_of
        loaded_c = loaded_s[so]
        tis_c = tis[so]
        loaded_ok_s = loaded_s & (tis >= self.min_loaded_s)
        # Rise rules run throughout the loaded phase (no gap after setting) and late in setting.
        rise_ok_s = loaded_s | (setting_s & (tis >= self.min_setting_s))
        loaded_ok = loaded_ok_s[so]
        rise_ok = rise_ok_s[so]

        X = np.full((C, len(FEATURE_NAMES)), np.nan)
        X[:, 0] = v

        with np.errstate(invalid="ignore", divide="ignore"):
            # dev_own: v - median(valid h[k], k in [30, 180], k <= time in the current loaded run)
            tis_samples = np.floor(tis_c / self.dt + 1e-9)
            need = self.k_base_lo + self.base_min - 1
            elig = np.flatnonzero(loaded_c & valid & (tis_samples >= need))
            if elig.size:
                W = R[-1 - self.k_base_hi: -self.k_base_lo][:, elig]
                mask = self._kvec[:, None] <= tis_samples[elig][None, :]
                med, cnt = nanmedian_cols(np.where(mask, W, np.nan))
                ok = cnt >= self.base_min
                X[elig[ok], 1] = v[elig[ok]] - med[ok]

            # slope_5 = (v - h[5]) / 5
            X[:, 2] = (v - R[-1 - self.k_short]) / (self.k_short * self.dt)

            # slope_30 = (median(h[0..4]) - median(h[25..29])) / 25  (robust to single spikes).
            # The spike reference median(h[2..6]) is computed in the same batched sort.
            ks, kl = self.k_short, self.k_long
            a, b = SPIKE_MED_LAGS
            if b - a + 1 == ks:
                med3, cnt3 = nanmedian_cols(np.concatenate(
                    (R[-ks:], R[-kl: -kl + ks], R[-1 - b: -a]), axis=1))
                m0, m1, med5 = med3[:C], med3[C:2 * C], med3[2 * C:]
                n0, n1, c5 = cnt3[:C], cnt3[C:2 * C], cnt3[2 * C:]
            else:
                m0, n0 = nanmedian_cols(R[-ks:])
                m1, n1 = nanmedian_cols(R[-kl: -kl + ks])
                med5, c5 = nanmedian_cols(R[-1 - b: -a])
            s30 = (m0 - m1) / ((kl - ks) * self.dt)
            s30[(n0 < MIN_MEDIAN_VALID) | (n1 < MIN_MEDIAN_VALID)] = np.nan
            X[:, 3] = s30

            # imbalance_frac and shield means
            V = v.reshape(S, L)
            Vv = valid.reshape(S, L)
            Vz = np.where(Vv, V, 0.0)
            ssum = Vz.sum(axis=1)
            scnt = Vv.sum(axis=1)
            shield_mean = np.where(scnt > 0, ssum / scnt, np.nan)
            other = (ssum[:, None] - Vz) / (scnt[:, None] - 1)
            imb = (V - other) / shield_mean[:, None]
            imb_ok = Vv & (scnt[:, None] >= 2) & (shield_mean[:, None] != 0)
            X[:, 4] = np.where(imb_ok, imb, np.nan).reshape(-1)

            # neighbor_dev: shield mean - median of loaded, valid neighbours within the radius
            if self._nb_idx.shape[1]:
                nb_ok = self._nb_inrange & (loaded_s & np.isfinite(shield_mean))[self._nb_idx]
                NB = np.where(nb_ok, shield_mean[self._nb_idx], np.nan)
                nmed, ncnt = nanmedian_cols(NB.T)
                nd = np.where(loaded_s & (ncnt > 0), shield_mean - nmed, np.nan)
                X[:, 5] = nd[so]

            # roll_std: 1.4826 * MAD of the last `roll_n` valid values (needs >= 5). The per-channel
            # ring holds exactly those values (NaN until filled); median/MAD ignore their order.
            vc = np.flatnonzero(valid)
            if vc.size:
                self._vbuf[self._vcount[vc] % self.roll_n, vc] = v[vc]
                self._vcount[vc] += 1
            rmed, rcnt = nanmedian_cols(self._vbuf)
            mad, _ = nanmedian_cols(np.abs(self._vbuf - rmed))
            X[:, 6] = np.where(rcnt >= MIN_ROLL_VALID, MAD_TO_SD * mad, np.nan)

            # missing_frac over the last 30 samples (only samples since t = 0 count)
            na = min(self.miss_n, self.n)
            X[:, 7] = np.count_nonzero(np.isnan(R[-na:]), axis=0) / na

            # spike (evaluates the previous sample): |h1 - med| > jump and |v - med| < return,
            # med = median(h[2..6]) computed above
            sp_ok = valid & np.isfinite(h1) & (c5 >= MIN_MEDIAN_VALID)
            jump = np.where(sp_ok, h1 - med5, np.nan)
            spike = sp_ok & (np.abs(jump) > self.spike_jump_bar) & (np.abs(v - med5) < self.spike_return_bar)

        scoreable = valid & loaded_ok & np.isfinite(X[:, 1]) & np.isfinite(X[:, 2]) & np.isfinite(X[:, 3])

        return FeatureFrame(
            t=float(t),
            X=X,
            values=v,
            valid=valid,
            states=st,
            time_in_state=tis.copy(),
            loaded_ok=loaded_ok,
            rise_ok=rise_ok,
            flat_run=self.flat_run,
            missing_run=self.missing_run,
            spike=spike,
            scoreable=scoreable,
            shield_mean=shield_mean,
            spike_jump=jump,
            legs=L,
        )
