"""Configuration loading and validation of user-facing simulation parameters."""

from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the JSON configuration (default: config/default.json).

    A relative ``path`` is resolved against the project root (the parent of the
    ``mineshield`` package). A fresh dict is returned on every call.
    """
    p = DEFAULT_CONFIG_PATH if path is None else Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _is_int(x: Any) -> bool:
    return isinstance(x, Integral) and not isinstance(x, bool)


def _is_real(x: Any) -> bool:
    return isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(float(x))


def validate_sim_params(
    cfg: dict[str, Any],
    seed: Any,
    n_shields: Any,
    legs_per_shield: Any,
    noise_bar: Any,
    load_trend_bar_per_h: Any,
) -> dict[str, Any]:
    """Check user-facing simulation parameters against ``cfg["ranges"]``.

    Raises ``ValueError`` with a descriptive message on the first invalid value.
    Returns the normalised parameters (ints / floats) as a dict.
    """
    ranges = cfg["ranges"]

    if not _is_int(seed) or seed < 0:
        raise ValueError(f"seed must be an integer >= 0, got {seed!r}")

    lo, hi = ranges["n_shields"]
    if not _is_int(n_shields) or not lo <= n_shields <= hi:
        raise ValueError(f"n_shields must be an integer in [{lo}, {hi}], got {n_shields!r}")

    lo, hi = ranges["legs_per_shield"]
    if not _is_int(legs_per_shield) or legs_per_shield not in (2, 4) or not lo <= legs_per_shield <= hi:
        raise ValueError(f"legs_per_shield must be 2 or 4, got {legs_per_shield!r}")

    lo, hi = ranges["noise_bar"]
    if not _is_real(noise_bar) or not lo <= noise_bar <= hi:
        raise ValueError(f"noise_bar must be a number in [{lo}, {hi}] bar, got {noise_bar!r}")

    lo, hi = ranges["load_trend_bar_per_h"]
    if not _is_real(load_trend_bar_per_h) or not lo <= load_trend_bar_per_h <= hi:
        raise ValueError(
            f"load_trend_bar_per_h must be a number in [{lo}, {hi}] bar/h, got {load_trend_bar_per_h!r}"
        )

    return {
        "seed": int(seed),
        "n_shields": int(n_shields),
        "legs_per_shield": int(legs_per_shield),
        "noise_bar": float(noise_bar),
        "load_trend_bar_per_h": float(load_trend_bar_per_h),
    }
