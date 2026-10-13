"""Illustrative hydraulic leg physics: piston area and leg force (F = P x A)."""

from __future__ import annotations

from typing import Any

import numpy as np

BAR_TO_PA = 1e5


def _num(x: Any) -> Any:
    """Pass scalars and arrays through; turn lists/tuples into float arrays."""
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=float)
    return x


def piston_area_m2(bore_m: Any) -> Any:
    """Piston area in m^2 for a bore diameter in m (0.32 m -> 0.080425 m^2)."""
    bore = _num(bore_m)
    return np.pi * (bore / 2.0) ** 2


def force_mn(pressure_bar: Any, area_m2: Any) -> Any:
    """Leg force in MN for a pressure in bar acting on ``area_m2`` (NaN propagates)."""
    p = _num(pressure_bar)
    a = _num(area_m2)
    return p * BAR_TO_PA * a / 1e6
