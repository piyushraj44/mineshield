import math

import numpy as np
import pytest

from mineshield.physics import BAR_TO_PA, force_mn, piston_area_m2


def test_piston_area_for_032_m_bore():
    assert piston_area_m2(0.32) == pytest.approx(0.080425, abs=1e-6)
    assert piston_area_m2(0.32) == pytest.approx(math.pi * 0.16 ** 2)


def test_piston_area_from_config(cfg):
    assert piston_area_m2(cfg["piston_bore_m"]) == pytest.approx(0.0804, abs=1e-4)


def test_force_300_bar():
    area = piston_area_m2(0.32)
    assert BAR_TO_PA == 1e5
    assert force_mn(300.0, area) == pytest.approx(2.413, abs=1e-3)
    assert force_mn(0.0, area) == 0.0


def test_force_vectorised_and_nan_propagates():
    area = piston_area_m2(0.32)
    p = np.array([300.0, np.nan, 150.0])
    f = force_mn(p, area)
    assert f.shape == (3,)
    assert f[0] == pytest.approx(2.413, abs=1e-3)
    assert np.isnan(f[1])
    assert f[2] == pytest.approx(f[0] / 2)
    assert math.isnan(force_mn(float("nan"), area))
    assert np.isnan(force_mn(300.0, np.nan))


def test_lists_and_array_bores():
    areas = piston_area_m2([0.32, 0.4])
    assert isinstance(areas, np.ndarray)
    assert areas[1] == pytest.approx(math.pi * 0.2 ** 2)
    assert np.allclose(force_mn([300.0, 300.0], areas), 300.0 * 1e5 * areas / 1e6)
