# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from regions import CircleSkyRegion
from ...data import DataStore, Observations, ObservationStats
from ...utils.testing import requires_data
from ...background import ReflectedRegionsBackgroundEstimator


@pytest.fixture(scope="session")
def observations():
    data_store = DataStore.from_dir("$GAMMAPY_EXTRA/datasets/hess-dl3-dr1/")
    run_list = [23523, 23526]
    return Observations([data_store.obs(_) for _ in run_list])


@pytest.fixture(scope="session")
def on_region():
    pos = SkyCoord(83.63 * u.deg, 22.01 * u.deg)
    on_size = 0.3 * u.deg
    return CircleSkyRegion(pos, on_size)


@pytest.fixture(scope="session")
def stats(on_region, observations):
    obs = observations[0]
    bge = ReflectedRegionsBackgroundEstimator(on_region=on_region, observations=obs)
    bg = bge.process(obs)
    return ObservationStats.from_observation(obs, bg)


@pytest.fixture(scope="session")
def stats_stacked(on_region, observations):
    bge = ReflectedRegionsBackgroundEstimator(
        on_region=on_region, observations=observations
    )
    bge.run()

    return ObservationStats.stack(
        [
            ObservationStats.from_observation(obs, bg)
            for obs, bg in zip(observations, bge.result)
        ]
    )


@requires_data("gammapy-extra")
class TestObservationStats(object):
    def test_str(self, stats):
        text = str(stats)
        assert "Observation summary report" in text

    def test_to_dict(self, stats):
        data = stats.to_dict()
        assert data["n_on"] == 425
        assert data["n_off"] == 395
        assert_allclose(data["alpha"], 0.333, rtol=1e-2)
        assert_allclose(data["sigma"], 16.430, rtol=1e-3)

    def test_stack(self, stats_stacked):
        data = stats_stacked.to_dict()
        assert data["n_on"] == 900
        assert data["n_off"] == 766
        assert_allclose(data["alpha"], 0.333, rtol=1e-2)
        assert_allclose(data["sigma"], 25.244, rtol=1e-3)
