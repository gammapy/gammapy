# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from regions import CircleSkyRegion
from ...data import DataStore, ObservationList, ObservationStats, Target
from ...utils.testing import requires_data, requires_dependency
from ...background import ReflectedRegionsBackgroundEstimator
from ...image import SkyImage


def get_obs_list():
    data_store = DataStore.from_dir('$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    run_list = [23523, 23526]
    obs_list = ObservationList([data_store.obs(_) for _ in run_list])
    return obs_list


def get_obs(id):
    obs_list = get_obs_list()
    for obs in obs_list:
        if obs.obs_id == id:
            return obs


@pytest.fixture(scope='session')
def target():
    pos = SkyCoord(83.63 * u.deg, 22.01 * u.deg)
    on_size = 0.3 * u.deg
    on_region = CircleSkyRegion(pos, on_size)

    return Target(position=pos, on_region=on_region, name='Crab Nebula', tag='crab')


@pytest.fixture(scope='session')
def mask():
    return SkyImage.read('$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits')


@pytest.fixture(scope='session')
def stats(target, mask):
    obs = get_obs(23523)
    bge = ReflectedRegionsBackgroundEstimator(on_region=target.on_region,
                                              exclusion_mask=mask,
                                              obs_list=obs)
    bg = bge.process(obs)
    return ObservationStats.from_obs(obs, bg)


@pytest.fixture(scope='session')
def stats_stacked(target, mask):
    obs_list = get_obs_list()
    bge = ReflectedRegionsBackgroundEstimator(on_region=target.on_region,
                                              exclusion_mask=mask,
                                              obs_list=obs_list)
    bge.run()

    return ObservationStats.stack([
        ObservationStats.from_obs(obs, bg) for obs, bg in zip(obs_list, bge.result)

    ])


# TODO: parametrize tests using single and stacked stats!

@requires_data('gammapy-extra')
@requires_dependency('scipy')
class TestObservationStats(object):
    def test_str(self, stats):
        text = str(stats)
        assert 'Observation summary report' in text

    def test_to_dict(self, stats):
        data = stats.to_dict()
        assert data['n_on'] == 235
        assert data['n_off'] == 107
        assert_allclose(data['alpha'], 0.333, rtol=1e-2)
        assert_allclose(data['sigma'], 16.973577445630323, rtol=1e-3)

    def test_stack(self, stats_stacked):
        data = stats_stacked.to_dict()
        assert data['n_on'] == 454
        assert data['n_off'] == 226
        assert_allclose(data['alpha'], 0.333, rtol=1e-2)
        assert_allclose(data['sigma'], 22.89225735104067, rtol=1e-3)
