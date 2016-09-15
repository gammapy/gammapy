# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from regions import CircleSkyRegion
from ...data import DataStore, ObservationList, ObservationStats, Target
from ...utils.testing import requires_data, requires_dependency
from ...background import reflected_regions_background_estimate as refl
from ...image import SkyMask


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
    return SkyMask.read('$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits')


@pytest.fixture(scope='session')
def stats(target, mask):
    obs = get_obs(23523)
    bg = refl(target.on_region, obs.pointing_radec, mask, obs.events)
    return ObservationStats.from_target(obs, target, bg)


@pytest.fixture(scope='session')
def stats_stacked(target, mask):
    obs_list = get_obs_list()
    obs_stats = list()
    for obs in obs_list:
        bg = refl(target.on_region, obs.pointing_radec, mask, obs.events)
        obs_stats.append(ObservationStats.from_target(obs, target, bg))

    return ObservationStats.stack(obs_stats)


# TODO: parametrize tests using single and stacked stats!

@requires_data('gammapy-extra')
@requires_dependency('scipy')
class TestObservationStats(object):
    def test_str(self, stats):
        text = str(stats)
        assert 'Observation summary report' in text

    def test_to_dict(self, stats):
        data = stats.to_dict()
        assert data['sigma'] == 15.301017160023662

    def test_stack(self, stats_stacked):
        assert_allclose(stats_stacked.alpha, 0.284, rtol=1e-2)
        assert_allclose(stats_stacked.sigma, 23.5757575757, rtol=1e-3)
