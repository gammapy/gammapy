# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from ...extern.regions import CircleSkyRegion
from ...data import DataStore, ObservationList, ObservationStats, Target
from ...utils.testing import requires_data, requires_dependency
from ...background import reflected_regions_background_estimate as refl
from ...image import ExclusionMask


@requires_data('gammapy-extra')
def get_obs_list():
    data_store = DataStore.from_dir(
        '$GAMMAPY_EXTRA/datasets/hess-crab4-hd-hap-prod2/')
    run_list = [23523, 23526]
    obs_list = ObservationList([data_store.obs(_) for _ in run_list])
    return obs_list


@requires_data('gammapy-extra')
def get_obs(id):
    obs_list = get_obs_list()
    for run in obs_list:
        if run.obs_id == id:
            return run


@pytest.fixture
def target():
    pos = SkyCoord(83.63 * u.deg, 22.01 * u.deg, frame='icrs')
    on_size = 0.3 * u.deg
    on_region = CircleSkyRegion(pos, on_size)

    target = Target(position=pos,
                    on_region=on_region,
                    name='Crab Nebula',
                    tag='crab')
    return target


@requires_data('gammapy-extra')
@pytest.fixture
def get_mask():
    mask = ExclusionMask.read(
        '$GAMMAPY_EXTRA/datasets/exclusion_masks/tevcat_exclusion.fits')
    return mask


@requires_data('gammapy-extra')
@requires_dependency('scipy')
def test_str(target):
    run = get_obs(23523)
    bg = refl(target.on_region,
              run.pointing_radec, get_mask(), run.events)
    stats = ObservationStats.from_target(run, target, bg)
    text = str(stats)
    assert 'Observation summary report' in text


@requires_data('gammapy-extra')
@requires_dependency('scipy')
def test_stack(target):
    obs_list = get_obs_list()
    obs_stats = list()
    for run in obs_list:
        bg = refl(target.on_region,
                  run.pointing_radec, get_mask(), run.events)
        obs_stats.append(ObservationStats.from_target(run, target, bg))
    sum_obs_stats = ObservationStats.stack(obs_stats)
    assert_allclose(sum_obs_stats.alpha, 0.284, rtol=1.e-2)
    assert_allclose(sum_obs_stats.sigma, 23.48, rtol=1.e-3)
