# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from .. import Target, TargetSummary
from astropy.coordinates import SkyCoord
from astropy.tests.helper import pytest
import astropy.units as u
from regions.shapes import CircleSkyRegion
from ...utils.testing import data_manager

def test_targetsummary(data_manager):
    pos = SkyCoord(83.63 * u.deg, 22.01 * u.deg, frame='icrs')
    on_size = 0.3 * u.deg
    on_region = CircleSkyRegion(pos, on_size)
    target = Target(pos, on_region, name='Test Target')
    
    data_store = data_manager['hess-crab4-hd-hap-prod2']
    obs_ids = [23523, 23592]
    obs = [data_store.obs(_) for _ in obs_ids]
    
    summary = TargetSummary(target, obs)
    with pytest.raises(ValueError):
        summary.stats

    irad = 0.5 * u.deg
    orad = 0.7 * u.deg
    summary.estimate_background(method='ring', inner_radius=irad,
                                outer_radius=orad)

    stats = summary.stats
    assert stats.n_on == 432
