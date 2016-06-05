# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from .. import Target, TargetSummary
from astropy.coordinates import SkyCoord
from astropy.tests.helper import pytest
import astropy.units as u
from ...extern.regions.shapes import CircleSkyRegion
from ...utils.testing import data_manager, requires_data, requires_dependency

@requires_dependency('yaml')
@requires_data('gammapy-extra')
def test_targetsummary(data_manager):
    pos = SkyCoord(83.63 * u.deg, 22.01 * u.deg, frame='icrs')
    on_size = 0.3 * u.deg
    on_region = CircleSkyRegion(pos, on_size)
    target = Target(pos, on_region, name='Test Target', obs_id=[23523, 23592])
    
    data_store = data_manager['hess-crab4-hd-hap-prod2']
    target.add_obs_from_store(data_store) 

    irad = 0.5 * u.deg
    orad = 0.7 * u.deg
    target.estimate_background(method='ring', inner_radius=irad, outer_radius=orad)

    summary = TargetSummary(target)

    stats = summary.stats
    assert stats.n_on == 432
