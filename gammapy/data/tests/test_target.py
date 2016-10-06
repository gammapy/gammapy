# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from .. import Target
from astropy.coordinates import SkyCoord
import astropy.units as u
from regions import CircleSkyRegion
from ...utils.testing import data_manager, requires_data, requires_dependency


@requires_dependency('yaml')
@requires_data('gammapy-extra')
def test_targetsummary(data_manager):
    pos = SkyCoord(83.63 * u.deg, 22.01 * u.deg, frame='icrs')
    on_size = 0.3 * u.deg
    on_region = CircleSkyRegion(pos, on_size)
    target = Target(on_region, name='Test Target', obs_id=[23523, 23592])

    assert 'Target' in str(target)
