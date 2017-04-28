# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from .. import Target
from astropy.coordinates import SkyCoord
import astropy.units as u
from regions import CircleSkyRegion


def test_target():
    pos = SkyCoord(83.63 * u.deg, 22.01 * u.deg, frame='icrs')
    on_size = 0.3 * u.deg
    on_region = CircleSkyRegion(pos, on_size)
    target = Target(on_region, name='Test Target', obs_id=[23523, 23592])

    assert 'Target' in str(target)
