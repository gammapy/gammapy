# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from numpy.testing import assert_allclose
from astropy.units import Quantity
from ...datasets import get_path
from ...data import TelescopeArray
from ...obs import observatory_locations

filename = get_path('hess/run_0023037_hard_eventlist.fits.gz')


def test_TelescopeArray():
    telescope_array = TelescopeArray.read(filename, hdu='TELARRAY')
    assert 'Telescope array info' in telescope_array.summary
    location = telescope_array.observatory_earth_location.geocentric
    hess_location = observatory_locations.HESS.geocentric
    offset = Quantity(location) - Quantity(hess_location)
    assert_allclose(offset, 0, atol=1e-3)  # meter
