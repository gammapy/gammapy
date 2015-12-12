# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.units import Quantity
from ...utils.testing import requires_data
from ...datasets import gammapy_extra
from ...data import TelescopeArray, observatory_locations


@requires_data('gammapy-extra')
def test_TelescopeArray():
    filename = gammapy_extra.filename('test_datasets/unbundled/hess/run_0023037_hard_eventlist.fits.gz')
    telescope_array = TelescopeArray.read(filename, hdu='TELARRAY')
    assert 'Telescope array info' in telescope_array.summary
    location = telescope_array.observatory_earth_location.geocentric
    hess_location = observatory_locations.HESS.geocentric
    offset = Quantity(location) - Quantity(hess_location)
    assert_allclose(offset.value, 0, atol=1e-3)
