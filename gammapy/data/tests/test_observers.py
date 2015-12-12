# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.coordinates import Angle
from ...data import observatory_locations


def test_ObservatoryLocations():
    # Check if attribute and key access works
    # and if all fields are set correctly for one example
    location = observatory_locations.HESS
    assert_allclose(location.longitude.deg, Angle('16d30m00.8s').deg)

    location = observatory_locations['HESS']
    assert_allclose(location.latitude.deg, Angle('-23d16m18.4s').deg)

    assert_allclose(location.height.to('meter').value, 1835)
