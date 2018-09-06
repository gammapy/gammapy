# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.coordinates import Angle
from ...data import observatory_locations


def test_observatory_locations():
    # Check one example
    location = observatory_locations["hess"]
    assert_allclose(location.lon.deg, Angle("16d30m00.8s").deg)
    assert_allclose(location.lat.deg, Angle("-23d16m18.4s").deg)
    assert_allclose(location.height.value, 1835)
    assert str(location.height.unit) == "m"
