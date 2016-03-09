# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from ..maps import SkyMap


class TestSkyMap():
    """
    Test sky map class.
    """
    def setup(self):
        self.skymap = SkyMap.empty()
        
    def test_empty():
        assert self.skymap.data.shape = (200, 200)

    def test_read():
        pass

    def test_write():
        pass

    def test_lookup():
        pass

    def test_coordinates():
        pass

    def test_info():
        pass

    def test_to_quantity():
        pass

