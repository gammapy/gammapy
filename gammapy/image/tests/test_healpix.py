# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing.utils import assert_allclose
from ...utils.testing import requires_dependency
from ..core import SkyImage
from ..healpix import SkyImageHealpix, WCSHealpix


@requires_dependency('healpy')
class TestSkyImageHealpix:
    def setup(self):
        wcs = WCSHealpix(nside=2)
        data = np.ones(wcs.npix)
        self.skyimage_hpx = SkyImageHealpix(name='test', data=data, wcs=wcs)

    @requires_dependency('reproject')
    def test_reproject(self):
        reference = SkyImage.empty(nxpix=21, nypix=11, binsz=1.)
        reprojected = self.skyimage_hpx.reproject(reference)
        assert_allclose(reprojected.data, 1.)
