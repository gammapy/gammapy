# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from astropy.tests.helper import pytest
from astropy.io import fits
from .. import ReflectedRegionMaker
from ...utils.testing import requires_data
from ...datasets import gammapy_extra

@requires_data('gammapy-extra')
def test_ReflectedRegionMaker():
    exclfile = gammapy_extra.filename('test_datasets/spectrum/dummy_exclusion.fits')
    exclusion = fits.open(exclfile, hdu = 0)[0]
    fov = {'x' : 82.87, 'y' : 23.24, 'r' : 10}
    maker = ReflectedRegionMaker(exclusion, fov)
    x_on, y_on, r_on = 80.2, 23.5, 0.3
    regions = maker.compute(x_on, y_on, r_on)
