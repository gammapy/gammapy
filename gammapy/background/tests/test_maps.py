# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
TODO: add tests for different cases of theta and is_off_correlated
"""
from __future__ import print_function, division
from astropy.tests.helper import pytest
import unittest
import numpy as np
from astropy.io import fits
from ..maps import Maps

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.mark.xfail
@pytest.mark.skipif('not HAS_SCIPY')
class TestMaps(unittest.TestCase):
    # TODO: use astropy temp file utils
    dir = '/tmp/'
    filename_basic = dir + 'maps_basic.fits'
    filename_derived = dir + 'maps_derived.fits'

    def setUp(self):
        """Make an example file containing a Maps object."""
        # Parameters
        shape = (300, 300)
        background = 1
        pos1 = (100, 100)
        signal1 = 100
        pos2 = (200, 200)
        signal2 = 100
        a_on = 1
        exclusion_dist = 10
        # Make an example on map containing background and two sources
        n_on_data = background * np.ones(shape)
        n_on_data[pos1] += signal1
        n_on_data[pos2] += signal2
        n_on_hdu = fits.ImageHDU(n_on_data, name='n_on')
        # Make and example onexposure map
        onexposure_data = a_on * np.ones(shape)
        onexposure_hdu = fits.ImageHDU(onexposure_data, name='a_on')
        # Make an example exclusion map that excludes source 1,
        # but not source 2
        y, x = np.indices(shape)
        dist1 = np.sqrt((x - pos1[0]) ** 2 + (y - pos1[1]) ** 2)
        exclusion_data = np.where(dist1 < exclusion_dist, 0, 1)
        exclusion_hdu = fits.ImageHDU(exclusion_data, name='exclusion')
        # Make a BgMaps object and write it to FITS file
        maps = Maps([n_on_hdu, onexposure_hdu, exclusion_hdu])
        maps.writeto(self.filename_basic, clobber=True)

    def test_make_derived_maps(self):
        """Read Maps containing only basic maps from file,
        add derived maps and write to another file."""
        maps = Maps(fits.open(self.filename_basic))
        maps.theta = 10
        maps.is_off_correlated = False
        maps.make_derived_maps()
        maps.writeto(self.filename_derived, clobber=True)
