"""@todo: add tests for different cases of theta and is_off_correlated"""
import unittest
import numpy as np
from astropy.io import fits
from ..maps import Maps


class TestMaps(unittest.TestCase):
    dir = '/tmp/'
    filename_basic = dir + 'bgmaps_basic.fits'
    filename_derived = dir + 'bgmaps_derived.fits'

    def setUp(self):
        """Make an example file containing a BgMaps object."""
        # Parameters
        shape = (300, 300)
        background = 1
        pos1 = (100, 100)
        signal1 = 100
        pos2 = (200, 200)
        signal2 = 100
        onexposure = 1
        exclusion_dist = 10
        # Make an example on map containing background and two sources
        on_data = background * np.ones(shape)
        on_data[pos1] += signal1
        on_data[pos2] += signal2
        on_hdu = fits.ImageHDU(on_data, name='on')
        # Make and example onexposure map
        onexposure_data = onexposure * np.ones(shape)
        onexposure_hdu = fits.ImageHDU(onexposure_data, name='onexposure')
        # Make an example exclusion map that excludes source 1,
        # but not source 2
        y, x = np.indices(shape)
        dist1 = np.sqrt((x - pos1[0]) ** 2 + (y - pos1[1]) ** 2)
        exclusion_data = np.where(dist1 < exclusion_dist, 0, 1)
        exclusion_hdu = fits.ImageHDU(exclusion_data, name='exclusion')
        # Make a BgMaps object and write it to FITS file
        maps = Maps([on_hdu, onexposure_hdu, exclusion_hdu])
        maps.writeto(self.filename_basic, clobber=True)

    def test_make_derived_maps(self):
        """Read BgMaps containing only basic maps from file,
        add derived maps and write to another file."""
        maps = Maps(fits.open(self.filename_basic))
        maps.theta = 10
        maps.is_off_correlated = False
        maps.make_derived_maps()
        maps.writeto(self.filename_derived, clobber=True)

    def tearDown(self):
        """Remove the example files"""
        # @todo Implement
        pass
