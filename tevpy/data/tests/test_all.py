# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from .. import poisson_stats_image

def test_poisson_stats_image():
    """Get the data file via the tevpy.data.poisson_stats_image function"""
    data = poisson_stats_image()
    assert data.sum() == 40896

def test_poisson_stats_image_direct():
    """Get the data file directly via get_pkg_data_filename"""
    filename = get_pkg_data_filename('../poisson_stats_image/counts.fits.gz')
    data = fits.getdata(filename)
    assert data.sum() == 40896
