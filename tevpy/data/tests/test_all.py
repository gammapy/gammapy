# Licensed under a 3-clause BSD style license - see LICENSE.rst
from numpy.testing import assert_almost_equal
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


def test_poisson_stats_extra_info():
    images = poisson_stats_image(extra_info=True)
    refs = dict(counts=40896, model=41000, source=1000, background=40000)
    for name, expected in refs.items():
        assert_almost_equal(images[name].sum(), expected) 