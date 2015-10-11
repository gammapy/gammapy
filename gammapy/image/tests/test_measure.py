# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_almost_equal
from astropy.tests.helper import pytest
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from ...image import (measure_labeled_regions,
                      make_empty_image, lookup,
                      measure_containment_radius,
                      measure_image_moments,
                      measure_containment,
                      measure_curve_of_growth,
                      coordinates)

BINSZ = 0.02

try:
    import scipy
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def generate_example_image():
    """
    Generate some greyscale image to run the detection on.
    """
    x = y = np.linspace(0, 3 * np.pi, 100)
    X, Y = np.meshgrid(x, y)
    image = X * Y * np.sin(X) ** 2 * np.sin(Y) ** 2
    return image


def set_header(image):
    image.header['SIMPLE'] = 'T'
    image.header['BITPIX'] = -64
    image.header['NAXIS'] = 2
    image.header['NAXIS1'] = 201
    image.header['NAXIS2'] = 201
    image.header['CTYPE1'] = 'GLON-CAR'
    image.header['CTYPE2'] = 'GLAT-CAR'
    image.header['CRPIX1'] = 100
    image.header['CRPIX2'] = 100
    image.header['CRVAL1'] = 0
    image.header['CRVAL2'] = 0
    image.header['CDELT1'] = -BINSZ
    image.header['CDELT2'] = BINSZ
    image.header['CUNIT1'] = 'deg'
    image.header['CUNIT2'] = 'deg'
    return image


def generate_gaussian_image():
    """
    Generate some greyscale image to run the detection on.
    """
    image = fits.ImageHDU(data=np.zeros((201, 201)))
    image = set_header(image)
    GLON, GLAT = coordinates(image, lon_sym=True)
    sigma = 0.2
    source = Gaussian2D(1. / (2 * np.pi * (sigma / BINSZ) ** 2), 0, 0, sigma, sigma)
    image.data += source(GLON, GLAT)
    return image


@pytest.mark.skipif('not HAS_SCIPY')
def test_measure():
    image = generate_example_image()
    labels = np.zeros_like(image, dtype=int)
    labels[10:20, 20:30] = 1
    results = measure_labeled_regions(image, labels)
    # TODO: check output!


def test_measure_image_moments():
    """Test measure_image_moments function"""
    image = generate_gaussian_image()
    moments = measure_image_moments(image)
    assert_almost_equal(moments, [1, 0, 0, 0.2, 0.2, 0.2])


def test_measure_containment():
    """Test measure_containment function"""
    image = generate_gaussian_image()
    frac = measure_containment(image, 0, 0, 0.2 * np.sqrt(2 * np.log(5)))
    assert_allclose(frac, 0.8, rtol=0.01)


@pytest.mark.skipif('not HAS_SCIPY')
def test_measure_containment_radius():
    """Test measure_containment_radius function"""
    image = generate_gaussian_image()
    rad = measure_containment_radius(image, 0, 0, 0.8)
    assert_allclose(rad, 0.2 * np.sqrt(2 * np.log(5)), rtol=0.01)


def test_measure_curve_of_growth():
    """Test measure_curve_of_growth function"""
    image = generate_gaussian_image()
    radius, containment = measure_curve_of_growth(image, 0, 0, 0.6, 0.05)
    sigma = 0.2
    containment_ana = 1 - np.exp(-0.5 * (radius / sigma) ** 2)
    assert_allclose(containment, containment_ana, rtol=0.1)


class _TestImageCoordinates(object):
    def setUp(self):
        self.image = make_empty_image(nxpix=3, nypix=2,
                                      binsz=10, proj='CAR')
        self.image.dat = np.arange(3 * 2).reshape(self.image.dat.shape)

    def test_lookup(self):
        self.assertEqual(lookup(self.image, 1, 1, world=False), 0)
        assert_equal(lookup(self.image, 5, 1, world=False), np.nan)
        self.assertEqual(lookup(self.image, 3, 2, world=False), 5)
        assert_equal(lookup(self.image, [1, 5], [1, 1], world=False), [0, np.nan])
