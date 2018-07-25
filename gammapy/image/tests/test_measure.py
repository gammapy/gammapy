# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pytest
from astropy.units import Quantity
from astropy.modeling.models import Gaussian2D
from astropy.coordinates import SkyCoord
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_dependency
from ...image import (measure_containment_radius,
                      measure_image_moments,
                      measure_containment,
                      measure_curve_of_growth, SkyImage)

BINSZ = 0.02



@pytest.fixture
def gaussian_image():
    """
    Generate some greyscale image to run the detection on.
    """
    image = SkyImage.empty(nxpix=201, nypix=201, binsz=0.02)
    coordinates = image.coordinates()
    l = coordinates.data.lon.wrap_at("180d")
    b = coordinates.data.lat
    sigma = 0.2
    source = Gaussian2D(1. / (2 * np.pi * (sigma / BINSZ) ** 2), 0, 0, sigma, sigma)
    image.data += source(l.degree, b.degree)
    image.data = Quantity(image.data, 'cm-2 s-1')
    return image


def test_measure_image_moments(gaussian_image):
    """Test measure_image_moments function"""
    moments = measure_image_moments(gaussian_image)
    reference = [Quantity(1, 'cm-2 s-1'),
                 Quantity(0, 'deg'),
                 Quantity(0, 'deg'),
                 Quantity(0.2, 'deg'),
                 Quantity(0.2, 'deg'),
                 Quantity(0.2, 'deg')]

    for val, ref in zip(moments, reference):
        assert_quantity_allclose(val, ref, atol=Quantity(1e-12, val.unit))


def test_measure_containment(gaussian_image):
    """Test measure_containment function"""
    position = SkyCoord(0, 0, frame='galactic', unit='deg')
    radius = Quantity(0.2 * np.sqrt(2 * np.log(5)), 'deg')
    frac = measure_containment(gaussian_image, position, radius)
    ref = Quantity(0.8, 'cm-2 s-1')
    assert_quantity_allclose(frac, ref, rtol=0.01)


@requires_dependency('scipy')
def test_measure_containment_radius(gaussian_image):
    """Test measure_containment_radius function"""
    position = SkyCoord(0, 0, frame='galactic', unit='deg')
    rad = measure_containment_radius(gaussian_image, position, 0.8)
    ref = Quantity(0.2 * np.sqrt(2 * np.log(5)), 'deg')
    assert_quantity_allclose(rad, ref, rtol=0.01)


def test_measure_curve_of_growth(gaussian_image):
    """Test measure_curve_of_growth function"""
    position = SkyCoord(0, 0, frame='galactic', unit='deg')
    radius_max = Quantity(0.6, 'deg')
    radius, containment = measure_curve_of_growth(gaussian_image, position, radius_max)
    sigma = Quantity(0.2, 'deg')
    containment_ana = Quantity(1 - np.exp(-0.5 * (radius / sigma) ** 2).value, 'cm-2 s-1')
    assert_quantity_allclose(containment, containment_ana, rtol=0.1)
