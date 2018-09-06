# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pytest
from astropy import units as u
from astropy.modeling.models import Gaussian2D
from astropy.coordinates import SkyCoord
from ...utils.testing import assert_quantity_allclose, requires_dependency
from ...maps import WcsGeom, WcsNDMap
from ...image import (
    measure_containment_radius,
    measure_image_moments,
    measure_containment,
    measure_curve_of_growth,
)


@pytest.fixture(scope="session")
def gaussian_image():
    """Generate gaussian test image.
    """
    binsz = 0.02
    sigma = 0.2
    geom = WcsGeom.create(npix=(201, 201), binsz=binsz, coordsys="GAL")
    gauss = Gaussian2D(1. / (2 * np.pi * (sigma / binsz) ** 2), 0, 0, sigma, sigma)
    coord = geom.get_coord().skycoord
    l = coord.data.lon.wrap_at("180d")
    b = coord.data.lat
    data = gauss(l.degree, b.degree)
    return WcsNDMap(geom=geom, data=data, unit="cm-2 s-1")


def test_measure_image_moments(gaussian_image):
    """Test measure_image_moments function"""
    moments = measure_image_moments(gaussian_image)

    reference = [
        1 * u.Unit("cm-2 s-1"),
        0 * u.deg,
        0 * u.deg,
        0.2 * u.deg,
        0.2 * u.deg,
        0.2 * u.deg,
    ]

    for val, ref in zip(moments, reference):
        assert_quantity_allclose(val, ref, atol=1e-12 * val.unit)


def test_measure_containment(gaussian_image):
    """Test measure_containment function"""
    position = SkyCoord(0, 0, frame="galactic", unit="deg")
    radius = u.Quantity(0.2 * np.sqrt(2 * np.log(5)), "deg")
    frac = measure_containment(gaussian_image, position, radius)
    ref = u.Quantity(0.8, "cm-2 s-1")
    assert_quantity_allclose(frac, ref, rtol=0.01)


@requires_dependency("scipy")
def test_measure_containment_radius(gaussian_image):
    """Test measure_containment_radius function"""
    position = SkyCoord(0, 0, frame="galactic", unit="deg")
    rad = measure_containment_radius(gaussian_image, position, 0.8)
    ref = 0.2 * np.sqrt(2 * np.log(5)) * u.deg
    assert_quantity_allclose(rad, ref, rtol=0.01)


def test_measure_curve_of_growth(gaussian_image):
    """Test measure_curve_of_growth function"""
    position = SkyCoord(0, 0, frame="galactic", unit="deg")
    radius_max = 0.6 * u.deg
    radius, containment = measure_curve_of_growth(gaussian_image, position, radius_max)
    sigma = 0.2 * u.deg
    containment_ana = u.Quantity(
        1 - np.exp(-0.5 * (radius / sigma) ** 2).value, "cm-2 s-1"
    )
    assert_quantity_allclose(containment, containment_ana, rtol=0.1)
