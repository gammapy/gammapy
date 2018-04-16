# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals

from ..new import SkyGaussian2D, SkyPointSource
from astropy.tests.helper import assert_quantity_allclose
import pytest
import astropy.units as u


def test_skygauss2D():
    model = SkyGaussian2D(
        x_0=359 * u.deg,
        y_0=88 * u.deg,
        sigma=1 * u.deg,
    )
    # Coordinates are chose such that 360 - 0 offset is tested
    lon = 1 * u.deg
    lat = 89 * u.deg
    actual = model(lon, lat)
    desired = 0.0964148382898712 / u.deg ** 2,
    assert_quantity_allclose(actual, desired)


def test_skypointsource():
    model = SkyPointSource(
        x_0=359 * u.deg,
        y_0=88 * u.deg,
    )
    lon = 359 * u.deg
    lat = 88 * u.deg
    actual = model(lon, lat)
    desired = 1
    assert_quantity_allclose(actual, desired)

    lon = 359.1 * u.deg
    lat = 88 * u.deg
    actual = model(lon, lat)
    desired = 0
    assert_quantity_allclose(actual, desired)
