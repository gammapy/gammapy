# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import astropy.units as u
import numpy as np
from ....utils.testing import assert_quantity_allclose
from ....utils.testing import requires_dependency, requires_data
from ..new import (
    SkyGaussian2D, SkyPointSource, SkyDisk2D, SkyShell2D, SkyTemplate2D,
)


def test_skygauss2D():
    model = SkyGaussian2D(
        lon_0='359 deg',
        lat_0='88 deg',
        sigma='1 deg',
    )
    # Coordinates are chose such that 360 - 0 offset is tested
    lon = 1 * u.deg
    lat = 89 * u.deg
    actual = model(lon, lat)
    desired = 0.0964148382898712 / u.deg ** 2,
    assert_quantity_allclose(actual, desired)

    lons = np.arange(-5, 5) * u.deg
    lats = np.arange(0, 5) * u.deg
    lon, lat = np.meshgrid(lons, lats)
    actual = model(lon, lat)
    assert actual.shape == (5, 10)


def test_skypointsource():
    model = SkyPointSource(
        lon_0='359 deg',
        lat_0='88 deg',
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


def test_skydisk2D():
    model = SkyDisk2D(
        lon_0='359 deg',
        lat_0='88 deg',
        r_0='2 deg',
    )
    lon = 180 * u.deg
    lat = 88 * u.deg
    actual = model(lon, lat)
    desired = 0 / u.deg ** 2
    assert_quantity_allclose(actual, desired)

    lon = 358 * u.deg
    lat = 88 * u.deg
    actual = model(lon, lat)
    desired = 261.26395634890207 / u.deg ** 2
    assert_quantity_allclose(actual, desired)

    lons = np.arange(-5, 5) * u.deg
    lats = np.arange(0, 5) * u.deg
    lon, lat = np.meshgrid(lons, lats)
    actual = model(lon, lat)
    assert actual.shape == (5, 10)


def test_skyshell2D():
    model = SkyShell2D(
        lon_0='359 deg',
        lat_0='88 deg',
        r_i='2 deg',
        r_o='4 deg',
    )

    lon = 180 * u.deg
    lat = 88 * u.deg
    actual = model(lon, lat)
    desired = 0.0002976757280439522 / u.deg ** 2
    assert_quantity_allclose(actual, desired)

    lons = np.arange(-5, 5) * u.deg
    lats = np.arange(0, 5) * u.deg
    lon, lat = np.meshgrid(lons, lats)
    actual = model(lon, lat)
    assert actual.shape == (5, 10)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_template2d():
    filename = ('$GAMMAPY_EXTRA/datasets/catalogs/fermi/Extended_archive_v18'
                '/Templates/HESSJ1841-055.fits')
    template = SkyTemplate2D.read(filename)
    lon = 26.7 * u.deg
    lat = 0 * u.deg
    actual = template(lon, lat)[0]
    desired = 0.00017506744188722223 / u.deg ** 2
    #desired = 1.1553735159851262 / u.deg ** 2
    assert_allclose(actual, desired)
