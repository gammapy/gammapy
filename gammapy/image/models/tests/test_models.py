# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
import astropy.units as u
from ....utils.testing import requires_dependency, requires_data
from ..new import (
    SkyPointSource,
    SkyGaussian,
    SkyDisk,
    SkyShell,
    SkyDiffuseConstant,
    SkyDiffuseMap,
)


def test_sky_point_source():
    model = SkyPointSource(
        lon_0='1 deg',
        lat_0='45 deg',
    )
    lon = [1, 1.1] * u.deg
    lat = [45, 45.2] * u.deg
    val = model(lon, lat)
    assert val.unit == 'sr-1'
    assert_allclose(val.value, [1, 0])


def test_sky_gaussian():
    model = SkyGaussian(
        lon_0='1 deg',
        lat_0='45 deg',
        sigma='1 deg',
    )
    lon = [1, 359] * u.deg
    lat = 46 * u.deg
    val = model(lon, lat)
    assert val.unit == 'sr-1'
    assert_allclose(val.value, [316.8970202, 118.6505303])


def test_sky_disk():
    model = SkyDisk(
        lon_0='1 deg',
        lat_0='45 deg',
        r_0='2 deg',
    )
    lon = [1, 5, 359] * u.deg
    lat = 46 * u.deg
    val = model(lon, lat)
    assert val.unit == 'sr-1'
    desired = [261.263956, 0, 261.263956]
    assert_allclose(val.value, desired)


def test_sky_shell():
    model = SkyShell(
        lon_0='1 deg',
        lat_0='45 deg',
        radius='2 deg',
        width='2 deg',
    )

    lon = [1, 2, 4] * u.deg
    lat = 45 * u.deg
    val = model(lon, lat)
    assert val.unit == 'sr-1'
    desired = [55.979449, 57.831651, 94.919895]
    assert_allclose(val.value, desired)


def test_sky_diffuse_constant():
    model = SkyDiffuseConstant(
        value='42 sr-1'
    )
    lon = [1, 2] * u.deg
    lat = 45 * u.deg
    val = model(lon, lat)
    assert val.unit == 'sr-1'
    assert_allclose(val.value, 42)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_sky_diffuse_map():
    filename = ('$GAMMAPY_EXTRA/datasets/catalogs/fermi/Extended_archive_v18'
                '/Templates/RXJ1713_2016_250GeV.fits')
    model = SkyDiffuseMap.read(filename)
    lon = [258.5, 0] * u.deg
    lat = -39.8 * u.deg
    val = model(lon, lat)
    assert val.unit == 'sr-1'
    desired = [3348.0417, 460.92587]
    assert_allclose(val.value, desired)

    # TODO: add more tests:
    # - different model `norm` parameter values / units and map units
    # - evaluate outside the map: why do we above not get 0 or NaN?
    # - make an input map from scratch with known values
