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
    SkyTemplate,
)


def test_sky_point_source():
    model = SkyPointSource(
        lon_0='1 deg',
        lat_0='45 deg',
    )
    lon = [1, 1.1] * u.deg
    lat = 45 * u.deg
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
        r_i='2 deg',
        r_o='4 deg',
    )

    lon = [1, 2, 4] * u.deg
    lat = 45 * u.deg
    val = model(lon, lat)
    assert val.unit == 'sr-1'
    desired = [55.979449, 57.831651, 94.919895]
    assert_allclose(val.value, desired)


@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_sky_template():
    filename = ('$GAMMAPY_EXTRA/datasets/catalogs/fermi/Extended_archive_v18'
                '/Templates/HESSJ1841-055.fits')
    template = SkyTemplate.read(filename)
    lon = 26.7 * u.deg
    lat = 0 * u.deg
    actual = template(lon, lat)[0]
    desired = 0.00017506744188722223 / u.deg ** 2
    # desired = 1.1553735159851262 / u.deg ** 2
    assert_allclose(actual, desired)
